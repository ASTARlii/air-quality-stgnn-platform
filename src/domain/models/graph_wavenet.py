"""Graph WaveNet 骨架（参考 nnzhan/Graph-WaveNet 二次开发，不引入其框架依赖）。

架构（开题报告 6.1）：
    输入 (B, C_in=9, N=24, T_in=24)
        │ start_conv (1×1)：升维到 residual_channels
        ▼
    [ blocks × layers ] 个 Gated-TCN + Graph-Conv 单元
        ├─ Gated TCN: tanh(filter(x)) * sigmoid(gate(x))
        ├─ skip:    每层产出累加到 skip tensor
        ├─ GCN:     A @ X，A = 静态图（地理距离）+ 自适应邻接（E1@E2ᵀ softmax）
        ├─ Residual: x += residual（时序对齐取末端）
        └─ Norm:    LayerNorm
        ▼
    end_conv_1 → end_conv_2：输出 (B, T_out, N, 1)，T_out=horizon

关键设计：
- 自适应邻接矩阵：node_embeddings (N, apt_dim)，A_adp = softmax(relu(E1 @ E2ᵀ))
- 4 个独立单输出模型：PM2.5 / PM10 / NO2 / O3 各训一个，仅 target 不同
- 直接多输出：一次前向返回未来 horizon 步，避免误差累积
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 基础组件 ----------

class NConv(nn.Module):
    """图扩散卷积：X' = Σ_k A^k @ X。k=0 时退化为恒等映射。"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N, T), adj: (N, N)。返回 (B, C, N, T)。"""
        # 在节点维做矩阵乘：(N,N) @ (B,C,N,T) -> 先 reshape 到 (B*C*T, N) 再右乘 adj.T
        B, C, N, T = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B, C, T, N)   # (B,C,T,N)
        x = x @ adj.transpose(-1, -2)                    # 节点维乘
        return x.permute(0, 1, 3, 2)                     # 回 (B,C,N,T)


class GraphConv(nn.Module):
    """nconv + Linear，并支持静态 + 自适应多邻接求和。"""

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0, order: int = 2) -> None:
        super().__init__()
        self.nconv = NConv()
        self.order = order
        self.mlp = nn.Linear((order + 1) * c_in, c_out)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        """x: (B,C,N,T)，supports: 邻接矩阵列表 [(N,N), ...]。

        多支撑（静态图 + 自适应图）时逐跳对各支撑的扩散结果取平均，
        通道维恒为 (order+1)*C，不随支撑数量变化（单支撑时与原实现完全一致）。
        """
        xs = [x]
        for _ in range(1, self.order + 1):
            acc = None
            for adj in supports:
                t = self.nconv(xs[-1], adj)
                acc = t if acc is None else acc + t
            xs.append(acc / len(supports))
        h = torch.cat(xs, dim=1)                         # (B, (order+1)*C, N, T)
        h = h.permute(0, 2, 3, 1)                        # (B,N,T,C*)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h.permute(0, 3, 1, 2)                     # (B,C_out,N,T)


class GatedTCN(nn.Module):
    """门控时序卷积：tanh(filter) * sigmoid(gate)，dilation 逐层翻倍。"""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation  # 因果 padding
        self.filter_conv = nn.Conv2d(c_in, c_out, (1, kernel_size), dilation=(1, dilation), padding=(0, pad))
        self.gate_conv = nn.Conv2d(c_in, c_out, (1, kernel_size), dilation=(1, dilation), padding=(0, pad))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,C,N,T)。注意 padding 在左（因果）。"""
        f = torch.tanh(self.filter_conv(x))
        g = torch.sigmoid(self.gate_conv(x))
        # 去掉右侧多余 padding 保证因果
        if f.size(-1) != x.size(-1):
            f = f[..., -x.size(-1):]
            g = g[..., -x.size(-1):]
        out = f * g
        return F.dropout(out, self.dropout, training=self.training)


# ---------- 主模型 ----------

class GraphWaveNet(nn.Module):
    """Graph WaveNet 主类。

    参数默认值参考开题报告：in_dim=9（4 污染物 + 5 气象），n_nodes=24，horizon=24，
    隐藏维度 32，blocks=4 / layers=2（感受野 = 2 + 2*kernel*(2^layers-1)*blocks ≥ T_in）。
    """

    def __init__(
        self,
        in_dim: int = 9,
        out_dim: int = 24,         # 直接多输出：horizon
        n_nodes: int = 24,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 256,
        end_channels: int = 512,
        kernel_size: int = 2,
        blocks: int = 4,
        layers: int = 2,
        dropout: float = 0.3,
        apt_dim: int = 10,         # 自适应嵌入维度（消融对象）
        use_adaptive_adj: bool = True,
        use_graph_conv: bool = True,   # False=纯 TCN 消融（跳过图卷积，仅保留门控时序卷积）
        static_supports: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_nodes = n_nodes
        self.blocks = blocks
        self.layers = layers
        self.use_adaptive_adj = use_adaptive_adj
        self.use_graph_conv = use_graph_conv
        self.dropout = dropout

        if not use_graph_conv and residual_channels != dilation_channels:
            raise ValueError(
                "纯 TCN（use_graph_conv=False）要求 residual_channels == dilation_channels"
            )

        # 自适应邻接：节点嵌入 E1/E2，A = softmax(relu(E1 @ E2ᵀ))
        if use_adaptive_adj:
            self.emb1 = nn.Parameter(torch.randn(n_nodes, apt_dim), requires_grad=True)
            self.emb2 = nn.Parameter(torch.randn(n_nodes, apt_dim), requires_grad=True)

        self.static_supports = static_supports or []

        # 1×1 升维
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))

        # blocks × layers 个 (GatedTCN + GraphConv)
        self.tcn_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        receptive_field = 1
        for b in range(blocks):
            for l in range(layers):
                dilation = 2 ** l
                self.tcn_layers.append(
                    GatedTCN(residual_channels, dilation_channels, kernel_size, dilation, dropout)
                )
                if use_graph_conv:
                    self.gcn_layers.append(
                        GraphConv(dilation_channels, residual_channels, dropout=dropout, order=2)
                    )
                else:
                    self.gcn_layers.append(nn.Identity())   # 纯 TCN 占位，保持逐层 zip 对齐
                self.norm_layers.append(nn.LayerNorm(residual_channels))
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
                )
                receptive_field += (kernel_size - 1) * dilation
        self.receptive_field = receptive_field

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N, T_in)。返回 (B, out_dim, N, 1)。"""
        # 时间窗裁剪：感受野以内
        if x.size(-1) > self.receptive_field:
            x = x[..., -self.receptive_field:]

        x = self.start_conv(x)
        skip = None

        # 自适应邻接（每步重算，跟随 emb 更新）
        supports = list(self.static_supports)
        if self.use_adaptive_adj:
            adp = F.softmax(F.relu(self.emb1 @ self.emb2.T), dim=1)
            supports = supports + [adp]

        for tcn, gcn, norm, skip_c in zip(
            self.tcn_layers, self.gcn_layers, self.norm_layers, self.skip_convs
        ):
            residual = x
            x = tcn(x)                                  # (B, dilation_C, N, T)
            s = skip_c(x)                               # (B, skip_C, N, T)
            skip = s if skip is None else skip + s      # 累加跳跃连接
            if self.use_graph_conv:
                x = gcn(x, supports)                    # (B, residual_C, N, T)
            x = x + residual[..., -x.size(-1):]         # 残差（取末端对齐）
            x = norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LayerNorm 在 channel 维

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)                          # (B, out_dim, N, T)
        x = x[..., -1:]                                 # 取最后一个时间步，得到 (B, out_dim, N, 1)
        return x


def build_model_from_config(cfg: dict, static_supports: list[torch.Tensor] | None = None) -> GraphWaveNet:
    """从 configs/model.yaml 构造模型。

    static_supports: 可选的静态邻接（如转移熵图），与自适应邻接共同参与图卷积。
    """
    return GraphWaveNet(
        in_dim=cfg.get("in_dim", 9),
        out_dim=cfg.get("horizon", 24),
        n_nodes=cfg.get("n_nodes", 24),
        residual_channels=cfg.get("residual_channels", 32),
        dilation_channels=cfg.get("dilation_channels", 32),
        skip_channels=cfg.get("skip_channels", 256),
        end_channels=cfg.get("end_channels", 512),
        kernel_size=cfg.get("kernel_size", 2),
        blocks=cfg.get("blocks", 4),
        layers=cfg.get("layers", 2),
        dropout=cfg.get("dropout", 0.3),
        apt_dim=cfg.get("apt_dim", 10),
        use_adaptive_adj=cfg.get("use_adaptive_adj", True),
        use_graph_conv=cfg.get("use_graph_conv", True),
        static_supports=static_supports,
    )
