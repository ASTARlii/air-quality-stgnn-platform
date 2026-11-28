"""基线模型骨架（神经网络类；统计/机器学习基线走 sklearn，另立文件）。

参考开题报告 6.2 对比基线：
- LSTM：单站点时序基线，证明空间结构有用（关键消融）
- STGCN：同赛道时空图基线
- ARIMA / XGBoost 不在本文件（非 PyTorch 模型）
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """单站点 LSTM：对每个站点独立预测，不共享空间信息。

    输入 (B, N, T_in, C_in)，输出 (B, N, T_out)。
    内部把 (B*N) 视为独立 batch，即 LSTM 不感知空间。
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden: int = 64,
        num_layers: int = 2,
        horizon: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T_in, C_in)。返回 (B, N, T_out)。"""
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C)
        h, _ = self.lstm(x)
        out = self.fc(h[:, -1, :])               # (B*N, T_out)
        return out.reshape(B, N, self.horizon)


class STGCNBaseline(nn.Module):
    """STGCN（Yu et al. 2018）：三明治结构 TCN-GCN-TCN。

    与 Graph WaveNet 的差异：无自适应邻接、无门控、无跳跃累加。
    骨架阶段先实现占位 forward，留待后续细化。
    """

    def __init__(
        self,
        in_dim: int = 9,
        hidden: int = 64,
        horizon: int = 24,
        n_nodes: int = 24,
        static_supports: list[torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.horizon = horizon
        self.n_nodes = n_nodes
        self.static_supports = static_supports or []

        # 占位：实际应为 ST-Conv Block × 2 + Output Layer
        self.start = nn.Conv2d(in_dim, hidden, kernel_size=(1, 1))
        self.end = nn.Conv2d(hidden, horizon, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C_in, N, T_in)。返回 (B, horizon, N, 1)。"""
        # TODO: 实现完整 ST-Conv Block（TCN → GCN → TCN），骨架阶段先用端到端简化路径
        h = self.start(x)
        h = torch.relu(h)
        h = self.end(h)
        return h[..., -1:]                        # (B, horizon, N, 1)
