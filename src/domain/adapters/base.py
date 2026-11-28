"""两个核心适配器接口（对应概要设计 03 第 3 节决策 #2）。

- ModelAdapter：统一 Graph WaveNet 与各基线的预测入口，让上层 service 不感知模型差异；
- NetworkBuilder：统一转移熵、滞后相关、风场先验等网络构建实现。

所有适配器实现必须遵循这两个 Protocol；替换算法时不动 service 层。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np
import torch


# ---------- 类型别名 ----------

# 张量形状约定：
#   batch_x: (B, T_in, N, C_in)      B=批大小  T_in=输入窗长  N=站点数  C_in=输入通道
#   batch_y: (B, T_out, N, 1)        T_out=预测步长（直接多输出，一次出未来各步）
#   edges:   List[Edge]              网络边集合（含方向、权重、lag、显著性）
Tensor = torch.Tensor
NArray = np.ndarray


@runtime_checkable
class ModelAdapter(Protocol):
    """预测模型适配器。

    实现类：GraphWaveNetAdapter / LSTMAdapter / STGCNAdapter / ...
    """

    name: str            # 模型名，如 "graph_wavenet"
    device: torch.device # 推理设备
    n_nodes: int         # 节点数
    in_dim: int          # 输入通道（4 污染物 + 5 气象 = 9）
    horizon: int         # 预测步长（如 12 或 24）

    def load(self, ckpt_path: str) -> None:
        """加载权重。实现类必须幂等，重复 load 同一文件不重复分配显存。"""
        ...

    @torch.no_grad()
    def predict(self, batch_x: Tensor) -> Tensor:
        """前向推理。batch_x 形状见上文，返回 (B, T_out, N, 1)。

        实现类负责：
          1. 张量搬到 self.device；
          2. 按本模型约定整形（Graph WaveNet 需要 (B,C,N,T)，适配器内部 permute）；
          3. 输出统一回 (B, T_out, N, 1)。
        """
        ...


@runtime_checkable
class NetworkBuilder(Protocol):
    """关联网络构建适配器。

    实现类：TransferEntropyBuilder / LaggedCorrBuilder / WindPriorBuilder / ...
    """

    method: str          # 算法名，如 "transfer_entropy"
    directed: bool       # 是否有向
    has_lag: bool        # 是否携带滞后标签

    def build(
        self,
        series: NArray,             # (T, N) 单污染物时间序列矩阵，T=时间步
        coords: NArray | None = None,  # (N, 2) 经纬度（仅 WindPrior 等需要）
        **params,
    ) -> list[dict]:
        """返回边列表，每条边为 dict：
            {
                "src": int,            # 源节点索引
                "dst": int,            # 目标节点索引
                "weight": float,       # 边权（TE 值 / 相关系数 / 风场强度）
                "lag": int | None,     # 滞后步长（has_lag=True 时）
                "p_value": float | None,  # 显著性（surrogate test 给出）
                "fdr_pass": bool | None,  # 是否通过 FDR 校正
            }

        实现类必须：
          1. 接收 (T, N) 矩阵，内部不感知全局配置；
          2. 返回边列表已按 weight 降序；
          3. 若 directed=False，每条无向边只输出一次（i<j）。
        """
        ...


# ---------- 占位错误 ----------

class AdapterNotImplementedError(NotImplementedError):
    """占位适配器被调用时抛出，提示尚未实现。"""
