"""风场先验网络构建器（占位，待 ERA5 接入后填充）。

设计：基于风速风向 + 站点地理坐标，判断 i 是否处于 j 的上风向，且风速足以在 lag 小时内输送。
weight = 风向匹配度 × 风速强度。
"""
from __future__ import annotations

import numpy as np


class WindPriorBuilder:
    """风场先验构建器（骨架）。

    待 ERA5 数据接入后填充：
      1. 输入 wind_u (T, N), wind_v (T, N) + coords (N, 2)
      2. 计算时段平均风向 / 风速
      3. 对每个 (i, j) 判断 i→j 是否与风向一致（cos(方向差) > 0.5）
      4. weight = cos(方向差) × 风速 / 参考风速
    """

    method = "wind_prior"
    directed = True
    has_lag = True

    def __init__(self, top_k: int | None = None) -> None:
        self.top_k = top_k

    def build(
        self,
        series: np.ndarray,            # 风场：(T, N, 2) 含 u/v 分量
        coords: np.ndarray | None = None,
        **_,
    ) -> list[dict]:
        """骨架阶段返回空列表，等 ERA5 数据接入后实现。"""
        return []
