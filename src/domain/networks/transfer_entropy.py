"""转移熵网络构建器。

输入：(T, N) 单污染物时间序列矩阵
输出：有向边列表，weight=TE 值，lag 默认 1，含 surrogate p 值与 FDR 标记
"""
from __future__ import annotations

import numpy as np

from src.domain.networks.common import transfer_entropy, surrogate_test, fdr_bh


class TransferEntropyBuilder:
    """转移熵网络构建器（实现 NetworkBuilder Protocol）。"""

    method = "transfer_entropy"
    directed = True
    has_lag = True

    def __init__(
        self,
        lag: int = 1,
        n_bins: int = 8,
        n_surrogates: int = 50,
        alpha: float = 0.05,
        top_k: int | None = None,
        seed: int = 42,
    ) -> None:
        self.lag = lag
        self.n_bins = n_bins
        self.n_surrogates = n_surrogates
        self.alpha = alpha
        self.top_k = top_k          # 若设置，只保留前 K 条边
        self.seed = seed

    def build(
        self,
        series: np.ndarray,            # (T, N)
        coords: np.ndarray | None = None,
        **_,
    ) -> list[dict]:
        """对 N×(N-1) 有向边计算 TE + surrogate + FDR。"""
        T, N = series.shape
        edges: list[dict] = []
        p_values: list[float] = []

        for src in range(N):
            for dst in range(N):
                if src == dst:
                    continue
                x = series[:, src]
                y = series[:, dst]
                # 跳过全 NaN
                if np.isnan(x).all() or np.isnan(y).all():
                    continue

                te, p = surrogate_test(
                    x, y,
                    fn=transfer_entropy,
                    n_surrogates=self.n_surrogates,
                    seed=self.seed,
                    lag=self.lag, n_bins=self.n_bins,
                )
                edges.append({
                    "src": src, "dst": dst,
                    "weight": round(float(te), 5),
                    "lag": self.lag,
                    "p_value": round(float(p), 4),
                    "fdr_pass": None,  # 待 FDR
                })
                p_values.append(p)

        # FDR 校正
        if p_values:
            passes = fdr_bh(np.array(p_values), alpha=self.alpha)
            for e, ok in zip(edges, passes):
                e["fdr_pass"] = bool(ok)

        # 按 weight 降序
        edges.sort(key=lambda e: -e["weight"])
        if self.top_k:
            edges = edges[: self.top_k]
        return edges
