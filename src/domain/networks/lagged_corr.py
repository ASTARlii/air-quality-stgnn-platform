"""滞后相关网络构建器。

对每个 (i, j) 对在候选 lag 上算 Pearson 相关，取绝对值最大者。
weight 用 |corr|，但保留 sign 字段（正/负）。
"""
from __future__ import annotations

import numpy as np

from src.domain.networks.common import lagged_correlation, surrogate_test, fdr_bh


def _abs_corr_fn(x, y, lags):
    c, _ = lagged_correlation(x, y, lags=lags)
    return abs(c)


class LaggedCorrBuilder:
    """滞后相关网络构建器（实现 NetworkBuilder Protocol）。"""

    method = "lagged_correlation"
    directed = True   # i→j 表示 i 领先 j，有向
    has_lag = True

    def __init__(
        self,
        lag_candidates: list[int] | None = None,
        n_surrogates: int = 50,
        alpha: float = 0.05,
        top_k: int | None = None,
        seed: int = 42,
    ) -> None:
        self.lag_candidates = lag_candidates or [1, 2, 3, 4]
        self.n_surrogates = n_surrogates
        self.alpha = alpha
        self.top_k = top_k
        self.seed = seed

    def build(
        self,
        series: np.ndarray,            # (T, N)
        coords: np.ndarray | None = None,
        **_,
    ) -> list[dict]:
        T, N = series.shape
        edges: list[dict] = []
        p_values: list[float] = []

        for src in range(N):
            for dst in range(N):
                if src == dst:
                    continue
                x = series[:, src]
                y = series[:, dst]
                if np.isnan(x).all() or np.isnan(y).all():
                    continue

                # 观测值（含最佳 lag）
                corr, best_lag = lagged_correlation(x, y, lags=self.lag_candidates)
                # surrogate test 用 |corr|
                _, p = surrogate_test(
                    x, y,
                    fn=_abs_corr_fn,
                    n_surrogates=self.n_surrogates,
                    seed=self.seed,
                    lags=self.lag_candidates,
                )
                edges.append({
                    "src": src, "dst": dst,
                    "weight": round(abs(corr), 4),
                    "sign": int(np.sign(corr)) if corr != 0 else 0,
                    "lag": best_lag,
                    "p_value": round(float(p), 4),
                    "fdr_pass": None,
                })
                p_values.append(p)

        if p_values:
            passes = fdr_bh(np.array(p_values), alpha=self.alpha)
            for e, ok in zip(edges, passes):
                e["fdr_pass"] = bool(ok)

        edges.sort(key=lambda e: -e["weight"])
        if self.top_k:
            edges = edges[: self.top_k]
        return edges
