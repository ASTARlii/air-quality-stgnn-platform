"""主网络构建器：转移熵 + 滞后相关 + 风场先验 融合（开题报告主网络）。

策略：三种方法独立建边，按权重融合：
  final_weight = w_te * norm(TE) + w_corr * norm(|corr|) + w_wind * norm(wind)
最终边列表按 final_weight 降序，并标注每个分量方法的贡献。

风场分量在 ERA5 接入前权重设为 0（自动降级为 TE + 滞后相关双源）。
"""
from __future__ import annotations

import numpy as np

from src.domain.networks.transfer_entropy import TransferEntropyBuilder
from src.domain.networks.lagged_corr import LaggedCorrBuilder
from src.domain.networks.wind_prior import WindPriorBuilder


def _normalize_weight(edges: list[dict], key: str = "weight") -> dict[tuple[int, int], float]:
    """把边权重归一化到 [0, 1]，按 (src, dst) 索引。"""
    if not edges:
        return {}
    weights = np.array([e[key] for e in edges])
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min < 1e-9:
        return {(e["src"], e["dst"]): 1.0 for e in edges}
    return {
        (e["src"], e["dst"]): (e[key] - w_min) / (w_max - w_min)
        for e in edges
    }


class CompositeBuilder:
    """主网络融合构建器。"""

    method = "composite"
    directed = True
    has_lag = True

    def __init__(
        self,
        w_te: float = 0.5,
        w_corr: float = 0.5,
        w_wind: float = 0.0,          # 默认 0（等 ERA5）
        top_k: int | None = None,
        n_surrogates: int = 50,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.w_te = w_te
        self.w_corr = w_corr
        self.w_wind = w_wind
        self.top_k = top_k
        self.te_builder = TransferEntropyBuilder(n_surrogates=n_surrogates, alpha=alpha, seed=seed)
        self.corr_builder = LaggedCorrBuilder(n_surrogates=n_surrogates, alpha=alpha, seed=seed)
        self.wind_builder = WindPriorBuilder()

    def build(
        self,
        series: np.ndarray,            # (T, N)
        coords: np.ndarray | None = None,
        wind_uv: np.ndarray | None = None,  # (T, N, 2) 可选
        **_,
    ) -> list[dict]:
        # 各源独立构建
        te_edges = self.te_builder.build(series, coords)
        corr_edges = self.corr_builder.build(series, coords)
        wind_edges = self.wind_builder.build(wind_uv, coords) if wind_uv is not None and self.w_wind > 0 else []

        # 归一化权重
        te_w = _normalize_weight(te_edges)
        corr_w = _normalize_weight(corr_edges)
        wind_w = _normalize_weight(wind_edges)

        # 融合
        all_keys = set(te_w) | set(corr_w) | set(wind_w)
        # 携带元信息
        meta = {}
        for e in te_edges + corr_edges + wind_edges:
            k = (e["src"], e["dst"])
            if k not in meta:
                meta[k] = {"lag": e.get("lag"), "p_value": e.get("p_value"), "fdr_pass": e.get("fdr_pass")}

        fused = []
        for k in all_keys:
            src, dst = k
            score = (
                self.w_te * te_w.get(k, 0.0) +
                self.w_corr * corr_w.get(k, 0.0) +
                self.w_wind * wind_w.get(k, 0.0)
            )
            fused.append({
                "src": src, "dst": dst,
                "weight": round(float(score), 5),
                "components": {
                    "te": round(te_w.get(k, 0.0), 4),
                    "corr": round(corr_w.get(k, 0.0), 4),
                    "wind": round(wind_w.get(k, 0.0), 4),
                },
                **meta[k],
            })

        fused.sort(key=lambda e: -e["weight"])
        if self.top_k:
            fused = fused[: self.top_k]
        return fused
