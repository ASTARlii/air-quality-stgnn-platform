"""网络构建的公共工具：分箱、条件熵、surrogate test、FDR 校正。

算法选择（开题报告 6.1 + 6.4）：
- 分箱：等频（quantile）分箱，对 PM2.5 等偏态分布稳健；默认 bins=8
- TE：一阶马尔可夫，离散直方图估计
- Surrogate：相位随机化（保留功率谱，破坏相位耦合）
- FDR：Benjamini-Hochberg
"""
from __future__ import annotations

import numpy as np


# ---------- 分箱 ----------

def quantile_bin(x: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """等频分箱：返回 0..n_bins-1 的整数序列。NaN 保留为 -1（不参与统计）。"""
    out = np.full(len(x), -1, dtype=np.int32)
    valid = ~np.isnan(x)
    if valid.sum() < n_bins * 2:
        return out
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.nanpercentile(x[valid], qs)
    out[valid] = np.clip(np.searchsorted(edges, x[valid], side="right"), 0, n_bins - 1)
    return out


# ---------- 转移熵（一阶马尔可夫） ----------

def transfer_entropy(x: np.ndarray, y: np.ndarray, lag: int = 1, n_bins: int = 8) -> float:
    """TE(x→y)：x 的过去对 y 的未来的信息量。

    TE = H(y_future | y_past) - H(y_future | y_past, x_past)

    x, y: 长度 T 的一维序列（连续值）。
    实现：等频分箱后做条件熵估计。
    """
    # 等频分箱
    xb = quantile_bin(x, n_bins)
    yb = quantile_bin(y, n_bins)
    T = len(x)

    # 构造三元组 (y_{t+lag}, y_t, x_t)
    # 有效索引：t + lag < T 且三个值都不是 -1
    ts = np.arange(T - lag)
    valid = (yb[ts] >= 0) & (yb[ts + lag] >= 0) & (xb[ts] >= 0)
    if valid.sum() < 30:                       # 样本太少，返回 0
        return 0.0
    ts = ts[valid]
    y_past = yb[ts]
    y_future = yb[ts + lag]
    x_past = xb[ts]
    n = len(ts)

    # H(y_future | y_past) = -Σ p(y_past, y_future) log p(y_future | y_past)
    # 用联合计数
    n_joint_yy = np.zeros((n_bins, n_bins))
    np.add.at(n_joint_yy, (y_past, y_future), 1)
    p_yy = n_joint_yy / n
    p_y_past = p_yy.sum(axis=1, keepdims=True)
    # 防止 0 log 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where((p_yy > 0) & (p_y_past > 0), np.log2(p_yy / p_y_past), 0.0)
    h_y_given_y = -np.sum(p_yy * log_ratio)

    # H(y_future | y_past, x_past)
    n_joint_xyy = np.zeros((n_bins, n_bins, n_bins))
    np.add.at(n_joint_xyy, (x_past, y_past, y_future), 1)
    p_xyy = n_joint_xyy / n
    p_xy_past = p_xyy.sum(axis=2, keepdims=True)  # p(x_past, y_past)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where((p_xyy > 0) & (p_xy_past > 0), np.log2(p_xyy / p_xy_past), 0.0)
    h_y_given_xy = -np.sum(p_xyy * log_ratio)

    return float(max(h_y_given_y - h_y_given_xy, 0.0))


# ---------- 滞后相关 ----------

def lagged_correlation(x: np.ndarray, y: np.ndarray, lags: list[int] = (1, 2, 3, 4)) -> tuple[float, int]:
    """对多个 lag 计算 Pearson 相关，取绝对值最大者。

    返回 (max_corr, best_lag)。corr 带符号，但排序用 |corr|。
    """
    best_corr, best_lag = 0.0, lags[0]
    for lag in lags:
        T = len(x)
        xs, ys = x[: T - lag], y[lag:]
        valid = ~(np.isnan(xs) | np.isnan(ys))
        if valid.sum() < 30:
            continue
        xs, ys = xs[valid], ys[valid]
        if xs.std() < 1e-6 or ys.std() < 1e-6:
            continue
        c = float(np.corrcoef(xs, ys)[0, 1])
        if abs(c) > abs(best_corr):
            best_corr, best_lag = c, lag
    return best_corr, best_lag


# ---------- Surrogate test（相位随机化） ----------

def phase_randomize(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """相位随机化：FFT → 保留幅度谱，相位替换为随机 → IFFT。

    保留功率谱（自相关结构），破坏与其他序列的耦合。NaN 用均值填充后参与 FFT。
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x, dtype=np.float64)
    nan_mask = np.isnan(x)
    if nan_mask.any():
        x = x.copy()
        x[nan_mask] = np.nanmean(x)

    T = len(x)
    fft = np.fft.rfft(x)
    magnitude = np.abs(fft)
    # 随机相位（除 DC 与 Nyquist 分量外）
    phase = rng.uniform(0, 2 * np.pi, len(fft))
    phase[0] = 0.0
    if T % 2 == 0:
        phase[-1] = 0.0
    surrogate = np.fft.irfft(magnitude * np.exp(1j * phase), n=T)
    # 把 NaN 位置补回
    surrogate[nan_mask] = np.nan
    return surrogate


def surrogate_test(
    x: np.ndarray, y: np.ndarray,
    fn,                                   # fn(x, y) -> float
    n_surrogates: int = 50,
    seed: int = 42,
    **fn_kwargs,
) -> tuple[float, float]:
    """对 x 做相位随机化，重算 fn(x_surr, y)，得到 null 分布。

    返回 (observed_value, p_value)。p = (#{surr >= observed} + 1) / (n + 1)
    """
    observed = fn(x, y, **fn_kwargs)
    rng = np.random.default_rng(seed)
    n_exceed = 0
    for _ in range(n_surrogates):
        x_s = phase_randomize(x, rng)
        s = fn(x_s, y, **fn_kwargs)
        if s >= observed:
            n_exceed += 1
    p = (n_exceed + 1) / (n_surrogates + 1)
    return observed, p


# ---------- FDR 校正（Benjamini-Hochberg） ----------

def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正。

    返回 bool 数组：True = 通过 FDR（拒绝原假设）。
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    # 最大 k 使 ranked[k-1] <= thresh[k-1]
    passes = ranked <= thresh
    if not passes.any():
        return np.zeros(m, dtype=bool)
    k = np.where(passes)[0].max() + 1
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out
