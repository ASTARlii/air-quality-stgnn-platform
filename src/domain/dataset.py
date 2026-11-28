"""空气质量时空数据集（PyTorch Dataset）。

职责（对应开题报告 6.1 + 数据说明 07 第 5 节）：
  1. 加载 data/processed/obs.npz（含 values/timestamps/codes/pollutants）；
  2. 按自然年切分 train/valid/test，禁止随机划分；
  3. 缺失填充：连续缺失 ≤6h 用前向填充（ffill），剩余用训练集均值（标准化后 = 0）；
  4. 归一化：按训练集统计量（每站点每污染物独立 z-score），valid/test 复用同一参数；
  5. 滑动窗采样：T_in=24 输入、horizon=24 输出；
  6. 单输出：每次返回指定 target 污染物的预测标签。

对外契约（与 GraphWaveNetAdapter 对齐）：
  __getitem__ 返回 (x, y, mask)：
    x    : (T_in, N, C_in)  float32，已标准化
    y    : (horizon, N, 1)  float32，已标准化（仅 target 污染物）
    mask : (horizon, N)     bool，标记 y 的每个位置是否有真值（供 loss 加权，mask=False 的位置 loss=0）

mask 设计要点：mask=False 表示该位置原始数据缺失，训练/评估时把对应 loss 置零。
  - 3671A 在 valid 后 5 月、test 前 7 月全 False；
  - 其他站正常缺失段也按真值有无设置 mask；
  - 输入侧 x 的 NaN 已被 ffill+均值填充，mask 只影响输出侧 y 的 loss。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# 站点索引常量（与 codes 顺序一致）
STATION_INDEX = {code: i for i, code in enumerate([
    '1001A', '1002A', '1003A', '1004A', '1005A', '1006A', '1007A', '1008A',
    '1009A', '1010A', '1011A', '1012A', '3281A', '3417A', '3418A', '3671A',
    '3672A', '3673A', '3674A', '3675A', '3694A', '3695A', '3696A', '3697A',
])}
POLLUTANT_INDEX = {'pm25': 0, 'pm10': 1, 'no2': 2, 'o3': 3}

# 3671A 缺测段（UTC 时间戳，与 obs.npz timestamps 一致）
GAP_3671A_UTC = [
    (1690761600, 1716595199),  # 2023-08-01 00:00 ~ 2024-05-24 23:59 (UTC+8 转 UTC)
    (1716768000, 1722887999),  # 2024-05-27 00:00 ~ 2024-08-05 23:59 (UTC+8 转 UTC)
]


@dataclass
class Normalizer:
    """每站点每污染物的 z-score 归一化参数。"""
    mean: np.ndarray  # (N, P)
    std: np.ndarray   # (N, P)

    @classmethod
    def fit(cls, values: np.ndarray) -> "Normalizer":
        """values: (T, N, P)，按 T 维求 mean/std，NaN 不参与统计。"""
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-6, 1.0, std)  # 防 0 除
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        """z-score 标准化。NaN 保留 NaN（待 ffill/均值填充）。"""
        return (values - self.mean[None]) / self.std[None]

    def inverse(self, values: np.ndarray, target: str) -> np.ndarray:
        """单污染物反归一化（用于评估时把预测值转回物理量）。"""
        p = POLLUTANT_INDEX[target]
        return values * self.std[None, :, p] + self.mean[None, :, p]


def _fill_missing(values: np.ndarray, ffill_limit: int = 6) -> np.ndarray:
    """缺失填充：先 ffill（限 ffill_limit 小时），剩余 NaN 用 0（标准化后 = 训练集均值）。

    实现：用 pandas 风格的 ffill（手写避免 pandas 依赖）。
    values: (T, N, P)，原地修改并返回。
    """
    T, N, P = values.shape
    # 前向填充：每个 (n, p) 序列独立
    out = values.copy()
    last_valid = np.full((N, P), np.nan, dtype=np.float32)
    run = np.zeros((N, P), dtype=np.int32)
    for t in range(T):
        v = out[t]
        valid = ~np.isnan(v)
        # 用上一次的有效值填充当前 NaN（仅在 ffill_limit 内）
        can_fill = (~valid) & (~np.isnan(last_valid)) & (run < ffill_limit)
        v = np.where(can_fill, last_valid, v)
        # 更新 last_valid / run
        new_valid = ~np.isnan(v)
        last_valid = np.where(new_valid, v, last_valid)
        run = np.where(valid, 0, run + 1)
        out[t] = v
    # 剩余 NaN 用 0 填充（标准化后 = 均值）
    out = np.nan_to_num(out, nan=0.0)
    return out


def _build_truth_mask(values: np.ndarray) -> np.ndarray:
    """记录原始真值 mask（True = 有数据）。"""
    return ~np.isnan(values)


class AirQualityDataset(Dataset):
    """单目标污染物的时空 Dataset。

    切分（按时间戳）：
      train: 2021-01-01 ~ 2022-12-31
      valid: 2023-01-01 ~ 2023-12-31
      test : 2024-01-01 ~ 2024-12-31
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        target: str = "pm25",
        input_window: int = 24,
        horizon: int = 24,
        normalizer: Normalizer | None = None,
        ffill_limit: int = 6,
    ) -> None:
        assert split in ("train", "valid", "test"), f"未知 split: {split}"
        assert target in POLLUTANT_INDEX, f"未知 target: {target}"

        self.split = split
        self.target = target
        self.input_window = input_window
        self.horizon = horizon
        self.ffill_limit = ffill_limit

        # 1. 加载
        d = np.load(npz_path, allow_pickle=True)
        values = d["values"].astype(np.float32)        # (T_total, N, P)
        timestamps = d["timestamps"].astype(np.int64)  # (T_total,)
        self.codes = [str(c) for c in d["codes"]]
        self.pollutants = [str(p) for p in d["pollutants"]]
        self.T_total, self.N, self.P = values.shape

        # 2. 时间切分（按自然年，UTC+8 边界）
        # timestamps 是 UTC 秒；2021/2023/2024 年初的 UTC 时间戳：
        SPLIT_BOUNDARIES = {
            "train": (1609430400, 1672502399),  # 2021-01-01 00:00 UTC ~ 2022-12-31 23:59 UTC
            "valid": (1672531200, 1704067199),  # 2023-01-01 00:00 UTC ~ 2023-12-31 23:59 UTC
            "test":  (1704067200, 1735689599),  # 2024-01-01 00:00 UTC ~ 2024-12-31 23:59 UTC
        }
        ts_lo, ts_hi = SPLIT_BOUNDARIES[split]
        mask_split = (timestamps >= ts_lo) & (timestamps <= ts_hi)
        self.split_start = int(np.argmax(mask_split))  # 该 split 起始索引
        self.split_end = self.split_start + int(mask_split.sum())  # 该 split 结束索引（不含）

        # 切出该 split 的数据
        self.values_raw = values[self.split_start:self.split_end]   # (T_split, N, P)
        self.timestamps = timestamps[self.split_start:self.split_end]

        # 3. 归一化
        if normalizer is None:
            # 训练集自己 fit（只 fit train split）
            assert split == "train", "valid/test 必须传入 train 的 normalizer"
            self.normalizer = Normalizer.fit(self.values_raw)
        else:
            self.normalizer = normalizer
        self.values_norm = self.normalizer.transform(self.values_raw)  # (T_split, N, P)

        # 4. 真值 mask（用于 loss）
        self.truth_mask = _build_truth_mask(self.values_raw)  # (T_split, N, P)

        # 5. 缺失填充（在标准化后的值上做，剩余 NaN → 0）
        self.values_filled = _fill_missing(self.values_norm, ffill_limit=ffill_limit)

        # 6. 采样索引：i 表示输入窗起始位置，输入 [i, i+T_in)，输出 [i+T_in, i+T_in+horizon)
        self.n_samples = len(self.values_raw) - input_window - horizon + 1
        assert self.n_samples > 0, f"{split} 样本数不足: {self.n_samples}"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        """返回 (x, y, mask)。

        x shape: (T_in, N, C_in=9)——4 污染物 + 5 气象占位（全 0）。
        待 ERA5 数据接入后，把占位气象替换为真实气象。
        """
        i = idx
        # 4 污染物
        x_pol = self.values_filled[i:i + self.input_window]      # (T_in, N, 4)
        # 5 气象占位（骨架阶段全 0；后续替换为 ERA5 插值结果）
        x_met = np.zeros((self.input_window, self.N, 5), dtype=np.float32)
        x = np.concatenate([x_pol, x_met], axis=-1)              # (T_in, N, 9)
        # 输出：target 污染物
        p = POLLUTANT_INDEX[self.target]
        y_start = i + self.input_window
        y_end = y_start + self.horizon
        y = self.values_filled[y_start:y_end, :, p:p+1]          # (horizon, N, 1)
        mask = self.truth_mask[y_start:y_end, :, p]              # (horizon, N)

        return (
            torch.from_numpy(x).float(),                         # (T_in, N, 9)
            torch.from_numpy(y).float(),                         # (horizon, N, 1)
            torch.from_numpy(mask).bool(),                       # (horizon, N)
        )

    def get_station_mask(self) -> np.ndarray:
        """返回该 split 下各站点是否可用的 mask（供 loss 加权或剔除 3671A）。"""
        # 整段缺测站点判定：真值比例 < 50% 的站点
        ratio = self.truth_mask.mean(axis=0).mean(axis=1)   # (N,)
        return ratio > 0.5                                   # (N,) bool


def build_datasets(
    npz_path: str | Path = "data/processed/obs.npz",
    target: str = "pm25",
    input_window: int = 24,
    horizon: int = 24,
) -> tuple[AirQualityDataset, AirQualityDataset, AirQualityDataset, Normalizer]:
    """构造 train/valid/test 三个 Dataset + 共享 Normalizer。"""
    train = AirQualityDataset(npz_path, "train", target, input_window, horizon)
    valid = AirQualityDataset(npz_path, "valid", target, input_window, horizon, normalizer=train.normalizer)
    test = AirQualityDataset(npz_path, "test", target, input_window, horizon, normalizer=train.normalizer)
    return train, valid, test, train.normalizer
