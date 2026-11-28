"""预测用例：站点 + 污染物 + 历史窗口 → 多步浓度 + 分级。

骨架阶段：模型已加载就真算，未加载则抛 503（由路由层捕获转为友好错误）。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

from src.domain.dataset import AirQualityDataset, POLLUTANT_INDEX
from src.domain.grading import grade
from src.infra.model_pool import get_model_pool
from src.infra.settings import get_settings


class PredictionInputError(ValueError):
    """预测输入不合法（路由层捕获后转为 400）。"""


# Dataset 构造要读 npz + 全量归一化 + 缺失填充，单次约 700ms。
# 常驻缓存，避免每次请求重建（地图视图一次要打 24 次请求）。
_DS_CACHE: dict[str, tuple[AirQualityDataset, object]] = {}
_DS_LOCK = threading.Lock()


def _get_test_dataset(pollutant: str) -> AirQualityDataset:
    """取（缓存的）test Dataset，复用 train 的 normalizer。"""
    hit = _DS_CACHE.get(pollutant)
    if hit is not None:
        return hit[0]

    with _DS_LOCK:
        hit = _DS_CACHE.get(pollutant)
        if hit is not None:
            return hit[0]

        s = get_settings()
        train_ds = AirQualityDataset(
            npz_path=s["obs_npz"], split="train", target=pollutant,
            input_window=s["input_window"], horizon=s["horizons"][-1],
        )
        test_ds = AirQualityDataset(
            npz_path=s["obs_npz"], split="test", target=pollutant,
            input_window=s["input_window"], horizon=s["horizons"][-1],
            normalizer=train_ds.normalizer,
        )
        # 只保留 test 集与 normalizer，释放 train 集的大数组
        _DS_CACHE[pollutant] = (test_ds, train_ds.normalizer)
        return test_ds


CN_TZ = timezone(timedelta(hours=8))


def _parse_as_of(as_of: str) -> int:
    """把 ISO 时间字符串转成数据时间轴的 epoch 秒。

    数据时间轴按北京整点记（见 build_features.py）：不带时区按北京时间解释，
    带时区（含 Z / +08:00）先换算到北京时间，再映射为同一坐标轴。
    """
    try:
        dt = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PredictionInputError(f"as_of 时间格式不合法: {as_of}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=CN_TZ)
    beijing = dt.astimezone(CN_TZ)
    return int(beijing.replace(tzinfo=timezone.utc).timestamp())


def _to_iso(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _resolve_window(test_ds: AirQualityDataset, as_of: str | None) -> tuple[int, int]:
    """按 as_of 定位输入窗口，返回 (样本索引, 窗口末时刻)。

    as_of 为空取最新窗口；早于 test 集起点或窗口不足 input_window 小时时抛错；
    晚于数据末尾则截断到最新可用窗口。
    """
    ts = test_ds.timestamps
    if ts.size == 0:
        raise PredictionInputError("测试集为空，无法预测")

    if as_of is None:
        # 用最近的 input_window 小时作为输入，不依赖 test 集样本索引上界
        return len(ts) - test_ds.input_window, int(ts[-1])

    cutoff = _parse_as_of(as_of)
    if cutoff < int(ts[0]):
        raise PredictionInputError(
            f"as_of 早于数据起点 {_to_iso(ts[0])}，可用窗口不足"
        )

    pos = int(np.searchsorted(ts, cutoff, side="right")) - 1
    pos = min(pos, len(ts) - 1)                  # 晚于末尾 → 取最新
    idx = pos + 1 - test_ds.input_window
    if idx < 0:
        raise PredictionInputError(
            f"输入窗口不足 {test_ds.input_window} 小时"
            f"（as_of={_to_iso(ts[pos])} 之前仅 {pos + 1} 小时）"
        )
    return idx, int(ts[pos])


def predict(
    station_id: str,
    pollutant: str,
    steps: list[int],
    as_of: str | None = None,
    seed: int = 42,
) -> dict:
    """执行预测。

    station_id: 站点编码（如 "1001A"）
    pollutant:  pm25 / pm10 / no2 / o3
    steps:      预测步长列表，如 [1,3,6,12,24]
    as_of:      截止时间 ISO 字符串，取不晚于该时刻的最近整点作为窗口末；
                None 表示用最新可用窗口；早于数据起点或不足 24h 抛 400
    seed:       模型 seed（默认 42，多 seed 模式由调用方决定）
    """
    s = get_settings()
    if pollutant not in POLLUTANT_INDEX:
        raise PredictionInputError(f"不支持的污染物: {pollutant}")

    # 加载（缓存的）测试集 Dataset
    test_ds = _get_test_dataset(pollutant)
    normalizer = _DS_CACHE[pollutant][1]

    # 找到站点索引
    if station_id not in test_ds.codes:
        raise PredictionInputError(f"未知站点: {station_id}")
    st_idx = test_ds.codes.index(station_id)

    # 按 as_of 定位输入窗口
    idx, window_end = _resolve_window(test_ds, as_of)
    x, _, _ = test_ds[idx]
    x = x.unsqueeze(0)                            # (1, T_in, N, C_in)

    # 前向
    adapter = get_model_pool().get(pollutant, seed=seed)
    t0 = time.time()
    pred_n = adapter.predict(x)                   # (1, horizon, N, 1) 标准化
    latency_ms = int((time.time() - t0) * 1000)

    # 反归一化到物理量
    p = POLLUTANT_INDEX[pollutant]
    mean = normalizer.mean[:, p]                  # (N,)
    std = normalizer.std[:, p]
    pred = pred_n[0, :, :, 0].cpu().numpy() * std + mean   # (horizon, N)

    # 提取该站点 + 各 steps
    series = []
    for h in steps:
        if h < 1 or h > pred.shape[0]:
            raise PredictionInputError(f"步长超出范围: {h}（horizon={pred.shape[0]}）")
        value = float(pred[h - 1, st_idx])
        series.append({"h": h, "value": round(value, 2)})

    return {
        "station_id": station_id,
        "pollutant": pollutant,
        "series": series,
        "model_id": f"gw_{pollutant}_seed{seed}",
        "latency_ms": latency_ms,
        "as_of": _to_iso(window_end),
    }


def predict_and_grade(
    station_id: str,
    pollutant: str,
    steps: list[int],
    as_of: str | None = None,
    seed: int = 42,
) -> dict:
    """预测 + 分级（开题报告三类 AI 服务的复合）。"""
    pred = predict(station_id, pollutant, steps, as_of=as_of, seed=seed)
    for item in pred["series"]:
        item["grading"] = grade(pollutant, item["value"])
    return pred
