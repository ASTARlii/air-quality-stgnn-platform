"""GB 3095-2012 空气质量分级（纯函数，可单测）。

参考：环境空气质量标准 GB 3095-2012 二级浓度限值 + 环境空气质量指数（AQI）技术规定 HJ 633-2012。

按 AQI 分指数 IAQI 分级（PM2.5 / PM10 / NO2 / O3）：
  优（0–50）→ 良（51–100）→ 轻度（101–150）→ 中度（151–200）→ 重度（201–300）→ 严重（>300）

注意：本平台分级为「辅助研判提示」，不等同于官方预警（开题报告声明）。
"""
from __future__ import annotations


# IAQI 分段表（按污染物类型），单位 μg/m³（CO 为 mg/m³，本骨架不含 CO）
# 每段为 (浓度下限, 浓度上限, IAQI 下限, IAQI 上限)
IAQI_BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    # PM2.5 24h 平均
    "pm25": [
        (0, 35, 0, 50), (35, 75, 50, 100), (75, 115, 100, 150),
        (115, 150, 150, 200), (150, 250, 200, 300), (250, 350, 300, 400), (350, 500, 400, 500),
    ],
    # PM10 24h 平均
    "pm10": [
        (0, 50, 0, 50), (50, 150, 50, 100), (150, 250, 100, 150),
        (250, 350, 150, 200), (350, 420, 200, 300), (420, 500, 300, 400), (500, 600, 400, 500),
    ],
    # NO2 1h 平均
    "no2": [
        (0, 100, 0, 50), (100, 200, 50, 100), (200, 700, 100, 150),
        (700, 1200, 150, 200), (1200, 2340, 200, 300), (2340, 3090, 300, 400), (3090, 3840, 400, 500),
    ],
    # O3 1h 平均
    "o3": [
        (0, 160, 0, 50), (160, 200, 50, 100), (200, 300, 100, 150),
        (300, 400, 150, 200), (400, 800, 200, 300), (800, 1000, 300, 400), (1000, 1200, 400, 500),
    ],
}

LEVEL_NAMES = ["优", "良", "轻度污染", "中度污染", "重度污染", "严重污染"]
LEVEL_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#99004c", "#7e0023"]


def compute_iaqi(pollutant: str, concentration: float) -> float | None:
    """计算单污染物 IAQI 分指数。浓度超出范围时返回 None（视为数据异常）。"""
    if pollutant not in IAQI_BREAKPOINTS:
        raise ValueError(f"不支持的污染物: {pollutant}")
    if concentration < 0:
        return None
    for lo_c, hi_c, lo_i, hi_i in IAQI_BREAKPOINTS[pollutant]:
        if lo_c <= concentration < hi_c:
            return lo_i + (concentration - lo_c) / (hi_c - lo_c) * (hi_i - lo_i)
    # 超出最大断点：取最大 IAQI（封顶 500）
    return 500.0


def iaqi_to_level(iaqi: float) -> tuple[int, str, str]:
    """IAQI → (等级数字, 等级名称, 颜色)。"""
    if iaqi <= 50:
        return 1, LEVEL_NAMES[0], LEVEL_COLORS[0]
    if iaqi <= 100:
        return 2, LEVEL_NAMES[1], LEVEL_COLORS[1]
    if iaqi <= 150:
        return 3, LEVEL_NAMES[2], LEVEL_COLORS[2]
    if iaqi <= 200:
        return 4, LEVEL_NAMES[3], LEVEL_COLORS[3]
    if iaqi <= 300:
        return 5, LEVEL_NAMES[4], LEVEL_COLORS[4]
    return 6, LEVEL_NAMES[5], LEVEL_COLORS[5]


def grade(pollutant: str, concentration: float) -> dict:
    """对单次浓度分级，返回 {iaqi, level, level_name, color, disclaimer}。"""
    iaqi = compute_iaqi(pollutant, concentration)
    if iaqi is None:
        return {"iaqi": None, "level": None, "level_name": "数据异常", "color": "#cccccc",
                "disclaimer": "辅助研判提示，不等同于官方预警"}
    level, name, color = iaqi_to_level(iaqi)
    return {
        "iaqi": round(iaqi, 1),
        "level": level,
        "level_name": name,
        "color": color,
        "disclaimer": "辅助研判提示，不等同于官方预警",
    }


def grade_series(pollutant: str, concentrations: dict[int, float]) -> dict:
    """对多步预测浓度批量分级。concentrations: {horizon_h: value}。"""
    return {
        f"h{h}": grade(pollutant, v) for h, v in concentrations.items()
    }
