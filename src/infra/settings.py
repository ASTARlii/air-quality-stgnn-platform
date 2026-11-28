"""后端配置加载（从 configs/ 读 data.yaml + model.yaml，禁止硬编码路径）。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def get_data_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "data.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_model_config() -> dict:
    with open(PROJECT_ROOT / "configs" / "model.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_settings() -> dict:
    """聚合配置，供 FastAPI Depends 注入。"""
    data = get_data_config()
    model = get_model_config()
    return {
        "project_root": PROJECT_ROOT,
        "data": data,
        "model": model,
        "pollutants": data["features"]["pollutants"],
        "horizons": data["features"]["horizons_h"],
        "input_window": data["features"]["input_window_h"],
        "stations_file": PROJECT_ROOT / data["air_quality"]["station_list"]["file"],
        "obs_npz": PROJECT_ROOT / data["paths"]["processed"] / "obs.npz",
        "checkpoint_dir": PROJECT_ROOT / model["train"]["checkpoint_dir"],
        "device": model["train"]["device"],
    }
