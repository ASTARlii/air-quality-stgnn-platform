"""关联网络缓存：磁盘 JSON，key = hash(pollutant, from, to, method, surrogates)。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.infra.settings import get_settings


CACHE_DIR = get_settings()["project_root"] / "data" / "processed" / "networks"


def compute_cache_key(pollutant: str, frm: str, to: str, method: str, surrogates: int) -> str:
    payload = json.dumps(
        {"pollutant": pollutant, "from": frm, "to": to, "method": method, "surrogates": surrogates},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def get_network(key: str) -> dict | None:
    """从缓存读网络，不存在返回 None。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def put_network(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{key}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def list_cached() -> list[str]:
    if not CACHE_DIR.exists():
        return []
    return [p.stem for p in CACHE_DIR.glob("*.json")]
