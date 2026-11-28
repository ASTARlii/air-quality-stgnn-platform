"""解释服务：网络构建 / 溯源 / 关键站点遴选（真实实现）。

工作流（开题报告 3.3）：
  GET /networks:    缓存命中 → 返回；未命中 → 自动提交异步任务，返回 202 + task_key
  POST /recompute:  显式触发异步重算（幂等）
  GET /trace:       基于缓存网络做反向溯源
  GET /key-stations: 基于缓存网络算中心性
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.domain.networks.composite import CompositeBuilder
from src.domain.networks.metrics import composite_score, degree_centrality, pagerank
from src.domain.networks.trace import trace_upstream
from src.infra import network_cache
from src.infra.settings import get_settings
from src.infra.task_queue import get_task_queue, Task


# ---------- 公共工具 ----------

def get_stations() -> list[dict]:
    """24 站元信息。"""
    s = get_settings()
    with open(s["stations_file"], encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_series(pollutant: str, frm: str | None, to: str | None) -> np.ndarray:
    """加载 obs.npz，按 from/to 切时间，返回 (T, N) 单污染物矩阵。"""
    s = get_settings()
    d = np.load(s["obs_npz"], allow_pickle=True)
    values = d["values"]
    timestamps = d["timestamps"].astype(np.int64)
    pollutants = [str(p) for p in d["pollutants"]]
    p = pollutants.index(pollutant)

    # 时间过滤（ISO 字符串 → UTC 秒）
    if frm:
        ts_lo = int(datetime.fromisoformat(frm).replace(tzinfo=timezone.utc).timestamp())
        mask = timestamps >= ts_lo
    else:
        mask = np.ones(len(timestamps), dtype=bool)
    if to:
        ts_hi = int(datetime.fromisoformat(to).replace(tzinfo=timezone.utc).timestamp())
        mask = mask & (timestamps <= ts_hi)

    return values[mask, :, p]


def _serialize_edges(edges: list[dict], codes: list[str]) -> list[dict]:
    """把 src/dst 索引转换为站点编码。"""
    out = []
    for e in edges:
        out.append({
            **e,
            "src": codes[e["src"]],
            "dst": codes[e["dst"]],
            "src_idx": e["src"],
            "dst_idx": e["dst"],
        })
    return out


# ---------- 网络构建（异步任务执行体） ----------

def _do_build_network(task: Task) -> dict:
    """异步任务：构建网络并写缓存。在 TaskQueue.run_async 中调用。"""
    p = task.params
    pollutant = p["pollutant"]
    frm, to = p["from"], p["to"]
    method = p.get("method", "composite")
    surrogates = p.get("surrogates", 50)

    series = _load_series(pollutant, frm, to)
    T, N = series.shape

    if method == "composite":
        builder = CompositeBuilder(w_te=0.5, w_corr=0.5, w_wind=0.0, n_surrogates=surrogates, top_k=None)
    else:
        raise ValueError(f"未知 method: {method}")

    edges = builder.build(series)
    stations = get_stations()
    codes = [r["code"] for r in stations]

    result = {
        "pollutant": pollutant,
        "from": frm, "to": to,
        "method": method,
        "surrogates": surrogates,
        "nodes": [{"id": r["code"], "name": r["name"],
                   "lon": float(r["lon"]), "lat": float(r["lat"])} for r in stations],
        "edges": _serialize_edges(edges, codes),
        "metrics": {
            "n_edges": len(edges),
            "n_fdr_pass": sum(1 for e in edges if e.get("fdr_pass")),
            "T": T, "N": N,
        },
        "mock": False,
    }

    # 写缓存
    key = network_cache.compute_cache_key(pollutant, frm or "", to or "", method, surrogates)
    network_cache.put_network(key, result)
    return {"cache_key": key, "n_edges": len(edges)}


# ---------- 网络查询（缓存优先） ----------

def get_network(
    pollutant: str = "pm25",
    frm: str | None = None,
    to: str | None = None,
    method: str = "composite",
    surrogates: int = 50,
    auto_submit: bool = True,
) -> tuple[dict | None, Task | None]:
    """返回 (network, task)。
      - 缓存命中: (network, None)
      - 未命中且 auto_submit: (None, task) — 已自动提交异步任务
      - 未命中且不提交: (None, None)
    """
    key = network_cache.compute_cache_key(pollutant, frm or "", to or "", method, surrogates)
    cached = network_cache.get_network(key)
    if cached:
        return cached, None

    if not auto_submit:
        return None, None

    # 自动提交异步任务
    queue = get_task_queue()
    params = {"pollutant": pollutant, "from": frm, "to": to, "method": method, "surrogates": surrogates}
    task = queue.submit(kind="network_recompute", params=params)
    # 启动后台执行（如果还没启动）
    if task.status == "pending":
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            asyncio.create_task(queue.run_async(task.key, _do_build_network))
        else:
            loop.run_until_complete(queue.run_async(task.key, _do_build_network))
    return None, task


def submit_network_recompute(pollutant: str, frm: str, to: str, method: str = "composite", surrogates: int = 50) -> Task:
    """显式提交重算（幂等）。"""
    queue = get_task_queue()
    params = {"pollutant": pollutant, "from": frm, "to": to, "method": method, "surrogates": surrogates}
    task = queue.submit(kind="network_recompute", params=params)
    if task.status == "pending":
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            asyncio.create_task(queue.run_async(task.key, _do_build_network))
        else:
            loop.run_until_complete(queue.run_async(task.key, _do_build_network))
    return task


# ---------- 溯源 ----------

def get_trace(
    station_id: str,
    pollutant: str = "pm25",
    frm: str | None = None, to: str | None = None,
    method: str = "composite", surrogates: int = 50,
    max_depth: int = 2, top_k: int = 5,
) -> dict | None:
    """基于缓存的网络做反向溯源。无缓存返回 None。"""
    key = network_cache.compute_cache_key(pollutant, frm or "", to or "", method, surrogates)
    cached = network_cache.get_network(key)
    if cached is None:
        return None

    stations = get_stations()
    codes = [r["code"] for r in stations]
    if station_id not in codes:
        return None
    target = codes.index(station_id)

    # 把编码边转回索引（trace 算法用索引）
    code2idx = {c: i for i, c in enumerate(codes)}
    idx_edges = [
        {**e, "src": code2idx[e["src"]], "dst": code2idx[e["dst"]]}
        for e in cached["edges"]
    ]
    candidates = trace_upstream(idx_edges, target, max_depth=max_depth, top_k=top_k)
    return {
        "target": station_id,
        "pollutant": pollutant,
        "candidates": [
            {
                "station_id": codes[c["station_id"]],
                "name": stations[c["station_id"]]["name"],
                "score": c["score"],
                "lag_h": c["lag_h"],
                "depth": c["depth"],
                "path": [codes[i] for i in c["path"]],
            }
            for c in candidates
        ],
        "note": "关联不等于因果，仅为候选来源提示",
        "mock": False,
    }


# ---------- 关键站点遴选 ----------

def get_key_stations(
    top: int = 10,
    pollutant: str = "pm25",
    frm: str | None = None, to: str | None = None,
    method: str = "composite", surrogates: int = 50,
) -> dict | None:
    """基于缓存的网络算中心性。无缓存返回 None。"""
    key = network_cache.compute_cache_key(pollutant, frm or "", to or "", method, surrogates)
    cached = network_cache.get_network(key)
    if cached is None:
        return None

    stations = get_stations()
    codes = [r["code"] for r in stations]
    code2idx = {c: i for i, c in enumerate(codes)}
    idx_edges = [
        {**e, "src": code2idx[e["src"]], "dst": code2idx[e["dst"]]}
        for e in cached["edges"]
    ]
    N = len(stations)
    score = composite_score(idx_edges, N)
    deg = degree_centrality(idx_edges, N)
    pr = pagerank(idx_edges, N)

    order = np.argsort(-score)[:top]
    return {
        "top": top,
        "items": [
            {
                "station_id": codes[i],
                "name": stations[i]["name"],
                "score": round(float(score[i]), 4),
                "degree": round(float(deg[i]), 4),
                "pagerank": round(float(pr[i]), 4),
                "rank": rank + 1,
            }
            for rank, i in enumerate(order)
        ],
        "mock": False,
    }
