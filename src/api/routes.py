"""API 路由：预测 / 网络 / 溯源 / 关键站点 / 分级 / 任务 / 平台。

路由层不含数值计算（开题报告决策 #1），全部委托 service。
响应统一 dict（FastAPI 自动序列化），骨架阶段不上 Pydantic 模型。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.infra.task_queue import get_task_queue
from src.infra.model_pool import get_model_pool
from src.infra.settings import get_settings
from src.service import prediction as predict_service
from src.service import network as network_service

router = APIRouter()


# ---------- 请求体（仅复杂 POST 用 Pydantic） ----------

class PredictBody(BaseModel):
    station_id: str = "1001A"
    pollutant: str = "pm25"
    steps: list[int] = [1, 6, 24]
    as_of: Optional[str] = None
    seed: int = 42


class RecomputeBody(BaseModel):
    pollutant: str = "pm25"
    from_time: Optional[str] = None
    to_time: Optional[str] = None
    method: str = "composite"
    surrogates: int = 50


# ---------- 健康与平台 ----------

@router.get("/health", tags=["平台"])
def health():
    import torch
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@router.get("/datasets", tags=["平台"])
def list_datasets():
    s = get_settings()
    return [{
        "name": "北京24站4污染物",
        "stations": 24,
        "pollutants": s["pollutants"],
        "range": {"from": "2021-01-01", "to": "2024-12-31"},
        "horizons_h": s["horizons"],
        "input_window_h": s["input_window"],
    }]


@router.get("/models", tags=["平台"])
def list_models():
    pool = get_model_pool()
    return {"loaded": pool.list_loaded()}


@router.get("/stations", tags=["平台"])
def list_stations():
    """24 个国控站元信息（前端地图/下拉框用，不触发网络构建）。"""
    return [
        {"code": r["code"], "name": r["name"], "lon": float(r["lon"]),
         "lat": float(r["lat"]), "is_reference": r["is_reference"] == "Y"}
        for r in network_service.get_stations()
    ]


# ---------- 预测与分级 ----------

@router.post("/predictions", tags=["预测"])
def predictions(body: PredictBody, with_grading: bool = False):
    try:
        if with_grading:
            return predict_service.predict_and_grade(
                station_id=body.station_id, pollutant=body.pollutant,
                steps=body.steps, as_of=body.as_of, seed=body.seed,
            )
        return predict_service.predict(
            station_id=body.station_id, pollutant=body.pollutant,
            steps=body.steps, as_of=body.as_of, seed=body.seed,
        )
    except predict_service.PredictionInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/grading", tags=["预测"])
def grading(station_id: str, pollutant: str = "pm25", steps: str = "1"):
    """简化的分级接口：先预测再分级（开题报告三类 AI 服务的复合）。"""
    step_list = [int(s) for s in steps.split(",")]
    try:
        return predict_service.predict_and_grade(station_id, pollutant, step_list)
    except predict_service.PredictionInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------- 关联网络（异步 + 缓存） ----------

@router.get("/networks", tags=["解释"])
def get_network(
    pollutant: str = "pm25",
    from_time: Optional[str] = Query(None, alias="from"),
    to_time: Optional[str] = Query(None, alias="to"),
    method: str = "composite",
    surrogates: int = 50,
):
    """缓存命中返回真值；未命中自动提交异步任务，返回 202 + task_key。"""
    cached, task = network_service.get_network(
        pollutant=pollutant, frm=from_time, to=to_time,
        method=method, surrogates=surrogates, auto_submit=True,
    )
    if cached:
        return cached
    return JSONResponse(
        status_code=202,
        content={
            "task_key": task.key,
            "status": task.status,
            "pollutant": pollutant,
            "message": "网络未缓存，已自动提交异步构建任务，请轮询 /api/v1/tasks/{task_key}",
        },
    )


@router.post("/networks/recompute", status_code=202, tags=["解释"])
def recompute_network(body: RecomputeBody):
    task = network_service.submit_network_recompute(
        pollutant=body.pollutant, frm=body.from_time, to=body.to_time,
        method=body.method, surrogates=body.surrogates,
    )
    return {"task_key": task.key, "status": task.status, "pollutant": body.pollutant,
            "duplicate": task.status != "pending"}


# ---------- 溯源 ----------

@router.get("/trace", tags=["解释"])
def get_trace(
    station_id: str,
    pollutant: str = "pm25",
    from_time: Optional[str] = Query(None, alias="from"),
    to_time: Optional[str] = Query(None, alias="to"),
    max_depth: int = 2, top_k: int = 5,
    method: str = "composite", surrogates: int = 50,
):
    result = network_service.get_trace(
        station_id=station_id, pollutant=pollutant,
        frm=from_time, to=to_time, max_depth=max_depth, top_k=top_k,
        method=method, surrogates=surrogates,
    )
    if result is None:
        return JSONResponse(
            status_code=202,
            content={
                "message": "网络未缓存，请先调用 GET /api/v1/networks 触发构建",
                "pollutant": pollutant, "mock": False,
            },
        )
    return result


# ---------- 关键站点 ----------

@router.get("/key-stations", tags=["解释"])
def get_key_stations(
    top: int = 10,
    pollutant: str = "pm25",
    from_time: Optional[str] = Query(None, alias="from"),
    to_time: Optional[str] = Query(None, alias="to"),
    method: str = "composite", surrogates: int = 50,
):
    result = network_service.get_key_stations(
        top=top, pollutant=pollutant, frm=from_time, to=to_time,
        method=method, surrogates=surrogates,
    )
    if result is None:
        return JSONResponse(
            status_code=202,
            content={
                "message": "网络未缓存，请先调用 GET /api/v1/networks 触发构建",
                "pollutant": pollutant, "mock": False,
            },
        )
    return result


# ---------- 任务中心 ----------

@router.get("/tasks", tags=["任务"])
def list_tasks(page: int = 1, page_size: int = 20):
    queue = get_task_queue()
    all_tasks = sorted(queue.list(), key=lambda t: t.created_at, reverse=True)
    total = len(all_tasks)
    items = [t.to_dict() for t in all_tasks[(page-1)*page_size : page*page_size]]
    return {"items": items, "page": page, "page_size": page_size, "total": total}


@router.get("/tasks/{task_key}", tags=["任务"])
def get_task(task_key: str):
    queue = get_task_queue()
    t = queue.get(task_key)
    if t is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_key}")
    return t.to_dict()
