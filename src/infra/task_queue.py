"""异步任务队列骨架（内存版）。

设计（开题报告 3.3）：
- 幂等键 = hash(站点范围 + 时间窗 + 污染物 + 算法参数)，同参重复提交返回同一 task；
- 状态机 pending → running → success/failed；
- 骨架用 in-memory dict + asyncio.Task；后续可换 PostgreSQL + Celery。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class Task:
    key: str
    kind: str
    params: dict[str, Any]
    status: str = "pending"           # pending / running / success / failed
    progress: float = 0.0             # 0–1
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_key(kind: str, params: dict[str, Any]) -> str:
    """幂等键：stable JSON + sha256。"""
    payload = json.dumps({"kind": kind, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class TaskQueue:
    """内存任务队列骨架。线程安全。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}

    def submit(self, kind: str, params: dict[str, Any]) -> Task:
        """提交任务（幂等）。已存在同 key 任务则直接返回。"""
        key = compute_key(kind, params)
        with self._lock:
            if key in self._tasks:
                return self._tasks[key]
            t = Task(key=key, kind=kind, params=params)
            self._tasks[key] = t
            return t

    def get(self, key: str) -> Task | None:
        return self._tasks.get(key)

    def list(self, status: str | None = None) -> list[Task]:
        with self._lock:
            tasks = list(self._tasks.values())
        return [t for t in tasks if status is None or t.status == status]

    def update(self, key: str, **kwargs) -> None:
        with self._lock:
            t = self._tasks.get(key)
            if t is None:
                return
            for k, v in kwargs.items():
                setattr(t, k, v)

    async def run_async(self, key: str, fn: Callable[[Task], Any]) -> None:
        """异步执行 fn，更新任务状态。"""
        t = self.get(key)
        if t is None:
            return
        self.update(key, status="running", started_at=time.time())
        try:
            result = await asyncio.to_thread(fn, t)
            self.update(key, status="success", result=result, progress=1.0, finished_at=time.time())
        except Exception as e:  # noqa: BLE001
            self.update(key, status="failed", error=str(e), finished_at=time.time())


# 全局单例
_queue: TaskQueue | None = None
_queue_lock = threading.Lock()


def get_task_queue() -> TaskQueue:
    global _queue
    if _queue is None:
        with _queue_lock:
            if _queue is None:
                _queue = TaskQueue()
    return _queue
