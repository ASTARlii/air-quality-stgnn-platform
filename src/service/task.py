"""任务用例：提交、查询、列表（薄封装 infra/task_queue）。"""
from __future__ import annotations

from src.infra.task_queue import get_task_queue, Task


def get_task(task_key: str) -> Task | None:
    return get_task_queue().get(task_key)


def list_tasks(status: str | None = None) -> list[Task]:
    return get_task_queue().list(status=status)
