"""模型常驻单例池：每个 (target, seed) 对应一个常驻 adapter，避免重复加载。

参考开题报告 3.3（2）：模型常驻单例、预测同步、网络重算异步。
"""
from __future__ import annotations

import threading

import numpy as np
import torch

from src.domain.adapters.graph_wavenet import GraphWaveNetAdapter
from src.infra.settings import get_settings


class ModelPool:
    """常驻模型池。key = (target, seed)。

    优先加载带静态邻接的版本（checkpoints/{target}_seed{n}_static），
    与 E7 验证过的忠实度配置保持一致；不存在时回退到纯自适应邻接版本。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._adapters: dict[tuple[str, int], GraphWaveNetAdapter] = {}

    def get(self, target: str, seed: int = 42) -> GraphWaveNetAdapter:
        """获取或加载（首次）模型。线程安全。"""
        key = (target, seed)
        if key in self._adapters:
            return self._adapters[key]

        with self._lock:
            if key in self._adapters:
                return self._adapters[key]

            s = get_settings()
            device = torch.device(s["device"] if torch.cuda.is_available() else "cpu")

            # 静态图版本优先
            static_ckpt = s["checkpoint_dir"] / f"{target}_seed{seed}_static" / "best.pt"
            static_adj = s["project_root"] / "data" / "processed" / f"static_adj_{target}.npz"
            supports = None
            if static_ckpt.exists() and static_adj.exists():
                adj = np.load(static_adj)["adj"]
                supports = [torch.from_numpy(adj).float()]
                ckpt = static_ckpt
            else:
                ckpt = s["checkpoint_dir"] / f"{target}_seed{seed}" / "best.pt"

            if not ckpt.exists():
                raise FileNotFoundError(f"模型权重不存在: {ckpt}（请先训练 {target} seed={seed}）")

            adapter = GraphWaveNetAdapter.from_config(
                s["model"]["model"], device=device, static_supports=supports,
            )
            adapter.load(str(ckpt))
            self._adapters[key] = adapter
            return adapter

    def list_loaded(self) -> list[str]:
        return [f"{t}_seed{s}" for (t, s) in self._adapters.keys()]

    def clear(self) -> None:
        with self._lock:
            self._adapters.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# 全局单例
_pool: ModelPool | None = None
_pool_lock = threading.Lock()


def get_model_pool() -> ModelPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ModelPool()
    return _pool
