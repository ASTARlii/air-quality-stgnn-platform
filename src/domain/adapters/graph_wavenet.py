"""Graph WaveNet 的 ModelAdapter 实现。

适配器职责：把上层 (B, T_in, N, C_in) 的数据契约，翻译成模型需要的 (B, C_in, N, T_in)，
推理完再翻译回 (B, T_out, N, 1)。上层 service 完全感知不到 Graph WaveNet 的存在。
"""
from __future__ import annotations

import torch

from src.domain.adapters.base import ModelAdapter
from src.domain.models.graph_wavenet import GraphWaveNet, build_model_from_config


class GraphWaveNetAdapter:
    """Graph WaveNet 适配器（实现 ModelAdapter Protocol）。"""

    name = "graph_wavenet"

    def __init__(
        self,
        model: GraphWaveNet,
        device: torch.device,
        n_nodes: int,
        in_dim: int,
        horizon: int,
    ) -> None:
        # 静态邻接是常量（非参数），需手动搬到目标设备
        model.static_supports = [
            s.to(device).detach() if isinstance(s, torch.Tensor) else s
            for s in model.static_supports
        ]
        self.model = model.to(device).eval()
        self.device = device
        self.n_nodes = n_nodes
        self.in_dim = in_dim
        self.horizon = horizon

    @classmethod
    def from_config(
        cls, cfg: dict, device: torch.device | str = "cuda",
        static_supports: list[torch.Tensor] | None = None,
    ) -> "GraphWaveNetAdapter":
        """从配置构造（configs/model.yaml）。"""
        dev = torch.device(device) if isinstance(device, str) else device
        model = build_model_from_config(cfg, static_supports=static_supports)
        return cls(
            model=model,
            device=dev,
            n_nodes=cfg.get("n_nodes", 24),
            in_dim=cfg.get("in_dim", 9),
            horizon=cfg.get("horizon", 24),
        )

    def load(self, ckpt_path: str) -> None:
        """加载权重，幂等：重复 load 同一路径不重新分配显存。

        兼容三种 checkpoint 格式：
          - 直接 state_dict
          - {"state_dict": ..., ...}（常见约定）
          - {"model": ..., "epoch": ..., "valid_loss": ...}（train.py 保存的格式）
        """
        if getattr(self, "_loaded_ckpt", None) == ckpt_path:
            return
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            sd = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            sd = state["model"]
        else:
            sd = state
        self.model.load_state_dict(sd)
        self.model.eval()
        self._loaded_ckpt = ckpt_path

    @torch.no_grad()
    def predict(self, batch_x: torch.Tensor) -> torch.Tensor:
        """batch_x: (B, T_in, N, C_in) → (B, T_out, N, 1)。"""
        x = batch_x.to(self.device)
        x = x.permute(0, 3, 2, 1)                  # (B, C_in, N, T_in)
        out = self.model(x)                         # (B, out_dim, N, 1)
        return out.permute(0, 1, 2, 3)              # 已是 (B, T_out, N, 1)
