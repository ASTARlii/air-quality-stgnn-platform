"""训练入口骨架。

骨架阶段先验证：
  1. 配置可加载
  2. obs.npz 可读取并切窗
  3. 模型可前向（dry-run：用随机张量走一遍 forward，确认形状）
  4. 优化器/损失函数可创建

后续填充：完整训练循环（含按 epoch 保存检查点、早停、多随机种子）。

用法：
    python scripts/train.py --dry-run                  # 只跑前向验证，不训练
    python scripts/train.py --target pm25              # 训练 PM2.5 单输出模型
    python scripts/train.py --target pm25 --seeds 3    # 多随机种子重复
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import random
import time

from torch.utils.data import DataLoader

from src.domain.adapters.graph_wavenet import GraphWaveNetAdapter
from src.domain.dataset import build_datasets, STATION_INDEX


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """带 mask 的 MSE：mask=False 的位置 loss=0。

    pred: (B, horizon, N, 1)
    y:    (B, horizon, N, 1)
    mask: (B, horizon, N)
    """
    mask = mask.unsqueeze(-1).float()                 # (B, horizon, N, 1)
    se = (pred - y) ** 2 * mask
    return se.sum() / mask.sum().clamp(min=1.0)       # 防 0 除


def evaluate(adapter, loader, device) -> float:
    """在 valid/test 上评估 masked MSE。"""
    adapter.model.eval()
    total_se, total_n = 0.0, 0
    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            # adapter.predict 期望 (B, T_in, N, C_in)，输入 x 已是此形状
            pred = adapter.predict(x)                 # (B, horizon, N, 1)
            m = mask.unsqueeze(-1).float()
            total_se += (((pred - y) ** 2) * m).sum().item()
            total_n += m.sum().item()
    return total_se / max(total_n, 1.0)


def train_one_seed(
    data_cfg, model_cfg, target: str, seed: int, resume: bool = True,
    static_adj: Path | None = None, tag: str = "",
) -> int:
    """单 seed 完整训练循环。

    - 按 epoch 保存 last.pt（断点续训）
    - valid loss 最优时保存 best.pt
    - 早停 patience=15
    - 训练日志写入 logs/{target}_seed{seed}.jsonl
    - static_adj: 稀疏静态邻接 npz（转移熵图），存在时作为 static_supports 传入
    - tag: checkpoint/日志目录后缀，用于区分实验变体（如 "_static"）
    """
    print(f"\n=== 训练 {target} seed={seed} ===")
    # 1. 随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Dataset / DataLoader
    train_ds, valid_ds, _, _ = build_datasets(
        npz_path=data_cfg["paths"]["processed"] + "/obs.npz",
        target=target,
        input_window=data_cfg["features"]["input_window_h"],
        horizon=data_cfg["features"]["horizons_h"][-1],
    )
    train_loader = DataLoader(
        train_ds, batch_size=model_cfg["train"]["batch_size"], shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=model_cfg["train"]["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"  train batches: {len(train_loader)}, valid batches: {len(valid_loader)}")

    # 3. 模型 + 优化器
    device = torch.device(model_cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    static_supports = None
    if static_adj is not None:
        if not Path(static_adj).exists():
            raise FileNotFoundError(f"静态邻接不存在: {static_adj}")
        adj = np.load(static_adj)["adj"]
        static_supports = [torch.from_numpy(adj).float()]
        print(f"  static_adj={static_adj} 非零边={int((adj > 0).sum())}")
    adapter = GraphWaveNetAdapter.from_config(
        model_cfg["model"], device=device, static_supports=static_supports,
    )
    adapter.model.train()                              # 切回训练模式
    optimizer = torch.optim.Adam(
        adapter.model.parameters(),
        lr=model_cfg["train"]["lr"],
        weight_decay=model_cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5,
    )

    # 4. checkpoint / log 路径
    run_name = f"{target}_seed{seed}{tag}"
    ckpt_dir = Path(model_cfg["train"]["checkpoint_dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(model_cfg["train"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.jsonl"

    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"

    # 5. 断点续训
    start_epoch = 0
    best_valid = float("inf")
    no_improve = 0
    if resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        adapter.model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_valid = ck["best_valid"]
        no_improve = ck["no_improve"]
        print(f"  从 epoch {start_epoch} 续训，best_valid={best_valid:.4f}")

    # 6. 训练循环
    epochs = model_cfg["train"]["epochs"]
    patience = model_cfg["train"]["early_stop_patience"]
    grad_clip = model_cfg["train"]["grad_clip"]

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        adapter.model.train()
        total_loss, total_n = 0.0, 0
        for bi, (x, y, mask) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = adapter.model(x.permute(0, 3, 2, 1))  # (B,horizon,N,1)
            loss = masked_mse(pred, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_n += x.size(0)

            if bi % 50 == 0:
                print(f"    epoch {epoch} batch {bi}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        train_loss = total_loss / max(total_n, 1)
        valid_loss = evaluate(adapter, valid_loader, device)
        scheduler.step(valid_loss)
        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        print(f"  epoch {epoch}: train={train_loss:.4f} valid={valid_loss:.4f} lr={cur_lr:.2e} time={elapsed:.1f}s", flush=True)

        # 日志
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss,
                "lr": cur_lr, "time_s": elapsed, "ts": time.time(),
            }) + "\n")

        # 早停 + best
        if valid_loss < best_valid:
            best_valid = valid_loss
            no_improve = 0
            torch.save({"model": adapter.model.state_dict(), "epoch": epoch, "valid_loss": valid_loss},
                       best_ckpt)
        else:
            no_improve += 1

        # last checkpoint（断点续训）
        torch.save({
            "model": adapter.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "best_valid": best_valid, "no_improve": no_improve,
        }, last_ckpt)

        if no_improve >= patience:
            print(f"  早停：连续 {patience} epoch 未改善，停止于 epoch {epoch}")
            break

    print(f"=== {target} seed={seed} 完成，best_valid={best_valid:.4f} ===")
    return 0


def dry_run_dataset(data_cfg: dict, target: str = "pm25") -> int:
    """验证 Dataset：样本数、形状、归一化、3671A mask。"""
    print(f"[dry-run-dataset] target={target}")
    train, valid, test, normalizer = build_datasets(
        npz_path=data_cfg["paths"]["processed"] + "/obs.npz",
        target=target,
        input_window=data_cfg["features"]["input_window_h"],
        horizon=data_cfg["features"]["horizons_h"][-1],
    )
    print(f"  train samples: {len(train)} (期望 ~{2*365*24 - 24 - 24 + 1})")
    print(f"  valid samples: {len(valid)} (期望 ~{365*24 - 24 - 24 + 1})")
    print(f"  test  samples: {len(test)} (期望 ~{366*24 - 24 - 24 + 1})")

    # 取一个样本
    x, y, mask = train[0]
    print(f"  train[0]: x={tuple(x.shape)}, y={tuple(y.shape)}, mask={tuple(mask.shape)}")
    assert x.shape == (24, 24, 9), f"x 形状异常: {x.shape}"
    assert y.shape == (24, 24, 1), f"y 形状异常: {y.shape}"
    assert mask.shape == (24, 24), f"mask 形状异常: {mask.shape}"
    assert x.dtype == torch.float32 and y.dtype == torch.float32 and mask.dtype == torch.bool

    # 归一化参数验证（训练集 z-score 后均值应接近 0）
    print(f"  normalizer.mean shape: {normalizer.mean.shape}, std shape: {normalizer.std.shape}")
    assert normalizer.mean.shape == (24, 4), f"normalizer.mean 形状异常"
    print(f"  train.values_norm 全站 PM2.5 均值（应接近 0）: {np.nanmean(train.values_norm[:,:,0]):.4f}")
    print(f"  train.values_norm 全站 PM2.5 标准差（应接近 1）: {np.nanstd(train.values_norm[:,:,0]):.4f}")

    # 3671A mask 验证
    idx_3671A = STATION_INDEX["3671A"]
    # valid 集 3671A 缺失率（应约 5/12 ≈ 42%）
    valid_mask_3671A = valid.truth_mask[:, idx_3671A, 0]   # (T_valid,)
    valid_ratio = valid_mask_3671A.mean()
    print(f"  valid 集 3671A 真值比例: {valid_ratio*100:.1f}% (期望 ~58%，因后 5 月缺失)")
    # test 集 3671A 缺失率（应约 8/12 ≈ 67%）
    test_mask_3671A = test.truth_mask[:, idx_3671A, 0]
    test_ratio = test_mask_3671A.mean()
    print(f"  test  集 3671A 真值比例: {test_ratio*100:.1f}% (期望 ~40%，因前 7 月缺失)")
    # 训练集 3671A 应几乎完整
    train_ratio = train.truth_mask[:, idx_3671A, 0].mean()
    print(f"  train 集 3671A 真值比例: {train_ratio*100:.1f}% (期望 ~90%)")

    # get_station_mask
    sm_train = train.get_station_mask()
    sm_valid = valid.get_station_mask()
    sm_test = test.get_station_mask()
    print(f"  get_station_mask: train 可用站 {sm_train.sum()}/24, valid {sm_valid.sum()}/24, test {sm_test.sum()}/24")

    print("[dry-run-dataset] ✓ 全部通过")
    return 0


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dry_run(data_cfg: dict, model_cfg: dict) -> int:
    """骨架验证：随机张量前向，打印形状，确认无 OOM。"""
    print("[dry-run] 开始骨架验证")

    # 1. obs.npz 加载
    obs = np.load(data_cfg["paths"]["processed"] + "/obs.npz", allow_pickle=True)
    values = obs["values"]           # (35064, 24, 4)
    T, N, P = values.shape
    print(f"[dry-run] obs.npz: T={T}, N={N}, P={P}")
    assert N == model_cfg["model"]["n_nodes"], f"节点数不匹配: {N} != {model_cfg['model']['n_nodes']}"
    assert P == len(model_cfg["single_output_targets"]), f"污染物数不匹配: {P}"

    # 2. 构造随机 batch 模拟一次前向
    B = model_cfg["train"]["batch_size"]
    T_in = data_cfg["features"]["input_window_h"]
    C_in = data_cfg["features"]["input_channels"]
    device = torch.device(model_cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    adapter = GraphWaveNetAdapter.from_config(model_cfg["model"], device=device)
    batch_x = torch.randn(B, T_in, N, C_in, dtype=torch.float32)

    print(f"[dry-run] 输入 batch_x: {tuple(batch_x.shape)}, device={device}")
    out = adapter.predict(batch_x)
    print(f"[dry-run] 输出 out: {tuple(out.shape)}")
    assert out.shape == (B, model_cfg["model"]["horizon"], N, 1), f"输出形状异常: {out.shape}"

    # 3. 损失/优化器创建
    optimizer = torch.optim.Adam(
        adapter.model.parameters(),
        lr=model_cfg["train"]["lr"],
        weight_decay=model_cfg["train"]["weight_decay"],
    )
    criterion = nn.MSELoss()
    print(f"[dry-run] 优化器/损失已创建: Adam(lr={model_cfg['train']['lr']}), MSELoss")

    # 4. 参数量与显存
    n_params = sum(p.numel() for p in adapter.model.parameters())
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"[dry-run] 参数量: {n_params:,}，前向后显存峰值: {mem_mb:.1f} MB")
    else:
        print(f"[dry-run] 参数量: {n_params:,}")

    print("[dry-run] ✓ 骨架验证通过")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 Graph WaveNet 单输出模型")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--target", choices=["pm25", "pm10", "no2", "o3"], help="目标污染物")
    parser.add_argument("--seeds", type=int, default=3, help="随机种子数量")
    parser.add_argument("--seed-list", type=int, nargs="+", default=None,
                        help="显式指定种子列表（覆盖 --seeds）")
    parser.add_argument("--dry-run", action="store_true", help="只跑前向验证，不训练")
    parser.add_argument("--dry-run-dataset", action="store_true", help="只验证 Dataset，不跑模型")
    parser.add_argument("--epochs", type=int, help="覆盖 model.yaml 中的 epochs（用于快验）")
    parser.add_argument("--no-resume", action="store_true", help="不从 last.pt 续训")
    parser.add_argument("--static-adj", type=Path, default=None,
                        help="稀疏静态邻接 npz（如 data/processed/static_adj_pm25.npz）")
    parser.add_argument("--tag", default="", help="checkpoint 目录后缀，如 _static")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)

    if args.dry_run_dataset:
        return dry_run_dataset(data_cfg, target=args.target or "pm25")
    if args.dry_run:
        return dry_run(data_cfg, model_cfg)

    if not args.target:
        parser.error("非 dry-run 模式必须指定 --target")

    # 覆盖 epochs（快验）
    if args.epochs:
        model_cfg["train"]["epochs"] = args.epochs

    # 多 seed 训练
    base_seed = model_cfg["train"]["seed"]
    seeds = args.seed_list if args.seed_list else [base_seed + i for i in range(args.seeds)]
    print(f"[info] target={args.target} seeds={seeds} epochs={model_cfg['train']['epochs']}")

    for seed in seeds:
        rc = train_one_seed(
            data_cfg, model_cfg, args.target, seed,
            resume=not args.no_resume, static_adj=args.static_adj, tag=args.tag,
        )
        if rc != 0:
            print(f"[error] seed={seed} 训练失败")
            return rc

    print(f"[done] {args.target} 全部 seed 训练完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
