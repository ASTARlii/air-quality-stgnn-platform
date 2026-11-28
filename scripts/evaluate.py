"""测试集评估脚本（开题报告 6.4 评价指标）。

指标：RMSE / MAE / MAPE / R2（按 horizon=1/3/6/12/24h 分别报告）
口径：
  - full    : 全 24 站（含 3671A 等，mask 自动屏蔽缺测段）
  - drop3671A: 剔除 3671A 门头沟三家店（23 站）

多 seed 汇总：对每个 (horizon × 口径 × 指标) 输出 mean ± std。

输出：
  results/{target}_test_metrics.csv    # 每 seed × horizon × 口径 × 指标
  results/{target}_test_summary.csv    # 跨 seed mean±std

用法：
    python scripts/evaluate.py --target pm25 --seeds 42 43 44
    python scripts/evaluate.py --target pm25 --seeds 42 --split valid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domain.adapters.graph_wavenet import GraphWaveNetAdapter
from src.domain.dataset import AirQualityDataset, Normalizer, STATION_INDEX


HORIZON_REPORT = [1, 3, 6, 12, 24]   # 报告的预测步长
STATION_3671A = STATION_INDEX["3671A"]


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def predict_all(adapter, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """全 split 前向，返回反归一化后的 (pred, y, mask)。

    pred / y: (N_samples, horizon, N, 1)   物理量
    mask    : (N_samples, horizon, N)      bool
    """
    adapter.model.eval()
    preds, ys, masks = [], [], []
    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        pred = adapter.predict(x).cpu().numpy()        # (B,horizon,N,1) 标准化空间
        preds.append(pred)
        ys.append(y.numpy())
        masks.append(mask.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(masks)


def inverse_norm(arr: np.ndarray, normalizer: Normalizer, target: str) -> np.ndarray:
    """标准化 → 物理量。arr: (..., N, 1)。"""
    p = {"pm25": 0, "pm10": 1, "no2": 2, "o3": 3}[target]
    mean = normalizer.mean[:, p]                       # (N,)
    std = normalizer.std[:, p]                         # (N,)
    # arr[..., n, 0] * std[n] + mean[n]
    return arr[..., 0] * std + mean


def compute_metrics(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    """pred/y: (N, ...) 已展平，mask: (N, ...) 同形。

    返回 {rmse, mae, mape}。
    """
    p = pred[mask]
    t = y[mask]
    if len(p) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "r2": float("nan")}
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    # MAPE 只对真值 > 1 μg/m³ 计算（避免除零/小值失真）
    nonzero = t > 1.0
    mape = float(np.mean(np.abs((p[nonzero] - t[nonzero]) / t[nonzero])) * 100) if nonzero.any() else float("nan")
    # R2 = 1 - SS_res / SS_tot；目标为常数（SS_tot=0）时无定义，记为 nan
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    r2 = float(1.0 - np.sum((p - t) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def eval_one_seed(
    data_cfg: dict, model_cfg: dict, target: str, seed: int,
    split: str = "test", batch_size: int = 64,
    static_adj: Path | None = None, tag: str = "",
) -> pd.DataFrame:
    """单 seed 在指定 split 上评估，返回 tidy DataFrame。"""
    # 训练时 fit 的 normalizer：从 train 集重算（因为 Normalizer 没存到 checkpoint）
    train_ds = AirQualityDataset(
        npz_path=data_cfg["paths"]["processed"] + "/obs.npz",
        split="train", target=target,
        input_window=data_cfg["features"]["input_window_h"],
        horizon=data_cfg["features"]["horizons_h"][-1],
    )
    eval_ds = AirQualityDataset(
        npz_path=data_cfg["paths"]["processed"] + "/obs.npz",
        split=split, target=target,
        input_window=data_cfg["features"]["input_window_h"],
        horizon=data_cfg["features"]["horizons_h"][-1],
        normalizer=train_ds.normalizer,
    )

    device = torch.device(model_cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    static_supports = None
    if static_adj is not None:
        adj = np.load(static_adj)["adj"]
        static_supports = [torch.from_numpy(adj).float()]
    adapter = GraphWaveNetAdapter.from_config(
        model_cfg["model"], device=device, static_supports=static_supports,
    )
    ckpt_path = Path(model_cfg["train"]["checkpoint_dir"]) / f"{target}_seed{seed}{tag}" / "best.pt"
    adapter.load(str(ckpt_path))
    print(f"  seed={seed} ckpt={ckpt_path}")

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    pred_n, y_n, mask = predict_all(adapter, loader, device)   # 标准化空间

    # 反归一化到物理量
    pred = inverse_norm(pred_n, train_ds.normalizer, target)   # (Ns, horizon, N)
    y = inverse_norm(y_n, train_ds.normalizer, target)

    rows = []
    for h in HORIZON_REPORT:
        hi = h - 1                                             # horizon 索引 0-based
        for caliber, stations in [
            ("full", np.arange(eval_ds.N)),
            ("drop3671A", np.array([i for i in range(eval_ds.N) if i != STATION_3671A])),
        ]:
            p_h = pred[:, hi, :][:, stations]                  # (Ns, len(stations))
            y_h = y[:, hi, :][:, stations]
            m_h = mask[:, hi, :][:, stations]
            m = compute_metrics(p_h, y_h, m_h)
            rows.append({
                "target": target, "seed": seed, "split": split,
                "horizon_h": h, "caliber": caliber,
                "rmse": m["rmse"], "mae": m["mae"], "mape": m["mape"], "r2": m["r2"],
                "n_valid": int(m_h.sum()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="测试集评估")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--target", required=True, choices=["pm25", "pm10", "no2", "o3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--static-adj", type=Path, default=None,
                        help="模型使用的静态邻接 npz（需与 checkpoint 匹配）")
    parser.add_argument("--tag", default="", help="checkpoint 目录后缀，如 _static")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[info] target={args.target} split={args.split} seeds={args.seeds} tag={args.tag or '-'}")

    all_dfs = []
    for seed in args.seeds:
        ckpt = Path(model_cfg["train"]["checkpoint_dir"]) / f"{args.target}_seed{seed}{args.tag}" / "best.pt"
        if not ckpt.exists():
            print(f"  [skip] seed={seed} 无 checkpoint: {ckpt}")
            continue
        df = eval_one_seed(
            data_cfg, model_cfg, args.target, seed, args.split,
            static_adj=args.static_adj, tag=args.tag,
        )
        all_dfs.append(df)

    if not all_dfs:
        print("[error] 没有可用 checkpoint")
        return 1

    metrics = pd.concat(all_dfs, ignore_index=True)
    tag_str = args.tag or ""
    metrics_path = args.out / f"{args.target}_{args.split}{tag_str}_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"[done] 明细已保存: {metrics_path}")

    # 汇总：mean ± std
    summary = (
        metrics.groupby(["horizon_h", "caliber"])[["rmse", "mae", "mape", "r2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["horizon_h", "caliber"] + [
        f"{m}_{s}" for m in ["rmse", "mae", "mape", "r2"] for s in ["mean", "std"]
    ]
    summary_path = args.out / f"{args.target}_{args.split}{tag_str}_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[done] 汇总已保存: {summary_path}")

    # 控制台打印紧凑摘要
    print("\n=== 汇总（mean ± std）===")
    for caliber in ["full", "drop3671A"]:
        sub = summary[summary.caliber == caliber]
        print(f"\n口径 = {caliber}（{'全 24 站' if caliber=='full' else '剔除 3671A，23 站'}）")
        print(f"{'h':>3}  {'RMSE':>14}  {'MAE':>14}  {'MAPE(%)':>14}  {'R2':>10}")
        for _, r in sub.iterrows():
            r2_str = f"{r.r2_mean:>6.3f} ± {r.r2_std:<4.3f}" if not np.isnan(r.r2_mean) else "     nan"
            print(f"{int(r.horizon_h):>3}  "
                  f"{r.rmse_mean:>6.2f} ± {r.rmse_std:<5.2f}  "
                  f"{r.mae_mean:>6.2f} ± {r.mae_std:<5.2f}  "
                  f"{r.mape_mean:>6.2f} ± {r.mape_std:<5.2f}  "
                  f"{r2_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
