"""E2 空间结构消融（仅 PM2.5）：纯 TCN vs 固定距离图 vs 完整模型。

三组配置：
  1. 纯 TCN            ：去掉图卷积与自适应邻接（use_graph_conv=False, use_adaptive_adj=False）
  2. 固定距离图         ：自适应邻接 + 经纬度高斯距离图（行归一化），静态图来源=距离
  3. 完整模型（本课题） ：自适应邻接 + 转移熵静态图 —— 直接复用 results/pm25_test_static_summary.csv

为与主模型严格可比，训练/评估直接复用 scripts/train.py 的 train_one_seed 与
scripts/evaluate.py 的 eval_one_seed（同切分、同标准化、同 24h→24h、同 5 步长、
同 full/drop3671A 口径、同 3 seed 42/43/44）。

用法：
  python scripts/run_e2.py --variant dist --seeds 42            # 冒烟：距离图单 seed
  python scripts/run_e2.py --variant all  --seeds 42 43 44      # 全量：两变体 × 3 seed
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))     # scripts/ 下的 train、evaluate

from train import train_one_seed                              # noqa: E402
from evaluate import eval_one_seed                            # noqa: E402


def load_yaml(p: Path) -> dict:
    import yaml
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_distance_adj(codes_order: list[str], stations_csv: Path) -> np.ndarray:
    """经纬度高斯核距离图，行归一化（与转移熵静态图同归一化方式），对角置 0。"""
    df = pd.read_csv(stations_csv)
    df["code"] = df["code"].astype(str)
    sub = df.set_index("code").loc[list(codes_order)]
    lon = sub["lon"].to_numpy(np.float32)
    lat = sub["lat"].to_numpy(np.float32)
    lat0 = np.radians(lat.mean())
    x = np.radians(lon) * 6371.0 * np.cos(lat0)
    y = np.radians(lat) * 6371.0
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    sigma = float(np.median(d[d > 0]))
    A = np.exp(-(d ** 2) / (sigma ** 2)).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    row = A.sum(axis=1, keepdims=True)
    A = np.divide(A, row, out=np.zeros_like(A), where=row > 1e-9)
    return A


def ensure_distance_npz(npz: str, stations_csv: Path, out_npz: Path) -> Path:
    if out_npz.exists():
        print(f"[dist] 复用已有距离图: {out_npz}")
        return out_npz
    d = np.load(npz, allow_pickle=True)
    codes = [str(c) for c in d["codes"]]
    A = build_distance_adj(codes, stations_csv)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, adj=A, codes=np.array(codes))
    print(f"[dist] 已构建距离图: {out_npz}  shape={A.shape}  "
          f"非零={int((A > 0).sum())}  行和≈{A.sum(1).mean():.3f}")
    return out_npz


def save_outputs(metrics: pd.DataFrame, target: str, tag: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / f"{target}_test{tag}_metrics.csv"
    metrics.to_csv(mpath, index=False, encoding="utf-8-sig")
    summary = (
        metrics.groupby(["horizon_h", "caliber"])[["rmse", "mae", "mape", "r2"]]
        .agg(["mean", "std"]).reset_index()
    )
    summary.columns = ["horizon_h", "caliber"] + [
        f"{m}_{s}" for m in ["rmse", "mae", "mape", "r2"] for s in ["mean", "std"]
    ]
    spath = out / f"{target}_test{tag}_summary.csv"
    summary.to_csv(spath, index=False, encoding="utf-8-sig")
    print(f"[done] {mpath}\n[done] {spath}")


def run_variant(variant: str, seeds: list[int], target: str,
                data_cfg: dict, model_cfg: dict, out: Path,
                dist_npz: Path) -> None:
    tag = f"_{variant}"
    cfg = copy.deepcopy(model_cfg)
    static_adj = None
    if variant == "dist":
        static_adj = dist_npz
        cfg["model"]["use_adaptive_adj"] = True
        cfg["model"]["use_graph_conv"] = True
        desc = "自适应邻接 + 经纬度高斯距离图"
    elif variant == "tcn":
        cfg["model"]["use_adaptive_adj"] = False
        cfg["model"]["use_graph_conv"] = False
        desc = "纯 TCN（无图卷积、无自适应邻接）"
    elif variant == "static_only":
        static_adj = Path(data_cfg["paths"]["processed"]) / f"static_adj_{target}.npz"
        cfg["model"]["use_adaptive_adj"] = False
        cfg["model"]["use_graph_conv"] = True
        desc = "仅转移熵静态图（无自适应邻接）"
    else:
        raise ValueError(f"未知变体: {variant}")

    print(f"\n===== E2 变体 {variant}（{desc}）target={target} seeds={seeds} =====")
    dfs = []
    for seed in seeds:
        ckpt = Path(cfg["train"]["checkpoint_dir"]) / f"{target}_seed{seed}{tag}" / "best.pt"
        if ckpt.exists():
            print(f"  [skip-train] seed={seed} 已有 checkpoint: {ckpt}")
        else:
            train_one_seed(data_cfg, cfg, target, seed, resume=False,
                           static_adj=static_adj, tag=tag)
        df = eval_one_seed(data_cfg, cfg, target, seed, split="test",
                           static_adj=static_adj, tag=tag)
        dfs.append(df)
    metrics = pd.concat(dfs, ignore_index=True)
    save_outputs(metrics, target, tag, out)

    sub = metrics[(metrics.horizon_h == 24) & (metrics.caliber == "full")]
    print(f"  [{variant}] h=24 full RMSE = {sub.rmse.mean():.2f} ± {sub.rmse.std():.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 空间结构消融")
    ap.add_argument("--variant", choices=["dist", "tcn", "static_only", "all"], default="all")
    ap.add_argument("--target", default="pm25", choices=["pm25", "pm10", "no2", "o3", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--stations", type=Path, default=Path("data/raw/stations.csv"))
    ap.add_argument("--dist-npz", type=Path, default=Path("data/processed/dist_adj.npz"))
    args = ap.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    obs_npz = data_cfg["paths"]["processed"] + "/obs.npz"

    variants = ["dist", "tcn"] if args.variant == "all" else [args.variant]
    targets = ["pm25", "pm10", "no2", "o3"] if args.target == "all" else [args.target]
    dist_npz = args.dist_npz
    if "dist" in variants:
        dist_npz = ensure_distance_npz(obs_npz, args.stations, args.dist_npz)

    print(f"[info] variants={variants} targets={targets} seeds={args.seeds}")
    for target in targets:
        for v in variants:
            run_variant(v, args.seeds, target, data_cfg, model_cfg, args.out, dist_npz)

    print("\n[done] E2 消融全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
