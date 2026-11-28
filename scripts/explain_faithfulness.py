"""E7 解释忠实度检验（开题报告 6.4 实验 E7）。

问题：解释模块给出的"高权上游站点"是否真的影响模型预测？
方法：输入消融（input ablation）——把指定站点的输入序列置零（标准化后 = 训练集均值，
      等价于"切断该站的信息输入"），观察目标站点预测误差的上升幅度。

三个条件（每个目标站点，K=5）：
  upstream_topK : 置零 TE 网络中以该站为终点的 Top-K 上游站
  downstream_topK: 置零以该站为起点的 Top-K 下游站（方向性对照）
  random        : 置零 K 个随机站（重复 R 次取均值±标准差，作为"无信息"基线）

忠实度判据：ΔRMSE(upstream) 显著大于 ΔRMSE(random)，且大于 ΔRMSE(downstream)，
           说明网络给出的方向性高权边确实对应模型依赖的信息通路。

输出：results/e7_faithfulness_{target}.csv
用法：python scripts/explain_faithfulness.py --target pm25 --k 5 --n-windows 1000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domain.adapters.graph_wavenet import GraphWaveNetAdapter
from src.domain.dataset import AirQualityDataset, POLLUTANT_INDEX
from src.infra import network_cache


HORIZONS = 24


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_windows(ds: AirQualityDataset, n_windows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """均匀抽取 n_windows 个窗口，返回 (X, Y, M)。

    X: (S, T_in, N, C_in)  Y: (S, horizon, N)  M: (S, horizon, N)
    """
    idx = np.linspace(0, len(ds) - 1, min(n_windows, len(ds))).astype(int)
    xs, ys, ms = [], [], []
    for i in idx:
        x, y, m = ds[int(i)]
        xs.append(x.numpy())
        ys.append(y.numpy()[..., 0])
        ms.append(m.numpy())
    return np.stack(xs), np.stack(ys), np.stack(ms)


@torch.no_grad()
def forward_pred(adapter, X: np.ndarray, batch: int = 128) -> np.ndarray:
    """X: (S, T_in, N, C) → 标准化预测 (S, horizon, N)。"""
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float()
        out.append(adapter.predict(xb).cpu().numpy()[..., 0])
    return np.concatenate(out)


def rmse_single(pred2d: np.ndarray, Y: np.ndarray, M: np.ndarray, st: int) -> float:
    """单站点 RMSE。pred2d: (S, horizon) 物理量；Y/M: (S, horizon, N)。"""
    m = M[:, :, st]
    p, t = pred2d[m], Y[:, :, st][m]
    if len(p) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def sign_test_p(n_pos: int, n: int) -> float:
    """单侧符号检验：H0 中位数差=0。"""
    if n == 0:
        return float("nan")
    p = sum(math.comb(n, k) for k in range(n_pos, n + 1)) / (2 ** n)
    return float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description="E7 解释忠实度检验")
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    ap.add_argument("--target", default="pm25", choices=["pm25", "pm10", "no2", "o3"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5, help="消融邻居数")
    ap.add_argument("--n-windows", type=int, default=1000)
    ap.add_argument("--n-random", type=int, default=10, help="随机基线重复次数")
    ap.add_argument("--net-from", default="2024-10-01")
    ap.add_argument("--net-to", default="2024-11-30")
    ap.add_argument("--net-surrogates", type=int, default=100)
    ap.add_argument("--adj-source", default="network", choices=["network", "adaptive"],
                    help="network=TE 融合网络；adaptive=模型自带自适应邻接（诊断用）")
    ap.add_argument("--static-adj", type=Path, default=None,
                    help="模型训练时使用的静态邻接 npz（需与 checkpoint 匹配）")
    ap.add_argument("--ckpt-tag", default="", help="checkpoint 目录后缀，如 _static")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. 数据
    npz = data_cfg["paths"]["processed"] + "/obs.npz"
    train_ds = AirQualityDataset(
        npz_path=npz, split="train", target=args.target,
        input_window=data_cfg["features"]["input_window_h"], horizon=HORIZONS,
    )
    test_ds = AirQualityDataset(
        npz_path=npz, split="test", target=args.target,
        input_window=data_cfg["features"]["input_window_h"], horizon=HORIZONS,
        normalizer=train_ds.normalizer,
    )
    codes = test_ds.codes
    N = test_ds.N
    X, Yn, M = collect_windows(test_ds, args.n_windows)
    print(f"[data] 窗口数={len(X)}  站点={N}  目标={args.target}")

    # 反归一化用（标准化 → 物理量）
    p = POLLUTANT_INDEX[args.target]
    mean, std = train_ds.normalizer.mean[:, p], train_ds.normalizer.std[:, p]

    # 2. 模型
    device = torch.device(model_cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    static_supports = None
    if args.static_adj is not None:
        adj = np.load(args.static_adj)["adj"]
        static_supports = [torch.from_numpy(adj).float()]
    adapter = GraphWaveNetAdapter.from_config(
        model_cfg["model"], device=device, static_supports=static_supports,
    )
    ckpt = (Path(model_cfg["train"]["checkpoint_dir"])
            / f"{args.target}_seed{args.seed}{args.ckpt_tag}" / "best.pt")
    adapter.load(str(ckpt))
    print(f"[model] {ckpt}  device={device}  static_adj={args.static_adj}")

    # 3. 网络 / 邻接
    code2idx = {c: i for i, c in enumerate(codes)}
    upstream: dict[int, list[tuple[int, float]]] = {i: [] for i in range(N)}
    downstream: dict[int, list[tuple[int, float]]] = {i: [] for i in range(N)}

    if args.adj_source == "adaptive":
        # 诊断：用模型自己学到的自适应邻接（A[m,n] = m→n）
        with torch.no_grad():
            adp = torch.softmax(
                torch.relu(adapter.model.emb1 @ adapter.model.emb2.t()), dim=1
            ).cpu().numpy()
        for dst in range(N):
            order = np.argsort(-adp[:, dst])
            for src in order:
                if src == dst:
                    continue
                upstream[dst].append((int(src), float(adp[src, dst])))
                downstream[int(src)].append((dst, float(adp[src, dst])))
        print(f"[adj] 使用模型自适应邻接（诊断模式）")
    else:
        cache_key = network_cache.compute_cache_key(
            args.target, args.net_from, args.net_to, "composite", args.net_surrogates,
        )
        net = network_cache.get_network(cache_key)
        if net is None:
            print(f"[error] 网络缓存不存在（key={cache_key}），请先调用 GET /api/v1/networks 构建")
            return 1
        use_edges = [e for e in net["edges"] if e.get("fdr_pass")] or net["edges"]
        print(f"[net] 边数={len(net['edges'])}  使用(FDR通过)={len(use_edges)}")
        for e in use_edges:
            s, d = code2idx[e["src"]], code2idx[e["dst"]]
            upstream[d].append((s, e["weight"]))
            downstream[s].append((d, e["weight"]))

    for dct in (upstream, downstream):
        for k in dct:
            dct[k].sort(key=lambda t: -t[1])

    # 4. 基线（全部窗口一次前向，得到所有站点预测）
    Y = Yn * std[None, None, :] + mean[None, None, :]                 # 物理量 (S,horizon,N)
    base_phys = forward_pred(adapter, X) * std[None, None, :] + mean[None, None, :]

    def intervened_rmse(st: int, cut: list[int]) -> float:
        Xc = X.copy()
        Xc[:, :, cut, :] = 0.0                        # 置零 = 切断该站信息输入
        pr = forward_pred(adapter, Xc)[:, :, st] * std[st] + mean[st]
        return rmse_single(pr, Y, M, st)

    rng = np.random.default_rng(args.seed)
    rows = []
    for s in range(N):
        up = [i for i, _ in upstream[s][:args.k]]
        down = [i for i, _ in downstream[s][:args.k]]
        if len(up) < args.k:
            continue

        r_base = rmse_single(base_phys[:, :, s], Y, M, s)
        r_up = intervened_rmse(s, up)
        r_dn = intervened_rmse(s, down) if down else float("nan")

        rand_rmse = []
        pool = [i for i in range(N) if i != s]
        for _ in range(args.n_random):
            pick = rng.choice(pool, size=args.k, replace=False).tolist()
            rand_rmse.append(intervened_rmse(s, pick))

        rows.append({
            "station": codes[s],
            "rmse_base": r_base,
            "rmse_up": r_up,
            "rmse_down": r_dn,
            "d_up": r_up - r_base,
            "d_down": r_dn - r_base,
            "rmse_rand_mean": float(np.mean(rand_rmse)),
            "rmse_rand_std": float(np.std(rand_rmse)),
            "d_rand": float(np.mean(rand_rmse)) - r_base,
            "upstream_stations": ";".join(codes[i] for i in up),
        })
        print(f"  {codes[s]}  基线={r_base:6.2f}  上游+{rows[-1]['d_up']:+6.2f}  "
              f"下游+{rows[-1]['d_down']:+6.2f}  随机+{rows[-1]['d_rand']:+6.2f}")

    df = pd.DataFrame(rows)
    suffix = f"{args.adj_source}{args.ckpt_tag}"
    out_path = args.out / f"e7_faithfulness_{args.target}_{suffix}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 5. 汇总
    n = len(df)
    n_up_gt_rand = int((df.d_up > df.d_rand).sum())
    n_up_gt_down = int((df.d_up > df.d_down).sum())
    p_rand = sign_test_p(n_up_gt_rand, n)
    p_down = sign_test_p(n_up_gt_down, n)
    ratio = float(df.d_up.mean() / df.d_rand.mean()) if abs(df.d_rand.mean()) > 1e-9 else float("nan")

    summary = {
        "target": args.target, "seed": args.seed, "k": args.k,
        "adj_source": args.adj_source,
        "n_windows": len(X), "n_stations": n,
        "mean_d_up": float(df.d_up.mean()), "mean_d_down": float(df.d_down.mean()),
        "mean_d_rand": float(df.d_rand.mean()),
        "ratio_up_over_rand": ratio,
        "n_up_gt_rand": n_up_gt_rand, "p_sign_up_gt_rand": p_rand,
        "n_up_gt_down": n_up_gt_down, "p_sign_up_gt_down": p_down,
    }
    sum_path = args.out / f"e7_faithfulness_{args.target}_{suffix}_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== E7 汇总 ===")
    print(f"  站点数={n}, K={args.k}, 窗口={len(X)}")
    print(f"  平均 ΔRMSE：上游 {df.d_up.mean():+.3f} | 下游 {df.d_down.mean():+.3f} | 随机 {df.d_rand.mean():+.3f}")
    print(f"  上游/随机 倍数 = {ratio:.2f}×")
    print(f"  上游 > 随机 的站点：{n_up_gt_rand}/{n}（符号检验 p={p_rand:.4f}）")
    print(f"  上游 > 下游 的站点：{n_up_gt_down}/{n}（符号检验 p={p_down:.4f}）")
    print(f"[done] {out_path}\n[done] {sum_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
