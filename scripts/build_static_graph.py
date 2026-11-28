"""构建稀疏静态邻接（供 Graph WaveNet 作为 static_supports 使用）。

关键约束：**只能用训练期数据建图**（2021-01-01 ~ 2022-12-31），
否则邻接里含测试期信息，属于数据泄漏。

构图：转移熵 + 滞后相关融合网络 → 每站保留 Top-K 入边 → 对称化 → 行归一化。

输出：
  data/processed/static_adj_{target}.npz   # adj (N,N) float32 + codes
  data/processed/networks/{key}.json       # 网络缓存（E7 可直接复用）

用法：
  python scripts/build_static_graph.py --target pm25 --from 2022-01-01 --to 2022-06-30 --top-k 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domain.networks.composite import CompositeBuilder
from src.infra import network_cache


def main() -> int:
    ap = argparse.ArgumentParser(description="构建稀疏静态邻接")
    ap.add_argument("--target", default="pm25", choices=["pm25", "pm10", "no2", "o3"])
    ap.add_argument("--from", dest="frm", default="2022-01-01")
    ap.add_argument("--to", dest="to", default="2022-06-30")
    ap.add_argument("--surrogates", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=5, help="每站保留的入边数")
    ap.add_argument("--npz", default="data/processed/obs.npz")
    ap.add_argument("--stations", default="data/raw/stations.csv")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    values = d["values"]
    ts = d["timestamps"].astype(np.int64)
    codes = [str(c) for c in d["codes"]]
    pollutants = [str(p) for p in d["pollutants"]]
    p = pollutants.index(args.target)

    lo = int(datetime.fromisoformat(args.frm).replace(tzinfo=timezone.utc).timestamp())
    hi = int(datetime.fromisoformat(args.to).replace(tzinfo=timezone.utc).timestamp())
    mask = (ts >= lo) & (ts <= hi)
    series = values[mask, :, p]
    N = series.shape[1]
    print(f"[data] {args.frm} ~ {args.to}  T={series.shape[0]}h  N={N}  缺失率={np.isnan(series).mean()*100:.1f}%")
    assert hi < 1672502400, "静态图必须只用训练期数据（< 2023-01-01），否则数据泄漏"

    print(f"[net] 构建融合网络（surrogates={args.surrogates}）…")
    edges = CompositeBuilder(
        w_te=0.5, w_corr=0.5, w_wind=0.0, n_surrogates=args.surrogates, seed=42,
    ).build(series)
    n_pass = sum(1 for e in edges if e["fdr_pass"])
    print(f"[net] 边={len(edges)}  FDR通过={n_pass}")

    # 每站保留 Top-K 入边（按融合权重），不强制 FDR（稀疏化本身就是筛选）
    incoming: dict[int, list[tuple[int, float]]] = {i: [] for i in range(N)}
    for e in edges:
        incoming[e["dst"]].append((e["src"], e["weight"]))
    W = np.zeros((N, N), dtype=np.float32)      # W[src, dst]
    kept = 0
    for dst, lst in incoming.items():
        lst.sort(key=lambda t: -t[1])
        for src, w in lst[: args.top_k]:
            W[src, dst] = w
            kept += 1
    print(f"[adj] 保留 {kept} 条边（每站 Top-{args.top_k}）")

    # 对称化 + 行归一化
    A = np.maximum(W, W.T)
    row = A.sum(axis=1, keepdims=True)
    A = np.divide(A, row, out=np.zeros_like(A), where=row > 1e-9)
    np.fill_diagonal(A, 0.0)
    print(f"[adj] 对称化后非零={int((A > 0).sum())}  行和范围=[{A.sum(1).min():.3f}, {A.sum(1).max():.3f}]")

    # 保存
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"static_adj_{args.target}.npz"
    np.savez(npz_path, adj=A, codes=np.array(codes))

    # 同时写网络缓存，供 E7 复用同一张图
    stations = list(csv.DictReader(open(args.stations, encoding="utf-8-sig")))
    result = {
        "pollutant": args.target, "from": args.frm, "to": args.to,
        "method": "composite", "surrogates": args.surrogates,
        "nodes": [{"id": r["code"], "name": r["name"],
                   "lon": float(r["lon"]), "lat": float(r["lat"])} for r in stations],
        "edges": [
            {**{k: v for k, v in e.items() if k not in ("src", "dst")},
             "src": codes[e["src"]], "dst": codes[e["dst"]],
             "src_idx": e["src"], "dst_idx": e["dst"]}
            for e in edges
        ],
        "metrics": {
            "n_edges": len(edges), "n_fdr_pass": n_pass,
            "T": int(series.shape[0]), "N": N, "static_top_k": args.top_k,
        },
        "mock": False,
    }
    key = network_cache.compute_cache_key(args.target, args.frm, args.to, "composite", args.surrogates)
    network_cache.put_network(key, result)

    print(f"[done] {npz_path}")
    print(f"[done] 缓存 key={key}（E7 请用 --net-from {args.frm} --net-to {args.to} --net-surrogates {args.surrogates}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
