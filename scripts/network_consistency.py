"""E6 关联网络结构一致性（开题报告 6.3 E6）。

对比两个网络：
  1. 解释主网络：转移熵 + 滞后相关融合网络（训练期数据构建）
  2. 模型自适应邻接：A = softmax(relu(E1 @ E2ᵀ), dim=1)

指标：
  - 边权 Spearman 秩相关（552 条有向边）
  - Top-K 边重叠 Jaccard（K = 24/60/120/240）
  - 关键站点中心性一致性（Top-120 子图上的综合中心性 Spearman）
  - 两网络各自的关键站点 Top-5

方向约定统一为 A[m, n] = m → n（与模型 NConv 中 x @ adj.T 的语义一致）。

用法：python scripts/network_consistency.py --targets pm25 pm10 no2 o3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domain.networks.metrics import composite_score
from src.infra import network_cache


def _rank(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列值）。"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def load_yaml(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def te_weight_matrix(net: dict, codes: list[str]) -> np.ndarray:
    """TE 网络 → (N, N) 权重矩阵，W[m, n] = m→n。"""
    idx = {c: i for i, c in enumerate(codes)}
    N = len(codes)
    W = np.zeros((N, N), dtype=np.float64)
    for e in net["edges"]:
        W[idx[e["src"]], idx[e["dst"]]] = e["weight"]
    return W


def adp_weight_matrix(state: dict, N: int) -> np.ndarray:
    """从 checkpoint 的节点嵌入算自适应邻接。"""
    e1 = state["emb1"].float()
    e2 = state["emb2"].float()
    return torch.softmax(torch.relu(e1 @ e2.t()), dim=1).numpy().astype(np.float64)


def topk_jaccard(a: np.ndarray, b: np.ndarray, off: np.ndarray, k: int) -> float:
    """按权重取 Top-K 有向边，返回 Jaccard。off 为 off-diagonal 的扁平索引。"""
    ia = off[np.argsort(-a.ravel()[off])[:k]]
    ib = off[np.argsort(-b.ravel()[off])[:k]]
    sa, sb = set(ia.tolist()), set(ib.tolist())
    return len(sa & sb) / len(sa | sb) if sa | sb else float("nan")


def subgraph_edges(W: np.ndarray, k: int | None) -> tuple[list[dict], int]:
    """取 Top-K 边构造 edges 列表（k=None 表示用全部非零边）。"""
    N = W.shape[0]
    tri = [(W[i, j], i, j) for i in range(N) for j in range(N) if i != j]
    tri.sort(reverse=True)
    if k is not None:
        tri = tri[:k]
    edges = [{"src": i, "dst": j, "weight": float(w)} for w, i, j in tri if w > 0]
    return edges, len(tri)


def run_one(target: str, model_cfg: dict, net_from: str, net_to: str, surrogates: int,
            seed: int, codes: list[str], out_dir: Path) -> dict:
    N = len(codes)
    ckpt = Path(model_cfg["train"]["checkpoint_dir"]) / f"{target}_seed{seed}" / "best.pt"
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model", state.get("state_dict", state))
    if "emb1" not in sd:
        return {"target": target, "error": "checkpoint 无节点嵌入（use_adaptive_adj=False）"}
    A = adp_weight_matrix(sd, N)

    key = network_cache.compute_cache_key(target, net_from, net_to, "composite", surrogates)
    net = network_cache.get_network(key)
    if net is None:
        return {"target": target, "error": f"网络缓存缺失 key={key}"}
    W = te_weight_matrix(net, codes)

    off = np.array([i * N + j for i in range(N) for j in range(N) if i != j])

    # 1) 边权秩相关
    rho = spearman(W.ravel()[off], A.ravel()[off])

    # 2) Top-K 边重叠
    jac = {str(k): round(topk_jaccard(W, A, off, k), 4) for k in (24, 60, 120, 240)}

    # 3) Top-120 子图上的关键站点一致性（与静态图稀疏度一致）
    e_te, _ = subgraph_edges(W, 120)
    e_adp, _ = subgraph_edges(A, 120)
    s_te = composite_score(e_te, N)
    s_adp = composite_score(e_adp, N)
    rho_cent = spearman(s_te, s_adp)

    top5_te = [codes[i] for i in np.argsort(-s_te)[:5]]
    top5_adp = [codes[i] for i in np.argsort(-s_adp)[:5]]
    overlap_top5 = len(set(top5_te) & set(top5_adp))

    res = {
        "target": target, "seed": seed,
        "net_window": f"{net_from} ~ {net_to}",
        "rho_edge_weight": round(rho, 4),
        "jaccard_topk": jac,
        "rho_centrality_top120": round(rho_cent, 4),
        "top5_te": top5_te, "top5_adp": top5_adp,
        "overlap_top5": overlap_top5,
    }

    print(f"\n=== {target.upper()} ===")
    print(f"  边权 Spearman ρ = {rho:+.4f}  （552 条有向边；ρ≈0 说明两者几乎无关）")
    print(f"  Top-K Jaccard: " + "  ".join(f"K={k}: {v:.3f}" for k, v in jac.items()))
    print(f"  中心性 Spearman ρ = {rho_cent:+.4f}（Top-120 子图）")
    print(f"  TE  关键站 Top5: {top5_te}")
    print(f"  ADP 关键站 Top5: {top5_adp}  （重合 {overlap_top5}/5）")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"e6_consistency_{target}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="E6 关联网络结构一致性")
    ap.add_argument("--targets", nargs="+", default=["pm25", "pm10", "no2", "o3"])
    ap.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--net-from", default="2022-01-01")
    ap.add_argument("--net-to", default="2022-06-30")
    ap.add_argument("--net-surrogates", type=int, default=100)
    ap.add_argument("--stations", default="data/raw/stations.csv")
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    import csv
    codes = [r["code"] for r in csv.DictReader(open(args.stations, encoding="utf-8-sig"))]
    model_cfg = load_yaml(args.model_config)

    results = []
    for t in args.targets:
        r = run_one(t, model_cfg, args.net_from, args.net_to,
                    args.net_surrogates, args.seed, codes, args.out)
        results.append(r)

    ok = [r for r in results if "error" not in r]
    if ok:
        print("\n=== 汇总 ===")
        print(f"{'污染物':<8}{'边权ρ':>10}{'J@120':>10}{'中心性ρ':>10}{'Top5重合':>10}")
        for r in ok:
            print(f"{r['target']:<8}{r['rho_edge_weight']:>+10.4f}"
                  f"{r['jaccard_topk']['120']:>10.3f}{r['rho_centrality_top120']:>+10.4f}"
                  f"{r['overlap_top5']:>8}/5")
    for r in results:
        if "error" in r:
            print(f"[skip] {r['target']}: {r['error']}")
    print(f"\n[done] 结果已写入 {args.out}/e6_consistency_*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
