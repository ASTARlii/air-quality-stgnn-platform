"""反向溯源：沿有向边向上游追溯（关联非因果，仅供辅助研判）。

算法：从目标站点 dst 出发，沿所有 src→dst 的边向上游追溯（限定深度），按累计路径权重排序。
"""
from __future__ import annotations

from collections import defaultdict


def trace_upstream(
    edges: list[dict],
    target: int,
    max_depth: int = 2,
    top_k: int = 5,
) -> list[dict]:
    """对 target 节点向上游溯源。

    返回：[{station_id, score, lag_h, depth, path}, ...]
      score = 累计路径权重（沿 path 各边权重相乘）
      depth = 溯源深度（1=直接上游，2=二级上游）
      path  = [target, ..., src] 完整路径（节点索引）
    """
    # 入边索引：dst → [(src, weight, lag), ...]
    # 优先 FDR 通过的边；若全失败（如 surrogate 数过少），fallback 用全部边
    in_edges: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
    fdr_edges = [e for e in edges if e.get("fdr_pass", True)]
    use = fdr_edges if fdr_edges else edges
    for e in use:
        in_edges[e["dst"]].append((e["src"], e["weight"], e.get("lag", 1)))

    candidates: list[dict] = []

    def dfs(node: int, path: list[int], acc_weight: float, acc_lag: int, depth: int):
        if depth > max_depth:
            return
        for src, w, lag in in_edges.get(node, []):
            if src in path:               # 避免环
                continue
            new_w = acc_weight * w
            new_lag = acc_lag + lag
            new_path = [src] + path
            candidates.append({
                "station_id": src,
                "score": round(float(new_w), 4),
                "lag_h": new_lag,
                "depth": depth,
                "path": new_path,
            })
            dfs(src, new_path, new_w, new_lag, depth + 1)

    dfs(target, [target], 1.0, 0, 1)
    candidates.sort(key=lambda c: -c["score"])
    return candidates[:top_k]
