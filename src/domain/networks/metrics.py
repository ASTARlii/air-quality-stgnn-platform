"""网络中心性指标（不依赖 networkx，纯 numpy）。

提供：
- degree_centrality：加权度（出度 + 入度）
- pagerank：简化版 PageRank
- betweenness：介数（基于最短路径，O(N³) BFS，N=24 完全可接受）
"""
from __future__ import annotations

import numpy as np
from collections import deque


def _to_adjacency(edges: list[dict], n_nodes: int) -> np.ndarray:
    """edges → 加权邻接矩阵 (N, N)。adj[i,j] = i→j 的权重。

    策略：优先用 FDR 通过的边；若无任何边通过 FDR（如 surrogate 数过少），fallback 用全部边。
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    fdr_edges = [e for e in edges if e.get("fdr_pass", True)]
    use = fdr_edges if fdr_edges else edges
    for e in use:
        adj[e["src"], e["dst"]] = e["weight"]
    return adj


def degree_centrality(edges: list[dict], n_nodes: int) -> np.ndarray:
    """加权度中心性（出度 + 入度）/ 2。"""
    adj = _to_adjacency(edges, n_nodes)
    out_deg = adj.sum(axis=1)
    in_deg = adj.sum(axis=0)
    return (out_deg + in_deg) / 2.0


def pagerank(edges: list[dict], n_nodes: int, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """加权 PageRank。"""
    adj = _to_adjacency(edges, n_nodes)
    # 行归一化（出边权重 → 概率转移）
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-9, 1.0, row_sum)
    P = adj / row_sum
    # 处理悬挂节点（无出边）：均匀分布
    dangling = adj.sum(axis=1) < 1e-9
    P[dangling] = 1.0 / n_nodes

    pr = np.ones(n_nodes) / n_nodes
    for _ in range(max_iter):
        pr_new = (1 - damping) / n_nodes + damping * P.T @ pr
        if np.abs(pr_new - pr).sum() < tol:
            break
        pr = pr_new
    return pr


def betweenness_centrality(edges: list[dict], n_nodes: int) -> np.ndarray:
    """加权介数中心性（用 BFS 最短路径，N=24 完全可接受）。

    骨架：用二值化（边是否存在）的最短路径，不按权重加权距离。
    """
    adj = _to_adjacency(edges, n_nodes)
    # 二值化（用于 BFS）
    b_adj = (adj > 0).astype(np.int32)
    btw = np.zeros(n_nodes, dtype=np.float64)

    for s in range(n_nodes):
        # BFS 从 s 出发
        dist = np.full(n_nodes, -1, dtype=np.int32)
        sigma = np.zeros(n_nodes, dtype=np.float64)   # 最短路径条数
        paths: list[list[int]] = [[] for _ in range(n_nodes)]  # 前驱
        dist[s] = 0
        sigma[s] = 1.0
        queue = deque([s])
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in range(n_nodes):
                if b_adj[v, w] == 0:
                    continue
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    paths[w].append(v)
        # 反向累加
        delta = np.zeros(n_nodes, dtype=np.float64)
        for w in reversed(order):
            for v in paths[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                btw[w] += delta[w]
    return btw / 2.0    # 有向图不除 2，但归一化保持一致


def composite_score(edges: list[dict], n_nodes: int, w_deg: float = 0.4, w_pr: float = 0.4, w_btw: float = 0.2) -> np.ndarray:
    """综合得分 = w_deg * norm(度) + w_pr * norm(PR) + w_btw * norm(介数)。"""
    deg = degree_centrality(edges, n_nodes)
    pr = pagerank(edges, n_nodes)
    btw = betweenness_centrality(edges, n_nodes)

    def norm(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-9 else np.zeros_like(v)

    return w_deg * norm(deg) + w_pr * norm(pr) + w_btw * norm(btw)
