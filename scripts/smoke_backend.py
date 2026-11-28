"""后端真实接口 smoke test：网络缓存命中 + 溯源 + 关键站点。"""
from __future__ import annotations

import json
import urllib.request


BASE = "http://127.0.0.1:8000/api/v1"
QS = "pollutant=pm25&from=2024-11-01&to=2024-11-30&surrogates=10"


def fetch(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def main():
    print("=== 1. 缓存命中：GET /networks ===")
    net = fetch(f"{BASE}/networks?{QS}")
    print(f"  mock={net['mock']}, nodes={len(net['nodes'])}, edges={len(net['edges'])}")
    print(f"  FDR 通过: {net['metrics']['n_fdr_pass']}/{net['metrics']['n_edges']}")
    print(f"  T={net['metrics']['T']}h, N={net['metrics']['N']}站")
    print(f"  Top 3 边:")
    for e in net["edges"][:3]:
        print(f"    {e['src']} → {e['dst']}  w={e['weight']:.4f}  lag={e['lag']}h  "
              f"te={e['components']['te']:.3f}  corr={e['components']['corr']:.3f}")

    print("\n=== 2. 溯源：GET /trace?station_id=1004A ===")
    tr = fetch(f"{BASE}/trace?station_id=1004A&{QS}&top_k=5")
    print(f"  目标: {tr['target']}  mock={tr['mock']}")
    print(f"  候选来源:")
    for c in tr["candidates"]:
        print(f"    [{c['depth']}级] {c['station_id']} ({c['name']})  "
              f"score={c['score']:.4f}  lag={c['lag_h']}h  path={'→'.join(c['path'])}")

    print("\n=== 3. 关键站点：GET /key-stations?top=10 ===")
    ks = fetch(f"{BASE}/key-stations?top=10&{QS}")
    print(f"  mock={ks['mock']}")
    print(f"  Top 10:")
    for it in ks["items"]:
        print(f"    #{it['rank']:2d}  {it['station_id']} ({it['name'][:8]:8s})  "
              f"score={it['score']:.4f}  deg={it['degree']:.3f}  pr={it['pagerank']:.4f}")

    print("\n=== 4. 任务列表 ===")
    tasks = fetch(f"{BASE}/tasks")
    print(f"  total={tasks['total']}")
    for t in tasks["items"][:3]:
        print(f"    {t['key']}  kind={t['kind']}  status={t['status']}  "
              f"耗时={(t['finished_at'] or 0)-(t['started_at'] or 0):.1f}s")

    print("\n=== 全部通过 ===")


if __name__ == "__main__":
    main()
