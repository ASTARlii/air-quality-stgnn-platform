"""并发压测：以固定并发（默认 20）对四类接口施压，统计 P50/P95/吞吐。

无需额外依赖（stdlib ThreadPoolExecutor + urllib），作为 locust 的轻量对照与
可复现结果来源。产物：results/load_test_report.json

用法：python scripts/load_test.py --concurrency 20 --rounds 10
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

BASE = "http://127.0.0.1:8000/api/v1"
QNET = "pollutant=pm25&from=2022-01-01&to=2022-06-30&surrogates=100"
PRED_BODY = {
    "station_id": "1001A", "pollutant": "pm25",
    "steps": [1, 3, 6, 12, 24], "as_of": "2024-12-31T23:00:00",
}


def one(method: str, path: str, body: dict | None) -> float | None:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    r = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            resp.read()
            if resp.status != 200:
                return None
            return (time.perf_counter() - t0) * 1000
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def run(name: str, method: str, path: str, body: dict | None,
        concurrency: int, rounds: int) -> dict:
    one(method, path, body)  # warmup（含模型加载/缓存）
    lat: list[float] = []
    errs = 0

    def worker(_: int) -> list[float]:
        out = []
        for _ in range(rounds):
            r = one(method, path, body)
            out.append(r if r is not None else -1.0)
        return out

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for res in ex.map(worker, range(concurrency)):
            for v in res:
                if v < 0:
                    errs += 1
                else:
                    lat.append(v)
    wall = time.perf_counter() - t0

    if not lat:
        return {"name": name, "concurrency": concurrency, "n": 0, "errors": errs}
    row = {
        "name": name, "concurrency": concurrency, "n": len(lat), "errors": errs,
        "p50_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 1),
        "max_ms": round(float(np.max(lat)), 1),
        "throughput_rps": round(len(lat) / wall, 1),
    }
    print(f"  {name:22s} c={concurrency} n={len(lat):4d} err={errs:3d}  "
          f"P50={row['p50_ms']:7.1f}  P95={row['p95_ms']:7.1f}  "
          f"max={row['max_ms']:7.1f}  {row['throughput_rps']:8.1f} req/s")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=10, help="每个并发用户的请求次数")
    ap.add_argument("--out", type=Path, default=Path("results/load_test_report.json"))
    args = ap.parse_args()

    try:
        one("GET", "/health", None)
    except Exception as e:  # noqa: BLE001
        print(f"[error] 后端不可用：{e}")
        return 1

    specs = [
        ("predictions", "POST", "/predictions", PRED_BODY),
        ("grading", "GET", "/grading?station_id=1001A&pollutant=pm25", None),
        ("networks", "GET", f"/networks?{QNET}", None),
        ("key-stations", "GET", f"/key-stations?top=10&{QNET}", None),
    ]
    print(f"=== 并发压测：并发={args.concurrency}，每用户 {args.rounds} 请求 ===")
    rows = [run(n, m, p, b, args.concurrency, args.rounds) for n, m, p, b in specs]

    summary = {"concurrency": args.concurrency, "rounds_per_user": args.rounds, "endpoints": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
