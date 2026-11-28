"""系统测试：功能用例 + 接口延迟压测（对应 docs/06-测试计划.md）。

对运行中的后端（默认 http://127.0.0.1:8000）执行：
  - 功能用例：健康、站点、预测（合法/非法）、分级、网络、溯源、关键站、任务、数据集、模型
  - 性能：predictions / grading / networks / key-stations 的 P50/P95 延迟

产物：results/system_test_report.json
用法：python scripts/system_test.py
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

BASE = "http://127.0.0.1:8000/api/v1"
QNET = "pollutant=pm25&from=2022-01-01&to=2022-06-30&surrogates=100"


def req(method: str, path: str, body: dict | None = None) -> tuple[int, dict | list | str, float]:
    url = f"{BASE}{path}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            dt = (time.perf_counter() - t0) * 1000
            try:
                return resp.status, json.loads(raw), dt
            except json.JSONDecodeError:
                return resp.status, raw, dt
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        dt = (time.perf_counter() - t0) * 1000
        try:
            return e.code, json.loads(raw), dt
        except json.JSONDecodeError:
            return e.code, raw, dt


CASES: list[dict] = []


def check(name: str, cond: bool, detail: str, na: bool = False) -> None:
    status = "NA" if na else ("PASS" if cond else "FAIL")
    CASES.append({"name": name, "status": status, "pass": bool(cond) and not na, "detail": detail})
    print(f"  [{status}] {name}  {detail}")


def functional() -> None:
    print("=== 功能用例 ===")
    st, b, _ = req("GET", "/health")
    check("TC-HEALTH /health", st == 200 and isinstance(b, dict) and b.get("status") == "ok", f"status={st}")

    st, b, _ = req("GET", "/stations")
    check("TC-STATIONS /stations", st == 200 and isinstance(b, list) and len(b) == 24, f"status={st}, n={len(b) if isinstance(b, list) else '?'}")

    st, b, _ = req("POST", "/predictions", {"station_id": "1001A", "pollutant": "pm25",
                                            "steps": [1, 3, 6, 12, 24], "as_of": "2024-12-31T23:00:00"})
    ok = st == 200 and isinstance(b, dict) and len(b.get("series", [])) == 5
    hs = sorted(s["h"] for s in b["series"]) if ok else []
    check("TC-PRED-001 合法预测", ok and hs == [1, 3, 6, 12, 24], f"status={st}, horizons={hs}")

    st, b, _ = req("POST", "/predictions", {"station_id": "1001A", "pollutant": "pm25",
                                            "steps": [1], "as_of": "2021-01-01T03:00:00"})
    st_mid, b_mid, _ = req("POST", "/predictions", {"station_id": "1001A", "pollutant": "pm25",
                                                    "steps": [1], "as_of": "2024-06-01T00:00:00"})
    st_end, b_end, _ = req("POST", "/predictions", {"station_id": "1001A", "pollutant": "pm25",
                                                    "steps": [1], "as_of": "2024-12-31T23:00:00"})
    differs = None
    if st_mid == 200 and st_end == 200 and b_mid.get("series") and b_end.get("series"):
        differs = b_mid["series"][0]["value"] != b_end["series"][0]["value"]
    check("TC-PRED-002 as_of 驱动窗口（含不足24h）", st == 400 and differs is True,
          f"early_status={st}, windows_differ={differs}")

    st, b, _ = req("POST", "/predictions", {"station_id": "1001A", "pollutant": "xx99",
                                            "steps": [1], "as_of": "2024-12-31T23:00:00"})
    check("TC-PRED-003 非法污染物枚举", st in (400, 422), f"status={st}（期望 400/422）")

    st, b, _ = req("GET", "/grading?station_id=1001A&pollutant=pm25")
    ok = st == 200 and isinstance(b, dict) and b.get("series")
    lvl = b["series"][0]["grading"]["level"] if ok else None
    check("TC-GRADING 分级", ok and 1 <= lvl <= 6, f"status={st}, level={lvl}")

    st, b, _ = req("GET", f"/networks?{QNET}")
    ok = st == 200 and isinstance(b, dict) and len(b.get("edges", [])) > 0
    check("TC-NET 关联网络（缓存命中）", ok, f"status={st}, edges={len(b.get('edges', [])) if isinstance(b, dict) else '?'}")

    st, b, _ = req("GET", f"/trace?station_id=1004A&{QNET}&top_k=5")
    ok = st == 200 and isinstance(b, dict) and isinstance(b.get("candidates"), list) and len(b["candidates"]) > 0
    check("TC-TRACE 污染溯源", ok, f"status={st}, candidates={len(b.get('candidates', [])) if isinstance(b, dict) else '?'}")

    st, b, _ = req("GET", f"/key-stations?top=10&{QNET}")
    ok = st == 200 and isinstance(b, dict) and len(b.get("items", [])) == 10
    check("TC-KEY 关键站点 Top-10", ok, f"status={st}, n={len(b.get('items', [])) if isinstance(b, dict) else '?'}")

    st, b, _ = req("GET", "/tasks")
    check("TC-TASKS 任务列表", st == 200 and isinstance(b, dict) and "items" in b, f"status={st}, total={b.get('total') if isinstance(b, dict) else '?'}")

    st, b, _ = req("GET", "/datasets")
    first = b[0] if isinstance(b, list) and b else (b if isinstance(b, dict) else {})
    ok = st == 200 and first.get("stations") == 24
    check("TC-DATASETS 数据集元信息", ok, f"status={st}, stations={first.get('stations')}")

    st, b, _ = req("GET", "/models")
    ok = st == 200 and isinstance(b, dict) and len(b.get("loaded", [])) > 0
    check("TC-MODELS 模型版本", ok, f"status={st}, loaded={b.get('loaded') if isinstance(b, dict) else '?'}")

    # 幂等：相同参数重复提交返回同一 task_key（小窗口、低 surrogate，避免重负载）
    body = {"pollutant": "pm25", "from": "2024-12-01", "to": "2024-12-02", "surrogates": 2}
    s1, b1, _ = req("POST", "/networks/recompute", body)
    s2, b2, _ = req("POST", "/networks/recompute", body)
    k1 = b1.get("task_key") if isinstance(b1, dict) else None
    k2 = b2.get("task_key") if isinstance(b2, dict) else None
    if s1 in (200, 202) and k1:
        check("TC-NET-001 重算幂等", k1 == k2, f"task_key={k1}")
    else:
        check("TC-NET-001 重算幂等", False, f"status={s1}, body={str(b1)[:80]}")


def perf() -> list[dict]:
    print("\n=== 性能测试（warm 后多次采样）===")
    specs = [
        ("predictions", "POST", "/predictions",
         {"station_id": "1001A", "pollutant": "pm25", "steps": [1, 3, 6, 12, 24],
          "as_of": "2024-12-31T23:00:00"}, 20),
        ("grading", "GET", "/grading?station_id=1001A&pollutant=pm25", None, 30),
        ("networks", "GET", f"/networks?{QNET}", None, 20),
        ("key-stations", "GET", f"/key-stations?top=10&{QNET}", None, 20),
    ]
    rows = []
    for name, method, path, body, n in specs:
        req(method, path, body)  # warmup
        lat = []
        for _ in range(n):
            st, _, dt = req(method, path, body)
            if st == 200:
                lat.append(dt)
        if not lat:
            rows.append({"name": name, "n": 0})
            print(f"  {name:12s} 无成功样本")
            continue
        row = {"name": name, "n": len(lat), "p50_ms": round(float(np.percentile(lat, 50)), 1),
               "p95_ms": round(float(np.percentile(lat, 95)), 1), "max_ms": round(float(np.max(lat)), 1)}
        rows.append(row)
        print(f"  {name:12s} n={len(lat):2d}  P50={row['p50_ms']:7.1f}  P95={row['p95_ms']:7.1f}  max={row['max_ms']:7.1f} ms")
    return rows


def main() -> int:
    try:
        req("GET", "/health")
    except Exception as e:  # noqa: BLE001
        print(f"[error] 后端未运行：{e}")
        return 1

    functional()
    rows = perf()
    passed = sum(c["pass"] for c in CASES)
    na = sum(c["status"] == "NA" for c in CASES)
    total = len(CASES)
    graded = total - na
    summary = {"n_cases": total, "n_pass": passed, "n_na": na,
               "pass_rate": round(passed / graded, 4) if graded else 0.0,
               "cases": CASES, "perf": rows}
    out = Path("results/system_test_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== 汇总：功能用例 {passed}/{graded} 通过（{passed/graded:.0%}），另有 {na} 项 N/A ===")
    print(f"[done] {out}")
    return 0 if passed == graded else 2


if __name__ == "__main__":
    raise SystemExit(main())
