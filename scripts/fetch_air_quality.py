"""下载 quotsoft 空气质量历史数据（单日 CSV，UTF-8 编码）。

数据源（已实测验证，2026-11）：
    https://quotsoft.net/air/data/{product}_{YYYYMMDD}.csv

可用 product：
    china_sites    全国监测点，全部 6 项污染物，列名为站点编码（如 1001A），约 1.9 MB/天
    beijing_all    北京站，仅 PM2.5/PM10/AQI，列名为中文站名，约 12 KB/天
    beijing_extra  北京站，SO2/NO2/O3/CO，列名为中文站名，约 16 KB/天
    china_cities   全国城市均值，约 400 KB/天

站点经纬度需另行下载官方"站点列表"（quotsoft 首页提供百度网盘链接，
字段：监测点编码, 监测点名称, 城市, 经度, 纬度, 对照点），
放在 data/raw/stations.csv 后即可与 china_sites 的列名 join。

用法：
    python scripts/fetch_air_quality.py --probe
    python scripts/fetch_air_quality.py --start 2021-01-01 --end 2024-12-31
    python scripts/fetch_air_quality.py --product beijing_all --start 2021-01-01 --end 2021-01-31

特性：断点续传、并发下载、失败重试、--probe 打印字段结构。
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

BASE_URL = "https://quotsoft.net/air/data"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
MIN_VALID_BYTES = 1024  # 小于此值视为错误页
# quotsoft 对单 IP 有突发限流（429），停一会即恢复，故用退避+抖动重试
BACKOFF_SECONDS = [2, 5, 10, 20, 40, 60, 90, 120]
SUCCESS_DELAY = 0.3  # 每次成功后短暂停顿，平滑整体请求速率


@dataclass
class Result:
    day: date
    status: str  # ok / skip / missing / fail
    size: int = 0
    detail: str = ""


def build_url(product: str, day: date) -> str:
    return f"{BASE_URL}/{product}_{day:%Y%m%d}.csv"


def target_path(out_dir: Path, product: str, day: date) -> Path:
    return out_dir / f"{product}_{day:%Y%m%d}.csv"


def download(url: str, dest: Path, timeout: int = 120) -> tuple[str, int, str]:
    """用系统 curl 下载（比 urllib 在弱网下更稳定），返回 (状态, 字节数, 说明)。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        proc = subprocess.run(
            ["curl.exe", "-sS", "-A", USER_AGENT, "--max-time", str(timeout),
             "-w", "%{http_code}", "-o", str(tmp), url],
            capture_output=True, text=True, timeout=timeout + 30,
        )
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return "fail", 0, f"{type(e).__name__}: {e}"

    code = (proc.stdout or "").strip()
    if code == "404":
        tmp.unlink(missing_ok=True)
        return "missing", 0, "404"
    if code != "200":
        tmp.unlink(missing_ok=True)
        return "fail", 0, f"curl http {code} {(proc.stderr or '')[:120]}"

    size = tmp.stat().st_size if tmp.exists() else 0
    if size < MIN_VALID_BYTES:
        tmp.unlink(missing_ok=True)
        return "fail", size, "内容过小，疑似错误页"
    tmp.replace(dest)
    return "ok", size, ""


def fetch_one(product: str, day: date, out_dir: Path, force: bool) -> Result:
    dest = target_path(out_dir, product, day)
    if dest.exists() and not force:
        size = dest.stat().st_size
        if size >= MIN_VALID_BYTES:
            return Result(day, "skip", size)

    last = ("fail", 0, "未执行")
    max_attempts = len(BACKOFF_SECONDS) + 1
    for attempt in range(max_attempts):
        last = download(build_url(product, day), dest)
        if last[0] == "ok":
            time.sleep(SUCCESS_DELAY)
            break
        if last[0] == "missing":
            break
        # fail：429/5xx/网络错误，退避（带抖动）后重试，避免惊群同时重打
        wait = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
        time.sleep(wait * (0.5 + random.random()))
    return Result(day, last[0], last[1], last[2])


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def probe(product: str, out_dir: Path) -> int:
    """下载一天并打印字段结构，用于确认污染物类型与站点数。"""
    day = date(2024, 1, 1)
    res = fetch_one(product, day, out_dir, force=False)
    print(f"[probe] {build_url(product, day)} -> {res.status}, {res.size} 字节")
    if res.status not in ("ok", "skip"):
        print("[probe] 下载失败，请检查网络或源站状态")
        return 1

    import csv
    from collections import Counter

    path = target_path(out_dir, product, day)
    # 本站 CSV 实测为 UTF-8，保留 GBK 回退以兼容早期文件
    text = None
    for enc in ("utf-8", "gb18030"):
        try:
            text = path.read_text(encoding=enc)
            print(f"[probe] 解码：{enc}")
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        print("[probe] 无法解码")
        return 1

    rows = list(csv.DictReader(text.splitlines()))
    cols = [c for c in rows[0].keys() if c not in ("date", "hour", "type")]
    types = sorted(set(r["type"] for r in rows))
    print(f"[probe] 数据行：{len(rows)}，站点列：{len(cols)}")
    print(f"[probe] 污染物类型：{', '.join(types)}")
    print(f"[probe] 站点列示例：{cols[:5]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 quotsoft 空气质量历史数据")
    parser.add_argument("--product", default="china_sites",
                        choices=["china_sites", "beijing_all", "beijing_extra", "china_cities"],
                        help="数据产品，默认 china_sites（全国站点、全污染物）")
    parser.add_argument("--start", type=parse_date, help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=parse_date, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="输出目录")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数，默认 4")
    parser.add_argument("--force", action="store_true", help="覆盖已存在文件")
    parser.add_argument("--probe", action="store_true", help="只下载一天并检查结构")
    args = parser.parse_args()

    if args.probe:
        return probe(args.product, args.out)

    if not args.start or not args.end:
        parser.error("需要同时提供 --start 与 --end（或使用 --probe）")
    if args.start > args.end:
        parser.error("--start 不能晚于 --end")

    days = list(daterange(args.start, args.end))
    print(f"[info] product={args.product} 天数={len(days)} 输出={args.out} 并发={args.workers}")

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_one, args.product, d, args.out, args.force) for d in days]
        for i, fut in enumerate(as_completed(futures), 1):
            results.append(fut.result())
            if i % 20 == 0 or i == len(days):
                print(f"[progress] {i}/{len(days)}", flush=True)

    ok = sum(r.status == "ok" for r in results)
    skip = sum(r.status == "skip" for r in results)
    missing = [r for r in results if r.status == "missing"]
    fail = [r for r in results if r.status == "fail"]
    total = sum(r.size for r in results) / 1024 / 1024

    print("\n[summary]")
    print(f"  新增 {ok} 天，跳过 {skip} 天，缺失 {len(missing)} 天，失败 {len(fail)} 天")
    print(f"  累计 {total:.2f} MB")
    if missing:
        print("  缺失日期（源站无数据）：" + ", ".join(f"{r.day:%Y-%m-%d}" for r in sorted(missing, key=lambda x: x.day)[:20]))
    if fail:
        print("  失败样例：")
        for r in sorted(fail, key=lambda x: x.day)[:10]:
            print(f"    {r.day:%Y-%m-%d} {r.detail}")

    return 0 if not fail else 2


if __name__ == "__main__":
    sys.exit(main())
