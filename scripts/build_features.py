"""将 quotsoft 单日 CSV 汇总为建模用的时空张量，并输出数据质量报告。

输入：
    data/raw/china_sites_YYYYMMDD.csv   （由 fetch_air_quality.py 下载）
    data/raw/stations.csv               （24 个北京国控站的编码、名称、经纬度）

输出：
    data/processed/obs.npz               values[t, s, p] + timestamps + codes + pollutants
    data/processed/quality.csv           每站每污染物的缺失率、最长连续缺失、异常比例

处理规则（对应 docs/07 数据说明第 5 节）：
    1. 时间轴按完整小时补齐，源站缺天/缺小时的位置保持 NaN；
    2. 负值与超出物理量程的值置为 NaN；
    3. MAD 稳健异常检测（k=3）只标记不删除，标记比例写入质量报告；
    4. 缺失填充不在本脚本做，留给训练时的 Dataset（避免用未来信息）。

用法：
    python scripts/build_features.py
    python scripts/build_features.py --start 2021-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

POLLUTANTS = {"PM2.5": "pm25", "PM10": "pm10", "NO2": "no2", "O3": "o3"}
VALID_MAX = {"pm25": 1000.0, "pm10": 2000.0, "no2": 2000.0, "o3": 2000.0}
MAD_K = 3.0


def load_stations(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return [r for r in rows if r.get("code")]


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_day(path: Path, codes: list[str]) -> dict[str, np.ndarray]:
    """读取一天，返回 {污染物: (24, n_codes) 数组}，缺失为 NaN。"""
    usecols = lambda c: c in ("date", "hour", "type") or c in codes
    try:
        df = pd.read_csv(path, encoding="utf-8", usecols=usecols, low_memory=False)
    except Exception:
        return {}

    if df.empty or "type" not in df.columns:
        return {}

    result: dict[str, np.ndarray] = {}
    for raw_name, key in POLLUTANTS.items():
        sub = df[df["type"] == raw_name]
        if sub.empty:
            continue
        sub = sub.set_index("hour")
        # 补齐 0-23 小时，站点顺序与 codes 一致
        sub = sub.reindex(range(24))
        cols = [c for c in codes if c in sub.columns]
        arr = np.full((24, len(codes)), np.nan, dtype=np.float32)
        if cols:
            mat = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            idx = [codes.index(c) for c in cols]
            arr[:, idx] = mat
        result[key] = arr
    return result


def longest_gap(mask: np.ndarray) -> int:
    """mask: True 表示缺失，返回最长连续缺失长度（小时）。"""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def build(raw_dir: Path, station_file: Path, out_dir: Path,
          start: date, end: date) -> int:
    stations = load_stations(station_file)
    codes = [s["code"] for s in stations]
    names = {s["code"]: s["name"] for s in stations}
    print(f"[info] 站点 {len(codes)} 个，时间 {start} ~ {end}")

    n_days = (end - start).days + 1
    n_hours = n_days * 24
    n_poll = len(POLLUTANTS)
    keys = list(POLLUTANTS.values())
    values = np.full((n_hours, len(codes), n_poll), np.nan, dtype=np.float32)

    found = 0
    missing_days: list[date] = []
    cur = start
    while cur <= end:
        path = raw_dir / f"china_sites_{cur:%Y%m%d}.csv"
        if not path.exists():
            missing_days.append(cur)
        else:
            day_data = parse_day(path, codes)
            if day_data:
                found += 1
                base = (cur - start).days * 24
                for p_idx, key in enumerate(keys):
                    if key in day_data:
                        values[base:base + 24, :, p_idx] = day_data[key]
        cur += timedelta(days=1)

    print(f"[info] 成功解析 {found} 天，源站缺失 {len(missing_days)} 天")

    # 量程过滤
    for p_idx, key in enumerate(keys):
        v = values[:, :, p_idx]
        bad = (v < 0) | (v > VALID_MAX[key])
        v[bad] = np.nan
        values[:, :, p_idx] = v

    # 质量统计 + MAD 标记
    quality_rows = []
    flagged_total = 0
    for s_idx, code in enumerate(codes):
        row = {"code": code, "name": names.get(code, "")}
        flagged = 0
        for p_idx, key in enumerate(keys):
            v = values[:, s_idx, p_idx]
            mask = np.isnan(v)
            row[f"{key}_missing"] = round(float(mask.mean()), 4)
            row[f"{key}_max_gap_h"] = longest_gap(mask)
            valid = v[~mask]
            if valid.size > 0:
                med = np.median(valid)
                mad = np.median(np.abs(valid - med))
                if mad > 0:
                    flag = np.abs(v - med) > MAD_K * 1.4826 * mad
                    flagged += int(np.nansum(flag))
        row["mad_flagged"] = flagged
        flagged_total += flagged
        quality_rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamps = pd.date_range(start=start, periods=n_hours, freq="h")
    np.savez_compressed(
        out_dir / "obs.npz",
        values=values,
        timestamps=timestamps.astype("int64").to_numpy(),
        codes=np.array(codes),
        pollutants=np.array(keys),
    )
    pd.DataFrame(quality_rows).to_csv(out_dir / "quality.csv", index=False, encoding="utf-8-sig")

    overall_missing = float(np.isnan(values).mean())
    print(f"[done] 张量 {values.shape}，总体缺失率 {overall_missing:.2%}，MAD 标记 {flagged_total} 个点")
    print(f"[done] 输出：{out_dir/'obs.npz'}、{out_dir/'quality.csv'}")
    if missing_days:
        preview = ", ".join(f"{d:%Y-%m-%d}" for d in missing_days[:10])
        print(f"[warn] 源站缺天 {len(missing_days)} 个（前 10：{preview}）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总空气质量数据为时空张量")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--station-file", type=Path, default=Path("data/raw/stations.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--start", type=parse_date, default=date(2021, 1, 1))
    parser.add_argument("--end", type=parse_date, default=date(2024, 12, 31))
    args = parser.parse_args()

    if not args.station_file.exists():
        print(f"[error] 站点表不存在：{args.station_file}")
        return 1
    return build(args.raw_dir, args.station_file, args.out_dir, args.start, args.end)


if __name__ == "__main__":
    raise SystemExit(main())
