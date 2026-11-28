"""汇总 results/ 下的 *_summary.csv 为一张总表，供论文表格与复核使用。

扫描规则（文件名即实验语义）：
  baseline_{model}_{target}_test_summary.csv   → E1 基线
  {target}_test_summary.csv                    → 主模型（自适应邻接，无静态图）
  {target}_test_static_summary.csv             → 主模型 + TE 静态图（完整模型）
  {target}_test_static_only_summary.csv        → 仅 TE 静态图（E2 消融）
  {target}_test_dist_summary.csv               → 自适应 + 距离图（E2 消融）
  {target}_test_tcn_summary.csv                → 纯 TCN（E2 消融）

只取 caliber=full 口径（不剔除 3671A），输出长表：
  results/summary_all.csv
用法：
  python scripts/collect_results.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TAG_LABEL = {
    "": "gwnet-adaptive",
    "_static": "gwnet-static-te",
    "_static_only": "gwnet-static-only",
    "_dist": "gwnet-distance",
    "_tcn": "gwnet-tcn",
}
BASE_LABEL = {"arima": "ARIMA", "lstm": "LSTM", "stgcn": "STGCN", "xgboost": "XGBoost"}
TARGETS = ("pm25", "pm10", "no2", "o3")


def parse_name(path: Path) -> tuple[str, str, str, str] | None:
    """返回 (experiment, model, pollutant, variant)；无法识别时返回 None。"""
    stem = path.stem
    if not stem.endswith("_summary"):
        return None
    core = stem[: -len("_summary")]

    m = re.match(r"^baseline_([a-z0-9]+)_(pm25|pm10|no2|o3)_test$", core)
    if m:
        model = BASE_LABEL.get(m.group(1), m.group(1).upper())
        return "E1-baseline", model, m.group(2), "baseline"

    m = re.match(r"^(pm25|pm10|no2|o3)_test(_[a-z_]+)?$", core)
    if m:
        tag = m.group(2) or ""
        if tag not in TAG_LABEL:
            return None
        if tag in ("_dist", "_tcn", "_static_only"):
            exp = "E2-ablation"
        elif tag == "_static":
            exp = "E3/E4/E5/E7-static"
        else:
            exp = "E1/E4/E5-adaptive"
        return exp, TAG_LABEL[tag], m.group(1), tag or "base"

    return None


def collect(results_dir: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for path in sorted(results_dir.glob("*_summary.csv")):
        if path.name == "summary_all.csv":
            continue
        meta = parse_name(path)
        if meta is None:
            print(f"[skip] {path.name}")
            continue
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "caliber" in df.columns:
            df = df[df["caliber"] == "full"]
        if df.empty:
            print(f"[skip] {path.name}: no full caliber rows")
            continue
        df = df.rename(
            columns={
                "horizon_h": "horizon_h",
                "mae_mean": "mae", "mae_std": "mae_std",
                "rmse_mean": "rmse", "rmse_std": "rmse_std",
                "mape_mean": "mape", "mape_std": "mape_std",
                "r2_mean": "r2", "r2_std": "r2_std",
            }
        )
        for col in ("mae", "mae_std", "rmse", "rmse_std", "mape", "mape_std", "r2", "r2_std"):
            if col not in df.columns:
                df[col] = float("nan")
        out = df[["horizon_h", "mae", "mae_std", "rmse", "rmse_std", "mape", "mape_std", "r2", "r2_std"]].copy()
        out.insert(0, "experiment", meta[0])
        out.insert(1, "model", meta[1])
        out.insert(2, "pollutant", meta[2])
        out.insert(3, "variant", meta[3])
        out["source_file"] = path.name
        records.append(out)
    if not records:
        raise SystemExit("no summary csv found under results/")
    return pd.concat(records, ignore_index=True).sort_values(
        ["experiment", "pollutant", "model", "variant", "horizon_h"]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="collect *_summary.csv into one table")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/summary_all.csv"))
    args = ap.parse_args()

    table = collect(args.results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[ok] files parsed -> {args.out}  rows={len(table)}  "
          f"groups={table.groupby(['experiment', 'model', 'pollutant']).ngroups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
