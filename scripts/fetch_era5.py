"""下载 ERA5-Land 再分析气象数据（按北京区域裁剪，逐年 NetCDF）。

数据源：Copernicus Climate Data Store (CDS)
    dataset: reanalysis-era5-land
    变量（取自 configs/data.yaml 的 meteorology.variables）：
        2m_temperature           -> 温度
        2m_dewpoint_temperature  -> 由温度/露点换算相对湿度
        10m_u_component_of_wind  -> 由 u/v 换算风速/风向
        10m_v_component_of_wind
        surface_pressure         -> 气压
    区域 [N, W, S, E]：北京及周边
    时间：逐小时，2021-01-01 ~ 2024-12-31

前置条件（运行前必须满足）：
    1. 注册 CDS 账号 https://cds.climate.copernicus.eu/ ，并在账号页
       接受 "ERA5-Land monthly averaged / hourly" 数据集条款；
    2. 获取 API key，写入用户主目录的 ~/.cdsapirc （或设置环境变量
       CDSAPI_URL / CDSAPI_KEY），文件内容形如：
           url: https://cds.climate.copernicus.eu/api
           key: <uid>:<api-key>
    3. pip install cdsapi（已在 requirements.txt）

用法：
    python scripts/fetch_era5.py --dry-run     # 只打印请求，不提交（先核对再下）
    python scripts/fetch_era5.py               # 逐年下载到 data/raw/era5/
    python scripts/fetch_era5.py --start 2021 --end 2022

特性：断点续传——已存在且大小正常的年份文件自动跳过，可中断后重跑。
注意：CDS 端对单用户并发请求有限制，故逐年顺序提交；单个请求排队+下载
      可能需要数分钟到数十分钟，建议后台运行。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

# 一年 5 变量 × 8760 小时的 NetCDF 在北京小区域下约数十 MB，小于此值视为未完成
MIN_VALID_BYTES = 5 * 1024 * 1024
CONFIG_PATH = Path("configs/data.yaml")


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_hours() -> list[str]:
    return [f"{h:02d}:00" for h in range(24)]


def build_request(cfg: dict, year: int) -> dict:
    met = cfg["meteorology"]
    area = met["area"]  # [N, W, S, E]
    return {
        "product_type": "reanalysis",
        "variable": list(met["variables"]),
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": all_hours(),
        "area": area,
        "format": "netcdf",
    }


def target_path(out_dir: Path, year: int) -> Path:
    return out_dir / f"era5_land_{year}.nc"


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 ERA5-Land 逐小时气象数据（逐年 NetCDF）")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--start", type=int, help="起始年份（默认取配置 scope.start）")
    parser.add_argument("--end", type=int, help="结束年份（默认取配置 scope.end）")
    parser.add_argument("--out", type=Path, default=Path("data/raw/era5"), help="输出目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印请求，不提交 CDS")
    args = parser.parse_args()

    cfg = load_config(args.config)
    start_year = args.start or datetime.strptime(cfg["scope"]["start"], "%Y-%m-%d").year
    end_year = args.end or datetime.strptime(cfg["scope"]["end"], "%Y-%m-%d").year
    if start_year > end_year:
        parser.error("--start 不能晚于 --end")

    years = list(range(start_year, end_year + 1))
    print(f"[info] dataset=reanalysis-era5-land 年份={years} 输出={args.out} dry_run={args.dry_run}")

    if not args.dry_run:
        try:
            import cdsapi  # noqa: WPS433
        except ImportError:
            print("[error] 未安装 cdsapi，请先执行：.venv\\Scripts\\python.exe -m pip install cdsapi")
            return 1
        client = cdsapi.Client()

    args.out.mkdir(parents=True, exist_ok=True)
    failed: list[int] = []

    for year in years:
        dest = target_path(args.out, year)
        if dest.exists() and dest.stat().st_size >= MIN_VALID_BYTES:
            print(f"[skip] {year} 已存在（{dest.stat().st_size / 1024 / 1024:.1f} MB）：{dest}")
            continue

        request = build_request(cfg, year)
        print(f"[request] {year} 变量={request['variable']} 区域={request['area']}")
        if args.dry_run:
            print(f"[dry-run] 将提交：{request}")
            continue

        try:
            client.retrieve("reanalysis-era5-land", request, str(dest))
            size = dest.stat().st_size / 1024 / 1024 if dest.exists() else 0
            print(f"[ok] {year} -> {dest}（{size:.1f} MB）")
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {year}: {type(e).__name__}: {e}")
            dest.unlink(missing_ok=True)
            failed.append(year)

    print("\n[summary]")
    print(f"  年份 {start_year}-{end_year}，失败 {len(failed)} 年" +
          (f"：{failed}" if failed else ""))
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
