# 基于时空图神经网络的多站点空气质量预测与污染传播可视分析平台

本科毕业设计（软件工程专业）。以北京国控监测站点为对象，用 Graph WaveNet 做多站点多步污染物浓度预测，用转移熵 + 滞后相关 + 风场先验构建污染关联网络，并对模型空间表征做结构一致性与解释忠实度双重检验，最终交付一套可独立部署的交互式 Web 分析系统。

## 目录结构

```
docs/            设计文档（需求、架构、数据库、接口、测试、数据、实验记录）
data/raw/        原始数据（空气质量、ERA5-Land），不入版本库
data/processed/  预处理产物与张量缓存，不入版本库
src/             源码（数据、模型、服务、前端）
scripts/         数据下载、实验运行等脚本
tests/           单元测试、接口测试、端到端测试
configs/         数据源、超参数、实验配置
```

## 文档索引

| 编号 | 文档 | 状态 |
| --- | --- | --- |
| 01 | [开题报告](docs/01-开题报告.md) | 已完成 |
| 02 | [需求规格说明书](docs/02-需求规格说明书.md) | 已完成 |
| 03 | [概要设计说明书](docs/03-概要设计说明书.md) | 已完成 |
| 04 | [数据库设计说明书](docs/04-数据库设计说明书.md) | 已完成 |
| 05 | [接口说明](docs/05-接口说明.md) | 已完成 |
| 06 | [测试计划](docs/06-测试计划.md) | 已完成 |
| 07 | [数据说明](docs/07-数据说明.md) | 已完成 |
| 08 | [实验记录规范](docs/08-实验记录规范.md) | 已完成 |
| 09 | [开发规范](docs/09-开发规范.md) | 已完成 |
| 10 | [实验与结果分析](docs/10-实验与结果分析.md) | 已完成 |

## 技术栈

- 模型：Python 3.10 + PyTorch 2.x（Graph WaveNet 原生实现，参考官方开源版本）
- 后端：FastAPI + PostgreSQL
- 前端：React/Vue3 + ECharts
- 测试：pytest + locust
- 部署：Docker Compose

## 快速开始

以下命令在仓库根目录、Windows PowerShell 下执行。

### 1 环境安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2 数据获取与预处理

```powershell
# 空气质量原始数据（每天一个 CSV）
python scripts/fetch_air_quality.py --product china_sites --start 2021-01-01 --end 2024-12-31

# 气象数据（需配置 CDS API key；--dry-run 只打印请求）
python scripts/fetch_era5.py --dry-run

# 站点对齐、清洗、张量构建，输出 data/processed/obs.npz 与 quality.csv
python scripts/build_features.py
```

站点清单为 `data/raw/stations.csv`（24 个国控站）。

### 3 模型训练与评估

```powershell
# 主模型（自适应邻接）
python scripts/train.py --target pm25 --seed-list 42 43 44
python scripts/evaluate.py --target pm25 --seeds 42 43 44

# 完整模型（接入转移熵静态图）
python scripts/build_static_graph.py --target pm25 --from 2022-01-01 --to 2022-06-30
python scripts/train.py --target pm25 --seed-list 42 43 44 `
    --static-adj data/processed/static_adj_pm25.npz --tag _static
python scripts/evaluate.py --target pm25 --seeds 42 43 44 `
    --static-adj data/processed/static_adj_pm25.npz --tag _static
```

四污染物把 `--target` 换成 `pm10` / `no2` / `o3`；实验脚本与参数约定见 [实验记录规范](docs/08-实验记录规范.md) §9。

### 4 启动后端

```powershell
.venv\Scripts\python.exe -m uvicorn src.main:app --host 127.0.0.1 --port 8000
```

- Swagger：http://127.0.0.1:8000/docs
- 健康检查：`GET /api/v1/health`
- 数据库（可选，演示可走本地缓存）：`psql -U postgres -d aq_platform -f scripts/init_db.sql`

### 5 前端

```powershell
cd web
npm install
npm run dev      # 开发模式，127.0.0.1:5173，/api 代理到 8000
npm run build    # 生产产物 web/dist
```

### 6 测试与压测

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\system_test.py
.venv\Scripts\python.exe scripts\load_test.py --concurrency 20
```

### 7 结果汇总

```powershell
python scripts/collect_results.py    # results/*_summary.csv → results/summary_all.csv
python scripts/make_figures.py       # 生成 docs/figures/4-1 ~ 4-3
```

## 关键约定

- 数据与模型权重不进入版本库，通过脚本与配置重建；
- 每次实验必须登记 run_id（见 `docs/08-实验记录规范.md`）；
- 文档先行：需求、接口变更先改文档再改代码。
