"""E1 基线模型训练与评估：ARIMA / XGBoost / 单站 LSTM / STGCN。

与主模型 Graph WaveNet 完全可比（口径对齐 scripts/evaluate.py）：
  - 相同 train/valid/test 切分（自然年，见 AirQualityDataset）
  - 相同 normalizer（train 统计量），相同输入窗 24h / 输出 24h
  - 相同报告步长 1/3/6/12/24、相同 mask（真值缺失不计入）、相同口径 full / drop3671A
  - 相同指标 RMSE / MAE / MAPE / R2（复用 evaluate.compute_metrics）

防泄露契约（所有基线）：
  - 特征只能从输入窗 [i, i+24) 派生；标签来自未来 [i+24, i+48)
  - normalizer 只在 train 上 fit；模型只在 train 上拟合；valid 仅用于早停
  - 监督学习（XGBoost/LSTM/STGCN）训练时剔除目标缺失的样本/位置（mask=False 不计 loss）

输出：
  results/baseline_{model}_{target}_test_metrics.csv   # 每 (seed,horizon,口径) 一行
  results/baseline_{model}_{target}_test_summary.csv   # 跨 seed mean±std

用法：
  python scripts/run_baselines.py --targets pm25 --models arima --seeds 0      # 单基线快验
  python scripts/run_baselines.py --targets pm25 pm10 no2 o3 --models arima xgboost lstm stgcn --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))        # scripts/  可 import evaluate

from evaluate import compute_metrics, HORIZON_REPORT              # noqa: E402
from src.domain.dataset import (                                  # noqa: E402
    AirQualityDataset, POLLUTANT_INDEX, STATION_INDEX, build_datasets,
)

IDX_3671A = STATION_INDEX["3671A"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 数据辅助
# ----------------------------------------------------------------------------
def load_data(target: str, npz: str):
    train, valid, test, norm = build_datasets(npz, target, 24, 24)
    return train, valid, test, norm


def input_windows(ds: AirQualityDataset) -> np.ndarray:
    """返回 (S, T_in=24, N, P) 标准化填充后的输入窗。"""
    T = len(ds.values_filled)
    S = T - 24 - 24 + 1
    i0 = np.arange(S)[:, None]                             # (S,1)
    tt = np.arange(24)[None, :]                            # (1,24)
    return ds.values_filled[i0 + tt]                       # (S,24,N,P)


def target_arrays(ds: AirQualityDataset) -> tuple[np.ndarray, np.ndarray]:
    """返回 (y_norm, mask)，各 (S, 24, N)，目标污染物。"""
    T = len(ds.values_filled)
    S = T - 24 - 24 + 1
    p = POLLUTANT_INDEX[ds.target]
    i0 = np.arange(S)[:, None]
    hh = np.arange(24)[None, :]
    idx = i0 + 24 + hh                                      # (S,24)
    y = ds.values_filled[idx, :, p]                          # (S,24,N)
    m = ds.truth_mask[idx, :, p]                             # (S,24,N)
    return y, m


def score_predictions(pred_norm: np.ndarray, ds: AirQualityDataset, target: str,
                      model: str, seed: int) -> list[dict]:
    """pred_norm: (S,24,N) 标准化预测 → 转物理量后按 evaluate 口径打分。"""
    p = POLLUTANT_INDEX[target]
    mean = ds.normalizer.mean[:, p]
    std = ds.normalizer.std[:, p]
    y_norm, mask = target_arrays(ds)
    pred_phys = pred_norm * std[None, None, :] + mean[None, None, :]
    y_phys = y_norm * std[None, None, :] + mean[None, None, :]
    N = pred_norm.shape[2]
    rows = []
    for h in HORIZON_REPORT:
        hi = h - 1
        for cal, stations in [
            ("full", np.arange(N)),
            ("drop3671A", np.array([i for i in range(N) if i != IDX_3671A])),
        ]:
            ph = pred_phys[:, hi, :][:, stations]
            yh = y_phys[:, hi, :][:, stations]
            mh = mask[:, hi, :][:, stations]
            m = compute_metrics(ph, yh, mh)
            rows.append({
                "model": model, "target": target, "seed": seed,
                "horizon_h": h, "caliber": cal,
                "rmse": m["rmse"], "mae": m["mae"], "mape": m["mape"], "r2": m["r2"],
                "n_valid": int(mh.sum()),
            })
    return rows


def save_results(rows: list[dict], model: str, target: str, out: Path):
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"baseline_{model}_{target}_test_metrics.csv", index=False, encoding="utf-8-sig")
    summary = (df.groupby(["horizon_h", "caliber"])[["rmse", "mae", "mape", "r2"]]
                 .agg(["mean", "std"]).reset_index())
    summary.columns = ["horizon_h", "caliber"] + [
        f"{m}_{s}" for m in ["rmse", "mae", "mape", "r2"] for s in ["mean", "std"]]
    summary.to_csv(out / f"baseline_{model}_{target}_test_summary.csv", index=False, encoding="utf-8-sig")
    print(f"  [done] {out / f'baseline_{model}_{target}_test_summary.csv'}")


# ----------------------------------------------------------------------------
# 1. ARIMA(24,1,0)：每站 OLS 拟合差分后的 24 阶自回归，numpy 递推多步预测（确定性）
# ----------------------------------------------------------------------------
def run_arima(target: str, npz: str, out: Path) -> list[dict]:
    from sklearn.linear_model import LinearRegression
    train_ds, _, test_ds, _ = load_data(target, npz)
    p = POLLUTANT_INDEX[target]
    N = train_ds.N
    S = test_ds.n_samples
    pred_norm = np.zeros((S, 24, N), dtype=np.float32)

    for n in range(N):
        z_train = train_ds.values_filled[:, n, p]            # (Ttr,)
        dz = np.diff(z_train)                                 # (Ttr-1,)
        # 构造训练矩阵：用 dz[t-24..t-1] 预测 dz[t]
        Ttr = len(dz)
        X, y = [], []
        for t in range(24, Ttr):
            X.append(dz[t - 24:t][::-1])                      # lag1..lag24
            y.append(dz[t])
        X, y = np.asarray(X, np.float32), np.asarray(y, np.float32)
        reg = LinearRegression().fit(X, y)

        # 测试集递推预测（向量化跨全部 origin）
        z_test = test_ds.values_filled[:, n, p]               # (Tte,)
        dz_test = np.diff(z_test)                             # (Tte-1,)
        i0 = np.arange(S)                                     # origin 起始
        e = i0 + 23                                           # 输入窗最后一个时刻索引
        # 初始 24 个 lag（lag1 = dz[e], lag24 = dz[e-23]）
        L = np.stack([dz_test[e - k] for k in range(24)], axis=1).astype(np.float32)  # (S,24)
        z_end = z_test[e]                                     # (S,) 最后已知水平
        cum = np.zeros(S, dtype=np.float32)
        for h in range(24):
            dz_hat = L @ reg.coef_ + reg.intercept_           # (S,)
            cum += dz_hat
            pred_norm[:, h, n] = z_end + cum
            L = np.roll(L, -1, axis=1)                         # 左移：lag1<-lag2 ...
            L[:, -1] = dz_hat                                  # 最新预测成为 lag24
    return score_predictions(pred_norm, test_ds, target, "arima", 0)


# ----------------------------------------------------------------------------
# 2. XGBoost：全局（跨站池化）直接多步回归，含站点 one-hot，3 seed
# ----------------------------------------------------------------------------
def _xgb_features(ds: AirQualityDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (X: (S*N, 52), y: (S*N, 24), keep: (S*N,24) mask)；按站先序排列。"""
    p = POLLUTANT_INDEX[ds.target]
    X_in = input_windows(ds)                                   # (S,24,N,P)
    y_norm, mask = target_arrays(ds)                           # (S,24,N)
    S, _, N, _ = X_in.shape
    feats = []
    labels = []
    keeps = []
    for n in range(N):
        lag = X_in[:, :, n, p]                                  # (S,24) 目标污染物 24 滞后
        cur = X_in[:, -1, n, :4]                                # (S,4)  当前 4 污染物
        oh = np.zeros((S, N), np.float32); oh[:, n] = 1.0
        feats.append(np.concatenate([lag, cur, oh], axis=1))    # (S, 24+4+N=52)
        labels.append(y_norm[:, :, n])                          # (S,24)
        keeps.append(mask[:, :, n])
    return np.concatenate(feats, 0), np.concatenate(labels, 0), np.concatenate(keeps, 0)


def _xgb_device() -> tuple[str, str]:
    # XGBoost 3.x 在本机探测不到可用 GPU 时 device="cuda" 会静默回退并产出异常预测，
    # 统一使用 CPU（速度足够，4 污染物 3 seed 合计约数分钟）。
    return "hist", "cpu"


def run_xgboost(target: str, npz: str, seeds: list[int], out: Path) -> list[dict]:
    import xgboost as xgb
    train_ds, _, test_ds, _ = load_data(target, npz)
    Xtr, ytr, ktr = _xgb_features(train_ds)                    # (M,52),(M,24),(M,24)
    Xte, _, _ = _xgb_features(test_ds)
    N = train_ds.N
    S_te = test_ds.n_samples
    tm, dev = _xgb_device()
    print(f"  [xgb] device={dev} tree_method={tm} train_samples={Xtr.shape[0]}")

    all_rows = []
    for seed in seeds:
        pred_norm = np.zeros((S_te, 24, N), dtype=np.float32)
        for h in HORIZON_REPORT:
            hi = h - 1
            keep = ktr[:, hi].astype(bool)                     # 只在真值处训练
            model = xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                tree_method=tm, device=dev, n_jobs=8,
                random_state=seed, verbosity=0,
            )
            model.fit(Xtr[keep], ytr[keep, hi])
            pred = model.predict(Xte)                           # (S_te*N,) 按 station 块排列
            pred_norm[:, hi, :] = pred.reshape(N, S_te).T       # → (S_te, N) 对齐 [origin, station]
        all_rows += score_predictions(pred_norm, test_ds, target, "xgboost", seed)
        print(f"  [xgb] target={target} seed={seed} done")
    return all_rows


# ----------------------------------------------------------------------------
# 3. 单站 LSTM（全局共享权重，无空间图）：输入 (24,4) → 输出 (24,)
# ----------------------------------------------------------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, in_dim=4, hidden=64, layers=2, dropout=0.1, horizon=24):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):                                       # (B,24,4)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])                          # (B,24)


def masked_mse(pred, y, mask):
    m = mask.float()
    return (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def _train_torch(model, train_ds, valid_ds, make_batch, seed, epochs, patience, tag):
    """通用训练：make_batch(ds, idx) -> (x,y,mask) torch on DEVICE。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-5)
    S_tr = train_ds.n_samples
    S_va = valid_ds.n_samples
    best, bad = float("inf"), 0
    bs = 256
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(S_tr)
        tl = 0.0
        for i in range(0, S_tr, bs):
            idx = perm[i:i + bs]
            x, y, m = make_batch(train_ds, idx)
            opt.zero_grad()
            loss = masked_mse(model(x), y, m)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tl += loss.item() * len(idx)
        # valid
        model.eval()
        with torch.no_grad():
            vse, vn = 0.0, 0.0
            for i in range(0, S_va, bs):
                idx = np.arange(i, min(i + bs, S_va))
                x, y, m = make_batch(valid_ds, idx)
                p = model(x)
                mf = m.float()
                vse += (((p - y) ** 2) * mf).sum().item(); vn += mf.sum().item()
        vl = vse / max(vn, 1.0)
        sched.step(vl)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    [{tag}] seed={seed} ep={ep} train={tl/S_tr:.4f} valid={vl:.4f}", flush=True)
        if vl < best:
            best = vl; bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    [{tag}] early stop ep={ep}"); break
    model.load_state_dict(best_state)
    return model


def run_lstm(target, npz, seeds, out):
    train_ds, valid_ds, test_ds, _ = load_data(target, npz)

    def prep(ds):
        X = input_windows(ds)[:, :, :, :4]            # (S,24,N,4)
        y, m = target_arrays(ds)                      # (S,24,N)
        return X, y, m

    Xtr, ytr, mtr = prep(train_ds)
    Xva, yva, mva = prep(valid_ds)
    Xte, _, _ = prep(test_ds)
    S_te, _, N, _ = Xte.shape

    def make_batch(ds, idx):
        X, y, m = (Xtr, ytr, mtr) if ds is train_ds else (Xva, yva, mva)
        x = X[idx].transpose(0, 2, 1, 3)              # (B,N,24,4)
        B, Nn, T, C = x.shape
        x = x.reshape(B * Nn, T, C)                   # (B*N,24,4)
        yt = y[idx].transpose(0, 2, 1).reshape(B * Nn, 24)
        mt = m[idx].transpose(0, 2, 1).reshape(B * Nn, 24)
        return (torch.from_numpy(x).float().to(DEVICE),
                torch.from_numpy(yt).float().to(DEVICE),
                torch.from_numpy(mt).bool().to(DEVICE))

    all_rows = []
    for seed in seeds:
        model = _train_torch(LSTMForecaster(), train_ds, valid_ds, make_batch, seed,
                             epochs=40, patience=8, tag=f"lstm_{target}")
        model.eval()
        pred_norm = np.zeros((S_te, 24, N), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, S_te, 512):
                idx = np.arange(i, min(i + 512, S_te))
                x = Xte[idx].transpose(0, 2, 1, 3).reshape(-1, 24, 4)
                p = model(torch.from_numpy(x).float().to(DEVICE)).cpu().numpy()
                pred_norm[idx] = p.reshape(len(idx), N, 24).transpose(0, 2, 1)
        all_rows += score_predictions(pred_norm, test_ds, target, "lstm", seed)
        print(f"  [lstm] target={target} seed={seed} done")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return all_rows


# ----------------------------------------------------------------------------
# 4. STGCN：距离高斯图 + 时空卷积块
# ----------------------------------------------------------------------------
def build_distance_adj(codes_order):
    df = pd.read_csv("data/raw/stations.csv")
    df["code"] = df["code"].astype(str)
    sub = df.set_index("code").loc[codes_order]
    lon = sub["lon"].to_numpy(np.float32)
    lat = sub["lat"].to_numpy(np.float32)
    # 近似等距投影（纬度 ~40°）到平面距离 km
    lat0 = np.radians(lat.mean())
    x = np.radians(lon) * 6371 * np.cos(lat0)
    y = np.radians(lat) * 6371
    d = np.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)
    sigma = float(np.median(d[d > 0]))
    A = np.exp(-(d ** 2) / (sigma ** 2))
    np.fill_diagonal(A, 0.0)
    # 对称归一化 D^-1/2 A D^-1/2
    deg = A.sum(axis=1)
    deg = np.where(deg < 1e-6, 1.0, deg)
    D12 = 1.0 / np.sqrt(deg)
    return (A * D12[:, None] * D12[None, :]).astype(np.float32)


class STConvBlock(nn.Module):
    def __init__(self, c_in, c_out, adj, kernel=3):
        super().__init__()
        self.register_buffer("adj", torch.from_numpy(adj))
        self.tc1 = nn.Conv2d(c_in, c_out, (kernel, 1), padding=(kernel // 2, 0))
        self.tc2 = nn.Conv2d(c_out, c_out, (kernel, 1), padding=(kernel // 2, 0))
        self.sc = nn.Conv2d(c_out, c_out, (1, 1))
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):                          # x: (B, C, T, N)
        x = torch.tanh(self.tc1(x))
        x = torch.tanh(self.tc2(x))
        # 空间：(B, C, T, N) @ (N,N) -> (B, C, T, N)
        xs = torch.einsum("bctn,nm->bctm", x, self.adj)
        x = self.sc(xs)
        return self.bn(x)


class STGCN(nn.Module):
    def __init__(self, adj, in_dim=4, hidden=32, horizon=24):
        super().__init__()
        self.b1 = STConvBlock(in_dim, hidden, adj)
        self.b2 = STConvBlock(hidden, hidden, adj)
        self.b3 = STConvBlock(hidden, hidden, adj)
        self.out = nn.Conv2d(hidden, horizon, (1, 1))

    def forward(self, x):                          # x: (B, C, T, N)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        o = self.out(x)                            # (B, horizon, T, N)
        return o[:, :, -1, :]                       # (B, horizon, N)


def run_stgcn(target, npz, seeds, out):
    train_ds, valid_ds, test_ds, _ = load_data(target, npz)
    adj = build_distance_adj(train_ds.codes)
    print(f"  [stgcn] distance-adj shape={adj.shape} mean={adj.mean():.4f}")

    def prep(ds):
        X = input_windows(ds)[:, :, :, :4]            # (S,24,N,4)
        y, m = target_arrays(ds)                      # (S,24,N)
        return X, y, m

    Xtr, ytr, mtr = prep(train_ds)
    Xva, yva, mva = prep(valid_ds)
    Xte, _, _ = prep(test_ds)
    S_te, _, N, _ = Xte.shape

    def make_batch(ds, idx):
        X, y, m = (Xtr, ytr, mtr) if ds is train_ds else (Xva, yva, mva)
        x = X[idx].transpose(0, 3, 1, 2)              # (B, C=4, T=24, N)
        return (torch.from_numpy(x).float().to(DEVICE),
                torch.from_numpy(y[idx]).float().to(DEVICE),
                torch.from_numpy(m[idx]).bool().to(DEVICE))

    all_rows = []
    for seed in seeds:
        model = _train_torch(STGCN(adj), train_ds, valid_ds, make_batch, seed,
                             epochs=40, patience=8, tag=f"stgcn_{target}")
        model.eval()
        pred_norm = np.zeros((S_te, 24, N), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, S_te, 256):
                idx = np.arange(i, min(i + 256, S_te))
                x = torch.from_numpy(Xte[idx].transpose(0, 3, 1, 2)).float().to(DEVICE)
                pred_norm[idx] = model(x).cpu().numpy()   # (B, 24, N)
        all_rows += score_predictions(pred_norm, test_ds, target, "stgcn", seed)
        print(f"  [stgcn] target={target} seed={seed} done")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return all_rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="E1 基线模型训练与评估")
    ap.add_argument("--targets", nargs="+", default=["pm25"], choices=["pm25", "pm10", "no2", "o3"])
    ap.add_argument("--models", nargs="+", default=["arima"],
                    choices=["arima", "xgboost", "lstm", "stgcn"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    args = ap.parse_args()

    data_cfg = yaml.safe_load(open(args.data_config, encoding="utf-8"))
    npz = data_cfg["paths"]["processed"] + "/obs.npz"
    print(f"[info] device={DEVICE} targets={args.targets} models={args.models} seeds={args.seeds}")

    seeds_det = [0]  # ARIMA 确定性，单 seed
    for target in args.targets:
        for model in args.models:
            t0 = time.time()
            print(f"\n===== {target} / {model} =====")
            if model == "arima":
                rows = run_arima(target, npz, args.out)
            elif model == "xgboost":
                rows = run_xgboost(target, npz, args.seeds, args.out)
            elif model == "lstm":
                rows = run_lstm(target, npz, args.seeds, args.out)
            elif model == "stgcn":
                rows = run_stgcn(target, npz, args.seeds, args.out)
            save_results(rows, model, target, args.out)
            print(f"  elapsed {time.time()-t0:.1f}s")
    print("[done] all baselines finished")


if __name__ == "__main__":
    sys.exit(main())
