"""第四章配图生成：误差-步长曲线、E7 忠实度箱线图、E6 边权一致性散点。

产物（docs/figures/）：
  4-1_error_by_horizon.png   四污染物 RMSE / R² 随步长的衰减（主模型，full 口径）
  4-2_e7_faithfulness.png    E7 三类条件（上游/下游/随机）的站点级 ΔRMSE 箱线图
  4-3_e6_edge_consistency.png E6 转移熵网络 vs 自适应邻接的边权秩散点

用法：python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.infra import network_cache  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TARGETS = ["pm25", "pm10", "no2", "o3"]
TGT_CN = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "o3": "O3"}
COLORS = {"pm25": "#C0392B", "pm10": "#8E6C88", "no2": "#2E86C1", "o3": "#D68910"}
FIG = Path("docs/figures")


def fig_error_horizon(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for t in TARGETS:
        df = pd.read_csv(f"results/{t}_test_static_summary.csv")
        d = df[df.caliber == "full"].sort_values("horizon_h")
        h = d.horizon_h.values
        axes[0].plot(h, d.rmse_mean, "-o", color=COLORS[t], label=TGT_CN[t], lw=1.8, ms=5)
        axes[0].fill_between(h, d.rmse_mean - d.rmse_std, d.rmse_mean + d.rmse_std,
                             color=COLORS[t], alpha=0.15)
        axes[1].plot(h, d.r2_mean, "-s", color=COLORS[t], label=TGT_CN[t], lw=1.8, ms=5)
    for ax in axes:
        ax.set_xticks([1, 3, 6, 12, 24])
        ax.set_xlabel("预测步长 h（小时）")
        ax.grid(alpha=0.3, ls="--")
    axes[0].set_ylabel("RMSE（μg/m³）")
    axes[0].set_title("(a) 预测误差随步长上升")
    axes[1].set_ylabel("R²")
    axes[1].set_title("(b) 拟合优度随步长下降")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("图 4-1  主模型（Graph WaveNet + 静态图）分污染物、分步长测试集表现", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def fig_e7_box(out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    labels = ["上游 K", "下游 K", "随机 K"]
    cols = ["#C0392B", "#2E86C1", "#95A5A6"]
    for ax, t in zip(axes.ravel(), TARGETS):
        df = pd.read_csv(f"results/e7_faithfulness_{t}_network_static.csv")
        data = [df["d_up"].dropna().values, df["d_down"].dropna().values,
                df["d_rand"].dropna().values]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=1.2),
                        flierprops=dict(marker="o", ms=3, alpha=0.4))
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels)
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        # 均值点
        for i, d in enumerate(data, start=1):
            ax.scatter([i], [d.mean()], marker="D", s=28, color="black", zorder=5)
        ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{TGT_CN[t]}（n={len(df)} 站）")
        ax.set_ylabel("ΔRMSE（μg/m³）")
        ax.grid(axis="y", alpha=0.3, ls="--")
    fig.suptitle("图 4-2  E7 解释忠实度：三类输入消融条件的站点级 ΔRMSE 分布（菱形=均值）", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def _te_matrix(net: dict, codes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(codes)}
    N = len(codes)
    W = np.zeros((N, N), dtype=np.float64)
    for e in net["edges"]:
        W[idx[e["src"]], idx[e["dst"]]] = e["weight"]
    return W


def _adp_matrix(t: str, cfg: dict) -> np.ndarray:
    """复刻 E6：从 base 自适应模型（{t}_seed42）的节点嵌入算自适应邻接。"""
    ckpt = Path(cfg["train"]["checkpoint_dir"]) / f"{t}_seed42" / "best.pt"
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model", state.get("state_dict", state))
    e1, e2 = sd["emb1"].float(), sd["emb2"].float()
    return torch.softmax(torch.relu(e1 @ e2.t()), dim=1).numpy().astype(np.float64)


def fig_e6_scatter(out: Path, model_cfg: dict, codes: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    N = len(codes)
    off = np.array([i * N + j for i in range(N) for j in range(N) if i != j])
    for ax, t in zip(axes.ravel(), TARGETS):
        key = network_cache.compute_cache_key(t, "2022-01-01", "2022-06-30", "composite", 100)
        net = network_cache.get_network(key)
        if net is None:
            raise RuntimeError(f"网络缓存缺失 key={key}（{t}）")
        W = _te_matrix(net, codes)
        A = _adp_matrix(t, model_cfg)

        w_te = W.ravel()[off]
        w_adp = A.ravel()[off]
        rho = spearmanr(w_te, w_adp).statistic
        r_te = rankdata(w_te) / len(w_te)
        r_adp = rankdata(w_adp) / len(w_adp)

        ax.scatter(r_te, r_adp, s=9, alpha=0.45, color=COLORS[t], edgecolors="none")
        e6 = json.load(open(f"results/e6_consistency_{t}.json", encoding="utf-8"))
        jac = e6["jaccard_topk"]["120"]
        print(f"[e6] {t} 重算 rho={rho:+.4f}  JSON rho={e6['rho_edge_weight']:+.4f}")
        ax.set_title(f"{TGT_CN[t]}  ρ={rho:+.3f}  Jaccard@120={jac:.3f}")
        ax.set_xlabel("转移熵网络边权分位")
        ax.set_ylabel("自适应邻接边权分位")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, ls="--")
    fig.suptitle("图 4-3  E6 两套网络边权一致性：秩散点（点云无结构，即两网络几乎无关）", y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open("configs/model.yaml", encoding="utf-8"))
    import csv
    codes = [r["code"] for r in csv.DictReader(open("data/raw/stations.csv", encoding="utf-8-sig"))]

    fig_error_horizon(FIG / "4-1_error_by_horizon.png")
    fig_e7_box(FIG / "4-2_e7_faithfulness.png")
    fig_e6_scatter(FIG / "4-3_e6_edge_consistency.png", cfg, codes)
    print("[done] 三张图已生成于 docs/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
