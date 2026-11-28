"""NetworkBuilder 骨架验证。

用 PM2.5 训练集前 500 小时（21 天）做 smoke test，检查：
1. TE / 滞后相关 / 融合 三种 Builder 都能跑通
2. 输出格式正确（src/dst/weight/lag/p_value/fdr_pass）
3. 耗时在可接受范围
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.domain.networks.transfer_entropy import TransferEntropyBuilder
from src.domain.networks.lagged_corr import LaggedCorrBuilder
from src.domain.networks.composite import CompositeBuilder
from src.domain.networks.common import transfer_entropy, phase_randomize, fdr_bh


def main():
    print("=== NetworkBuilder 骨架验证 ===\n")

    # 加载 PM2.5 train 前 500 小时
    d = np.load("data/processed/obs.npz", allow_pickle=True)
    values = d["values"][:500, :, 0]                  # (500, 24) PM2.5
    codes = [str(c) for c in d["codes"]]
    print(f"数据形状: {values.shape} (T, N)，污染物=pm25")
    print(f"缺失率: {np.isnan(values).mean()*100:.1f}%\n")

    # ---- 1. 单元测试：TE 对称性破坏（应 TE(x→y) != TE(y→x)） ----
    print("--- 单元测试 ---")
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 500)
    y = np.roll(x, 2) + rng.normal(0, 0.3, 500)     # y 是 x 的延迟 + 噪声
    te_xy = transfer_entropy(x, y, lag=2, n_bins=8)
    te_yx = transfer_entropy(y, x, lag=2, n_bins=8)
    print(f"  构造耦合序列：TE(x→y)={te_xy:.4f}  TE(y→x)={te_yx:.4f}")
    assert te_xy > te_yx, "TE 应识别 x→y 方向"
    print("  ✓ TE 方向性测试通过")

    # ---- 2. 相位随机化保留功率谱 ----
    surr = phase_randomize(x, rng)
    power_orig = np.abs(np.fft.rfft(x)) ** 2
    power_surr = np.abs(np.fft.rfft(surr)) ** 2
    rel_err = np.abs(power_orig - power_surr).mean() / power_orig.mean()
    print(f"  相位随机化功率谱相对误差: {rel_err:.4f}（应接近 0）")
    assert rel_err < 0.01, "相位随机化应保留功率谱"
    print("  ✓ 相位随机化测试通过")

    # ---- 3. FDR 校正 ----
    # BH: k 位置的阈值为 alpha*k/m；选最大 k 使 p_(k) <= alpha*k/m
    # 用 p=[0.001,0.005,0.01,0.02,0.04,0.10,0.50]，alpha=0.05, m=7：
    # 阈值=[0.0071, 0.0143, 0.0214, 0.0286, 0.0357, 0.0429, 0.05]
    # 前 4 个通过，第 5 个 (0.04) 不通过
    p = np.array([0.001, 0.005, 0.01, 0.02, 0.04, 0.10, 0.50])
    fdr = fdr_bh(p, alpha=0.05)
    print(f"  FDR 测试: p={p.tolist()}")
    print(f"           通过={fdr.tolist()}（BH 期望前 4 个通过）")
    assert fdr[:4].all() and not fdr[4:].any(), f"FDR 结果异常: {fdr.tolist()}"
    print("  ✓ FDR 测试通过\n")

    # ---- 4. TransferEntropyBuilder ----
    print("--- TransferEntropyBuilder ---")
    t0 = time.time()
    te_builder = TransferEntropyBuilder(lag=1, n_bins=8, n_surrogates=10, top_k=20, seed=42)
    te_edges = te_builder.build(values)
    print(f"  耗时: {time.time()-t0:.1f}s（surrogates=10）")
    print(f"  边数: {len(te_edges)} (top_k=20)")
    print(f"  前 3 条边:")
    for e in te_edges[:3]:
        src, dst = codes[e["src"]], codes[e["dst"]]
        print(f"    {src} → {dst}  weight={e['weight']} p={e['p_value']} fdr={e['fdr_pass']}")
    fdr_pass_count = sum(e["fdr_pass"] for e in te_edges)
    print(f"  FDR 通过率: {fdr_pass_count}/{len(te_edges)} = {fdr_pass_count/len(te_edges)*100:.0f}%")

    # ---- 5. LaggedCorrBuilder ----
    print("\n--- LaggedCorrBuilder ---")
    t0 = time.time()
    corr_builder = LaggedCorrBuilder(lag_candidates=[1, 2, 3, 4], n_surrogates=10, top_k=20, seed=42)
    corr_edges = corr_builder.build(values)
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(f"  边数: {len(corr_edges)}")
    print(f"  前 3 条边:")
    for e in corr_edges[:3]:
        src, dst = codes[e["src"]], codes[e["dst"]]
        print(f"    {src} → {dst}  weight={e['weight']} lag={e['lag']}h sign={e['sign']} p={e['p_value']} fdr={e['fdr_pass']}")

    # ---- 6. CompositeBuilder ----
    print("\n--- CompositeBuilder（主网络） ---")
    t0 = time.time()
    composite = CompositeBuilder(w_te=0.5, w_corr=0.5, w_wind=0.0, top_k=20, n_surrogates=10, seed=42)
    fused_edges = composite.build(values)
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(f"  边数: {len(fused_edges)}")
    print(f"  前 3 条边:")
    for e in fused_edges[:3]:
        src, dst = codes[e["src"]], codes[e["dst"]]
        comp = e["components"]
        print(f"    {src} → {dst}  weight={e['weight']} (te={comp['te']}, corr={comp['corr']}, wind={comp['wind']})")

    # ---- 7. 耗时估算（全量 4 年） ----
    print(f"\n=== 全量耗时估算 ===")
    # TE 部分占总耗时大头
    # 当前 500h × surrogates=10，N=24
    # 全量 35064h × surrogates=50
    # 时间复杂度 ~ T × surrogates
    curr_seconds = time.time() - t0
    est_full = curr_seconds * (35064 / 500) * (50 / 10)
    print(f"  当前 composite 耗时: {curr_seconds:.1f}s")
    print(f"  全量（T=35064, surrogates=50）预估: {est_full/60:.1f} 分钟")

    print("\n=== 全部通过 ===")


if __name__ == "__main__":
    main()
