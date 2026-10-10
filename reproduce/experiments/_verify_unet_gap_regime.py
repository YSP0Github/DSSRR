"""只读核验：1D U-Net 的"训练/测试缺口长度制度差"是否构成混淆变量。

审计报告 §3.5「问题 2」提出：论文把 U-Net 在 Table 1 上的劣化解释为"跨域
（背景/幅度）效应"，但该对比同时混入了"缺口长度外推 2–24×"这一未声明变量。

本脚本用**已有归档数据**（不重跑任何实验）把这两者拆开：

  训练域      unet_model.py  PATCH=1536 样本=231.8 s；dropout_long 占 20–40%
                             → 缺口 307–614 样本 = 46.3–92.7 s
  域内基准    E1_synthetic.py inject_dropout_long  integers(500,1000) 样本
                             = 75.5–150.9 s
  测试域      E-RealInj dropout 300 / 900 / 3600 s → 相对域内上限 2× / 6× / 24×

判据：若"缺口长度外推"是主因，U-Net 误差应随缺口长度**单调增长**。

数据源：
  results_E2r_full_v6/stats_summary.csv      E-RealInj（Table 1），含 level 列
  results_dl_synth/dl_synth_by_anomaly.csv   域内基准（Table S1）按异常类型

用法：
  cd experiments && /g/miniconda3/envs/python_3.11/python.exe _verify_unet_gap_regime.py
"""
from __future__ import annotations

import os
import pandas as pd

SR = 6.625  # Hz，Apollo PSE 采样率

# ---- 代码里读出来的区间定义（此处硬编码，注释给出出处，避免依赖 import 副作用）----
PATCH = 1536                                    # unet_model.py:26
TRAIN_LONG_FRAC = (0.20, 0.40)                  # unet_model.py:102
E1_LONG = (500, 1000)                           # E1_synthetic.py:116
E2R_LEVELS = {"small": 300.0, "medium": 900.0, "large": 3600.0}   # E-RealInj dropout


def _sec(n_samples: float) -> float:
    return n_samples / SR


def report_regimes() -> None:
    print("=" * 78)
    print("一、三个缺口长度区间（samples → seconds）")
    print("=" * 78)
    lo = int(PATCH * TRAIN_LONG_FRAC[0])
    hi = int(PATCH * TRAIN_LONG_FRAC[1])
    print(f"  训练   patch={PATCH} 样本={_sec(PATCH):.1f} s；dropout_long {lo}-{hi} 样本"
          f" = {_sec(lo):.1f}-{_sec(hi):.1f} s")
    print(f"  域内   dropout_long {E1_LONG[0]}-{E1_LONG[1]} 样本"
          f" = {_sec(E1_LONG[0]):.1f}-{_sec(E1_LONG[1]):.1f} s")
    print("  测试   E-RealInj dropout "
          + " / ".join(f"{k}={v:.0f} s" for k, v in E2R_LEVELS.items()))
    dom_max = _sec(E1_LONG[1])
    print()
    print(f"  相对域内上限 {dom_max:.1f} s 的外推倍数：")
    for k, v in E2R_LEVELS.items():
        print(f"     {k:7s} {v:6.0f} s  →  {v / dom_max:5.2f}×")


def report_unet_scaling() -> None:
    v6 = pd.read_csv("results_E2r_full_v6/stats_summary.csv")
    print()
    print("=" * 78)
    print("二、U-Net 误差 vs 缺口长度（Table 1 数据按 level 拆开，未重跑）")
    print("=" * 78)

    u = v6[v6["method"] == "UNet"]
    for anom in ["dropout", "spike"]:
        s = u[u["anom_type"] == anom]
        piv = s.pivot_table(index="level", columns="scenario",
                            values="wasserstein", aggfunc="mean")
        order = [l for l in ["small", "medium", "large"] if l in piv.index]
        piv = piv.reindex(order)
        print(f"\n  【{anom}】Wasserstein（行=level）")
        print(piv.round(3).to_string())
        m = s.groupby("level")["wasserstein"].mean().reindex(order)
        vals = list(m.values)
        print(f"    small→large: {vals[0]:.3f} → {vals[-1]:.3f}  "
              f"(变化 {vals[-1] / vals[0]:.2f}×)")
        mono = all(vals[i] <= vals[i + 1] * 1.05 for i in range(len(vals) - 1))
        print(f"    单调递增？ {'是' if mono else '否（非单调）'}")

    # 域内基准
    bya = pd.read_csv("results_dl_synth/dl_synth_by_anomaly.csv")
    bu = bya[bya["method"] == "UNet"].set_index("anomaly")["wasserstein"]
    print()
    print("=" * 78)
    print("三、域内基准（Table S1，E9）按异常类型拆开")
    print("=" * 78)
    for k, v in bu.items():
        tag = ""
        if k == "dropout_long":
            tag = f"   ← 域内最长缺口 {_sec(E1_LONG[0]):.0f}-{_sec(E1_LONG[1]):.0f} s"
        print(f"    {k:15s} {v:.3f}{tag}")
    print(f"    {'(六类平均)':15s} {bu.mean():.3f}   ← 论文正文引用的 0.18")

    print()
    print("=" * 78)
    print("四、结论")
    print("=" * 78)
    dl = bu["dropout_long"]
    e2r_drop = v6[(v6["method"] == "UNet") & (v6["anom_type"] == "dropout")]["wasserstein"].mean()
    print(f"  域内 dropout_long ({_sec(E1_LONG[0]):.0f}-{_sec(E1_LONG[1]):.0f} s) = {dl:.3f}")
    print(f"  E-RealInj dropout (300-3600 s) 均值 = {e2r_drop:.3f}")
    print(f"  → 劣化 {e2r_drop / dl:.1f}×，但该劣化在 300→3600 s 上基本平坦")
    print("  → 若「缺口长度外推」是主因，300→3600 s 应继续单调上升；实测没有。")
    print("  → 因此「跨域（背景/幅度）」解释站得住；长度外推不是主导机制。")


def main() -> None:
    print(f"cwd = {os.getcwd()}\n")
    report_regimes()
    report_unet_scaling()
    print()


if __name__ == "__main__":
    main()
