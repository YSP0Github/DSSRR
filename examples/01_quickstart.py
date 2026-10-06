# -*- coding: utf-8 -*-
"""DSSRR 示例 01 —— 最小快速上手（合成数据，无需任何外部文件）。

这个脚本演示 DSSRR 的**最小可用路径**，只依赖 numpy：

1. 造一条「常数基线 + 有色噪声」的合成记录（模拟月震背景噪声）。
2. 人为挖掉一段（置零），当作需要修复的缺口。
3. 用 :class:`dssrr.DSSRR` 修复它。
4. 检查两条硬约束：
   - 缺口内不再是 0（确实被填上了）；
   - 缺口**以外**的样本逐位不变（DSSRR 从不改动健康数据）。
5. 用 :meth:`DSSRR.repair_with_report` 取回完整报告，打印参考窗与质量指标。

运行::

    python examples/01_quickstart.py

无需安装 dssrr（脚本会把仓库根目录加入 sys.path），也无需下载任何数据。
"""

import sys
from pathlib import Path

# 允许从源码目录直接运行，无需先 pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dssrr import DSSRR, auto_reference_length


def make_synthetic_trace(sr=6.625, duration_sec=1800, seed=0):
    """造一条带慢漂移的有色噪声记录（模拟行星地震背景）。

    参数
    ----
    sr : float
        采样率 Hz。
    duration_sec : float
        记录长度（秒）。
    seed : int
        随机种子，保证可复现。

    返回
    ----
    (data, sr) : 1-D ndarray 与采样率。
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    t = np.arange(n) / sr
    # 一阶自回归把白噪声染成偏低频的「有色」噪声
    white = rng.normal(0.0, 1.0, n)
    colored = np.zeros(n)
    for i in range(1, n):
        colored[i] = 0.98 * colored[i - 1] + 0.02 * white[i]
    # 叠一条缓慢漂移，让基线对齐这一步也有事可做
    drift = 5.0 * np.sin(2 * np.pi * 0.002 * t)
    return 450.0 + 3.0 * colored + drift


def main():
    sr = 6.625

    # ---------------------------------------------------------------
    # 1) 造数据并挖一个 10 分钟的缺口
    # ---------------------------------------------------------------
    data = make_synthetic_trace(sr=sr, duration_sec=1800, seed=0)
    gap_sec = 600.0
    start = int(400 * sr)
    end = start + int(gap_sec * sr) - 1

    corrupted = data.copy()
    corrupted[start:end + 1] = 0.0          # 缺口：一段全零
    print(f"trace      : {len(data)} samples @ {sr} Hz "
          f"({len(data) / sr / 60:.1f} min)")
    print(f"gap        : samples [{start}, {end}] "
          f"({gap_sec:.0f} s, {(end - start + 1)} samples)")

    # ---------------------------------------------------------------
    # 2) 按论文协议挑选参考窗长度：L = clip(anomaly/2, 120 s, 600 s)
    # ---------------------------------------------------------------
    ref_sec = auto_reference_length(end - start + 1, sr)
    print(f"reference  : {ref_sec:.1f} s per side (auto_reference_length)")

    # ---------------------------------------------------------------
    # 3) 修复
    # ---------------------------------------------------------------
    model = DSSRR(sr=sr, reference_sec=ref_sec, seed=42)
    repaired = model.repair(corrupted, start, end)

    # ---------------------------------------------------------------
    # 4) 检查两条硬约束
    # ---------------------------------------------------------------
    filled = not np.allclose(repaired[start:end + 1], 0.0)
    outside_untouched = (
        np.array_equal(repaired[:start], corrupted[:start])
        and np.array_equal(repaired[end + 1:], corrupted[end + 1:])
    )
    print(f"gap filled?          {filled}")
    print(f"outside untouched?   {outside_untouched}")

    # 与「真值」比较：DSSRR 保统计量而非相位，所以比的是标准差/均值是否接近
    truth = data[start:end + 1]
    est = repaired[start:end + 1]
    print(f"segment mean  true={truth.mean():8.3f}  repaired={est.mean():8.3f}")
    print(f"segment std   true={truth.std():8.3f}  repaired={est.std():8.3f}")

    # ---------------------------------------------------------------
    # 5) 完整报告（参考窗位置、基线对齐、质量校验、已知局限）
    # ---------------------------------------------------------------
    _, report = model.repair_with_report(
        corrupted, start, end,
        reference_before_sec=ref_sec,
        reference_after_sec=ref_sec,
        random_seed=42,
    )
    rb, ra = report["reference_before"], report["reference_after"]
    print("\n--- report ---")
    print(f"method            : {report['method']}")
    print(f"ref before        : [{rb['start']}, {rb['end']}]  ({rb['length']} smp)")
    print(f"ref after         : [{ra['start']}, {ra['end']}]  ({ra['length']} smp)")
    print(f"boundary blend    : {report['boundary_blend_samples']} samples")
    print(f"changed samples   : {report['changed_samples']}")

    v = report["verification"]
    print("\n--- quality diagnostics ---")
    print(f"energy_ratio      : {v['energy_ratio']:.4f}  "
          f"(1.0 = repaired energy matches the references)")
    print(f"envelope_continuity: {v['envelope_continuity']:.4f}  "
          f"(join discontinuity, lower is better)")
    print(f"dc_continuity     : {v['dc_continuity']:.4f}  "
          f"(mean-level step at the joins, lower is better)")
    print(f"spectral_continuity: {v['spectral_continuity']:.4f}  "
          f"(cosine similarity of the *raw* periodograms)")
    print(f"autocorr_consistency: {v['autocorr_consistency']:.4f}")
    print(f"verifier verdict  : passed={v['passed']}   ({v['details']})")
    # 说明：spectral_continuity 比较的是**单次实现的原始周期图**（未做 Welch
    # 平滑），因此对于宽带噪声，两条独立实现之间的余弦相似度天然只有 ~0.5；
    # 它衡量的是「逐实现是否一致」，而不是「谱形是否一致」。DEFAULT_CONFIG 里
    # 的 0.95 阈值是按窄带占优的记录标定的，对宽带噪声会恒定判 FAIL——所以
    # 这里把它当诊断量看，而不是当成最终验收结论。

    print("\nlimitations:")
    for line in report["limitations"]:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
