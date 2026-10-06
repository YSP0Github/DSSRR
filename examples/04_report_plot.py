# -*- coding: utf-8 -*-
"""DSSRR 示例 04 —— 生成一张「修复报告」对比图。

修复完成之后，通常需要一张能直接放进报告/论文的图：时域前后对比、频谱对比、
质量指标一览。这正是 :class:`~dssrr.repair_lib.report_plot.RepairReportPlotter`
做的事。

本脚本：

1. 用合成数据跑一次 DSSRR 修复，拿到 ``report``（含参考窗、校验指标）。
2. 调 :meth:`RepairReportPlotter.plot_report` 输出综合报告图。
3. 顺带演示一行式的便捷函数 :func:`plot_comparison_quick`。

输出：``04_report_plot.png`` 与 ``04_report_quick.png``。

运行::

    python examples/04_report_plot.py

依赖：numpy、scipy、matplotlib。脚本会把 matplotlib 后端钉成 ``Agg``，
因此可在无显示器的环境运行。
"""

import os
import sys
from pathlib import Path

# 本示例只出图、不开窗口：在导入 matplotlib / report_plot 之前先钉住后端。
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from dssrr import DSSRR, auto_reference_length
from dssrr.repair_lib.report_plot import RepairReportPlotter, plot_comparison_quick

HERE = Path(__file__).resolve().parent


def make_trace(sr=6.625, duration_sec=2400, seed=1):
    """合成一条有色噪声记录。"""
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    white = rng.normal(0.0, 1.0, n)
    colored = np.zeros(n)
    for i in range(1, n):
        colored[i] = 0.98 * colored[i - 1] + 0.02 * white[i]
    return 450.0 + 3.0 * colored


def main():
    sr = 6.625
    data = make_trace(sr=sr)

    # 挖一个 5 分钟缺口
    start = int(900 * sr)
    end = start + int(300 * sr) - 1
    corrupted = data.copy()
    corrupted[start:end + 1] = 0.0

    ref_sec = auto_reference_length(end - start + 1, sr)
    model = DSSRR(sr=sr, reference_sec=ref_sec, seed=42)

    repaired, report = model.repair_with_report(
        corrupted, start, end,
        reference_before_sec=ref_sec,
        reference_after_sec=ref_sec,
        random_seed=42,
    )

    # 从 report 里取出前后参考段，供报告图单独着色
    rb, ra = report["reference_before"], report["reference_after"]
    ref_before = corrupted[rb["start"]:rb["end"] + 1]
    ref_after = corrupted[ra["start"]:ra["end"] + 1]

    metadata = {
        "filename": "synthetic_demo.mseed",
        "station": "SYN",
        "channel": "MHZ",
        "method": report["method"],
        "reference_sec": ref_sec,
    }

    # ---------------------------------------------------------------
    # 1) 完整报告图（中文）
    # ---------------------------------------------------------------
    plotter = RepairReportPlotter(sr=sr, lang="zh")
    out1 = HERE / "04_report_plot.png"
    plotter.plot_report(
        original=corrupted,
        repaired=repaired,
        anomaly_start=start,
        anomaly_end=end,
        ref_before=ref_before,
        ref_after=ref_after,
        verification=report["verification"],
        metadata=metadata,
        save_path=str(out1),
        figsize=(16, 11),
        dpi=150,
        custom_title="DSSRR repair report (synthetic)",
    )
    print(f"full report figure -> {out1}")

    # ---------------------------------------------------------------
    # 2) 一行式便捷函数（英文）
    # ---------------------------------------------------------------
    out2 = HERE / "04_report_quick.png"
    plot_comparison_quick(
        original=corrupted,
        repaired=repaired,
        anomaly_start=start,
        anomaly_end=end,
        sr=sr,
        save_path=str(out2),
        lang="en",
        show_reference_segments=True,
        show_inset_zoom=False,
    )
    print(f"quick figure       -> {out2}")

    v = report["verification"]
    print(f"\nverification: energy_ratio={v['energy_ratio']:.4f}  "
          f"envelope_continuity={v['envelope_continuity']:.4f}  "
          f"dc_continuity={v['dc_continuity']:.4f}  "
          f"spectral_continuity={v['spectral_continuity']:.4f}")
    print("(the report figure shows the same metrics; see 01_quickstart.py for a "
          "note on how spectral_continuity is defined)")


if __name__ == "__main__":
    main()
