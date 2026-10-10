"""M4 附：C3（step + recovery）修复窗 ACF $L_2$ 超带的**窗口长度敏感性**检验。

背景
----
M4 主脚本（`M4_archival_metrics.py`）发现：C3 修复窗的 ACF $L_2$ 距离（0.317）落在
自然变异带的第 95.4 百分位，略高于带的上沿（0.307）——**唯一**超带的指标。

诊断假设：C3 的 step 之后有一段**缓慢的指数恢复**（约 25 min），而 spike 检测器只标出
陡峭的跃变、未覆盖缓慢的尾部；主导修复窗的右端因此切在**衰减尾**内部而非安静背景，
其自相关残留一个单调衰减分量 → 与其它健康窗的 ACF 距离偏大。

检验：固定窗口起点（= C3 主导窗起点 5.508 h），只改变窗长 L，观察修复窗 ACF $L_2$
相对自然带的位置如何变化。若窗长变短（不深入尾部）即回落到带内，则支持"检测器覆盖不足"
的解释，而非"合成样本有问题"。

输出：终端表格 + `results_obs/M4_window_sensitivity.csv`
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from obspy import read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from M4_archival_metrics import seg_metrics, OBS, BEFORE_DIR  # noqa: E402

SR = 6.625
FILE = "XA.S15.01.MHZ.19760113_061655-19760113_161655.mseed"
S_H = 5.508                     # 与主脚本 C3 的 s_h 一致
LENGTHS_MIN = [5.0, 8.8, 12.6, 17.4, 22.6, 30.2]


def main():
    b = read(os.path.join(BEFORE_DIR, FILE))[0]
    a = read(os.path.join(OBS, FILE))[0]
    n = min(b.stats.npts, a.stats.npts)
    bef = b.data[:n].astype(float)
    aft = a.data[:n].astype(float)

    s = int(round(S_H * 3600 * SR))
    rows = []
    print(f"{'L (min)':>8}{'L (samples)':>13}{'N_h':>5}{'pairs':>7}"
          f"{'repaired ACF':>14}{'band median':>13}{'band [5,95]':>22}{'pctile':>8}{'>band?':>8}")
    for Lm in LENGTHS_MIN:
        L = int(round(Lm * 60 * SR))
        e = min(s + L - 1, n - 1)
        L = e - s + 1
        starts = list(range(0, n - L + 1, L))
        healthy = [t for t in starts if t + L - 1 < s or t > e]
        H = [bef[t:t + L] for t in healthy]
        if len(H) < 2:
            continue

        hh = np.array([seg_metrics(H[i], H[j])["acf_l2"]
                       for i in range(len(H)) for j in range(i + 1, len(H))])
        rv = np.array([seg_metrics(aft[s:e + 1], h)["acf_l2"] for h in H])
        rep = float(np.median(rv))
        pct = float((hh < rep).mean() * 100)
        band95 = float(np.percentile(hh, 95))
        over = rep > band95
        print(f"{Lm:>8.1f}{L:>13d}{len(H):>5}{len(hh):>7}"
              f"{rep:>14.3f}{np.median(hh):>13.3f}"
              f"{'[%.3f, %.3f]' % (np.percentile(hh, 5), band95):>22}"
              f"{pct:>7.1f}%{'yes' if over else 'no':>8}")
        rows.append(dict(L_min=Lm, L_samples=L, n_healthy=len(H), n_pairs=len(hh),
                         repaired_acf=rep, band_median=float(np.median(hh)),
                         band_p05=float(np.percentile(hh, 5)), band_p95=band95,
                         pctile=pct, exceeds_band=bool(over)))

    out = os.path.join(OBS, "M4_window_sensitivity.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n写出 {out}")


if __name__ == "__main__":
    main()
