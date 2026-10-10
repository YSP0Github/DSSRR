"""新图 C：STA/LTA 下游可用性

论文 §Seam Boundary Fidelity 声称 DSSRR 的输出可直接进入 STA/LTA 检测流程。
本图给出定量证据，并把主张精确化为"检测器工作点保真"：

(a) 修复区间内的平均触发次数随触发阈值的变化（真值为参照）
(b) 阈值 5 时各方法保留的触发率（相对真值的百分比）

解读：真值背景本身在 1 s/30 s 的 STA/LTA 下就会稳定触发（≈20 次/窗口）。
DSSRR 复现了这一工作点（≈21 次，差 7%），而插值类与 UNet 把触发率压低 90% 以上
—— 即修复窗口对检测器而言接近"失明"。这比"是否产生假触发"更贴近实际后果。
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import sys
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import apply_srl_style, DOUBLE_COL_WIDTH, METHOD_COLORS, SAVE_KW, mlabel

apply_srl_style()

ROOT = r'G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6'
CLEAN_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h'
FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'
CSV_OUT = r'G:\SeisY\docs\dssrr_paper\experiments\v6_stalta_downstream.csv'

SR = 6.625
NSTA, NLTA = int(1.0 * SR), int(30.0 * SR)
PAD = 100
THRS = [3.0, 5.0, 8.0, 12.0]
METHODS = ['DSSRR', 'FFT', 'UNet', 'SSA', 'CubicSpline', 'Linear']
SERIES = ['Truth'] + METHODS
COL = {'Truth': METHOD_COLORS['GroundTruth'], 'DSSRR': METHOD_COLORS['DSSRR'],
       'FFT': METHOD_COLORS['FFT'], 'UNet': METHOD_COLORS['UNet'],
       'SSA': METHOD_COLORS['SSA'], 'CubicSpline': METHOD_COLORS['CubicSpline'],
       'Linear': METHOD_COLORS['Linear']}

clean_map = {}
for f in sorted(os.listdir(CLEAN_DIR)):
    if not f.endswith('.mseed'):
        continue
    p = f[:-6].split('.')
    clean_map[f'{p[1]}_{p[4].split("-")[0]}'] = os.path.join(CLEAN_DIR, f)

acc = {t: {m: [] for m in SERIES} for t in THRS}
rows = []
for ev in sorted(os.listdir(ROOT)):
    p_ev = os.path.join(ROOT, ev)
    if not os.path.isdir(p_ev):
        continue
    head, scenario = ev.rsplit('_', 1)
    station, start = head.split('_', 1)
    clean = read(clean_map[f'{station}_{start}'])[0].data.astype(float)

    for anom in sorted(os.listdir(p_ev)):
        p_anom = os.path.join(p_ev, anom)
        if not os.path.isdir(p_anom):
            continue
        for lvl in sorted(os.listdir(p_anom)):
            cdir = os.path.join(p_anom, lvl)
            cf_path = os.path.join(cdir, 'corrupted.mseed')
            if not os.path.exists(cf_path):
                continue
            corr = read(cf_path)[0].data.astype(float)
            n = min(len(corr), len(clean))
            d = np.abs(corr[:n] - clean[:n])
            m = d > 1e-3 * np.std(clean[:n])
            s, e = int(np.where(m)[0][0]), int(np.where(m)[0][-1])
            lo, hi = max(0, s - int(300 * SR)), min(n, e + 1 + int(300 * SR))
            i0, i1 = s - lo, e + 1 - lo

            for mn in SERIES:
                if mn == 'Truth':
                    series = clean[:n]
                else:
                    fp = os.path.join(cdir, f'repaired_{mn}.mseed')
                    if not os.path.exists(fp):
                        continue
                    series = read(fp)[0].data.astype(float)[:n]
                z = series[lo:hi] - series[lo:hi].mean()
                cf = classic_sta_lta(z, NSTA, NLTA)
                j0, j1 = max(0, i0 - PAD), min(len(cf), i1 + PAD)
                for t in THRS:
                    tr = trigger_onset(cf, t, t * 0.6)
                    k = int(np.sum((tr[:, 0] >= j0) & (tr[:, 0] <= j1))) if len(tr) else 0
                    rows.append(dict(event=ev, scenario=scenario, anom_type=anom, level=lvl,
                                     method=mn, threshold=t, n_trigger=k))
                    if anom == 'dropout':
                        acc[t][mn].append(k)

df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False, encoding='utf-8-sig')

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.75))

# ---- (a) triggers vs threshold ----
ax = axes[0]
for mn in SERIES:
    ys = [np.mean(acc[t][mn]) for t in THRS]
    lw = 1.5 if mn in ('Truth', 'DSSRR') else 1.0
    ls = '--' if mn == 'Truth' else '-'
    ax.plot(THRS, ys, marker='o', ms=3.5, color=COL[mn], lw=lw, ls=ls, label=mlabel(mn))
ax.set_yscale('log')
ax.set_xlabel('STA/LTA trigger threshold')
ax.set_ylabel('Mean triggers inside repaired window')
ax.set_title('(a) Detector operating point', fontsize=9, fontweight='bold', loc='left', pad=4)
ax.set_xticks(THRS)
ax.set_xticklabels([f'{t:.0f}' for t in THRS])
ax.legend(fontsize=6.2, handletextpad=0.3, borderaxespad=0.2, ncol=2)

# ---- (b) retention at thr = 5 ----
ax = axes[1]
t0 = 5.0
truth_mean = np.mean(acc[t0]['Truth'])
ret = [100.0 * np.mean(acc[t0][m]) / truth_mean for m in METHODS]
bars = ax.bar(np.arange(len(METHODS)), ret, color=[COL[m] for m in METHODS], width=0.65,
              edgecolor='none')
ax.axhline(100.0, color=METHOD_COLORS['GroundTruth'], lw=1.0, ls='--')
ax.text(len(METHODS) - 0.5, 104, 'true background', fontsize=6.2, color='0.25', ha='right')
ax.set_xticks(np.arange(len(METHODS)))
ax.set_xticklabels([mlabel(m) for m in METHODS], rotation=30, ha='right')
ax.set_ylabel('Trigger rate retained (% of truth)')
ax.set_title('(b) Detector response retained (thr = 5)', fontsize=9, fontweight='bold',
             loc='left', pad=4)
ax.set_ylim(0, 125)
for b, v in zip(bars, ret):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, f'{v:.0f}%', ha='center', va='bottom',
            fontsize=7)

plt.tight_layout(w_pad=1.4)
out_path = f'{FIG_DIR}/fig06_stalta_downstream.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"HD Figure saved to {out_path}")
print('\n--- dropout: mean triggers ---')
for mn in SERIES:
    print('%-12s' % mn + ''.join('%8.2f' % np.mean(acc[t][mn]) for t in THRS))
