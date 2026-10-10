"""新图 B：双侧参考谱融合的必要性

论文命名中的核心设计是"双侧（Dual-Sided）"参考。本图给出它的直接证据：
(a) 代表性案例的 PSD —— 真值 vs 左侧单侧估计 vs 右侧单侧估计 vs 双侧 log 域融合
(b) 全部 36 个 dropout 案例上，三种估计到真值 PSD 的对数谱距离（LSD）

参考段几何严格遵循论文协议：
    L = clip(anomaly_sec / 2, 120 s, 600 s),  G = reference_gap_sec = 1.0 s
    R_b = x[s-G-L : s-G],   R_a = x[e+G+1 : e+G+1+L]
    fused = exp( 0.5 * (log P_left + log P_right) )   （log 域等权融合）
只读取 v6 已落盘的 mseed 与真值，不重跑实验。
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import welch
from matplotlib.ticker import FixedLocator
import sys
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import apply_srl_style, DOUBLE_COL_WIDTH, METHOD_COLORS, SAVE_KW

apply_srl_style()

ROOT = r'G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6'
CLEAN_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h'
FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'
PANEL_DIR = os.path.join(FIG_DIR, '_panels')   # 合并用中间面板（非最终图）
os.makedirs(PANEL_DIR, exist_ok=True)
SR = 6.625
GAP_SEC = 1.0

REP = ('S12_19760113_070152_transient', 'dropout', 'large')

clean_map = {}
for f in sorted(os.listdir(CLEAN_DIR)):
    if not f.endswith('.mseed'):
        continue
    p = f[:-6].split('.')
    clean_map[f'{p[1]}_{p[4].split("-")[0]}'] = os.path.join(CLEAN_DIR, f)


def spectra(ev, anom, lvl, nperseg=1024):
    cdir = os.path.join(ROOT, ev, anom, lvl)
    head, _ = ev.rsplit('_', 1)
    station, start = head.split('_', 1)
    clean = read(clean_map[f'{station}_{start}'])[0].data.astype(float)
    corr = read(os.path.join(cdir, 'corrupted.mseed'))[0].data.astype(float)
    d = np.abs(corr - clean[:len(corr)])
    m = d > 1e-3 * np.std(clean[:len(corr)])
    s, e = int(np.where(m)[0][0]), int(np.where(m)[0][-1])
    L = int(np.clip((e - s + 1) / SR / 2.0, 120.0, 600.0) * SR)
    G = int(round(GAP_SEC * SR))
    lb, le = s - G - L, s - G
    ra, rb = e + G + 1, e + G + 1 + L
    if lb < 0 or rb > len(corr):
        return None
    nper = min(nperseg, e - s + 1)
    f, pt = welch(clean[s:e + 1], fs=SR, nperseg=nper)
    _, pl = welch(corr[lb:le], fs=SR, nperseg=min(nperseg, le - lb))
    _, pr = welch(corr[ra:rb], fs=SR, nperseg=min(nperseg, rb - ra))
    k = min(len(pt), len(pl), len(pr))
    pt, pl, pr = pt[:k], pl[:k], pr[:k]
    pf = np.exp(0.5 * (np.log(pl + 1e-30) + np.log(pr + 1e-30)))
    return f[:k], pt, pl, pr, pf


def lsd(a, b):
    m = (a > 0) & (b > 0)
    return float(np.sqrt(np.mean((10 * np.log10(a[m]) - 10 * np.log10(b[m])) ** 2)))


# ---- 汇总全部 dropout 案例 ----
rows = []
for ev in sorted(os.listdir(ROOT)):
    p_ev = os.path.join(ROOT, ev)
    if not os.path.isdir(p_ev):
        continue
    for lvl in sorted(os.listdir(os.path.join(p_ev, 'dropout'))):
        out = spectra(ev, 'dropout', lvl)
        if out is None:
            continue
        _, pt, pl, pr, pf = out
        rows.append((lsd(pt, pl), lsd(pt, pr), lsd(pt, pf)))
arr = np.array(rows)
print(f'n={len(arr)} dropout cases')
print('mean LSD  left=%.3f  right=%.3f  fused=%.3f' % tuple(arr.mean(axis=0)))
print('fused better than both sides in %d/%d cases'
      % (int(np.sum((arr[:, 2] < arr[:, 0]) & (arr[:, 2] < arr[:, 1]))), len(arr)))

# ---- 单侧参考估计专用配色 ----
# 原方案左/右都用灰系（#95a5a6 / #7f8c8d），两者亮度太接近，在含噪 PSD 上几乎无法区分，
# 灰度打印或色觉障碍读者更无法分辨。这里改用色盲安全、且与黑色真值/深蓝融合线都拉得开的
# 一对色（橙 / 青绿），线型（点线 / 点划线）作为冗余区分线索，透明度 0.9 避免过饱和。
C_LEFT = '#d95f02'    # 左侧单侧参考（橙色，点线）
C_RIGHT = '#1b9e77'   # 右侧单侧参考（青绿，点划线）
C_FUSED = METHOD_COLORS['DSSRR']   # 双侧对数域融合（深蓝，实线，最粗）

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 2.55))

# ---- (a) PSD ----
ax = axes[0]
f, pt, pl, pr, pf = spectra(*REP)
ax.semilogx(f[1:], 10 * np.log10(pt[1:]), color=METHOD_COLORS['GroundTruth'], lw=1.5, ls='--',
            label='Truth (anomaly segment)', zorder=3)
ax.semilogx(f[1:], 10 * np.log10(pl[1:]), color=C_LEFT, lw=1.1, ls=':',
            alpha=0.9, label='Left reference only', zorder=1)
ax.semilogx(f[1:], 10 * np.log10(pr[1:]), color=C_RIGHT, lw=1.1, ls='-.',
            alpha=0.9, label='Right reference only', zorder=1)
ax.semilogx(f[1:], 10 * np.log10(pf[1:]), color=C_FUSED, lw=1.9,
            label='Two-sided log-fused', zorder=2)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.set_title('(c) Reference spectrum estimation', fontsize=9, fontweight='bold', loc='left', pad=4)
ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0]))
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.set_xlim(0.005, SR / 2)
ax.legend(loc='lower left', fontsize=6.2, handletextpad=0.4, borderaxespad=0.2)

# ---- (b) LSD summary ----
ax = axes[1]
labels = ['Left\nonly', 'Right\nonly', 'Two-sided\nfused']
means = arr.mean(axis=0)
stds = arr.std(axis=0)
cols = [C_LEFT, C_RIGHT, C_FUSED]
bars = ax.bar(np.arange(3), means, yerr=stds, color=cols, width=0.6, edgecolor='none',
              error_kw=dict(lw=0.8, capsize=3, ecolor='0.35'))
ax.set_xticks(np.arange(3))
ax.set_xticklabels(labels)
ax.set_ylabel('Log-spectral distance to truth  (lower = better)')
ax.set_title('(d) All dropout cases (n=%d)' % len(arr), fontsize=9, fontweight='bold',
             loc='left', pad=4)
# 上限必须容下最高的误差棒上端（mean+1σ），否则最高那根会被坐标轴顶边截断。
_top = float((means + stds).max())
ax.set_ylim(0, _top * 1.14)
for b, v, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width() / 2, v + s + _top * 0.03,
            f'{v:.2f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout(w_pad=1.3)
out_path = f'{PANEL_DIR}/fig07cd_reference_fusion.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"HD Figure saved to {out_path}")
