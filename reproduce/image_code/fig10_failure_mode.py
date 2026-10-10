"""新图 A：smooth-bridge 失败模式诊断（波形 / 振幅分布 / 自相关）

用同一个 60-min dropout 案例把论文的核心范式主张一次说清：
插值修复在波形上看起来"平滑合理"，但振幅分布塌缩、时间结构被伪造。

数据源：v6 落盘的 repaired_Linear / repaired_DSSRR.mseed 与 clean_data_12h 真值。
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import read
from scipy.stats import wasserstein_distance
import sys
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import (apply_srl_style, DOUBLE_COL_WIDTH, METHOD_COLORS,
                       PANEL_LABEL_KW, SAVE_KW)

apply_srl_style()

CASE_DIR = os.environ.get(
    'FAILCASE',
    r'G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6\S12_19760113_070152_stationary\dropout\large')
CLEAN_FILE = r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h\XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed'
FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'

clean = read(CLEAN_FILE)[0].data.astype(float)
corr = read(os.path.join(CASE_DIR, 'corrupted.mseed'))[0].data.astype(float)
lin = read(os.path.join(CASE_DIR, 'repaired_Linear.mseed'))[0].data.astype(float)
dss = read(os.path.join(CASE_DIR, 'repaired_DSSRR.mseed'))[0].data.astype(float)
sr = 6.625

diff = np.abs(corr - clean[:len(corr)])
m = diff > 1e-3 * np.std(clean[:len(corr)])
s, e = int(np.where(m)[0][0]), int(np.where(m)[0][-1])
truth = clean[s:e + 1]
lin_seg = lin[s:e + 1]
dss_seg = dss[s:e + 1]

w_lin = wasserstein_distance(truth, lin_seg)
w_dss = wasserstein_distance(truth, dss_seg)
r_lin = lin_seg.std() / truth.std()
r_dss = dss_seg.std() / truth.std()
print(f'case span={e - s + 1} samples ({(e - s + 1) / sr / 60:.1f} min)')
print(f'  Linear  W={w_lin:.3f}  std_ratio={r_lin:.3f}')
print(f'  DSSRR   W={w_dss:.3f}  std_ratio={r_dss:.3f}')


def norm_acf(x, nlags):
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[len(x) - 1:]
    r = r / (r[0] + 1e-12)
    return r[:nlags]


SERIES = [
    ('Ground truth', truth, METHOD_COLORS['GroundTruth'], 1.1, '--'),
    ('Linear', lin_seg, METHOD_COLORS['Linear'], 1.1, '-'),
    ('DSSRR', dss_seg, METHOD_COLORS['DSSRR'], 1.3, '-'),
]

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.75))

# (a) waveform —— 异常段内部的 5-min 放大窗口：平坦桥 vs 真实噪声纹理
ax = axes[0]
zc = (s + e) // 2
z0, z1 = zc - int(150 * sr), zc + int(150 * sr)     # 300 s = 5 min
base = float(np.median(truth))
t = (np.arange(z0, z1) - s) / sr / 60.0
YLIM = 6.0
# 三条曲线一律使用全篇统一方法色号（Linear = 浅灰），避免同一方法在本图 (a) 是红色、
# 在 (b)(c) 是灰色这种自相矛盾。
ax.plot(t, clean[z0:z1] - base, color=METHOD_COLORS['GroundTruth'], lw=1.1, ls='--',
        label='Ground truth', zorder=3)
ax.plot(t, lin[z0:z1] - base, color=METHOD_COLORS['Linear'], lw=1.6, label='Linear', zorder=2)
ax.plot(t, dss[z0:z1] - base, color=METHOD_COLORS['DSSRR'], lw=1.3, label='DSSRR', zorder=4)
ax.set_ylim(-YLIM, YLIM)
ax.set_xlim(t[0], t[-1])
ax.set_xlabel('Time relative to anomaly start (min)')
ax.set_ylabel('Amplitude (DU, median-removed)')
ax.set_title('(a) Waveform (5-min zoom)', **PANEL_LABEL_KW)
# 注意：此处不再在坐标轴内部放图例——三条曲线几乎铺满整幅，(a) 内的图例必然压住波形。
# 改为整图共用一个顶部横向图例（三个面板的序列完全相同）。

# (b) amplitude distribution
ax = axes[1]
c0, sd = truth.mean(), truth.std()
for name, seg, color, lw, ls in SERIES:
    xs = np.sort(seg)
    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=color, lw=lw, ls=ls, label=name)
ax.set_xlim(c0 - 4 * sd, c0 + 4 * sd)
ax.set_ylim(0, 1)
ax.set_xlabel('Amplitude (DU)')
ax.set_ylabel('Cumulative probability')
ax.set_title('(b) Amplitude distribution', **PANEL_LABEL_KW)

# (c) autocorrelation
# 读法说明（2026-10-09 补充）：每条曲线用自身零延迟值归一化。线性插值把整个 60-min
# 缺口填成一条直线，去均值后是线性斜坡，其归一化自相关在整个滞后范围内都≈1
# （实测 lag1 = 0.9999、lag199 = 0.9750），即"任何两个样本都完全相关"——
# 这正是 smooth-bridge 失败模式在时间结构上的表现，也是本面板要展示的核心。
ax = axes[2]
nlags = 200
lags = np.arange(nlags) / sr
for name, seg, color, lw, ls in SERIES:
    ax.plot(lags, norm_acf(seg, nlags), color=color, lw=lw, ls=ls, label=name)
# 零参考线：细点线 + 更浅的灰，与 Linear 的灰色实线明确区分
ax.axhline(0.0, color='0.7', lw=0.6, ls=':', zorder=0)
ax.annotate('Linear fill: ACF $\\approx$ 1 at all lags\n(degenerate correlation structure)',
            xy=(14.0, 0.99), xytext=(1.0, 0.60), fontsize=5.8, color='0.25',
            ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color='0.45', lw=0.6))
ax.set_xlabel('Lag (s)')
ax.set_ylabel('Normalized ACF')
ax.set_title('(c) Autocorrelation', **PANEL_LABEL_KW)

fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.3)
handles, labs = axes[0].get_legend_handles_labels()
fig.legend(handles, labs, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
           fontsize=6.8, frameon=False, handlelength=2.2, columnspacing=1.8)
out_path = f'{FIG_DIR}/fig10_failure_mode.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"HD Figure saved to {out_path}")
