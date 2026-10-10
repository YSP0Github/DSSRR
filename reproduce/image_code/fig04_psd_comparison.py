"""图6：修复段三视角对比（PSD / 振幅直方图 / 归一化自相关）

把"DSSRR 保的是统计分布"这一核心主张变成可直接看见的证据：
(a) Welch PSD            —— 谱形保真
(b) 振幅直方图           —— 分布保真（论文的核心卖点）
(c) 归一化自相关函数      —— 时间结构保真

数据源：v6 落盘的 corrupted / repaired_<Method>.mseed 与 clean_data_12h 真值。
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import welch
from matplotlib.ticker import FixedLocator
import sys
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import apply_srl_style, DOUBLE_COL_WIDTH, METHOD_COLORS, SAVE_KW, mlabel

apply_srl_style()

CASE_DIR = os.environ.get(
    'E2R_CASE',
    r'G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6\S12_19760113_070152_stationary\dropout\medium')
CLEAN_FILE = r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h\XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed'
FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'

plot_methods = [
    ("Ground truth", None, METHOD_COLORS["GroundTruth"], 1.2, "--"),
    ("DSSRR", "repaired_DSSRR.mseed", METHOD_COLORS["DSSRR"], 1.3, "-"),
    ("FFT", "repaired_FFT.mseed", METHOD_COLORS["FFT"], 1.1, "-"),
    ("UNet", "repaired_UNet.mseed", METHOD_COLORS["UNet"], 1.0, "-"),
    ("SSA", "repaired_SSA.mseed", METHOD_COLORS["SSA"], 1.0, "-"),
]
# 基线中 SSA / UNet 在 ACF 面板振荡剧烈，降低其视觉权重以免遮挡主曲线
ALPHA = {"Ground truth": 1.0, "DSSRR": 1.0, "FFT": 0.95, "UNet": 0.6, "SSA": 0.6}

st_clean = read(CLEAN_FILE)
data_clean = st_clean[0].data.astype(float)
st_corrupt = read(os.path.join(CASE_DIR, 'corrupted.mseed'))
data_corrupt = st_corrupt[0].data.astype(float)
sr = st_corrupt[0].stats.sampling_rate

diff = np.abs(data_corrupt - data_clean[:len(data_corrupt)])
anom_mask = diff > 0.1 * np.std(data_corrupt)
anom_s = int(np.where(anom_mask)[0][0])
anom_e = int(np.where(anom_mask)[0][-1])

segments = {}
for name, fname, color, lw, ls in plot_methods:
    if fname is None:
        segments[name] = data_clean[anom_s:anom_e]
    else:
        segments[name] = read(os.path.join(CASE_DIR, fname))[0].data.astype(float)[anom_s:anom_e]


def norm_acf(x, nlags):
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[len(x) - 1:]
    r = r / (r[0] + 1e-12)
    return r[:nlags]


fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.75))

# ---------------- (a) PSD ----------------
ax = axes[0]
nperseg = 1024
for name, fname, color, lw, ls in plot_methods:
    freqs, psd = welch(segments[name], fs=sr, nperseg=nperseg)
    ax.semilogx(freqs[1:], 10 * np.log10(psd[1:]), color=color, lw=lw, ls=ls,
                alpha=ALPHA[name], zorder=(5 if name in ('Ground truth', 'DSSRR') else 3),
                label=mlabel(name))
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (dB/Hz)')
ax.set_title('(a) Power spectral density', fontsize=9, fontweight='bold', loc='left', pad=4)
ax.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0]))
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.set_xlim(0.005, sr / 2)
ax.legend(loc='lower left', handletextpad=0.4, borderaxespad=0.2, fontsize=6.5)

# ---------------- (b) Amplitude distribution (ECDF) ----------------
# ECDF 直接可视化 Wasserstein-1：两曲线之间的面积即该距离。
ax = axes[1]
truth = segments["Ground truth"]
c0, sd = truth.mean(), truth.std()
for name, fname, color, lw, ls in plot_methods:
    seg = np.sort(segments[name])
    ecdf = np.arange(1, len(seg) + 1) / len(seg)
    ax.plot(seg, ecdf, color=color, lw=lw, ls=ls,
            alpha=ALPHA[name], zorder=(5 if name in ('Ground truth', 'DSSRR') else 3),
            label=mlabel(name))
ax.set_xlabel('Amplitude (DU)')
ax.set_ylabel('Cumulative probability')
ax.set_title('(b) Amplitude distribution', fontsize=9, fontweight='bold', loc='left', pad=4)
ax.set_xlim(c0 - 4 * sd, c0 + 4 * sd)
ax.set_ylim(0, 1)

# ---------------- (c) Normalized ACF ----------------
# 读法说明（2026-10-09 补充）：每条曲线都用自己的零延迟值归一化，所以本面板只比较
# **衰减形状**，不比较幅度（幅度归 (b) 面板）。DSSRR 与 FFT 沿真值的衰减走；
# SSA 与 U-Net 的 ACF 在零附近剧烈振荡——它们的填充与周围背景没有共同的相位/相关结构，
# 归一化后剩下的就是噪声。两条振荡曲线若与主曲线同权重会把主曲线完全盖住，
# 故在 (c) 面板单独再降一档透明度与线宽，使其退为背景信息。
ax = axes[2]
nlags = 130
lags = np.arange(nlags) / sr
ALPHA_C = {"Ground truth": 1.0, "DSSRR": 1.0, "FFT": 0.95, "UNet": 0.35, "SSA": 0.35}
LW_C = {"UNet": 0.7, "SSA": 0.7}
for name, fname, color, lw, ls in plot_methods:
    ax.plot(lags, norm_acf(segments[name], nlags), color=color,
            lw=LW_C.get(name, lw), ls=ls, alpha=ALPHA_C[name],
            zorder=(5 if name in ('Ground truth', 'DSSRR') else 3), label=mlabel(name))
ax.set_xlabel('Lag (s)')
ax.set_ylabel(r'Normalized ACF')
ax.set_title('(c) Autocorrelation', fontsize=9, fontweight='bold', loc='left', pad=4)
ax.axhline(0.0, color='0.7', lw=0.6, zorder=0)
# 指出"看不懂"的那部分：两条基线为何在零附近来回振荡。
# 放在右上角（该区域无曲线：lag>10 时 DSSRR ≈ 0.3、真值 ≈ -0.1），
# 不用箭头，避免引线横穿 DSSRR 主曲线。
ax.text(0.97, 0.97, 'SSA and 1D U-Net (pale):\noscillate about zero',
        transform=ax.transAxes, fontsize=5.8, color='0.3',
        ha='right', va='top')

plt.tight_layout(w_pad=1.3)
out_path = f'{FIG_DIR}/fig04_psd_comparison.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"HD Figure saved to {out_path}")
