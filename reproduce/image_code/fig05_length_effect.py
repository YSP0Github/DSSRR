"""图7：缺口长度扫描（三联图）

(a) Wasserstein distance  vs 缺口长度  —— 分布保真随长度的演化
(b) ACF L2 distance       vs 缺口长度  —— 时间结构保真随长度的演化
(c) std(repaired)/std(truth) vs 缺口长度 —— 能量保真

(c) 面板把正文中"FFT 是形状正确但能量不足的填补、插值类则近乎丢失全部能量"
这段论证直接可视化。

数据源唯一：``results_length_scan/stats_summary.csv``（由 run_length_scan.py 落盘，
含 wasserstein / acf_l2 / std_ratio / length_min）。逐长度对 4 个事件做截尾平均
（去掉最好与最差），与图注口径一致。

  误差棒（2026-10-07，回应审稿意见 M2）：
  每个长度只有 **4 条背景记录**。极差（min--max）会被单条离群记录撑爆
  （Wasserstein 原始极差最大到 6.3，而截尾均值只在 0.05--2.0），故改用
  **±1 标准差（n = 4 条记录）**——这是 n=4 下既不失效又可读的离散度口径。

  **水平错开（2026-10-07 二次修改）**：19 个长度 × 4 条方法 = 76 根误差棒，
  同一 gap 上 4 根棒会互相压住、无法判断归属。故给每条方法一个固定的 x 比例
  因子（见 ``OFFSETS``），把整条曲线（含 marker 与误差棒）沿 x 方向平移开。
  在对数轴上 ``x -> k*x`` 是**固定像素平移**，因此曲线只会平行移开、不会倾斜；
  最大偏移 ±3.5% 远小于相邻长度之间 22% 的间隔，读者仍会读成同一个 gap。
  误差棒用**方法色淡染**（alpha 0.42、线宽 0.5）而非统一灰色：颜色使棒与其
  曲线同属一目了然，淡染使其不抢曲线的视觉权重。
  另：三个面板 y 轴下限均为 0，少数误差棒下沿本应伸至负值（(a) 上 DSSRR
  Wasserstein 最深 −1.76、Linear/SSA −0.43），这些棒在 0 处被裁切，图注已注明。

  配对检验（80 个 (记录,长度) 单元、4/4 记录方向一致、p ≤ 5e-6）见
  ``experiments/results_stats/paired_length_scan.csv``；逐长度完整离散度
  （median / IQR / SD / CI）见 ``results_stats/dispersion_length_scan.csv``，
  其 SD 列已整理为补充材料 Table S2--S3。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import apply_srl_style, DOUBLE_COL_WIDTH, METHOD_COLORS, PANEL_LABEL_KW, SAVE_KW

apply_srl_style()

FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'
SUMMARY_CSV = os.environ.get(
    'LENGTH_SUMMARY',
    r'G:\SeisY\docs\dssrr_paper\experiments\results_length_scan\stats_summary.csv')

# 剔除 12.14 min 采样假象点（该长度下缺口两端落在同一量化平台，插值类退化为常数填补，
# 而按 lag-0 归一化的 ACF 指标反而给这种彻底失败打高分）。原始数据完整保留在 CSV 中。
EXCLUDE_LENGTHS = [12.137540]

NAMES = ['DSSRR', 'Linear', 'FFT', 'SSA']
colors = {n: METHOD_COLORS[n] for n in NAMES}
markers = {'DSSRR': '*', 'Linear': 'o', 'FFT': '^', 'SSA': 's'}
# 缩小 marker：星形 DSSRR 视觉面积大，需要比其它 marker 设得稍大但显著小于原稿；
# 其它 marker 从 5–5.5 → 4.0–4.4。白色描边让点从浅灰误差棒里"浮"出来。
marker_sizes = {'DSSRR': 5.5, 'Linear': 4.0, 'FFT': 4.4, 'SSA': 3.8}
MARKER_EDGE = dict(color='white', width=0.5)

# 水平错开因子：对数轴上 x -> k*x 是固定像素平移（曲线不倾斜）。
# 幅度控制在 ±3.5% 以内 —— 相邻长度间隔约 22%，因此不会被误读成不同 gap。
OFFSETS = {'DSSRR': 1.035, 'Linear': 1.012, 'FFT': 0.988, 'SSA': 0.965}
# 误差棒：方法色淡染 + 细线 + 小 cap
ERRBAR_KW = dict(alpha=0.42, elinewidth=0.5, capsize=0.9, capthick=0.4)


def _trimmed(vals):
    v = np.sort(np.asarray(vals, dtype=float))
    return float(np.mean(v[1:-1])) if len(v) >= 4 else float(np.mean(v))


df = pd.read_csv(SUMMARY_CSV)
df = df[df['method'].isin(NAMES)].copy()
mask = np.zeros(len(df), dtype=bool)
for L in EXCLUDE_LENGTHS:
    mask |= np.isclose(df['length_min'].to_numpy(), L, atol=1e-6)
df = df[~mask]

results = {}
for n in NAMES:
    sub = df[df['method'] == n].groupby('length_min').agg(
        w=('wasserstein', _trimmed), a=('acf_l2', _trimmed), r=('std_ratio', _trimmed),
        w_sd=('wasserstein', 'std'), a_sd=('acf_l2', 'std'), r_sd=('std_ratio', 'std'),
    ).reset_index()
    results[n] = {'lens': sub['length_min'].values}
    for k in ('w', 'a', 'r'):
        results[n][k] = sub[k].values
        results[n][k + '_sd'] = sub[k + '_sd'].values

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.8))
xticks_major = [2, 5, 10, 20, 30, 60, 90]


def _style(ax, title, ylab):
    ax.set_xscale('log')
    ax.set_xlabel('Gap length (min)')
    ax.set_ylabel(ylab)
    ax.set_title(title, **PANEL_LABEL_KW)
    ax.xaxis.set_major_locator(FixedLocator(xticks_major))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(xticks_major)


def _plot(ax, key):
    for n in NAMES:
        L, v = results[n]['lens'], results[n][key]
        sd = results[n][key + '_sd']
        Lx = L * OFFSETS[n]          # 水平错开，避免同一 gap 上 4 根误差棒互相压住
        # 误差棒隔点绘制，进一步降低密度；颜色取方法色但淡染（alpha 0.42）
        sl = slice(None, None, 2)
        ax.errorbar(Lx[sl], v[sl], yerr=sd[sl],
                    fmt='none', ecolor=colors[n], zorder=1, **ERRBAR_KW)
        ax.plot(Lx, v, marker=markers[n], color=colors[n], linewidth=1.2,
                markersize=marker_sizes[n], markevery=2, label=n, zorder=3,
                markeredgecolor=MARKER_EDGE['color'],
                markeredgewidth=MARKER_EDGE['width'])


# ---- (a) Wasserstein ----
ax = axes[0]
_plot(ax, 'w')
_style(ax, '(a) Amplitude distribution fidelity', 'Wasserstein distance (lower = better)')
ymax_w = max(float(np.nanmax(results[n]['w'] + results[n]['w_sd'])) for n in NAMES)
ax.set_ylim(0.0, ymax_w * 1.04)
ax.legend(loc='upper left', handletextpad=0.3, borderaxespad=0.1, fontsize=6.5)
n_win = int(np.sum(np.all(
    [results['DSSRR']['w'] <= results[n]['w'] + 1e-12 for n in NAMES[1:]], axis=0)))
# 图内自注已删（审稿意见「次要 7」）："DSSRR lowest at 19/19 lengths" 改写入图注。
print(f"[check] DSSRR lowest Wasserstein at {n_win}/{len(results['DSSRR']['w'])} lengths")

# ---- (b) ACF L2 ----
ax = axes[1]
_plot(ax, 'a')
_style(ax, '(b) Temporal correlation fidelity', r'ACF $L_2$ distance (lower = better)')
ax.legend(loc='upper left', handletextpad=0.3, borderaxespad=0.1, fontsize=6.5)

# ---- (c) Energy ratio ----
ax = axes[2]
_plot(ax, 'r')
_style(ax, '(c) Energy fidelity', r'std(repaired) / std(truth)')
ax.axhline(1.0, color='0.35', lw=0.9, ls='--', zorder=0)
ymax_r = max(float(np.nanmax(results[n]['r'] + results[n]['r_sd'])) for n in NAMES)
ax.set_ylim(0.0, ymax_r * 1.04)
# 图内 "truth level" 自注已删：水平错开后它会压住 DSSRR 的数据点，
# 且 unity 的含义已写进图注（"dashed line at unity is an exact energy match"）。

plt.tight_layout(w_pad=1.1)
out_path = f'{FIG_DIR}/fig05_length_effect.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"HD Figure saved to {out_path}")
print('DSSRR lowest on Wasserstein at %d/%d lengths' % (n_win, len(results['DSSRR']['lens'])))

print('\n--- trimmed means per length ---')
tbl = pd.DataFrame({n: results[n]['w'] for n in NAMES},
                   index=[f'{L:.2f}' for L in results['DSSRR']['lens']])
print('Wasserstein:\n' + tbl.round(3).to_string())
tbl = pd.DataFrame({n: results[n]['r'] for n in NAMES},
                   index=[f'{L:.2f}' for L in results['DSSRR']['lens']])
print('\nstd_ratio:\n' + tbl.round(3).to_string())
