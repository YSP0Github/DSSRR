"""图5：E-RealInj 六方法核心统计保真指标对比（2x2，覆盖论文定义的四项指标）

(a) Wasserstein-1     振幅分布保真
(b) Envelope distance 包络/波动保真
(c) Spectral entropy  谱复杂度保真
(d) ACF L2            时间相关结构保真（唯一由基线领先的指标，如实呈现）

子集：stationary x dropout（与正文 Figure 5 口径一致）。
数值全部来自 v6 落盘的 stats_summary.csv，未做任何改动。

误差棒（2026-10-07，回应审稿意见 M2）：
  柱高 = 18 个注入单元的均值；误差棒 = **该均值的 bootstrap 95% 置信区间**
  （百分位法，20000 次重采样，seed=7）。柱间比较为**配对**比较（同一单元被
  六种方法各修一次），配对检验结果见 experiments/results_stats/paired_v6_stat_dropout.csv。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import (apply_srl_style, DOUBLE_COL_WIDTH, COLOR_LIST, METHOD_ORDER,
                       PANEL_LABEL_KW, SAVE_KW, mlabel)

apply_srl_style()

CSV_PATH = os.environ.get(
    'E2R_CSV',
    r'G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6\stats_summary.csv')
FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'
SEED = 7
N_BOOT = 20000


def bootstrap_ci_mean(x, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """百分位法 bootstrap 95% CI（对均值）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n_boot, len(x)))
    mus = x[idx].mean(axis=1)
    lo, hi = np.percentile(mus, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


df = pd.read_csv(CSV_PATH)
df_dropout = df[(df['anom_type'] == 'dropout') & (df['scenario'] == 'stationary')]
summary = df_dropout.groupby('method')[['wasserstein', 'envelope_dist',
                                        'spec_entropy_err', 'acf_l2']].mean()

PANELS = [
    ('wasserstein',      '(a) Amplitude distribution fidelity', 'Wasserstein distance'),
    ('envelope_dist',    '(b) Fluctuation (envelope) fidelity', 'Envelope distance'),
    ('spec_entropy_err', '(c) Spectral complexity fidelity',    'Spectral entropy error'),
    ('acf_l2',           '(d) Temporal correlation fidelity',   r'ACF $L_2$ distance'),
]

fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.35))
bar_width = 0.68
x = np.arange(len(METHOD_ORDER))

for ax, (col, title, ylab) in zip(axes.ravel(), PANELS):
    vals, errs = [], []
    for m in METHOD_ORDER:
        v = df_dropout.loc[df_dropout['method'] == m, col].to_numpy(dtype=float)
        vals.append(np.nanmean(v))
        lo, hi = bootstrap_ci_mean(v)
        errs.append([np.nanmean(v) - lo, hi - np.nanmean(v)])
    errs = np.array(errs).T                      # (2, n_methods)
    bars = ax.bar(x, vals, color=COLOR_LIST, width=bar_width, edgecolor='none')
    ax.errorbar(x, vals, yerr=errs, fmt='none', ecolor='0.25',
                elinewidth=0.8, capsize=2.0, capthick=0.8, zorder=5)
    ax.set_ylabel(ylab + '  (lower = better)', fontsize=8)
    ax.set_title(title, **PANEL_LABEL_KW)
    ax.set_xticks(x)
    ax.set_xticklabels([mlabel(m) for m in METHOD_ORDER], rotation=30, ha='right')
    top = max(vals)
    # 留白要同时容纳误差棒上端与数值标签，否则标签会压到子图标题上
    ymax = max(v + e[1] for v, e in zip(vals, errs.T)) * 1.16
    ax.set_ylim(0, ymax)
    ax.tick_params(axis='x', pad=2)
    fmt = '{:.3f}' if top < 0.5 else '{:.2f}'
    for bar, val, err in zip(bars, vals, errs.T):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + err[1] + ymax * 0.018,
                fmt.format(val), ha='center', va='bottom', fontsize=7)

plt.tight_layout(w_pad=1.6, h_pad=1.5)
out_path = f'{FIG_DIR}/fig03_method_comparison.png'
plt.savefig(out_path, **SAVE_KW)
print(f"HD Figure saved to {out_path}")
plt.close()
