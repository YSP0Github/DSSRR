"""图8：DSSRR组件消融实验（JGR标准高清版）
所有原始统计值不变，修正文件名、去掉网格、统一样式、1200dpi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import apply_srl_style, DOUBLE_COL_WIDTH, METHOD_COLORS, PANEL_LABEL_KW, SAVE_KW

apply_srl_style()

OUT_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\results_ablation_v3'
FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'
PANEL_DIR = os.path.join(FIG_DIR, '_panels')   # 合并用中间面板（非最终图）
os.makedirs(PANEL_DIR, exist_ok=True)

df = pd.read_csv(f'{OUT_DIR}/ablation_v3_results.csv')
summary = df.groupby('config')[['wasserstein', 'acf_l2']].mean()

# 纵轴用等宽短码（\u2212 = 被消融的组件），完整定义见论文图注。
# 长描述标签会撑宽左侧留白、行宽参差，视觉上不整齐；短码保证各行等长。
MINUS = '\u2212'
config_labels = {
    'full': 'Full',
    'one_sided': MINUS + 'ref',      # 双侧参考 -> 仅用左侧单侧参考
    'no_trend_align': MINUS + 'trend',   # 去掉趋势对齐
    'linear_fusion': MINUS + 'log',      # 去掉对数域融合（退化为线性域平均）
    'no_seam_blend': MINUS + 'seam',     # 去掉接缝混合
    'no_phase_template': MINUS + 'phase',  # 去掉相位模板
}
plot_order = ['full', 'one_sided', 'no_trend_align', 'linear_fusion', 'no_seam_blend', 'no_phase_template']
labels = [config_labels[c] for c in plot_order]
y_pos = np.arange(len(labels))
bar_h = 0.62
# 配色语义：Full = 深绯红（唯一深色，锚定 DSSRR 主色）；-ref = 中间调 + 深色描边
# （它是被消融项里贡献最大的那一个）；其余四个被消融项 = 浅绯红。
#
# 为什么不用"换色相"来高亮 -ref：原方案给 -ref 单独用 #c0392b，它在 RGB 空间里
# 离 Full 只有 ΔRGB≈29、离其他被消融柱却有 ΔRGB≈179 —— 视觉上被归到 "Full 那一类"，
# 与"它本身也是一个被消融项"的语义正好相反。
# 改成同色系三档明度后：Full↔-ref ΔRGB≈108、-ref↔其余 ΔRGB≈56、Full↔其余 ΔRGB≈164，
# -ref 明确落在"消融组"内，同时靠明度与描边保持突出。
ablation_color = "#EF9A9A"   # 被消融项统一色（浅绯红）
REF_COLOR = "#E57373"        # -ref 专属（同色系中间调）
HILITE_EDGE = "#7f1d1d"      # -ref 描边
BAR_COLORS = [METHOD_COLORS["DSSRR"], REF_COLOR] + [ablation_color] * 4
BAR_ALPHAS = [1.0] * 6       # 不再用透明度区分，改用明度（打印/灰度下同样可辨）

# 标准双栏尺寸
fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.2))

# (a) Wasserstein
ax = axes[0]
vals_wd = [summary.loc[c, 'wasserstein'] for c in plot_order]
bars = ax.barh(y_pos, vals_wd, color=BAR_COLORS, height=bar_h, edgecolor='none')
for i, b in enumerate(bars):
    if i == 1:                      # -ref：同色系中间调 + 加粗深色描边
        b.set_edgecolor(HILITE_EDGE)
        b.set_linewidth(1.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Wasserstein distance (lower = better)')
ax.set_title('(a) Amplitude distribution fidelity', **PANEL_LABEL_KW)
ax.axvline(x=vals_wd[0], color=METHOD_COLORS["DSSRR"], lw=1.0, ls='--', alpha=0.7)
ax.set_xlim(0, max(vals_wd)*1.22)
# 反转y轴让full在最上方
ax.invert_yaxis()
# 高亮标注：双侧参考是最大单点贡献
ax.annotate('dual-sided reference:\nremoving it degrades $W$ by $5.2\\times$',
            xy=(vals_wd[1], y_pos[1]), xytext=(max(vals_wd)*0.42, y_pos[1] - 1.15),
            fontsize=6.5, color=HILITE_EDGE, ha='left', va='center',
            arrowprops=dict(arrowstyle='-', color=HILITE_EDGE, lw=0.8))

def _pct(deg):
    """整数百分比；若取整后会变成 0 而实际非零（如 +0.4%），保留一位小数。

    否则 -seam 那根柱会被标成 "+0%"，与正文的 "~0.4% Wasserstein change" 冲突，
    读者会误读为"接缝混合对分布保真毫无影响"。
    """
    return f'{deg:.0f}' if (round(deg) != 0 or deg == 0) else f'{deg:.1f}'


for i, (bar, val) in enumerate(zip(bars, vals_wd)):
    if i == 0:
        text = f'{val:.3f}'
    else:
        deg = (val - vals_wd[0])/vals_wd[0]*100
        if deg < 0:
            text = f'{val:.3f} (-{_pct(abs(deg))}%)'
        else:
            text = f'{val:.3f} (+{_pct(deg)}%)'
    ax.text(val + max(vals_wd)*0.02, bar.get_y()+bar.get_height()/2,
            text, va='center', fontsize=7)

# (b) ACF L2
ax = axes[1]
vals_acf = [summary.loc[c, 'acf_l2'] for c in plot_order]
bars = ax.barh(y_pos, vals_acf, color=BAR_COLORS, height=bar_h, edgecolor='none')
for i, b in enumerate(bars):
    if i == 1:
        b.set_edgecolor(HILITE_EDGE)
        b.set_linewidth(1.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel(r'Normalized ACF $L_2$ distance (lower = better)')
ax.set_title('(b) Temporal correlation fidelity', **PANEL_LABEL_KW)
ax.axvline(x=vals_acf[0], color=METHOD_COLORS["DSSRR"], lw=1.0, ls='--', alpha=0.7)
ax.set_xlim(0, max(vals_acf)*1.28)
ax.invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, vals_acf)):
    if i == 0:
        text = f'{val:.3f}'
    else:
        deg = (val - vals_acf[0])/vals_acf[0]*100
        if deg < 0:
            text = f'{val:.3f} (-{_pct(abs(deg))}%)'
        else:
            text = f'{val:.3f} (+{_pct(deg)}%)'
    ax.text(val + max(vals_acf)*0.02, bar.get_y()+bar.get_height()/2,
            text, va='center', fontsize=7)

# 整图共用一个配色图例：明确 "Full 基准 / 被消融项 / 最大贡献项" 三类的区别，
# 避免读者把 -ref 的强调色误读成"与 Full 同类"。
#
# 图例位置：2026-10-09 由"整图底部"改为"整图顶部"。原因有二：
#   1) 本面板要与 fig07cd 上下拼成 Figure 7，图例放在面板底部时会落到拼图的
#      正中间（(a)(b) 与 (c)(d) 之间），读起来像 (c)(d) 的图例；
#   2) 与 Figure 8、Figure 10 的"整图顶部横向图例"保持一致。
# 顶部预留带高度 0.095×3.2 in ≈ 0.30 in，与 fig10（0.12×2.75 in）相当。
_leg = [Patch(facecolor=METHOD_COLORS['DSSRR'], edgecolor='none',
              label='Full model (baseline)'),
        Patch(facecolor=ablation_color, edgecolor='none',
              label='Ablated variant'),
        Patch(facecolor=REF_COLOR, edgecolor=HILITE_EDGE, lw=1.6,
              label='Largest single contributor')]
fig.legend(handles=_leg, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
           fontsize=6.4, frameon=False, handlelength=1.5, columnspacing=1.6)

plt.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.5)
out_path = f'{PANEL_DIR}/fig07ab_ablation.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"HD Figure saved to {out_path}")
