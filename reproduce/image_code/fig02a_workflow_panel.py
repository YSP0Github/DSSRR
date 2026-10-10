"""图3：DSSRR方法流程图（JGR标准高清版）
所有流程结构/文字完全保留，紧凑适配双栏印刷尺寸，1200dpi
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import apply_srl_style, DOUBLE_COL_WIDTH, SAVE_KW

# ⚠️ 必须调用：本面板与 fig02bc 拼成同一张 Figure 2，若此处漏掉 apply_srl_style()，
# 面板 (a) 会用 matplotlib 默认的 DejaVu Sans，而 (b)/(c) 用 Arial。
# DejaVu Sans Bold 比 Arial Bold 明显更宽更重，同为 fontsize=9 时 (a) 的标题看上去
# 比 (b)(c) 大一圈（实测 "D" 宽/高比 0.95 vs 0.79），2026-10-09 修。
apply_srl_style()

FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'
PANEL_DIR = os.path.join(FIG_DIR, '_panels')   # 合并用中间面板（非最终图）
os.makedirs(PANEL_DIR, exist_ok=True)

# 标准双栏尺寸，等比缩放原有坐标系
fig_w = DOUBLE_COL_WIDTH
fig_h = 2.55
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title('(a) DSSRR processing workflow', fontsize=9, fontweight='bold', loc='left', pad=4)

# 配色和原版本一致
c_input = '#1E88E5'
c_process = '#27ae60'
c_output = '#D62728'

steps = [
    {'x': 0.15, 'y': 1.7, 'w': 1.75, 'h': 1.5, 'color': c_input,
     'title': 'Input\nWaveform', 'subtitle': 'Raw seismogram'},
    {'x': 2.25, 'y': 1.7, 'w': 1.75, 'h': 1.5, 'color': c_process,
     'title': 'Anomaly\nDetection', 'subtitle': 'Identify $[s,e]$\nSafety gap $G$'},
    {'x': 4.35, 'y': 1.7, 'w': 1.75, 'h': 1.5, 'color': c_process,
     'title': 'Two-sided\nReference', 'subtitle': 'Extract $R_b,R_a$\nHealthy windows'},
    {'x': 6.45, 'y': 1.7, 'w': 1.75, 'h': 1.5, 'color': c_process,
     'title': 'PSD\nFusion', 'subtitle': 'Welch PSD\nLog-domain merge'},
    {'x': 8.55, 'y': 1.7, 'w': 1.75, 'h': 1.5, 'color': c_process,
     'title': 'Phase Template\nSynthesis', 'subtitle': 'Healthy phase\nsampling'},
    {'x': 10.65, 'y': 1.7, 'w': 1.75, 'h': 1.5, 'color': c_process,
     'title': 'Amplitude\nAlignment', 'subtitle': 'Statistical match\n+ seam blending'},
    {'x': 12.75, 'y': 1.7, 'w': 1.1, 'h': 1.5, 'color': c_output,
     'title': 'Output\nRepaired', 'subtitle': 'Statistically\nfaithful'},
]

for i, step in enumerate(steps):
    box = FancyBboxPatch(
        (step['x'], step['y']), step['w'], step['h'],
        boxstyle="round,pad=0.07",
        facecolor=step['color'], alpha=0.14,
        edgecolor=step['color'], linewidth=1.1, zorder=2
    )
    ax.add_patch(box)
    ax.text(step['x']+step['w']/2, step['y']+step['h']*0.68,
            step['title'], ha='center', va='center',
            fontsize=7.8, fontweight='bold', color=step['color'], zorder=3)
    ax.text(step['x']+step['w']/2, step['y']+step['h']*0.25,
            step['subtitle'], ha='center', va='center',
            fontsize=6.8, color='#444444', zorder=3)
    if i < len(steps)-1:
        nxt = steps[i+1]
        arrow = FancyArrowPatch(
            (step['x']+step['w'], step['y']+step['h']/2),
            (nxt['x'], nxt['y']+nxt['h']/2),
            arrowstyle='->,head_width=0.3,head_length=0.16',
            color='#555555', linewidth=1.1, zorder=1
        )
        ax.add_patch(arrow)

# 底部三个特性（紧凑）
feat_y = 0.55
features = [
    (2.3, 'Training-free\nNo labeled data'),
    (7, '$O(N\\log N)$ complexity\nSingle-pass fast'),
    (11.7, 'Statistical fidelity\nPSD, ACF, envelope'),
]
for x, text in features:
    box = FancyBboxPatch(
        (x-1.45, feat_y-0.32), 2.9, 0.65,
        boxstyle="round,pad=0.07", facecolor='#ecf0f1',
        edgecolor='#bdc3c7', linewidth=0.8, zorder=2
    )
    ax.add_patch(box)
    ax.text(x, feat_y, text, ha='center', va='center',
            fontsize=7, color='#333333')

plt.tight_layout(pad=0.1)
out_path = f'{PANEL_DIR}/fig02a_workflow.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"HD Figure saved to {out_path}")
