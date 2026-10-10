"""图4：DSSRR修复原理示意图（JGR标准高清版）
所有模拟数据/标注/扰动完全保留，仅适配印刷尺寸、统一字号、1200dpi
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import apply_srl_style, DOUBLE_COL_WIDTH, SAVE_KW

apply_srl_style()

FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'
PANEL_DIR = os.path.join(FIG_DIR, '_panels')   # 合并用中间面板（非最终图）
os.makedirs(PANEL_DIR, exist_ok=True)

# 数据生成逻辑和原版本100%一致，固定随机种子
np.random.seed(42)
fs = 100
duration = 24
t = np.arange(0, duration, 1/fs)
freqs = [0.3, 0.8, 1.5, 2.5, 4.0, 6.0, 8.0]
amps = [2.5, 2.0, 1.2, 0.8, 0.5, 0.3, 0.2]
bg = np.zeros_like(t)
for f, a in zip(freqs, amps):
    bg += a * np.sin(2*np.pi*f*t + np.random.uniform(0, 2*np.pi))
bg += 0.4 * np.random.randn(len(t))

anom_s, anom_e = 9.0, 15.0
anom_mask = (t >= anom_s) & (t <= anom_e)
gap = 0.6
lref_s, lref_e = anom_s - gap -4.0, anom_s - gap
rref_s, rref_e = anom_e + gap, anom_e + gap +4.0

before = bg.copy()
before[anom_mask] = 0
after = bg.copy()
lref = bg[(t>=lref_s)&(t<=lref_e)]
rref = bg[(t>=rref_s)&(t<=rref_e)]
ref_all = np.concatenate([lref, rref])
n_anom = np.sum(anom_mask)
phase_shift = 0.35
synth = np.zeros(n_anom)
for f, a in zip(freqs, amps):
    synth += a*np.sin(2*np.pi*f*t[anom_mask] + phase_shift*f)
synth += 0.3*np.random.randn(n_anom)
synth = synth * np.std(ref_all)/np.std(bg[anom_mask])
synth += np.mean(ref_all) - np.mean(bg[anom_mask])
after[anom_mask] = synth

blend_n = 120
anom_idx = np.where(anom_mask)[0]
for i in range(min(blend_n, len(anom_idx))):
    alpha = 0.5*(1-np.cos(np.pi*i/blend_n))
    li = anom_idx[i]
    ri = anom_idx[-(i+1)]
    after[li] = (1-alpha)*bg[li] + alpha*after[li]
    after[ri] = (1-alpha)*bg[ri] + alpha*after[ri]

# 标准双栏尺寸
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(DOUBLE_COL_WIDTH, 3.9), sharex=True)
# Signal Red：DSSRR repair 绯红加粗、Ground truth 黑、修复前浅灰；参考钴蓝 / 间隙灰 / 异常红
c_before, c_anom, c_ref, c_gap, c_repair = '#B0BEC5','#e74c3c','#1E88E5','#78909C','#D62728'
c_truth = '#000000'

# 上：修复前
ax1.plot(t, before, color=c_before, lw=0.7)
ax1.axvspan(lref_s,lref_e, alpha=0.12, color=c_ref, zorder=0)
ax1.axvspan(lref_e,anom_s, alpha=0.08, color=c_gap, zorder=0)
ax1.axvspan(anom_s,anom_e, alpha=0.18, color=c_anom, zorder=0)
ax1.axvspan(anom_e,rref_s, alpha=0.08, color=c_gap, zorder=0)
ax1.axvspan(rref_s,rref_e, alpha=0.12, color=c_ref, zorder=0)
ax1.axvline(anom_s, color=c_anom, ls='--', lw=0.9, alpha=0.8)
ax1.axvline(anom_e, color=c_anom, ls='--', lw=0.9, alpha=0.8)

y_main, y_sub = 5.0, 6.0
bbox = dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.95)
labels_top = [
    ((lref_s+lref_e)/2, 'Healthy reference', r'$R_b$', c_ref, 9),
    ((lref_e+anom_s)/2, 'Safety gap', r'$G$', c_gap, 9),
    ((anom_s+anom_e)/2, 'Anomaly segment', r'$[s,e]$', c_anom, 10),
    ((anom_e+rref_s)/2, 'Safety gap', r'$G$', c_gap, 9),
    ((rref_s+rref_e)/2, 'Healthy reference', r'$R_a$', c_ref, 9),
]
for x, txt, sym, col, sym_sz in labels_top:
    ax1.text(x, y_sub, txt, ha='center', va='bottom', fontsize=7.2, color=col, bbox=bbox)
    ax1.text(x, y_main, sym, ha='center', va='bottom', fontsize=sym_sz-1, color=col, fontweight='bold', bbox=bbox)

ax1.set_ylabel('Amplitude (DU)', fontsize=8)
ax1.set_title('(b) Before DSSRR repair', fontsize=9, fontweight='bold', loc='left', pad=4)
ax1.set_ylim(-4.5,7.0)
ax1.set_xlim(-0.5, duration+0.5)
leg = [Patch(facecolor=c_ref, alpha=0.3, label='Healthy reference'),
       Patch(facecolor=c_gap, alpha=0.3, label='Safety gap'),
       Patch(facecolor=c_anom, alpha=0.3, label='Anomaly segment')]
ax1.legend(handles=leg, loc='upper right', fontsize=6.8)

# 下：修复后
# 关键：DSSRR 只改写 [s,e] 内的样本，段外与输入逐比特相同。若把整条 after 曲线
# 都画成绯红，读者会以为整条记录都被"处理"过——图上绯红铺满全程，与图注
# "all samples outside [s,e] remain unchanged" 直接冲突。
# 因此按"是否被修改"分段着色：段外 = 未改动（沿用 (b) 的修复前浅灰，语义一致），
# 段内 = DSSRR 合成（绯红加粗），并叠一层真值背景虚线供直接比对。
mask_anom = (t >= anom_s) & (t <= anom_e)
ax2.axvspan(anom_s, anom_e, alpha=0.10, color=c_repair, zorder=0)
# 段外 = 未改动（沿用 (b) 的修复前浅灰，语义一致）；段内置 NaN 断开。
# 注意：**不能写 t[~mask_anom]**。布尔索引会把"段外左侧最后一点"和"段外右侧
# 第一点"压缩成数组中相邻的元素，matplotlib 于是直接把 9 s 与 15 s 两点连成
# 一条横跨整个异常段的灰色斜线；又因共用同一个 label，图例还会把这条伪影线
# 标成 "Unchanged input (bit-identical)"，造成 (c) 与 (b) 自相矛盾
# （(b) 里段内明明是 0，不是斜线）。2026-10-06 修复。
after_unchanged = after.copy()
after_unchanged[mask_anom] = np.nan
ax2.plot(t, after_unchanged, color=c_before, lw=0.9, zorder=2,
         label='Unchanged input (bit-identical)')
ax2.plot(t[mask_anom], after[mask_anom], color=c_repair, lw=1.6, zorder=3,
         label='DSSRR synthesized segment')
ax2.plot(t[mask_anom], bg[mask_anom], color=c_truth, lw=1.0, ls='--', alpha=0.85, zorder=4,
         label='True background (ground truth)')
ax2.axvline(anom_s, color=c_anom, ls='--', lw=0.9, alpha=0.8)
ax2.axvline(anom_e, color=c_anom, ls='--', lw=0.9, alpha=0.8)
ax2.text((anom_s+anom_e)/2, y_sub, 'DSSRR synthesized',
         ha='center', va='bottom', fontsize=7.5, color=c_repair, fontweight='bold', bbox=bbox)
ax2.text((anom_s+anom_e)/2, y_main, 'Reconstructed segment',
         ha='center', va='bottom', fontsize=7, color=c_repair, bbox=bbox)
ax2.set_xlabel('Time (s)', fontsize=8)
ax2.set_ylabel('Amplitude (DU)', fontsize=8)
ax2.set_title('(c) After DSSRR repair', fontsize=9, fontweight='bold', loc='left', pad=4)
ax2.set_ylim(-4.5,7.0)
ax2.set_xlim(-0.5, duration+0.5)
ax2.legend(loc='upper right', fontsize=6.5, handlelength=1.8, borderaxespad=0.3)

plt.tight_layout(h_pad=0.8, pad=0.1)
out_path = f'{PANEL_DIR}/fig02bc_repair_before_after.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"HD Figure saved to {out_path}")
