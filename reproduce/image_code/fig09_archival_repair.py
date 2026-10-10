"""图9：阿波罗档案三个修复案例
布局 3 行 2 列：左列全段概览 + 异常红标，右列上下分修复前/后（独立 y 轴）。

着色规则（与图 2(c) 一致）：**按"是否被修改"着色**，而不是按"修复前/后"着色。
- 灰  = 输入数据（DSSRR 未触碰的样本，与修复前逐比特相同）
- 绯红 = 被 DSSRR 改写过的样本（检测窗口内逐比特发生变化的那些，是窗口的真子集）
这样图面与图注 "Only samples inside the red anomaly window are modified; all samples
outside remain bit-identical to the input" 才一致；否则整段 10 小时记录都会被涂成红色，
而实际被改动的样本只有 2.5%–11.1%。
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from obspy import read
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import (apply_srl_style, DOUBLE_COL_WIDTH,
                       ANOMALY_FILL, ANOMALY_ALPHA, SAVE_KW, PANEL_LABEL_KW)

apply_srl_style()

OBS = Path(r"G:\SeisY\docs\dssrr_paper\experiments\results_obs")
FIG_DIR = Path(r"G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures")
BEFORE = "#555555"    # 修复前深灰
AFTER = "#D62728"     # 修复后绯红（Signal Red 主角色）
AFTER_ALPHA = 0.92

# 手动固定三个案例及异常/放大窗口（小时，相对记录起点）
CASES = [
    {
        "file": "XA.S15.00.MHZ.19720917_133809-19720917_233809.mseed",
        "anom_zoom_h": (0.919, 1.927),   # 精确检测：60.5min dropout
        "view_zoom_h": (0.7, 2.15),
        "label": "S15\n1972-09-17"
    },
    {
        "file": "XA.S12.01.MHZ.19760113_061654-19760113_161654.mseed",
        "anom_zoom_h": (4.819, 5.124),   # 精确检测：18.3min尖峰簇
        "view_zoom_h": (4.6, 5.35),
        "label": "S12\n1976-01-13"
    },
    {
        "file": "XA.S15.01.MHZ.19760113_061655-19760113_161655.mseed",
        # 2026-10-06 数据重处理：改动样本实测 5.5125--5.7932 h（16.8 min，
        # 含尖峰簇 + 缺数段 + 阶跃恢复），窗口相应由旧 (5.514, 5.770) 扩到覆盖全部改动。
        "anom_zoom_h": (5.508, 5.798),
        "view_zoom_h": (5.3, 5.95),
        "label": "S15\n1976-01-13"
    },
]

def load(name):
    b = read(str(OBS / "修复前" / name))[0]
    a = read(str(OBS / name))[0]
    n = min(b.stats.npts, a.stats.npts)
    sr = b.stats.sampling_rate
    t = np.arange(n)/sr/3600
    return t, b.data[:n].astype(float), a.data[:n].astype(float), sr

def pad_ylim(y, frac=0.12, top_frac=None):
    lo, hi = np.min(y), np.max(y)
    span = hi-lo
    if top_frac is None:
        top_frac = frac
    return lo - span*frac, hi + span*top_frac

fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, 4.85))
gs = GridSpec(3, 2, figure=fig,
              width_ratios=[2.2, 1.0],
              hspace=0.34, wspace=0.2,
              left=0.08, right=0.98, top=0.96, bottom=0.07)

for r, case in enumerate(CASES):
    t, b, a, sr = load(case["file"])
    # 被改动的样本掩码：DSSRR 只改写检测到的异常窗口，其余样本与输入逐比特相同。
    # 若把整条 after 曲线都涂成绯红，读者会以为整段 10 小时记录都被"处理"过
    # （实测真正被改动的样本只占 11.1% / 3.2% / 2.5%），与图注
    # "Only samples inside the red anomaly window are modified" 直接冲突。
    # 因此按"是否被修改"着色：未改动处用灰（就是输入本身），只有被改写的窗口用绯红。
    dmask = np.abs(a - b) > 1e-9
    a_repaired = np.where(dmask, a, np.nan)
    # 左列全段
    axL = fig.add_subplot(gs[r,0])
    axL.plot(t, b, color=BEFORE, lw=0.45, label="Before repair (input)", zorder=2)
    axL.plot(t, a_repaired, color=AFTER, lw=0.7, alpha=AFTER_ALPHA,
             label="After repair (repaired windows only)", zorder=3)
    s_h, e_h = case["anom_zoom_h"]
    axL.axvspan(s_h,e_h, color=ANOMALY_FILL, alpha=ANOMALY_ALPHA, lw=0,
                label="Detected anomaly window")
    axL.set_ylabel(case["label"], fontsize=7.5, linespacing=1.3)
    axL.set_ylim(pad_ylim(b, 0.08))
    axL.tick_params(labelsize=7)
    # 子图标号：(a)(c)(e) 为左列全段概览，(b)(d)(f) 为右列 before/after 成对放大图
    axL.set_title('(%s)' % chr(ord('a') + 2 * r), **PANEL_LABEL_KW)
    if r == 0:
        axL.legend(loc="upper right", fontsize=7, framealpha=0.9)
    if r == 2:
        axL.set_xlabel("Time (hours)", fontsize=8)
    # 放大窗标记
    vs_h, ve_h = case["view_zoom_h"]
    axL.axvspan(vs_h,ve_h, color="#000", alpha=0.04, lw=0)

    # 右列上下分栏
    inner = gs[r,1].subgridspec(2,1, hspace=0.45)
    axB = fig.add_subplot(inner[0,0])
    axA = fig.add_subplot(inner[1,0], sharex=axB)
    mask = (t >= vs_h) & (t <= ve_h)
    tz, bz, az = t[mask], b[mask], a[mask]
    am_s, am_e = s_h - vs_h, e_h - vs_h

    # 右列 before/after 共用一个标号，放在上方那块
    axB.set_title('(%s)' % chr(ord('a') + 2 * r + 1), **PANEL_LABEL_KW)
    _tag_bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                     edgecolor='none', alpha=0.8)

    # 同一着色规则：灰 = 输入（未改动），绯红 = 被改写的窗口。
    # 放大窗略宽于异常窗，若把整幅 after 都涂红，窗内未改动的部分会被误读为已修复。
    dmask_z = np.abs(az - bz) > 1e-9

    axB.plot(tz, bz, color=BEFORE, lw=0.9)
    axB.axvspan(s_h,e_h, color=ANOMALY_FILL, alpha=ANOMALY_ALPHA, lw=0, label="Anomaly")
    axB.set_ylim(pad_ylim(bz))
    axB.tick_params(labelsize=7)
    axB.text(0.02, 0.96, "Before repair", transform=axB.transAxes, fontsize=7,
             color=BEFORE, va="top", ha="left", zorder=6, bbox=_tag_bbox)
    axB.tick_params(labelbottom=False)

    axA.axvspan(s_h,e_h, color=ANOMALY_FILL, alpha=ANOMALY_ALPHA, lw=0, label="Anomaly")
    if dmask_z.any():
        axA.plot(tz, np.where(dmask_z, np.nan, az), color=BEFORE, lw=0.9, zorder=2)
        axA.plot(tz, np.where(dmask_z, az, np.nan), color=AFTER, lw=1.0,
                 alpha=AFTER_ALPHA, zorder=3)
    else:
        axA.plot(tz, az, color=BEFORE, lw=0.9, zorder=2)
    axA.set_ylim(pad_ylim(az))
    axA.tick_params(labelsize=7)
    axA.text(0.02, 0.96, "After repair", transform=axA.transAxes, fontsize=7,
             color=AFTER, va="top", ha="left", zorder=6, bbox=_tag_bbox)
    if r == 2:
        axA.set_xlabel("Time (hours)", fontsize=8)
    else:
        axA.tick_params(labelbottom=False)

out = FIG_DIR / "fig09_archival_repair.png"
plt.savefig(out, **SAVE_KW)
plt.close()
print(f"Restored fig9 saved to {out}")
