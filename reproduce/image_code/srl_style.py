"""SRL (Seismological Research Letters) 期刊图件统一标准样式
遵循 SSA/SRL 作者指南：双栏印刷宽度约 7.09 inch，所有图件按实际印刷尺寸生成，
插入论文无需缩放，字号不会被压缩。

配色采用 "Signal Red" 方案（2026-10-05 定稿）：
- DSSRR（本文方法，主角色）= 绯红 #D62728，线宽加粗，保证彩色 / 灰度 / 色盲下均可辨；
- Ground truth 全文统一为黑色虚线（唯一黑色角色）；
- Linear / CubicSpline 保持灰阶（配角）；FFT / SSA / UNet 用区分色但不抢眼；
- 异常段统一浅红填充。

注意：本机 matplotlib 默认后端为 qtagg。在无显示环境（CI、后台批量出图）里直接运行
本目录脚本会被静默 SIGTERM 杀掉，且不打印任何错误。批量出图请加环境变量：

    MPLBACKEND=Agg python figXX_xxx.py
（或直接运行 build_all_figures.py，它已用 Agg 子进程编排全部脚本）
"""
import matplotlib as mpl

# ---- 统一方法色号（全论文图件严格一致，Signal Red 方案）----
METHOD_COLORS = {
    "DSSRR": "#D62728",       # 绯红（本文方法，主角色，Okabe-Ito 色盲安全）
    "Linear": "#B0BEC5",      # 浅灰（线性插值）
    "CubicSpline": "#78909C", # 中灰（保形分段三次插值 PCHIP）
    "FFT": "#1E88E5",         # 钴蓝（FFT/STFT 外推）
    "SSA": "#8E24AA",         # 紫（奇异谱分析）
    "UNet": "#F57C00",        # 琥珀橙（U-Net）
    "GroundTruth": "#000000", # 黑（真值，全文统一虚线）
}
METHOD_ORDER = ["Linear", "CubicSpline", "FFT", "SSA", "UNet", "DSSRR"]
COLOR_LIST = [METHOD_COLORS[m] for m in METHOD_ORDER]

# ---- 图例 / 坐标轴显示名（内部键仍是 METHOD_ORDER 里的短码，数据文件不受影响）----
# UNet 显示为 "1D U-Net"：本文的基线是单通道一维 U-Net（nn.Conv1d），
# 与地震学里常用于多道 / 时频图的 2D U-Net 不是同一个设置。正文与图表统一标注维度，
# 避免读者把"本文测的基线"误读成"U-Net 这一类方法"。
# CubicSpline 显示为 "PCHIP"：该基线内部调用 scipy.interpolate.PchipInterpolator
# （保形分段三次 Hermite 插值），并非经典 C2 三次样条。正文与图表统一用 PCHIP，
# 与公开代码里的实际实现一致（见 BASELINE_CODE_AUDIT_2026-10-10.md §3.2）。
METHOD_LABELS = {"UNet": "1D U-Net", "CubicSpline": "PCHIP"}


def mlabel(name):
    """把内部方法短码映射为图例/刻度上的显示名。"""
    return METHOD_LABELS.get(name, name)

# 线宽：主角色加粗、基线常规（保证灰度 / 色盲下主角仍突出）
DSSRR_LW = 2.4   # DSSRR 曲线线宽
BASE_LW = 1.4    # 各基线曲线线宽

# 异常段统一填充色（alpha 混合后呈浅粉）
ANOMALY_FILL = "#e74c3c"
ANOMALY_ALPHA = 0.15

# 修复前后统一配色
BEFORE_COLOR = "#B0BEC5"  # 修复前：浅灰
AFTER_COLOR = "#D62728"   # 修复后：绯红（主角色）

# ---- 印刷尺寸常量 ----
DOUBLE_COL_WIDTH = 7.09  # 双栏图宽度（英寸，约 18cm）
SINGLE_COL_WIDTH = 3.4   # 单栏图宽度（英寸，约 8.6cm）

def apply_srl_style():
    """一键应用 SRL 全局样式"""
    # 字体：无衬线 Arial/Helvetica
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    # 字号（实际印刷字号，不会被缩放）
    mpl.rcParams["axes.labelsize"] = 8.5
    mpl.rcParams["xtick.labelsize"] = 7.5
    mpl.rcParams["ytick.labelsize"] = 7.5
    mpl.rcParams["legend.fontsize"] = 7.5
    mpl.rcParams["xtick.major.size"] = 4
    mpl.rcParams["ytick.major.size"] = 4
    mpl.rcParams["xtick.major.width"] = 0.8
    mpl.rcParams["ytick.major.width"] = 0.8
    mpl.rcParams["axes.linewidth"] = 0.8

    # 简洁风格：去上右边框、刻度向外、无网格
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["xtick.direction"] = "out"
    mpl.rcParams["ytick.direction"] = "out"
    mpl.rcParams["axes.grid"] = False

    # 图例默认无边框透明背景
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["legend.numpoints"] = 1
    mpl.rcParams["legend.handlelength"] = 1.8

# 子图标签统一样式
PANEL_LABEL_KW = dict(fontsize=9, fontweight="bold", loc="left", pad=4)
# 保存高清图默认参数
SAVE_KW = dict(dpi=1200, bbox_inches="tight", facecolor="white", pad_inches=0.05)
