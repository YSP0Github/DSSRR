# -*- coding: utf-8 -*-
"""
DbRepairViewer — 双库对比查看器（原始库 vs DSSRR 修复库）
=========================================================
按台站 / 日期 / 时间窗对比去异常前后的波形，用于人工验收 DSSRR 的修复质量。

功能
----
- 时间窗浏览：30m / 1h / 6h / 12h / 1d 快捷档位，可拖动起始位置
- 视图模式：分画布 / 同画布 / PSD / 振幅谱 / 综合 2×2
- 叠加标注：修复段（红）、保护事件（绿）、长缺口（灰）
- 目录联动：读取 ``E7_catalog_*.csv`` 自动标注每个修复段使用的方法

数据源
------
- 原始库根目录：由用户在界面上选择，也可用环境变量 ``DSSRR_RAW_ROOT``
  预置。**不假设任何特定机器上的路径。**
- 修复库根目录：默认 = 原始库 + ``_repaired``（存在时自动识别）。
- 两者均支持日文件 ``XA.{站}.{YYYYMMDD}.mseed`` 与片段文件
  ``XA.{站}.{频带}.{通道}.{起}-{止}.mseed``。

依赖：PyQt5、matplotlib、obspy、pandas。

本模块是独立工具窗口，由主窗口的 "Repaired DB Viewer" 导航项打开。
"""
from __future__ import annotations

import os
import sys

import numpy as np

from PyQt5.QtCore import Qt, QTimer, QPointF, QRect, QSize, QPoint
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QComboBox, QCompleter, QSpinBox, QPushButton, QRadioButton, QGroupBox,
    QLineEdit, QFileDialog, QMessageBox, QApplication, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog, QSplitter, QDoubleSpinBox,
    QScrollArea, QCheckBox, QTabWidget, QLayout, QSizePolicy,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import rcParams

from dssrr.discovery import parse_mseed_name, find_counterpart

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SR = 6.625
STATIONS_DEFAULT = ["S12", "S15", "S16"]
# 扫描不到任何文件时，"Stream" 下拉的占位项（Apollo 三个着陆点，日文件命名）
STREAMS_DEFAULT = ["XA.S12", "XA.S15", "XA.S16"]
VIEW_MODES = ("split", "overlay", "psd", "amp", "combo")

# 数据根目录输入框的占位提示。刻意使用通用示例路径而不是某台机器上的真实
# 路径，避免把个人目录结构带进开源发行版。实际路径由用户选择或环境变量
# DSSRR_RAW_ROOT / DSSRR_FIX_ROOT 预置。
RAW_ROOT_HINT = r"e.g. D:\apollo\database"
FIX_ROOT_HINT = r"e.g. D:\apollo\database_repaired"


class FlowLayout(QLayout):
    """左到右排布、放不下就自动换行的布局（Qt 没有内置流式布局）。

    为什么需要它：用一条 ``QHBoxLayout`` 放「视图模式单选 + 6 个操作按钮」时，
    Qt 会把该行所有子控件的宽度**求和**当作窗口的最小宽度（实测约 1400 px，
    叠加其它面板后整个窗口最小宽度超过 2500 px）。窗口比屏幕还宽时，Windows
    会在每次关闭模态对话框后重新"摆放"窗口——这正是主窗口莫名跳到屏幕底部
    的根源之一。换成流式布局后，最小宽度只取决于**单个**最宽的子控件，窄窗口
    下自动折成两行，既不裁切也不再逼着窗口超出屏幕。

    实现参考 Qt 官方 Flow Layout 示例，仅保留本项目用得到的部分。
    """

    def __init__(self, parent=None, margin=0, hspacing=6, vspacing=6):
        super().__init__(parent)
        self._items = []
        self._hspace = int(hspacing)
        self._vspace = int(vspacing)
        self.setContentsMargins(margin, margin, margin, margin)

    # ---- QLayout 必须实现的接口 ----
    def addItem(self, item):                      # noqa: N802 (Qt 命名)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):                      # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):                      # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):                # noqa: N802
        return Qt.Orientations(Qt.Orientation(0))

    # ---- 高度随宽度变化：QBoxLayout 靠这两个方法给出行高 ----
    def hasHeightForWidth(self):                  # noqa: N802
        return True

    def heightForWidth(self, width):              # noqa: N802
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):                  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):                           # noqa: N802
        """首选尺寸 = **单行**排布所需的宽度（子项宽度之和 + 间距 + 边距）。

        与 :meth:`minimumSize` 的区别很关键，两者不能混为一谈：

        - ``minimumSize()`` = "最宽的那一项"——窄到极限时逐项折行，这是**下限**，
          也正是它让窗口的最小宽度只取决于单个最宽子控件（避免窗口被撑到屏幕外）；
        - ``sizeHint()`` = "不折行时需要多宽"——父布局（QBoxLayout）按它分配宽度。

        若这里也返回 ``minimumSize()``，父布局就只给这么点宽度，于是这一组按钮
        每个各占一行、整组竖着排下来（把 FlowLayout 嵌进 QHBoxLayout 时必踩）。
        """
        m = self.contentsMargins()
        w, h = m.left() + m.right(), 0
        for i, item in enumerate(self._items):
            hint = item.sizeHint()
            w += hint.width() + (self._hspace if i else 0)
            h = max(h, hint.height())
        h += m.top() + m.bottom()
        # 单行宽度不可能小于下限（例如只有一个子项时两者相等）
        return QSize(max(w, self.minimumSize().width()), h)

    def minimumSize(self):                        # noqa: N802
        """最小尺寸 = 最宽子项 + 最矮子项（而不是所有子项之和）。"""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    # ---- 实际排布 ----
    def _do_layout(self, rect, test_only):
        """两遍排布：先切行、后定位。

        第一遍只决定"谁和谁在同一行、这一行多高"（行高 = 行内最高的一项），
        第二遍才真正摆放。分两遍是**必须**的：只有知道整行的高度之后，才能把
        行内较矮的子控件（例如 27 px 的按钮）在行内**垂直居中**。若像早期实现
        那样边遍历边摆放，矮控件只能贴在本行顶边，于是当它和 51 px 的
        ``QGroupBox``（View 选择器）同处一行时，按钮下方就会空出一条带——看起来
        就是"这一行比内容高"。
        """
        m = self.contentsMargins()
        eff = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        avail = eff.width()

        # ---- 第一遍：切行 ----
        lines = []                     # [[(item, w, h), ...], ...]
        cur, cur_w = [], 0             # cur_w = 本行已占宽度（含项间间距）
        for item in self._items:
            hint = item.sizeHint()
            # 单项比整行还宽时压到行宽，避免溢出到窗口外；被压窄的子控件如果
            # 自带 heightForWidth（例如内层也是 FlowLayout 的 QGroupBox），
            # 高度要按新宽度重算，否则它折行后会盖住下面一行。
            w = hint.width()
            if 0 < avail < w:
                w = avail
            h = hint.height()
            if w != hint.width() and item.hasHeightForWidth():
                h = max(h, item.heightForWidth(w))
            # 与旧实现同一条换行判据（旧：x + w > eff.right() + 1，其中
            # x = eff.x() + cur_w、eff.right() + 1 = eff.x() + avail）：
            # 本行已有内容、且加上这一项会越过右边界
            if cur and cur_w + w > avail:
                lines.append(cur)
                cur, cur_w = [], 0
            cur.append((item, w, h))
            cur_w += w + self._hspace
        if cur:
            lines.append(cur)

        # ---- 第二遍：逐行定位（行内垂直居中） ----
        if not lines:
            return m.top() + m.bottom()
        y = eff.y()
        for line in lines:
            line_h = max(h for _, _, h in line)
            x = eff.x()
            for item, w, h in line:
                if not test_only:
                    item.setGeometry(
                        QRect(QPoint(x, y + (line_h - h) // 2), QSize(w, h)))
                x += w + self._hspace
            y += line_h + self._vspace
        return (y - self._vspace) - rect.y() + m.bottom()


def visible_top_level(widget):
    """返回 ``widget`` 所属的**可见**顶层窗口；找不到返回 ``None``。"""
    if widget is None:
        return None
    p = widget.parent()
    while p is not None:
        if p.isWindow() and p.isVisible():
            return p
        p = p.parent()
    if widget.isWindow() and widget.isVisible():
        return widget
    return None


def run_modal_dialog(widget, fn):
    """执行一次模态对话框调用，并在前后保持宿主窗口的位置与层级。

    为什么需要：查看器 / 手动修复面板被嵌入主窗口后，**自身仍是顶层窗口**
    （只是从不 show）。若直接把它们当作 ``QFileDialog`` 的 parent，Windows
    会把对话框挂到一个不可见的窗口上；对话框关闭后窗口管理器重新"摆放"
    真正的窗口，表现为主窗口跳到屏幕底部、或被其它窗口盖住。

    这里做三件事：

    1. 用**可见的**顶层窗口作为对话框的 parent（``fn`` 会收到它）；
    2. 记录打开前的几何；
    3. 关闭后恢复几何并 ``raise_()`` / ``activateWindow()``，把它提回前台。
    """
    host = visible_top_level(widget) or widget
    geo = host.geometry() if host is not None else None
    try:
        return fn(host)
    finally:
        if host is not None:
            try:
                if geo is not None and not geo.isNull():
                    host.setGeometry(geo)
                host.raise_()
                host.activateWindow()
            except Exception:
                pass

# 修复明细中 kind/method 数据值的中文显示映射（数据本身存英文键，
# 界面显示统一走映射，保证国际化）
KIND_ZH = {
    "repair": "修复段",
    "event": "事件",
    "gap": "缺口",
}
METHOD_ZH = {
    "dssrr": "DSSRR 重建",
    "dssrr_fb_missing": "DSSRR 缺失兜底",
    "freeze_missing": "冻结缺失修复",
    "missing_runs": "缺失段修复",
    "spike": "尖峰修复",
    "zscore": "Z分数清理",
    "keep": "保留（缺口）",
    "catalog_protect": "月震目录保护",
}
METHOD_EN = {
    "dssrr": "DSSRR rebuild",
    "dssrr_fb_missing": "DSSRR + missing fallback",
    "freeze_missing": "Freeze missing-run fix",
    "missing_runs": "Missing-run fix",
    "spike": "Spike repair",
    "zscore": "Z-score clean",
    "keep": "Keep (gap)",
    "catalog_protect": "Catalog protection",
    # ---- 旧 catalog 兼容：老 CSV 没有 method 列，_infer_method() 会就地合成
    # 一段**中文**描述当 method 值。method_name() 拿不到映射就把原值原样吐出，
    # 于是英文界面的"Method"列会冒出中文。这里把那些中文原值也登记成 key，
    # 指向英文显示名（METHOD_ZH 不用登记：查不到时原样返回，正好是中文）。
    "信号保护（振荡/瞬态簇，非目录月震）":
        "Signal protection (oscillation/transient cluster)",
    "信号保护（检测器判定，非目录月震）":
        "Signal protection (detector-flagged)",
    "月震保护（目录命中）": "Moonquake protection (catalog hit)",
    "月震保护（不修复）": "Moonquake protection (left untouched)",
    "长缺口（未修复）": "Long gap (not repaired)",
    "月震保护型尖峰修复": "Moonquake-protected spike repair",
    "MAD 孤立尖峰修复": "Isolated spike repair (MAD)",
    "线性插值": "Linear interpolation",
    "Z-score 修复": "Z-score repair",
    "DSSRR 双侧谱参考重建": "DSSRR two-sided spectral reconstruction",
    "保留原值（冻结不重建）": "Keep original (frozen, no rebuild)",
}


KIND_EN = {
    "repair": "Repaired",
    "event": "Event",
    "gap": "Gap",
}


def kind_name(key):
    """异常类别显示名：跟界面语言（zh qm 安装时走中文翻译，否则英文）。"""
    from PyQt5.QtCore import QCoreApplication
    en = KIND_EN.get(key, key)
    s = QCoreApplication.translate("RepairKindNames", en)
    return s if s else KIND_ZH.get(key, key)


def method_name(key):
    """修复方法显示名：跟界面语言（zh qm 安装时走中文翻译，否则英文）。"""
    from PyQt5.QtCore import QCoreApplication
    en = METHOD_EN.get(key, key)
    s = QCoreApplication.translate("RepairMethodNames", en)
    return s if s else METHOD_ZH.get(key, key)
# 修复段按方法着色（画布选框、明细表方法列、统计文本共用同一套色，
# 浅/深主题下均清晰可读）；事件=绿、缺口=灰，保持语义稳定。
METHOD_COLORS = {
    "dssrr": "#EF4444",             # 红——核心 DSSRR 重建
    "dssrr_fb_missing": "#F97316",  # 橙——DSSRR 缺失兜底
    "freeze_missing": "#A855F7",    # 紫——冻结缺失修复
    "missing_runs": "#3B82F6",      # 蓝——缺失段修复
    "lin": "#06B6D4",               # 青——线性插值
    "zscore": "#EC4899",            # 粉——Z分数清理
    "spike": "#F59E0B",             # 琥珀——尖峰修复
    "catalog_protect": "#10B981",   # 绿——月震目录保护
    "keep": "#9CA3AF",              # 灰——保留（缺口）
}
DEFAULT_REPAIR_COLOR = "#EF4444"
EVENT_COLOR = "#22C55E"
GAP_COLOR = "#9CA3AF"
MAX_WIN_MIN = 1440          # 时间窗上限：1 天
MAX_START_MIN = 1440 * 40   # 起始位置上限（覆盖整月跨日查看）
# 窗口快捷档位（4-5 个常用档，按钮文字须完整显示；上限 1d = MAX_WIN_MIN）
WIN_PRESETS = [
    ("30m", 30, "Set window to 30 minutes (30 min)"),
    ("1h", 60, "Set window to 1 hour (60 min)"),
    ("6h", 360, "Set window to 6 hours (360 min)"),
    ("12h", 720, "Set window to 12 hours (720 min)"),
    ("1d", 1440, "Set window to 1 day (1440 min)"),
]


# ----------------------------------------------------------------------
# 数据与渲染核心（独立于 GUI，可离线测试）
# ----------------------------------------------------------------------
def load_trace(path, loc=None, cha=None):
    """读取文件里的目标 Trace，返回 ``(data, starttime)``。

    选择顺序：

    1. 若给了 ``loc`` / ``cha``，优先精确匹配（空串表示"该位不限"）；
    2. 否则回退到通道 ``MHZ``（Apollo 长周期垂向道的惯例，也是查看器一直
       以来假设的通道）；
    3. 再不行就取第一条 Trace。

    这样既支持"从 Stream 列表里精确选某一道"，也保留了旧行为：只给一个
    文件路径时自动取 MHZ 道。
    """
    from obspy import read
    st = read(path)
    if len(st) == 0:
        raise ValueError(f"No trace in {path}")
    loc = "" if loc is None else str(loc)
    cha = "" if cha is None else str(cha)
    pick = None
    if loc or cha:
        for tr in st:
            if loc and tr.stats.location != loc:
                continue
            if cha and tr.stats.channel != cha:
                continue
            pick = tr
            break
    if pick is None:
        mhz = [t for t in st if t.stats.channel == "MHZ"]
        pick = mhz[0] if mhz else st[0]
    return pick.data.astype(float), pick.stats.starttime


def load_mhz(path):
    """向后兼容的别名：取 MHZ 道（等价于 :func:`load_trace` 的默认行为）。"""
    return load_trace(path)


_REPAIR_LIB_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "repair_lib"))
# 独立包内没有 docs/dssrr_paper/experiments，直接指向随包分发的 repair_lib
_EXPERIMENTS_DIR = _REPAIR_LIB_DIR


def _find_catalog_csvs(fix_root=None):
    """返回所有 E7_catalog_*.csv 路径列表（按文件名排序）。

    查找顺序：
    1. 修复库根目录下的 catalogs/（批量去异常写入位置，优先）
    2. 随包 repair_lib / 源码 experiments（回退兼容旧数据）
    """
    found = []
    seen = set()
    seen_names = set()
    candidates = []
    if fix_root:
        cat_dir = os.path.join(str(fix_root).strip(), "catalogs")
        if os.path.isdir(cat_dir):
            candidates.append(cat_dir)
    fallbacks = [
        _REPAIR_LIB_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments"),
        _EXPERIMENTS_DIR,
    ]
    # 优先目录（fix_root/catalogs/，批量去异常权威写入位置）的同名文件为
    # 权威版本；fallback 中的同名旧副本（如 repair_lib/ 历史 catalog）一律
    # 跳过，否则新旧两版叠加会导致明细出现"幽灵段/重复段"
    for base in candidates:
        if not base or not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if fn.startswith("E7_catalog_") and fn.endswith(".csv"):
                fp = os.path.join(base, fn)
                if fp not in seen:
                    seen.add(fp)
                    seen_names.add(fn)
                    found.append(fp)
    for base in fallbacks:
        if not base or not os.path.isdir(base):
            continue
        for fn in sorted(os.listdir(base)):
            if fn.startswith("E7_catalog_") and fn.endswith(".csv"):
                if fn in seen_names:
                    continue
                fp = os.path.join(base, fn)
                if fp not in seen:
                    seen.add(fp)
                    seen_names.add(fn)
                    found.append(fp)
    return found


def _repair_catalog_path():
    """兼容旧调用：返回第一个可用 catalog CSV（无 fix_root 上下文）。"""
    paths = _find_catalog_csvs()
    return paths[0] if paths else os.path.join(_REPAIR_LIB_DIR, "E7_catalog_197601.csv")


def _spectrum_nperseg(n):
    """频谱估计的频率分辨率自适应：nperseg 取 2 的幂、上限 16384。

    频点数 ≈ nperseg/2，nperseg=16384 时 df≈0.4 mHz（SR=6.625），
    全天/小时窗口的谱线平滑密集，低频自由振荡频段有足够频点。
    """
    n = int(n)
    if n < 8:
        return max(8, n)
    p = 1
    while p * 2 <= n and p < 16384:
        p *= 2
    return max(256, min(16384, p))


def _spectrum_ready(x):
    """谱估计前的数据准备：缺失标记（≤-1、==0、NaN）线性插值填充。

    raw 数据中的缺失标记是伪值（-1/0），若直接参与 welch/FFT 会在
    低频产生虚假能量峰（实测 S12 01-03 低频峰 9.97 → 填充后 2.37，
    与修复后谱 2.37 一致）；填充仅用于谱估计，不改动显示的数据。
    """
    x = np.asarray(x, dtype=float).copy()
    miss = (x <= -1) | (x == 0) | ~np.isfinite(x)
    if miss.all():
        x[:] = 0.0
    elif miss.any():
        idx = np.arange(len(x))
        x[miss] = np.interp(idx[miss], idx[~miss], x[~miss])
    return x


def psd(x, sr):
    from scipy.signal import welch
    if len(x) < 8:
        return np.array([0.0]), np.array([0.0])
    x = _spectrum_ready(x)
    nperseg = _spectrum_nperseg(len(x))
    f, p = welch(x - x.mean(), fs=sr, nperseg=nperseg, noverlap=nperseg // 2)
    return f, p


def amplitude_spectrum(x, sr):
    """线性振幅谱：直接 FFT 单边振幅（无分段平均、无 nperseg 参数）。

    A(f) = 2·|X(f)| / (N·w̄)，X 为加 Hanning 窗后的 DFT；频点 N/2 个，
    频率分辨率 df = sr/N（全天 57 万样本 → df≈1.2e-5 Hz），
    适合直接查看自由振荡等窄频特征。缺失标记先线性填充
    （_spectrum_ready，仅用于谱估计）。
    """
    from numpy.fft import rfft, rfftfreq
    x = _spectrum_ready(x)
    n = len(x)
    if n < 8:
        return np.array([0.0]), np.array([0.0])
    x = x - x.mean()
    win = np.hanning(n)
    X = rfft(x * win)
    amp = 2.0 * np.abs(X) / (n * win.mean())
    amp[0] = amp[0] / 2.0  # DC 分量不乘 2
    f = rfftfreq(n, d=1.0 / sr)
    return f, amp


def _infer_method(kind, subtype, len_s, source=""):
    """由目录字段推断该段使用的修复方法（v4c 分级规则）。
    若 CSV 后续扩展 method 列，load_segments 会优先采用真实值。"""
    if kind == "event":
        # v4g：event 分来源——目录命中（moon）才是"月震保护"；
        # 振荡/瞬态簇保护（osc）与检测器判定事件（det）均为非目录
        # 疑似信号，不得标为月震保护。
        if source == "osc":
            return "信号保护（振荡/瞬态簇，非目录月震）"
        if source == "det":
            return "信号保护（检测器判定，非目录月震）"
        if source == "moon":
            return "月震保护（目录命中）"
        return "月震保护（不修复）"
    if kind == "gap":
        return "长缺口（未修复）"
    st = (subtype or "").lower()
    try:
        npts = float(len_s) * SR
    except Exception:
        npts = 0
    if st == "spike":
        return "月震保护型尖峰修复" if npts <= 3 else "MAD 孤立尖峰修复"
    if st == "zero":
        if npts <= 2:
            return "线性插值"
        if npts <= 20:
            return "Z-score 修复"
        return "DSSRR 双侧谱参考重建"
    if st == "freeze":
        # v4f：冻结段不整体重建，只填缺失 run、恒定部分保留原值
        return "保留原值（冻结不重建）"
    # step / burst / sat 及其他长段
    return "DSSRR 双侧谱参考重建"


def load_segments(catalog_path):
    """(station, day) -> {"repair":[dict], "event":[dict], "gap":[dict]}
    每段 dict: s,e(样本索引), subtype, len_s, start_time, end_time, method, kind
    catalog_path 可以是单个 CSV 路径或路径列表（多月份合并加载）。"""
    segs = {}
    if not catalog_path:
        return segs
    if isinstance(catalog_path, str):
        catalog_path = [catalog_path]
    paths = [p for p in catalog_path if p and os.path.exists(p)]
    if not paths:
        return segs
    # 显示层去重：同一 (台站,日,类别,方法) 且分钟级起止相同的行只保留一条。
    # write_catalog_csv 已按样本级 key 去重，但历史文件或边界微调（start_s 差
    # 几个样本）可能让两条记录在界面显示成完全相同的行，这里统一兜底。
    seen_disp = set()
    try:
        import pandas as pd
        for catalog_file in paths:
            df = pd.read_csv(catalog_file)
            # 兼容 BOM 列名
            df.columns = [c.lstrip("\ufeff") for c in df.columns]
            has_method = "method" in df.columns
            for _, r in df.iterrows():
                key = (str(r["station"]), str(r["day"]).replace("-", ""))
                bucket = segs.setdefault(key, {"repair": [], "event": [], "gap": []})
                s, e = int(round(r["start_s"])), int(round(r["end_s"]))
                kind = str(r["kind"])
                subtype = str(r.get("subtype", "")) if pd.notna(r.get("subtype", "")) else ""
                source = str(r.get("source", "")) if pd.notna(r.get("source", "")) else ""
                method = str(r["method"]) if has_method and pd.notna(r.get("method", "")) \
                    else _infer_method(kind, subtype, r.get("len_s", 0), source)
                seg = {
                    "s": s, "e": e,
                    "subtype": subtype,
                    "source": source,
                    "len_s": float(r.get("len_s", 0)),
                    "start_time": str(r.get("start_time", "")),
                    "end_time": str(r.get("end_time", "")),
                    "method": method,
                    "kind": kind,
                }
                disp_key = (key[0], key[1], kind, method,
                            round(s / SR / 60.0, 1), round(e / SR / 60.0, 1))
                if disp_key in seen_disp:
                    continue
                seen_disp.add(disp_key)
                bucket[kind if kind in bucket else "repair"].append(seg)
    except Exception as exc:  # pragma: no cover
        print("catalog load warn:", exc)
    return segs


def stream_label(net, sta, loc="", cha=""):
    """拼出 Stream 列表里显示的标签。

    - 文件名只带台站（日文件）→ ``XA.S12``
    - 文件名带定位码/通道（片段文件）→ ``XA.S12.01.MHZ``

    不带定位码/通道时不做猜测：查看器加载时会按 Apollo 惯例取 MHZ 道
    （见 :func:`load_trace`），标签只如实反映文件名里能读到的信息。
    """
    head = ".".join(p for p in (net, sta) if p)
    tail = ".".join(p for p in (loc, cha) if p)
    return f"{head}.{tail}" if tail else head


def split_stream_label(label):
    """把 ``XA.S12.01.MHZ`` 拆成 ``(net, sta, loc, cha)``；缺项补空串。"""
    parts = [p for p in str(label or "").split(".")]
    parts += [""] * (4 - len(parts))
    return parts[0], parts[1], parts[2], parts[3]


def station_of(label):
    """从 Stream 标签取出台站代码（``XA.S12.01.MHZ`` → ``S12``）。

    也接受裸台站名（``S12`` → ``S12``），以兼容"只给台站"的旧调用
    （如命令行导出、修复库路径拼接）。
    """
    s = str(label or "").strip()
    if "." not in s:
        return s
    return split_stream_label(s)[1]


def scan_days(raw_root):
    """递归扫描原始库。

    返回 ``(streams, days, index)``：

    - ``streams``: 排序后的 **Stream 标签** 列表（``NET.STA[.LOC.CHA]``），
      供界面的 "Stream" 下拉直接显示——这样定位码/通道不必再单独做两个
      下拉框，所有 Trace 都在这一个列表里。
    - ``days``:    排序后的全部日期列表（YYYYMMDD）
    - ``index``:   ``{(key, day): 文件绝对路径}``，其中 key 同时登记了两种粒度：
      **Stream 标签**（``XA.S12.01.MHZ``，精确定位到某个 Trace 的文件）与
      **台站代码**（``S12``，同名日期下取首个扫到的文件）。前者优先，
      后者用于"只知道台站"的旧调用（命令行导出、修复库路径拼接）。

    兼容两种文件名格式：
    - 日文件：``XA.{station}.{day}.mseed``（如 ``XA.S12.19760113.mseed``）
    - 片段文件：``XA.{station}.{band}.{channel}.{start}-{end}.mseed``
      （day 取 start 字段前 8 位，如 ``XA.S12.01.MHZ.19760113_070151-...``）
    """
    streams, days, index = [], [], {}
    if not raw_root or not os.path.isdir(raw_root):
        return streams, days, index
    seen_streams = set()
    try:
        for root, _dirs, files in os.walk(raw_root):
            for fn in sorted(files):
                if not (fn.startswith("XA.") and fn.endswith(".mseed")):
                    continue
                parts = fn.split(".")
                if len(parts) < 4 or parts[-1] != "mseed":
                    continue
                net, stn, day, label = parts[0], parts[1], None, None
                if len(parts) == 4:
                    # 日文件 XA.ST.DAY.mseed —— 文件名不含定位码/通道
                    day = parts[2]
                    label = stream_label(net, stn)
                elif len(parts) >= 6:
                    # 片段 XA.ST.LOC.CH.START-END.mseed
                    day = parts[4][:8]
                    label = stream_label(net, stn, parts[2], parts[3])
                if not day or not day.isdigit():
                    continue
                if label and label not in seen_streams:
                    seen_streams.add(label)
                    streams.append(label)
                full = os.path.join(root, fn)
                # 精确 key 优先：同一 (台站, 日期) 同时存在日文件与片段文件时，
                # 选 Stream 标签能唯一定位到想看的那个文件
                index.setdefault((label, day), full)
                index.setdefault((stn, day), full)
                if day not in days:
                    days.append(day)
    except Exception as exc:  # pragma: no cover
        print("scan warn:", exc)
    streams.sort()
    days.sort()
    return streams, days, index


def resolve_paths(stream, day, raw_root, fix_root, index):
    """由索引解析原始/修复库实际文件路径；index 缺失时回退旧式根目录拼接。

    ``stream`` 既可以是 Stream 标签（``XA.S12.01.MHZ``），也可以是裸台站名
    （``S12``）——先按标签精确查找，找不到再退回到台站粒度。

    返回 (raw_path, fix_path, error_code)；error_code 为 None 表示成功。
    error_code: RAW_MISSING / FIX_MISSING / RAW_SEGMENT

    修复库路径解析顺序（按代价递增）：
    1. **同布局镜像**（``os.path.relpath(raw_path, raw_root)`` 拼到 fix_root 下）：
       历史约定，扁平目录 ↔ 扁平、``YYYY/MM/`` ↔ ``YYYY/MM/`` 时最快。
    2. **批量修复的 ``YYYY/MM/`` 镜像约定**：批量去异常 / CLI 都写到
       ``<fix_root>/YYYY/MM/<filename>``，与原始库的目录结构无关；当 raw
       是扁平目录时这一约定与 (1) 不一致，所以单独再试一次。
    3. **递归扫描**（兜底）：遍历 ``fix_root`` 找同名的 ``.mseed``，匹配
       ``parse_mseed_name`` 解析出的 ``(station, day)`` 都对得上才算；目录
       结构完全任意都能找到。

    为什么不能完全靠 (3)：examples/data 里有重复站名/日的日文件 + 片段文件，
    直接按 basename 抓可能撞错名；``station_of(stream)`` 一致 + 日匹配更稳。
    """
    station = station_of(stream)
    raw_path = None
    if index:
        raw_path = index.get((stream, day)) or index.get((station, day))
    if raw_path is None:
        # 回退：根目录直接拼接（兼容旧用法）
        guess = os.path.join(raw_root, f"XA.{station}.{day}.mseed")
        if os.path.exists(guess):
            raw_path = guess
        else:
            import glob as _glob
            hits = _glob.glob(os.path.join(raw_root, "**",
                                           f"XA.{station}.*.*.{day}_*.mseed"),
                              recursive=True)
            if hits:
                return hits[0], None, "RAW_SEGMENT"
            return raw_path, None, "RAW_MISSING"
    fix_path = find_counterpart(raw_path, raw_root, fix_root)
    if fix_path is None:
        # 全部约定都失败：把"我们以为应该在那儿"的路径也带回去，
        # 方便界面给可读错误（不会抛异常）。
        try:
            rel = os.path.relpath(raw_path, raw_root)
        except Exception:
            rel = os.path.basename(raw_path)
        fix_path = os.path.join(fix_root, rel)
        return raw_path, fix_path, "FIX_MISSING"
    return raw_path, fix_path, None


def render_view(station, day, start_min, win_min, mode, raw_root, fix_root,
                segments=None, raw_path=None, fix_path=None, figure=None,
                dark=False, highlight=None, hidden_methods=None, tr_fn=None,
                loc=None, cha=None, label=None):
    """渲染一帧对比图，返回 Figure 与信息 dict。mode: split/overlay/psd/amp/combo

    dark=True 时使用深色画布配色（跟随 DSSRR 深色主题）。
    highlight: 可选 dict（含绝对样本索引 s/e），在时域子图上金色底纹 +
    红色虚线边界高亮该段，画布窗口保持不变。
    hidden_methods: 可选 set，被隐藏的修复方法（其选框/曲线不绘制，
    由统计文本开关控制，所有画布模式生效）。
    tr_fn: 可选翻译函数（界面 tr），用于画布子图标题的国际化。
    loc / cha: 可选定位码与通道；给了就从文件里精确取该道（见 load_trace），
    不给则按惯例取 MHZ 道。
    label: 可选显示名（如 ``XA.S12.01.MHZ``）；只影响画布标题，不参与取数。
    """
    tr = tr_fn if callable(tr_fn) else (lambda s: s)
    hidden_methods = set(hidden_methods or ())
    from matplotlib import rcParams as _rc
    _rc["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    _rc["axes.unicode_minus"] = False
    pal = dict(
        raw="#9CA3AF" if dark else "0.4",
        fix="#60A5FA" if dark else "#1f77b4",
        seg_raw="#F87171" if dark else "r",
        seg_fix="#38BDF8" if dark else "b",
        residual="#F87171" if dark else "crimson",
        grid="#3D3D3D" if dark else "#E5E7EB",
    )
    if dark:
        _rc.update({
            "figure.facecolor": "#1F1F1F",
            "axes.facecolor": "#1F1F1F",
            "axes.edgecolor": "#5C5C5C",
            "axes.labelcolor": "#D4C5A9",
            "xtick.color": "#D4C5A9",
            "ytick.color": "#D4C5A9",
            "text.color": "#D4C5A9",
            "legend.facecolor": "#1F1F1F",
            "legend.edgecolor": "#3D3D3D",
            "legend.labelcolor": "#D4C5A9",
        })
    else:
        _rc.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "legend.facecolor": "white",
            "legend.edgecolor": "#CCCCCC",
            "legend.labelcolor": "#111111",
        })
    if segments is None:
        segments = {}
    if raw_path is None:
        raw_path = os.path.join(raw_root, f"XA.{station}.{day}.mseed")
        if not os.path.exists(raw_path):
            import glob as _glob
            hits = _glob.glob(os.path.join(raw_root, "**",
                                           f"XA.{station}.*.*.{day}_*.mseed"),
                              recursive=True)
            if hits:
                raise FileNotFoundError(
                    "RAW_SEGMENT: " + os.path.basename(hits[0]))
            raise FileNotFoundError("RAW_MISSING: " + raw_path)
    if fix_path is None:
        fix_path = os.path.join(fix_root, f"XA.{station}.{day}.mseed")
    if not os.path.exists(fix_path):
        raise FileNotFoundError("FIX_MISSING: " + fix_path)
    raw, t0 = load_trace(raw_path, loc, cha)
    fix, _ = load_trace(fix_path, loc, cha)
    if len(fix) != len(raw):
        fix = fix[:len(raw)]
    s0 = int(start_min * 60 * SR)
    # start 超出文件长度时 clamp 到文件末尾窗口，避免空画布（用户把
    # Start 拉到很大时仍能看到末尾数据，而不是一片空白）
    if s0 >= len(raw):
        s0 = max(0, len(raw) - int(win_min * 60 * SR))
    s1 = min(len(raw), int((start_min + win_min) * 60 * SR))
    if s0 >= s1:
        s1 = min(len(raw), s0 + int(60 * SR))
    raw_win = raw[s0:s1]
    fix_win = fix[s0:s1]
    t = np.arange(len(raw_win)) / SR / 60.0
    bucket = segments.get((station, day), {"repair": [], "event": [], "gap": []})
    repair = bucket.get("repair", [])
    event = bucket.get("event", [])
    gap = bucket.get("gap", [])

    def clip(segl):
        out = []
        for seg in segl:
            s, e = seg["s"], seg["e"]
            if s < s0 + len(raw_win) and e >= s0:
                d = dict(seg)
                d["s"] = max(s, s0) - s0
                d["e"] = min(e, s0 + len(raw_win) - 1) - s0
                out.append(d)
        return out

    rep_c, ev_c, gap_c = clip(repair), clip(event), clip(gap)
    # 统计文本开关隐藏的方法/类别：其选框/曲线不绘制（画布标题的窗口内
    # 段数随之按"实际显示段"统计；隐藏键可为 kind(repair/event/gap)
    # 或 method 名）
    if hidden_methods:
        rep_c = [seg for seg in rep_c
                 if seg.get("kind", "") not in hidden_methods
                 and seg.get("method", "") not in hidden_methods]
        ev_c = [seg for seg in ev_c
                if seg.get("kind", "") not in hidden_methods]
        gap_c = [seg for seg in gap_c
                 if seg.get("kind", "") not in hidden_methods]

    fig = figure if figure is not None else Figure(figsize=(12, 7), dpi=100)
    fig.clear()
    # figure 复用场景：显式同步 facecolor（figure 创建时取的 rcParams
    # 可能已被主窗口主题污染，且创建后不自动跟随 rcParams）
    fig.patch.set_facecolor("#1F1F1F" if dark else "white")
    fig.set_facecolor("#1F1F1F" if dark else "white")

    def span(ax, color, alpha):
        """为每个异常/事件/缺口段画"填充 + 左右边界竖线"。

        repair 段按修复方法着色（METHOD_COLORS，与明细表/统计文本一致），
        事件=绿、缺口=灰。边界竖线保证短段、密集段在波形下也清晰可见。
        """
        for seg in rep_c:
            c = METHOD_COLORS.get(seg.get("method", ""), DEFAULT_REPAIR_COLOR)
            x0, x1 = t[seg["s"]], t[seg["e"]]
            ax.axvspan(x0, x1, color=c, alpha=alpha, lw=0)
            if x1 > x0:
                ax.axvline(x0, color=c, lw=1.1, alpha=min(alpha + 0.5, 0.95))
                ax.axvline(x1, color=c, lw=1.1, alpha=min(alpha + 0.5, 0.95))
        for seg in ev_c:
            x0, x1 = t[seg["s"]], t[seg["e"]]
            ax.axvspan(x0, x1, color=EVENT_COLOR, alpha=0.35, lw=0)
            if x1 > x0:
                ax.axvline(x0, color=EVENT_COLOR, lw=1.1, alpha=0.8)
                ax.axvline(x1, color=EVENT_COLOR, lw=1.1, alpha=0.8)
        for seg in gap_c:
            x0, x1 = t[seg["s"]], t[seg["e"]]
            ax.axvspan(x0, x1, color=GAP_COLOR, alpha=0.5, lw=0)
            if x1 > x0:
                ax.axvline(x0, color=GAP_COLOR, lw=1.1, alpha=0.8)
                ax.axvline(x1, color=GAP_COLOR, lw=1.1, alpha=0.8)

    def _method_patches(seen=None):
        """窗口内各修复方法的图例 patch（按 method 去重）。"""
        from matplotlib.patches import Patch
        handles = []
        marks = set()
        for seg in rep_c:
            m = seg.get("method", "") or ""
            if m in marks:
                continue
            marks.add(m)
            handles.append(Patch(
                facecolor=METHOD_COLORS.get(m, DEFAULT_REPAIR_COLOR),
                alpha=0.3, label=method_name(m)))
        return handles

    def seg_legend(ax):
        """异常区图例：窗口内各修复方法色 + event/gap 色说明。"""
        from matplotlib.patches import Patch
        handles = _method_patches()
        if ev_c:
            handles.append(Patch(facecolor=EVENT_COLOR, alpha=0.35,
                                 label="event"))
        if gap_c:
            handles.append(Patch(facecolor=GAP_COLOR, alpha=0.5, label="gap"))
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8,
                      framealpha=0.85)

    def hl(ax):
        """在时域子图上高亮选中段：金色底纹 + 红色虚线边界（窗口不变）。

        金色底纹 zorder=1.5，盖过默认异常 span（zorder≈1）但低于数据曲线
        （zorder=2），保证高亮段在各类标注之上清晰可见。
        """
        if not highlight:
            return
        hs = max(highlight["s"], s0) - s0
        he = min(highlight["e"], s0 + len(raw_win) - 1) - s0
        if hs >= len(raw_win) or he < 0 or hs > he:
            return
        ax.axvspan(t[hs], t[he], color="#FFC107", alpha=0.35, lw=0, zorder=1.5)
        ax.axvline(t[hs], color="#F87171", lw=1.4, ls="--", zorder=3)
        ax.axvline(t[he], color="#F87171", lw=1.4, ls="--", zorder=3)

    total_seg = len(repair) + len(event) + len(gap)
    in_seg = len(rep_c) + len(ev_c) + len(gap_c)
    title = f"{label or station} {day} [{start_min}-{start_min + win_min} min]"
    if total_seg:
        # 提示窗口内/全天段数，避免"表格 19 段画布只有 2 个框"的误解
        title += f" · {tr('window segments')} {in_seg}/{total_seg}"

    if mode == "split":
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax1.plot(t, raw_win, lw=0.5, color=pal["raw"])
        ax1.set_ylabel("counts")
        ax1.set_title(f"{tr('Raw')} {t0.strftime('%H:%M')} UTC")
        ax2.plot(t, fix_win, lw=0.5, color=pal["fix"])
        ax2.set_ylabel("counts")
        ax2.set_xlabel("min")
        ax2.set_title(tr("Repaired"))
        span(ax1, "red", 0.25)
        span(ax2, "orange", 0.25)
        seg_legend(ax1)
        hl(ax1)
        hl(ax2)
        fig.suptitle(title)
    elif mode == "overlay":
        ax = fig.add_subplot(111)
        ax.plot(t, raw_win, lw=0.6, color=pal["raw"], label="raw")
        ax.plot(t, fix_win, lw=0.6, color=pal["fix"], label="DSSRR")
        span(ax, "red", 0.25)
        from matplotlib.patches import Patch
        _h, _l = ax.get_legend_handles_labels()
        _h += _method_patches()
        if ev_c:
            _h += [Patch(facecolor=EVENT_COLOR, alpha=0.35, label="event")]
        if gap_c:
            _h += [Patch(facecolor=GAP_COLOR, alpha=0.5, label="gap")]
        ax.legend(handles=_h, loc="upper right", fontsize=8)
        ax.set_xlabel("min")
        ax.set_ylabel("counts")
        ax.set_title(title)
        hl(ax)
    elif mode == "psd":
        ax = fig.add_subplot(111)
        fr, pr = psd(raw_win, SR)
        ff, pf = psd(fix_win, SR)
        ax.semilogy(fr, pr, color=pal["raw"], lw=1.2, label="raw PSD")
        ax.semilogy(ff, pf, color=pal["fix"], lw=1.2, label="DSSRR PSD")
        ax.set_xlabel("Hz")
        ax.set_ylabel("PSD")
        ax.legend(loc="upper right")
        ax.set_title(title + "  PSD")
    elif mode == "amp":
        ax = fig.add_subplot(111)
        fr, ar = amplitude_spectrum(raw_win, SR)
        ff, af = amplitude_spectrum(fix_win, SR)
        ax.plot(fr, ar, color=pal["raw"], lw=1.2, label="raw amplitude")
        ax.plot(ff, af, color=pal["fix"], lw=1.2, label="DSSRR amplitude")
        ax.set_xlabel("Hz")
        ax.set_ylabel("amplitude (counts)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")
        ax.set_title(title + "  Amplitude")
    else:
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, sharex=ax1)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224, sharex=ax1)
        ax1.plot(t, raw_win, lw=0.5, color=pal["raw"])
        ax1.set_title(tr("Raw"))
        ax2.plot(t, fix_win, lw=0.5, color=pal["fix"])
        ax2.set_title(tr("Repaired"))
        span(ax1, "red", 0.25)
        span(ax2, "orange", 0.25)
        seg_legend(ax1)
        hl(ax1)
        hl(ax2)
        fr, pr = psd(raw_win, SR)
        ff, pf = psd(fix_win, SR)
        ax3.semilogy(fr, pr, color=pal["raw"], lw=1.1)
        ax3.semilogy(ff, pf, color=pal["fix"], lw=1.1)
        ax3.set_title(tr("PSD"))
        ax3.set_xlabel("Hz")
        ax4.plot(t, raw_win - fix_win, lw=0.5, color=pal["residual"])
        ax4.set_title(tr("Residual"))
        ax4.set_xlabel("min")
        span(ax4, "red", 0.25)
        hl(ax4)
        fig.suptitle(title)
    fig.tight_layout()
    all_segs = sorted(rep_c + ev_c + gap_c, key=lambda x: x["s"])
    return fig, {"repair": len(rep_c), "event": len(ev_c), "gap": len(gap_c),
                 "segments": all_segs}


# ----------------------------------------------------------------------
# 工具栏：zoom/pan 后同步共享 x 轴的兄弟 axes（sharex 通常已联动，此处兜底）
# ----------------------------------------------------------------------
class _SyncNavToolbar(NavigationToolbar2QT):
    def _sync_shared_x(self, ax):
        try:
            x1 = ax.get_xlim()
            for a in self.canvas.figure.axes:
                if a is not ax and a.get_shared_x_axes().joined(a, ax):
                    a.set_xlim(x1)
        except Exception:
            pass

    def release_zoom(self, event):
        ax = event.inaxes
        if ax is None:
            return super().release_zoom(event)
        x0 = ax.get_xlim()
        super().release_zoom(event)
        if tuple(x0) != tuple(ax.get_xlim()):
            self._sync_shared_x(ax)

    def release_pan(self, event):
        ax = event.inaxes
        if ax is None:
            return super().release_pan(event)
        x0 = ax.get_xlim()
        super().release_pan(event)
        if tuple(x0) != tuple(ax.get_xlim()):
            self._sync_shared_x(ax)


# ----------------------------------------------------------------------
# GUI 窗口
# ----------------------------------------------------------------------
class DbRepairViewerWindow(QMainWindow):
    def __init__(self, initial_raw_root=None, parent=None, language="en_US"):
        super().__init__(parent)
        self.setWindowTitle(self.tr("DSSRR Repaired DB Viewer"))
        self.resize(1400, 900)
        self._language = language
        self._segments = {}
        self._index = {}
        self._highlight_seg = None  # 表格点击选中的段（画布高亮，不缩放窗口）
        self._hidden_methods = set()  # 统计文本开关隐藏的修复方法（画布不绘其选框）
        self._hidden_series = set()   # 频谱图例点击隐藏的曲线（raw/DSSRR 谱）
        # 主题状态必须先初始化：_sync_theme(force=True) 首次调用时会读取
        # self._theme / self._dark（此前未初始化 → AttributeError，深色模式崩溃）
        self._theme = {}
        self._dark = False
        self._canvas_bg = "white"
        self._canvas_fg = "#333333"
        self._sync_theme(force=True)
        self._build_ui()
        # 环境变量预置：命令行 / VS Code launch.json 里设 DSSRR_RAW_ROOT 与
        # DSSRR_FIX_ROOT 就能直接指向某个数据目录，不必每次在界面上点 Browse。
        # （此前只有"批量去异常"对话框读了这两个变量，查看器页没读，与本模块
        # 文件头的说明不一致——这里补上。）
        env_fix = os.environ.get("DSSRR_FIX_ROOT")
        if env_fix and os.path.isdir(env_fix):
            self.ed_fix.setText(env_fix)
        self.set_raw_root(initial_raw_root or os.environ.get("DSSRR_RAW_ROOT"))
        self.refresh()
        self._center_on_screen()
        # 主窗口切主题时的兜底轮询（广播入口见 _apply_style）
        self._theme_timer = QTimer(self)
        self._theme_timer.setInterval(600)
        self._theme_timer.timeout.connect(self._poll_host_theme)
        self._theme_timer.start()

    # ---------------- 主题跟随 ----------------
    def _apply_style(self):
        """主题切换时的广播入口（即时换肤）。"""
        try:
            self._sync_theme(force=True)
            self.refresh()
        except Exception:
            pass

    def _poll_host_theme(self):
        """兜底轮询：主窗口主题令牌变化时自动跟随（无需重新打开窗口）。"""
        theme = self._collect_theme()
        if theme and theme != self._theme:
            self._sync_theme(force=True)
            self.refresh()

    def _collect_theme(self):
        p = self.parent()
        if p is None:
            return {}
        return {k: getattr(p, k, None) for k in (
            "theme_bg", "theme_fg", "theme_surface1", "theme_surface2",
            "theme_accent", "theme_mantle", "theme_crust", "theme_panel")}

    def _sync_theme(self, force=False):
        """读取父窗口主题令牌；变化时重建窗口 QSS。"""
        theme = self._collect_theme()
        if theme == getattr(self, "_theme", {}) and not force:
            return
        self._theme = theme
        bg = theme.get("theme_bg")
        if bg:
            self._dark = QColor(bg).lightness() < 128
        else:
            # 拿不到主窗口主题令牌（主窗口主题尚未初始化 / 独立运行）：
            # 默认浅色。绝不按系统 palette 判断——Windows 深色应用模式下
            # palette 为深色，会把查看器误判成全黑（用户"刚打开软件就打开
            # 查看器"即踩此坑）；稍后主窗口主题就绪时由广播/轮询自动换肤。
            self._dark = False
        self.setStyleSheet(_qss(self._dark, theme))
        self._apply_canvas_theme()
        self._refresh_nav_icons()

    def _nav_icon(self, kind):
        """用 QPainter 矢量绘制天/月/年步进按钮图标，颜色随主题（theme_fg）。"""
        fg = self._theme.get("theme_fg") or ("#D4C5A9" if self._dark else "#333333")
        color = QColor(fg)
        pm = QPixmap(22, 22)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(color)

        def tri(cx, cy, s, right):
            if right:
                pts = [QPointF(cx - 0.55 * s, cy - s),
                       QPointF(cx - 0.55 * s, cy + s),
                       QPointF(cx + 0.8 * s, cy)]
            else:
                pts = [QPointF(cx + 0.55 * s, cy - s),
                       QPointF(cx + 0.55 * s, cy + s),
                       QPointF(cx - 0.8 * s, cy)]
            return QPolygonF(pts)

        if kind == "day_prev":
            p.drawPolygon(tri(10.5, 11, 4.6, False))
        elif kind == "day_next":
            p.drawPolygon(tri(11.5, 11, 4.6, True))
        elif kind == "month_prev":
            p.drawPolygon(tri(7.5, 11, 3.6, False))
            p.drawPolygon(tri(14.5, 11, 3.6, False))
        elif kind == "month_next":
            p.drawPolygon(tri(7.5, 11, 3.6, True))
            p.drawPolygon(tri(14.5, 11, 3.6, True))
        elif kind == "year_prev":
            p.drawRect(4, 5, 2, 12)   # 最左竖线
            p.drawPolygon(tri(9.2, 11, 3.4, False))
            p.drawPolygon(tri(15.6, 11, 3.4, False))
        else:  # year_next
            p.drawRect(16, 5, 2, 12)  # 最右竖线
            p.drawPolygon(tri(6.4, 11, 3.4, True))
            p.drawPolygon(tri(12.8, 11, 3.4, True))
        p.end()
        return QIcon(pm)

    def _refresh_nav_icons(self):
        """按当前主题重绘天/月/年步进按钮图标（按钮尚未创建时跳过）。"""
        for name in ("day_prev", "day_next", "month_prev", "month_next",
                     "year_prev", "year_next"):
            btn = getattr(self, "btn_" + name, None)
            if btn is not None:
                btn.setIcon(self._nav_icon(name))

    def _canvas_colors(self):
        if self._dark:
            return "#1F1F1F", "#D4C5A9"   # 画布背景, 文字
        return "white", "#333333"

    def _apply_canvas_theme(self):
        """显式把画布（figure/canvas 控件）背景同步到当前主题。

        figure 创建时的 facecolor 取的是当时全局 rcParams（可能被宿主
        深色主题污染），且创建后不会自动跟随 rcParams —— 这是
        "半黑半白"（窗口一个色、画布另一个色）的根因，必须显式设置。
        """
        bg, fg = self._canvas_colors()
        self._canvas_bg, self._canvas_fg = bg, fg
        fig = getattr(self, "figure", None)
        if fig is not None:
            fig.patch.set_facecolor(bg)
            fig.set_facecolor(bg)
        cv = getattr(self, "canvas", None)
        if cv is not None:
            cv.setStyleSheet(f"background-color: {bg};")
        nav = getattr(self, "_nav", None)
        if nav is not None:
            nav.setStyleSheet(_toolbar_qss(self._dark))
        # central widget 显式设背景：QMainWindow 的 background-color 在 central widget
        # 区域不绘制（Qt 限制），而 QWidget{background:transparent} 会让路径区（无
        # 容器的 QGridLayout）透出底层黑色。直接给 central widget 上色最稳。
        cw = self.centralWidget()
        if cw is not None:
            win_bg = self._theme.get("theme_bg") or ("#1A1A1A" if self._dark else "#EEEEEE")
            cw.setStyleSheet(f"background-color: {win_bg};")
        # 底部明细面板的标签显式着色（QGroupBox 内 QLabel 不继承全局 QSS 颜色）
        detail_labels = [getattr(self, n, None) for n in
                         ("lbl_stats", "lbl_det_time", "lbl_det_type",
                          "lbl_det_method", "lbl_det_len")]
        for lbl in detail_labels:
            if lbl is not None:
                lbl.setStyleSheet(f"color: {fg};")

    def _paint_placeholder(self, text):
        """加载失败/未就绪时在画布上显示主题色提示，避免空白色块割裂。"""
        try:
            fig = self.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor(self._canvas_bg)
            ax.text(0.5, 0.5, str(text), ha="center", va="center",
                    color=self._canvas_fg, fontsize=12,
                    transform=ax.transAxes, wrap=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            self.canvas.draw()
        except Exception:
            pass

    def _center_on_screen(self):
        """初始居中：优先主窗口中心，否则屏幕可用区中心（防右下角越界）。"""
        try:
            pw = self.parent().window() if self.parent() is not None else None
            if pw is not None and pw.isVisible():
                geo = pw.frameGeometry()
                if geo.isValid():
                    self.move(geo.center().x() - self.width() // 2,
                              geo.center().y() - self.height() // 2)
                    return
            scr = self.screen()
            if scr is not None:
                avail = scr.availableGeometry()
                if avail.isValid():
                    self.move(avail.center().x() - self.width() // 2,
                              avail.center().y() - self.height() // 2)
        except Exception:
            pass

    # ---------------- UI ----------------
    def _build_ui(self):
        self.setStyleSheet(_qss(self._dark, getattr(self, "_theme", {})))
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QVBoxLayout(cw)
        root.setSpacing(10)

        # ---- 路径区：两行，标签在上、控件在下 ----
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.lbl_raw = QLabel(self.tr("Raw DB"))
        grid.addWidget(self.lbl_raw, 0, 0)
        self.ed_raw = QLineEdit()
        self.ed_raw.setPlaceholderText(RAW_ROOT_HINT)
        self.btn_raw = QPushButton(self.tr("Browse"))
        self.btn_raw.clicked.connect(lambda: self._browse("raw"))
        row_raw = QHBoxLayout()
        row_raw.addWidget(self.ed_raw, 1)
        row_raw.addWidget(self.btn_raw)
        grid.addLayout(row_raw, 1, 0, 1, 2)

        self.lbl_fix = QLabel(self.tr("Repaired DB"))
        grid.addWidget(self.lbl_fix, 0, 2)
        self.ed_fix = QLineEdit()
        self.ed_fix.setPlaceholderText(FIX_ROOT_HINT)
        self.btn_fix = QPushButton(self.tr("Browse"))
        self.btn_fix.clicked.connect(lambda: self._browse("fix"))
        row_fix = QHBoxLayout()
        row_fix.addWidget(self.ed_fix, 1)
        row_fix.addWidget(self.btn_fix)
        grid.addLayout(row_fix, 1, 2, 1, 2)
        root.addLayout(grid)
        # ---- 参数区：台站/日期/起始/窗口，每列标签在上、控件在下 ----
        self.grp_sel = QGroupBox(self.tr("Data Selection"))
        # 参数区整体最小宽度：防止窗口拉窄时 Window 快捷按钮行被压缩裁切
        # （曾出现右侧控件只剩"将窗"残字）
        self.grp_sel.setMinimumWidth(720)
        g = QGridLayout(self.grp_sel)
        g.setHorizontalSpacing(28)
        g.setVerticalSpacing(3)
        g.setContentsMargins(8, 4, 8, 6)
        # "Stream" 列表：把每个 Trace 作为一项列出（NET.STA 或 NET.STA.LOC.CHA），
        # 选中即同时确定了台站 / 定位码 / 通道，因此不再需要单独的
        # Station / Location / Channel 三个下拉。
        self.lbl_station = QLabel(self.tr("Stream"))
        self.cb_station = QComboBox()
        self.cb_station.setToolTip(self.tr(
            "All traces found under the raw DB root.\n"
            "NET.STA (day files) or NET.STA.LOC.CHA (segment files)."))
        self.cb_station.currentIndexChanged.connect(self._on_stream_changed)
        self.lbl_day = QLabel(self.tr("Day"))
        self.cb_day = QComboBox()
        # 可编辑 + 自动补全：输入日期数字即可过滤（如输入 13 / 0113 / 19760113）
        self.cb_day.setEditable(True)
        self.cb_day.setInsertPolicy(QComboBox.NoInsert)
        # 下拉选择日期（或匹配项）即刷新；手动输入回车见下方 returnPressed。
        # 步进按钮内显式 refresh 与信号重复触发无害（幂等）。
        self.cb_day.currentIndexChanged.connect(lambda *_: self.refresh())
        self._day_completer = QCompleter(self.cb_day.model(), self)
        self._day_completer.setFilterMode(Qt.MatchContains)
        self._day_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self._day_completer.setCompletionMode(QCompleter.PopupCompletion)
        self.cb_day.setCompleter(self._day_completer)
        self._day_completer.activated.connect(
            lambda txt: (self.cb_day.setCurrentText(txt), self.refresh()))
        # 手动输入日期后按回车直接刷新
        self.cb_day.lineEdit().returnPressed.connect(self.refresh)
        self.lbl_start = QLabel(self.tr("Start (min)"))
        self.sp_start = QSpinBox()
        self.sp_start.setRange(0, MAX_START_MIN)
        self.sp_start.setValue(0)
        self.sp_start.setSingleStep(10)
        self.lbl_win = QLabel(self.tr("Window (min)"))
        self.sp_win = QSpinBox()
        self.sp_win.setRange(1, MAX_WIN_MIN)
        self.sp_win.setValue(MAX_WIN_MIN)
        self.sp_win.setSingleStep(30)
        for w in (self.cb_station, self.cb_day, self.sp_start, self.sp_win):
            w.setFixedHeight(22)
        for lbl in (self.lbl_station, self.lbl_day, self.lbl_start, self.lbl_win):
            lbl.setObjectName("fieldLabel")
        g.addWidget(self.lbl_station, 0, 0)
        g.addWidget(self.cb_station, 1, 0)
        g.addWidget(self.lbl_day, 0, 1)
        # Day 列：可编辑下拉 + 天/月/年三级快速切换（矢量图标，随主题变色）
        day_row = QHBoxLayout()
        day_row.setSpacing(3)
        self.btn_day_prev = QPushButton()
        self.btn_day_prev.setToolTip(self.tr("Previous day"))
        self.btn_day_prev.setFixedSize(24, 22)
        self.btn_day_prev.clicked.connect(lambda: self._step_day(-1))
        self.btn_day_next = QPushButton()
        self.btn_day_next.setToolTip(self.tr("Next day"))
        self.btn_day_next.setFixedSize(24, 22)
        self.btn_day_next.clicked.connect(lambda: self._step_day(1))
        self.btn_month_prev = QPushButton()
        self.btn_month_prev.setToolTip(self.tr("Previous month"))
        self.btn_month_prev.setFixedSize(28, 22)
        self.btn_month_prev.clicked.connect(lambda: self._step_month(-1))
        self.btn_month_next = QPushButton()
        self.btn_month_next.setToolTip(self.tr("Next month"))
        self.btn_month_next.setFixedSize(28, 22)
        self.btn_month_next.clicked.connect(lambda: self._step_month(1))
        self.btn_year_prev = QPushButton()
        self.btn_year_prev.setToolTip(self.tr("Previous year"))
        self.btn_year_prev.setFixedSize(28, 22)
        self.btn_year_prev.clicked.connect(lambda: self._step_year(-1))
        self.btn_year_next = QPushButton()
        self.btn_year_next.setToolTip(self.tr("Next year"))
        self.btn_year_next.setFixedSize(28, 22)
        self.btn_year_next.clicked.connect(lambda: self._step_year(1))
        day_row.addWidget(self.cb_day, 1)
        day_row.addWidget(self.btn_day_prev)
        day_row.addWidget(self.btn_day_next)
        day_row.addWidget(self.btn_month_prev)
        day_row.addWidget(self.btn_month_next)
        day_row.addWidget(self.btn_year_prev)
        day_row.addWidget(self.btn_year_next)
        self._refresh_nav_icons()
        g.addLayout(day_row, 1, 1)
        g.addWidget(self.lbl_start, 0, 2)
        g.addWidget(self.sp_start, 1, 2)
        g.addWidget(self.lbl_win, 0, 3)
        # Window 列：分钟输入框 + 常用时长快捷按钮（一键改窗口并刷新）
        win_row = QHBoxLayout()
        win_row.setSpacing(3)
        self._win_btns = []
        win_row.addWidget(self.sp_win)
        for label, val, tip in WIN_PRESETS:
            btn = QPushButton(self.tr(label))
            btn.setObjectName("preset")
            # 高度 26px：与 QSS 上下 padding 3px 配合，保证文字完整不裁底
            btn.setFixedHeight(26)
            # 按钮宽度按文字长度自适应，保证文字完整显示不截断
            btn.setMinimumWidth(44)
            btn.setToolTip(self.tr(tip))
            btn.clicked.connect(
                lambda _=False, v=val: self._set_win(v))
            win_row.addWidget(btn)
            self._win_btns.append(btn)
        win_row.addStretch(1)
        g.addLayout(win_row, 1, 3)
        for c in range(4):
            g.setColumnStretch(c, 1)
        root.addWidget(self.grp_sel)
        # ---- 操作与视图 ----
        # 整行用一个 FlowLayout：窗口够宽时排成一行（[View 选择器][6 个按钮]），
        # 变窄时按钮自动折到下一行。
        #
        # 为什么不用 QHBoxLayout：它会把行内所有子控件的宽度**求和**当作窗口的
        # 最小宽度（实测把整窗最小宽度顶到 2500 px 以上）。窗口一旦宽过屏幕，
        # Windows 就会在每次关闭模态对话框后重新"摆放"它——那正是主窗口莫名
        # 跳到屏幕底部的根源之一。FlowLayout 的最小宽度只取决于**单个**最宽的
        # 子控件，所以窗口最小宽度降到了 1000 px 以内。
        #
        # View 选择器整体作为一个 item 放进流式布局（而不是和按钮一起塞进一个
        # 嵌套布局）：这样流式布局会先按它的"单行宽度"给足空间，5 个单选按钮
        # 才会排成一行；否则它会被压窄，按钮竖着堆成一列。
        row_ops = FlowLayout(hspacing=10, vspacing=6)
        self.grp_view = QGroupBox(self.tr("View"))
        vh = FlowLayout(self.grp_view, hspacing=14, vspacing=2)
        vh.setContentsMargins(10, 4, 10, 6)
        self.rb = {}
        labels = {
            "split": self.tr("Time · Split"),
            "overlay": self.tr("Time · Overlay"),
            "psd": self.tr("Frequency · PSD"),
            "amp": self.tr("Frequency · Amplitude"),
            "combo": self.tr("Combo 2×2"),
        }
        for key in VIEW_MODES:
            rb = QRadioButton(labels[key])
            rb.toggled.connect(self.refresh)
            self.rb[key] = rb
            vh.addWidget(rb)
        # 注意：不在构建期间 setChecked——会触发 refresh，而此时 canvas/
        # tbl_detail 尚未创建，且 _apply_canvas_theme 里的 cw.setStyleSheet
        # 会在布局未完成时递归重算样式导致 segfault。移到 _build_ui 末尾。
        row_ops.addWidget(self.grp_view)

        self.btn_refresh = QPushButton(self.tr("Refresh"))
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_export = QPushButton(self.tr("Export PNG"))
        self.btn_export.clicked.connect(self.export_png)
        self.btn_manual = QPushButton(self.tr("Manual Repair"))
        self.btn_manual.setToolTip(
            self.tr("Manually re-repair a segment of the current file "
                    "with real-time preview"))
        self.btn_manual.clicked.connect(self.open_manual_repair)
        self.btn_batch = QPushButton(self.tr("Batch Repair"))
        self.btn_batch.setToolTip(
            self.tr("Batch de-anomaly: select raw/fix roots, stations and "
                    "date range, pre-check the data, then run the full "
                    "pipeline with progress and auto verification"))
        self.btn_batch.clicked.connect(self.open_batch_repair)
        self.btn_set = QPushButton(self.tr("Settings"))
        self.btn_set.setToolTip(
            self.tr("Configure anomaly-detection and repair parameters "
                    "(applied to manual re-processing)"))
        self.btn_set.clicked.connect(self.open_repair_settings)
        self.btn_help = QPushButton(self.tr("Help"))
        self.btn_help.setToolTip(self.tr("Open the viewer and parameter help"))
        self.btn_help.clicked.connect(self.open_repair_help)
        row_ops.addWidget(self.btn_refresh)
        row_ops.addWidget(self.btn_export)
        row_ops.addWidget(self.btn_manual)
        row_ops.addWidget(self.btn_batch)
        row_ops.addWidget(self.btn_set)
        row_ops.addWidget(self.btn_help)
        root.addLayout(row_ops)
        # ---- 画布 + 工具栏 ----
        self.figure = Figure(figsize=(12, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(280)
        self.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self._nav = _SyncNavToolbar(self.canvas, self)
        self.addToolBar(self._nav)
        # matplotlib 工具栏的按钮名/提示是第三方硬编码英文，不走 self.tr()，
        # 必须在创建后手动本地化（中文界面下才会被替换）。
        self._localize_nav_toolbar()

        # ---- 底部修复明细面板（统计概览 + 段列表 + 选中段详情，与时间轴同步）----
        self._current_segments = []
        self.grp_detail = QGroupBox(self.tr("Repair Details"))
        self.grp_detail.setMinimumHeight(110)
        dl = QVBoxLayout(self.grp_detail)
        dl.setContentsMargins(6, 2, 6, 4)
        dl.setSpacing(3)

        # 统计概览栏：当前窗口各类异常/修复方法计数（每项可点击切换
        # 该颜色选框在画布上的显示/隐藏）
        self.lbl_stats = QLabel("")
        self.lbl_stats.setObjectName("detailStats")
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setTextFormat(Qt.RichText)
        self.lbl_stats.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.lbl_stats.linkActivated.connect(self._on_stats_link)
        dl.addWidget(self.lbl_stats)

        # 中段：左侧段列表表格 + 右侧选中段详情卡片
        mid = QHBoxLayout()
        mid.setSpacing(8)

        self.tbl_detail = QTableWidget(0, 6)
        self.tbl_detail.setHorizontalHeaderLabels([
            self.tr("#"), self.tr("Start (min)"), self.tr("End (min)"),
            self.tr("Len (s)"), self.tr("Anomaly"), self.tr("Method")])
        self.tbl_detail.horizontalHeader().setStretchLastSection(True)
        self.tbl_detail.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        # ResizeToContents 会让表格的"建议最小宽度"等于所有列内容之和（实测
        # 约 1200 px），把整窗最小宽度顶到屏幕之外。显式给一个较小的最小宽度，
        # 布局就可以收缩，列放不下时由表格自己出横向滚动条。
        self.tbl_detail.setMinimumWidth(320)
        self.tbl_detail.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_detail.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_detail.verticalHeader().setDefaultSectionSize(20)
        self.tbl_detail.verticalHeader().setVisible(False)
        self.tbl_detail.cellClicked.connect(self._on_detail_clicked)
        self.tbl_detail.itemSelectionChanged.connect(self._on_detail_selection_changed)
        mid.addWidget(self.tbl_detail, 3)

        # 右侧选中段详情卡片
        self.grp_detail_card = QGroupBox(self.tr("Selected Segment"))
        dc = QVBoxLayout(self.grp_detail_card)
        dc.setContentsMargins(8, 4, 8, 4)
        dc.setSpacing(2)
        self.lbl_det_time = QLabel("—")
        self.lbl_det_type = QLabel("—")
        self.lbl_det_method = QLabel("—")
        self.lbl_det_len = QLabel("—")
        for lbl in (self.lbl_det_time, self.lbl_det_type, self.lbl_det_method, self.lbl_det_len):
            lbl.setWordWrap(True)
            dc.addWidget(lbl)
        dc.addStretch(1)
        mid.addWidget(self.grp_detail_card, 2)

        dl.addLayout(mid)

        # 画布与修复明细之间用可拖动分割条（初始明细约 180px，可上下拖大拖小）
        self._main_splitter = QSplitter(Qt.Vertical)
        self._main_splitter.addWidget(self.canvas)
        self._main_splitter.addWidget(self.grp_detail)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 0)
        self._main_splitter.setSizes([660, 180])
        root.addWidget(self._main_splitter)

        self._apply_canvas_theme()
        self.statusBar().showMessage(self.tr("Ready"))
        # 所有控件创建完毕后再设默认选中，触发 refresh 安全
        self.rb["split"].setChecked(True)
    # ---------------- 数据 ----------------
    def _browse(self, which):
        start = (self.ed_raw.text() if which == "raw"
                 else self.ed_fix.text()).strip()
        path = run_modal_dialog(
            self,
            lambda host: QFileDialog.getExistingDirectory(
                host, self.tr("Select folder"), start))
        if not path:
            return
        if which == "raw":
            self.ed_raw.setText(path)
        else:
            self.ed_fix.setText(path)
        self._reload_index()
        self.refresh()   # 选完路径立即加载，避免停在 EMPTY 提示

    def set_raw_root(self, raw_root):
        """设置原始库路径。

        修复库的取值优先级：界面上已有的值（含 DSSRR_FIX_ROOT 预置）→ 由原
        路径推导出的 ``<raw>_repaired``（存在时）→ 保持为空由用户自己选。
        已有值不会被推导结果覆盖，否则 DSSRR_FIX_ROOT 预置会被静默丢弃。
        """
        if raw_root:
            self.ed_raw.setText(str(raw_root))
            if not self.ed_fix.text().strip():
                fix_guess = str(raw_root).rstrip("\\/") + "_repaired"
                if os.path.isdir(fix_guess):
                    self.ed_fix.setText(fix_guess)
        self._reload_index()
        if raw_root:
            self.refresh()   # __init__ 传 None 时由调用方统一 refresh

    def _reload_index(self):
        raw_root = self.ed_raw.text().strip()
        streams, days, index = scan_days(raw_root)
        self._index = index
        self._all_days = days
        self.cb_station.blockSignals(True)
        self.cb_station.clear()
        self.cb_station.addItems(streams or STREAMS_DEFAULT)
        self.cb_station.blockSignals(False)
        self._reload_days()
        # 目录标注（E7_catalog_*.csv）：优先修复库 catalogs/，回退随包数据
        self._segments = {}
        fix_root = self.ed_fix.text().strip()
        self._catalog_paths = _find_catalog_csvs(fix_root)
        self._segments = load_segments(self._catalog_paths)

    def _reload_days(self):
        """按当前 Stream 所属台站过滤日期，默认选中最新一天。"""
        station = self._current_station()
        # index 同时登记了 Stream 标签键（形如 XA.S12.01.MHZ）与裸台站键
        # （形如 S12）；这里只要台站粒度的日期，故排除含 "." 的标签键。
        days = sorted(d for (st, d) in self._index
                      if "." not in st and st == station)
        self._station_days = days
        self._highlight_seg = None  # 换台站/日期后清除旧高亮
        self.cb_day.blockSignals(True)
        self.cb_day.clear()
        self.cb_day.addItems(days)
        if days:
            self.cb_day.setCurrentIndex(len(days) - 1)  # 最新日期
        self.cb_day.blockSignals(False)

    def _set_win(self, minutes):
        """快捷设置时间窗宽度（1h/12h/1d）并自动刷新。"""
        self.sp_win.setValue(minutes)
        self._highlight_seg = None
        self.refresh()

    def _step_day(self, step):
        """前一天/后一天快速切换（在当前台站的日期列表内）。"""
        days = getattr(self, "_station_days", None)
        if not days:
            return
        cur = self.cb_day.currentText()
        if cur in days:
            i = days.index(cur)
            j = i + step
        else:
            # 当前文本不在列表（手动输入）→ 按数值就近取
            try:
                n = int(cur)
                cands = [d for d in days if int(d) >= n] if step > 0 \
                    else [d for d in days if int(d) <= n]
                j = days.index(min(cands)) if step > 0 else days.index(max(cands))
            except (ValueError, IndexError):
                j = len(days) - 1 if step > 0 else 0
        if 0 <= j < len(days):
            self.cb_day.setCurrentText(days[j])
            self._highlight_seg = None
            self.refresh()

    def _step_month(self, step):
        """上/下月快速切换：优先同号日，否则该月最新可用日。"""
        days = getattr(self, "_station_days", None)
        if not days:
            return
        cur = self.cb_day.currentText()
        try:
            y, m, d = int(cur[0:4]), int(cur[4:6]), int(cur[6:8])
        except (ValueError, IndexError):
            return
        ym = y * 12 + (m - 1) + step
        ty, tm = divmod(ym, 12)
        target = "%04d%02d" % (ty, tm + 1)
        cands = [x for x in days if x.startswith(target)]
        if not cands:
            return
        pref = target + "%02d" % d
        pick = pref if pref in cands else cands[-1]
        if pick != cur:
            self.cb_day.setCurrentText(pick)
            self._highlight_seg = None
            self.refresh()

    def _step_year(self, step):
        """上/下年快速切换：优先同月同日，否则该年最新可用日。"""
        days = getattr(self, "_station_days", None)
        if not days:
            return
        cur = self.cb_day.currentText()
        try:
            y, m, d = int(cur[0:4]), int(cur[4:6]), int(cur[6:8])
        except (ValueError, IndexError):
            return
        target = "%04d" % (y + step)
        cands = [x for x in days if x.startswith(target)]
        if not cands:
            return
        pref = target + "%02d%02d" % (m, d)
        pick = pref if pref in cands else cands[-1]
        if pick != cur:
            self.cb_day.setCurrentText(pick)
            self._highlight_seg = None
            self.refresh()

    def _on_stream_changed(self, *_):
        """Stream 下拉切换：重算该台站的日期列表并刷新画布。"""
        self._reload_days()
        self.refresh()

    # ---------------- 当前 Stream ----------------
    def _current_stream_label(self):
        """当前选中的 Stream 标签（如 ``XA.S12.01.MHZ``）。"""
        return self.cb_station.currentText().strip()

    def _current_station(self):
        """当前 Stream 对应的台站代码（``XA.S12.01.MHZ`` → ``S12``）。

        文件索引、修复库路径拼接、catalog 查找都以台站为键，所以界面上的
        Stream 选择最终都要归约到台站。
        """
        return station_of(self._current_stream_label())

    def _current_loc_cha(self):
        """当前 Stream 的 (定位码, 通道)；日文件命名下两者都是空串。"""
        _net, _sta, loc, cha = split_stream_label(self._current_stream_label())
        return loc, cha

    def _mode(self):
        for key, rb in self.rb.items():
            if rb.isChecked():
                return key
        return "split"

    def _paths(self):
        """返回 (raw_path, fix_path, err)"""
        raw_root = self.ed_raw.text().strip()
        fix_root = self.ed_fix.text().strip()
        stream = self._current_stream_label()
        day = self.cb_day.currentText()
        if not raw_root or not fix_root or not stream or not day:
            return None, None, "EMPTY"
        raw_path, fix_path, err = resolve_paths(
            stream, day, raw_root, fix_root, self._index)
        return raw_path, fix_path, err

    def refresh(self):
        self._sync_theme()
        # 无条件同步画布颜色：_sync_theme 在主题未变化时会提前返回，
        # 但画布可能因先前的深色窗口/渲染残留而失配（浅色模式黑背景问题）
        self._apply_canvas_theme()
        # 重新加载 catalog CSV（批量修复可能写了新明细）
        try:
            fix_root = self.ed_fix.text().strip()
            if fix_root:
                paths = _find_catalog_csvs(fix_root)
                self._segments = load_segments(paths)
        except Exception:
            pass
        raw_path, fix_path, err = self._paths()
        if err == "EMPTY":
            self._paint_placeholder(
                self.tr("Select raw/repaired DB and station/day"))
            self.statusBar().showMessage(
                self.tr("Select raw/repaired DB and station/day"))
            self._current_segments = []
            tbl = getattr(self, "tbl_detail", None)
            if tbl is not None:
                tbl.setRowCount(0)
            return
        if err:
            self._paint_placeholder(
                self.tr("No data loaded") + " — " +
                (self._path_error_text(err, raw_path, fix_path)))
            self._show_path_error(err, raw_path, fix_path)
            self._current_segments = []
            tbl = getattr(self, "tbl_detail", None)
            if tbl is not None:
                tbl.setRowCount(0)
            return
        loc, cha = self._current_loc_cha()
        try:
            fig, info = render_view(
                self._current_station(), self.cb_day.currentText(),
                self.sp_start.value(), self.sp_win.value(), self._mode(),
                self.ed_raw.text().strip(), self.ed_fix.text().strip(),
                self._segments, raw_path=raw_path, fix_path=fix_path,
                figure=self.figure, dark=self._dark,
                highlight=self._highlight_seg,
                hidden_methods=self._hidden_methods, tr_fn=self.tr,
                loc=loc, cha=cha, label=self._current_stream_label())
        except Exception as exc:
            self.statusBar().showMessage(f"{self.tr('Error')}: {exc}")
            return
        self._apply_series_visibility()
        self.canvas.draw()
        self._fill_detail_table(info)
        self.statusBar().showMessage(
            f"{self._current_stream_label()} {self.cb_day.currentText()} "
            f"[{self.sp_start.value()}-{self.sp_start.value() + self.sp_win.value()} min]  "
            f"{self.tr('repair')}={info['repair']}  "
            f"{self.tr('event')}={info['event']}  "
            f"{self.tr('gap')}={info['gap']}")

    def _redraw(self):
        """仅重绘画布（保持当前窗口不变），用于表格选中段高亮。

        render_view 会重建 axes 并把 x 轴重置为整窗——这里在渲染前记录
        各子图当前 xlim（用户 zoom/pan 后的视口），渲染后逐子图恢复，
        使"点击明细行更新高亮"不改变当前框选范围。
        """
        self._sync_theme()
        self._apply_canvas_theme()
        raw_path, fix_path, err = self._paths()
        if err:
            return
        saved_xlims = []
        for ax in self.canvas.figure.axes:
            try:
                saved_xlims.append(ax.get_xlim())
            except Exception:
                saved_xlims.append(None)
        loc, cha = self._current_loc_cha()
        try:
            fig, info = render_view(
                self._current_station(), self.cb_day.currentText(),
                self.sp_start.value(), self.sp_win.value(), self._mode(),
                self.ed_raw.text().strip(), self.ed_fix.text().strip(),
                self._segments, raw_path=raw_path, fix_path=fix_path,
                figure=self.figure, dark=self._dark,
                highlight=self._highlight_seg,
                hidden_methods=self._hidden_methods, tr_fn=self.tr,
                loc=loc, cha=cha, label=self._current_stream_label())
        except Exception as exc:
            self.statusBar().showMessage(f"{self.tr('Error')}: {exc}")
            return
        # 恢复用户 zoom/pan 后的 x 轴视口（子图数量/结构在两次调用间不变，
        # 按序恢复；数量变化则放弃恢复，回落到整窗）
        new_axes = self.canvas.figure.axes
        if len(new_axes) == len(saved_xlims):
            for ax, lim in zip(new_axes, saved_xlims):
                if lim is not None:
                    try:
                        ax.set_xlim(lim)
                    except Exception:
                        pass
        self._apply_series_visibility()
        self.canvas.draw()
        self.statusBar().showMessage(
            f"{self._current_stream_label()} {self.cb_day.currentText()} "
            f"[{self.sp_start.value()}-{self.sp_start.value() + self.sp_win.value()} min]  "
            f"{self.tr('repair')}={info['repair']}  "
            f"{self.tr('event')}={info['event']}  "
            f"{self.tr('gap')}={info['gap']}")

    def _fill_detail_table(self, info):
        """用当天全量段明细填充底部面板：统计概览 + 段列表 + 自适应高度。

        表格始终显示该台站/日期全天的所有异常段（不随画布时间窗过滤），
        保证点击任意段定位画布后，其他段依然可见可点。
        """
        station = self._current_station()
        day = self.cb_day.currentText()
        bucket = self._segments.get((station, day), {"repair": [], "event": [], "gap": []})
        segs = sorted(
            list(bucket.get("repair", [])) + list(bucket.get("event", [])) +
            list(bucket.get("gap", [])),
            key=lambda s: s["s"])
        self._current_segments = segs
        self.tbl_detail.setRowCount(len(segs))
        base_min = 0  # 全量段的分钟使用当天绝对位置（0 起点），不随窗口漂移

        # 统计：按 kind 和 method 分组计数；每项为可点击链接（点击切换
        # 该颜色选框在画布上的显示/隐藏），颜色与画布选框一致；已隐藏
        # 的项显示为灰色。
        def _cs(text, color, key=None):
            if key is None:
                return f'<span style="color:{color};">{text}</span>'
            if key in self._hidden_methods:
                return (f'<a href="toggle:{key}" style="color:#9CA3AF;'
                        f'text-decoration:none;">{text}</a>')
            return (f'<a href="toggle:{key}" style="color:{color};'
                    f'text-decoration:none;">{text}</a>')
        from collections import Counter
        kind_cnt = Counter(seg.get("kind", "") for seg in segs)
        method_cnt = Counter(seg.get("method", "") for seg in segs)
        stats_parts = []
        if kind_cnt.get("repair"):
            stats_parts.append(_cs(
                f"{kind_name('repair')} {kind_cnt['repair']}",
                DEFAULT_REPAIR_COLOR, "repair"))
        if kind_cnt.get("event"):
            stats_parts.append(_cs(
                f"{kind_name('event')} {kind_cnt['event']}", EVENT_COLOR, "event"))
        if kind_cnt.get("gap"):
            stats_parts.append(_cs(
                f"{kind_name('gap')} {kind_cnt['gap']}", GAP_COLOR, "gap"))
        method_str = "  |  ".join(
            _cs(f"{method_name(m)} {c}",
                METHOD_COLORS.get(m, DEFAULT_REPAIR_COLOR), m)
            for m, c in method_cnt.most_common())
        if method_str:
            stats_parts.append(method_str)
        total = len(segs)
        prefix = f"{self.tr('All day')}: {total} {self.tr('segments')}"
        self.lbl_stats.setText(
            (prefix + "  —  " + "  |  ".join(stats_parts)) if stats_parts else prefix)

        for i, seg in enumerate(segs):
            start_min = base_min + seg["s"] / SR / 60.0
            end_min = base_min + seg["e"] / SR / 60.0
            kind = seg.get("kind", "")
            anomaly = kind_name(kind) if kind in KIND_EN else (
                seg.get("subtype") or kind)
            method = method_name(seg.get("method", ""))
            items = [
                QTableWidgetItem(str(i + 1)),
                QTableWidgetItem(f"{start_min:.1f}"),
                QTableWidgetItem(f"{end_min:.1f}"),
                QTableWidgetItem(f"{seg.get('len_s', 0):.1f}"),
                QTableWidgetItem(anomaly),
                QTableWidgetItem(method),
            ]
            # 异常类型列(4)/方法列(5)按 kind/method 着色，与画布选框、
            # 统计文本颜色一致
            items[4].setForeground(QColor(
                EVENT_COLOR if kind == "event" else
                GAP_COLOR if kind == "gap" else DEFAULT_REPAIR_COLOR))
            items[5].setForeground(QColor(
                METHOD_COLORS.get(seg.get("method", ""),
                                  DEFAULT_REPAIR_COLOR)))
            for j, it in enumerate(items):
                it.setToolTip(
                    f"{seg.get('start_time', '')} ~ {seg.get('end_time', '')}\n"
                    f"kind={seg.get('kind', '')}  subtype={seg.get('subtype', '')}")
                self.tbl_detail.setItem(i, j, it)

        # 表格高度交给主 splitter 控制（用户可上下拖动分割条查看全部明细），
        # 只保留最小高度防止折叠到不可用
        self.tbl_detail.setMinimumHeight(56)

        # 仅在无任何段时清空详情卡片；有段时保留/恢复选中行的详情
        if not segs:
            self._update_detail_card(None)

    def _on_detail_selection_changed(self):
        """表格选中行变化 → 更新右侧详情卡片。"""
        rows = self.tbl_detail.selectionModel().selectedRows()
        if not rows:
            self._update_detail_card(None)
            return
        row = rows[0].row()
        if 0 <= row < len(self._current_segments):
            self._update_detail_card(self._current_segments[row])

    def _on_hover(self, event):
        """鼠标在画布上移动时，若停在某个异常段选框内，显示 tooltip。"""
        if event.inaxes is None or not getattr(self, "_current_segments", None):
            return
        x_min = event.xdata
        if x_min is None:
            return
        # x 轴单位是分钟
        for i, seg in enumerate(self._current_segments):
            s_min = seg["s"] / SR / 60.0
            e_min = seg["e"] / SR / 60.0
            if s_min <= x_min <= e_min:
                kind = seg.get("kind", "")
                method = method_name(seg.get("method", ""))
                kind_zh = kind_name(kind) if kind in KIND_EN else kind
                tip = (f"#{i+1}  {s_min:.1f}–{e_min:.1f} min  "
                       f"({seg.get('len_s',0):.1f} s)\n"
                       f"{kind_zh} · {method}")
                from PyQt5.QtWidgets import QToolTip
                QToolTip.showText(
                    self.canvas.mapToGlobal(
                        self.canvas.mapFromGlobal(
                            self.cursor().pos())),
                    tip, self.canvas)
                return
        from PyQt5.QtWidgets import QToolTip
        QToolTip.hideText()

    def _update_detail_card(self, seg):
        """更新右侧选中段详情卡片（分钟使用当天绝对位置）。"""
        if seg is None:
            self.lbl_det_time.setText("—")
            self.lbl_det_type.setText("—")
            self.lbl_det_method.setText("—")
            self.lbl_det_len.setText("—")
            return
        base_min = 0
        start_min = base_min + seg["s"] / SR / 60.0
        end_min = base_min + seg["e"] / SR / 60.0
        st = seg.get("start_time", "").replace("T", " ").replace("Z", " UTC")
        et = seg.get("end_time", "").replace("T", " ").replace("Z", " UTC")
        self.lbl_det_time.setText(
            f"{self.tr('Time')}: {st} ~ {et}")
        self.lbl_det_type.setText(
            f"{self.tr('Anomaly')}: "
            f"{kind_name(seg.get('kind', '—')) if seg.get('kind') in KIND_EN else seg.get('kind', '—')}"
            f"  ({method_name(seg.get('method', '—'))})")
        self.lbl_det_method.setText(
            f"{self.tr('Method')}: "
            f"{method_name(seg.get('method', '—')) if seg.get('method') else '—'}")
        self.lbl_det_len.setText(
            f"{self.tr('Length')}: {seg.get('len_s', 0):.1f} s"
            f"  ({start_min:.1f}–{end_min:.1f} min)")

    def _on_stats_link(self, href):
        """统计文本链接：切换某修复方法/类别的画布选框显示/隐藏。"""
        if not href.startswith("toggle:"):
            return
        key = href[len("toggle:"):]
        if key in self._hidden_methods:
            self._hidden_methods.discard(key)
        else:
            self._hidden_methods.add(key)
        self.refresh()

    def _apply_series_visibility(self):
        """渲染后应用频谱图例隐藏状态：为图例曲线启用点击(pick)，并按
        self._hidden_series 隐藏对应数据曲线（raw/DSSRR 谱）。"""
        for ax in self.canvas.figure.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            for ln in leg.get_lines():
                ln.set_picker(5)
            for ln in ax.get_lines():
                if ln.get_label() in self._hidden_series:
                    ln.set_visible(False)

    def _on_legend_pick(self, event):
        """点击频谱图例曲线：切换该曲线的显示/隐藏（仅重绘，不重建）。"""
        artist = getattr(event, "artist", None)
        if artist is None:
            return
        label = artist.get_label()
        if not label or label.startswith("_"):
            return
        if label in self._hidden_series:
            self._hidden_series.discard(label)
        else:
            self._hidden_series.add(label)
        vis = label not in self._hidden_series
        for ax in self.canvas.figure.axes:
            for ln in ax.get_lines():
                if ln.get_label() == label:
                    ln.set_visible(vis)
        self.canvas.draw_idle()

    def _current_view(self):
        """当前实际视口（当天绝对分钟）。

        优先取 matplotlib 时间轴子图的 xlim（用户用工具栏 zoom/pan 后
        的实际视口，_redraw 会恢复它）；PSD/振幅模式无时间轴子图时
        退化为 spinbox 窗口。x 轴为窗口相对分钟（t = i/SR/60），
        因此视口绝对分钟 = sp_start + xlim。
        """
        base = self.sp_start.value()
        time_lims = []
        for ax in self.canvas.figure.axes:
            try:
                yl = (ax.get_ylabel() or "").lower()
            except Exception:
                yl = ""
            if "psd" in yl or "amplitude" in yl:
                continue  # 频域子图（x 为 Hz，不是时间轴）
            try:
                x0, x1 = ax.get_xlim()
            except Exception:
                continue
            time_lims.append((x0, x1))
        if time_lims:
            return (base + min(t[0] for t in time_lims),
                    base + max(t[1] for t in time_lims))
        return base, base + self.sp_win.value()

    def _on_detail_clicked(self, row, col):
        """点击明细行 → 画布保持当前窗口，仅高亮该段 + 更新详情卡片。

        若该段完全在当前实际视口之外（含任一方向越界），自动扩展窗口到
        [当前视口 ∪ 点击段外侧取整到小时]，使原放大区域与点击段同时
        可见，边界保留一小时级冗余：如当前显示 9:00-10:00、点击段
        14:00-14:23 → 扩展为 9:00-15:00；点击段 5:30-6:00 →
        扩展为 5:00-10:00（点击段外侧均取整到整点）。

        当前视口优先取 matplotlib 实际 xlim（用户工具栏 zoom/pan 后的
        视口），避免"spinbox 仍是全天、实际视口已放大"时扩展失效。
        """
        if row < 0 or row >= len(self._current_segments):
            return
        seg = self._current_segments[row]
        cur_start, cur_end = self._current_view()
        s_min = seg["s"] / SR / 60.0
        e_min = seg["e"] / SR / 60.0
        if s_min < cur_start or e_min > cur_end:
            new_start = (int(s_min // 60.0) * 60
                         if s_min < cur_start else int(cur_start))
            new_end = (int(np.ceil(e_min / 60.0)) * 60
                       if e_min > cur_end else int(np.ceil(cur_end)))
            new_start = max(0, new_start)
            new_end = min(1440, new_end)
            if new_end - new_start < 1:
                new_end = min(1440, new_start + 1)
            self.sp_start.setValue(new_start)
            self.sp_win.setValue(new_end - new_start)
            self._highlight_seg = seg
            self._update_detail_card(seg)
            self.refresh()
            return
        self._highlight_seg = seg
        self._update_detail_card(seg)
        self._redraw()

    def _path_error_text(self, err, raw_path, fix_path):
        if err == "RAW_SEGMENT":
            return self.tr("Raw DB is a segment-file library for this day "
                           "(auto-merge not supported yet). Please point Raw DB "
                           "at a day-file database matching the repaired DB")
        if err == "RAW_MISSING":
            return self.tr("Raw DB has no day file for this station/day") + \
                "  " + str(raw_path)
        if err == "FIX_MISSING":
            return self.tr("Repaired DB has no day file for this station/day "
                           "(not repaired yet)") + "  " + str(fix_path)
        return str(err)

    def _show_path_error(self, err, raw_path, fix_path):
        self.statusBar().showMessage(self._path_error_text(err, raw_path, fix_path))

    def export_png(self):
        raw_path, fix_path, err = self._paths()
        if err == "EMPTY":
            return
        if err:
            self._show_path_error(err, raw_path, fix_path)
            return
        out, _ = run_modal_dialog(
            self,
            lambda host: QFileDialog.getSaveFileName(
                host, self.tr("Export PNG"),
                f"{self._current_station()}_{self.cb_day.currentText()}_compare.png",
                "PNG (*.png)"))
        if not out:
            return
        loc, cha = self._current_loc_cha()
        try:
            fig, _ = render_view(
                self._current_station(), self.cb_day.currentText(),
                self.sp_start.value(), self.sp_win.value(), self._mode(),
                self.ed_raw.text().strip(), self.ed_fix.text().strip(),
                self._segments, raw_path=raw_path, fix_path=fix_path,
                dark=self._dark, loc=loc, cha=cha,
                label=self._current_stream_label())
            fig.savefig(out, dpi=100)
            self.statusBar().showMessage(f"{self.tr('Exported')}: {out}")
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Error"), str(exc))

    def open_manual_repair(self):
        """打开手动重处理对话框：对当前文件的选中段/指定段重新去异常。"""
        from .manual_repair import ManualRepairDialog
        raw_path, fix_path, err = self._paths()
        if err == "EMPTY":
            self.statusBar().showMessage(
                self.tr("Select raw/repaired DB and station/day"))
            return
        if err:
            self._show_path_error(err, raw_path, fix_path)
            return
        seg = self._highlight_seg
        dlg = ManualRepairDialog(
            self, self._current_station(), self.cb_day.currentText(),
            raw_path, fix_path, default_seg=seg)
        dlg.exec_()

    def open_repair_settings(self):
        """打开去异常参数设置对话框：保存后应用到手动重处理（auto 模式）。"""
        dlg = RepairSettingsDialog(self)
        dlg.exec_()

    def open_batch_repair(self):
        """打开批量去异常对话框：导入数据/月震目录、检查数据、批量修复、
        自动验收（复用 E7 修复流水线，与论文口径一致）。"""
        from .batch_repair import BatchRepairDialog
        dlg = BatchRepairDialog(self)
        dlg.exec_()

    def open_repair_help(self):
        """打开查看器使用帮助（快速上手 + 参数说明 + 图例/方法/验收）。"""
        dlg = RepairHelpDialog(self)
        dlg.exec_()

    # ---------------- 国际化 ----------------
    def set_tool_language(self, lang_code, persist=False):
        self._language = lang_code
        self.retranslate_ui()

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("DSSRR Repaired DB Viewer"))
        self._retranslate_static_texts()
        # matplotlib 工具栏是第三方控件，按钮文字不走 self.tr()，需单独重设
        self._localize_nav_toolbar()
        # 动态内容（统计链接/表格异常列/详情卡片/画布标题/状态栏）在
        # refresh() 中按当前语言重新填充，随 qm 实时翻译生效
        try:
            self.refresh()
        except Exception:
            pass

    def _retranslate_static_texts(self):
        """重设 _build_ui 中创建的静态控件文本（控件引用均为 self.xxx）。"""
        # 路径区
        self.lbl_raw.setText(self.tr("Raw DB"))
        self.lbl_fix.setText(self.tr("Repaired DB"))
        self.btn_raw.setText(self.tr("Browse"))
        self.btn_fix.setText(self.tr("Browse"))
        # 参数区
        self.grp_sel.setTitle(self.tr("Data Selection"))
        self.lbl_station.setText(self.tr("Stream"))
        self.lbl_day.setText(self.tr("Day"))
        self.lbl_start.setText(self.tr("Start (min)"))
        self.lbl_win.setText(self.tr("Window (min)"))
        self.btn_day_prev.setToolTip(self.tr("Previous day"))
        self.btn_day_next.setToolTip(self.tr("Next day"))
        self.btn_month_prev.setToolTip(self.tr("Previous month"))
        self.btn_month_next.setToolTip(self.tr("Next month"))
        self.btn_year_prev.setToolTip(self.tr("Previous year"))
        self.btn_year_next.setToolTip(self.tr("Next year"))
        for btn, label, tip in zip(
                self._win_btns,
                [self.tr(label) for label, _v, _t in WIN_PRESETS],
                [self.tr(tip) for _l, _v, tip in WIN_PRESETS]):
            btn.setText(label)
            btn.setToolTip(tip)
        # 操作与视图
        self.grp_view.setTitle(self.tr("View"))
        _labels = {
            "split": self.tr("Time · Split"),
            "overlay": self.tr("Time · Overlay"),
            "psd": self.tr("Frequency · PSD"),
            "amp": self.tr("Frequency · Amplitude"),
            "combo": self.tr("Combo 2×2"),
        }
        for key, rb in self.rb.items():
            rb.setText(_labels.get(key, rb.text()))
        self.btn_refresh.setText(self.tr("Refresh"))
        self.btn_export.setText(self.tr("Export PNG"))
        self.btn_manual.setText(self.tr("Manual Repair"))
        self.btn_manual.setToolTip(
            self.tr("Manually re-repair a segment of the current file "
                    "with real-time preview"))
        self.btn_batch.setText(self.tr("Batch Repair"))
        self.btn_batch.setToolTip(
            self.tr("Batch de-anomaly: select raw/fix roots, stations and "
                    "date range, pre-check the data, then run the full "
                    "pipeline with progress and auto verification"))
        self.btn_set.setText(self.tr("Settings"))
        self.btn_set.setToolTip(
            self.tr("Configure anomaly-detection and repair parameters "
                    "(applied to manual re-processing)"))
        self.btn_help.setText(self.tr("Help"))
        self.btn_help.setToolTip(self.tr("Open the viewer and parameter help"))
        # 底部明细面板
        self.grp_detail.setTitle(self.tr("Repair Details"))
        self.tbl_detail.setHorizontalHeaderLabels([
            self.tr("#"), self.tr("Start (min)"), self.tr("End (min)"),
            self.tr("Len (s)"), self.tr("Anomaly"), self.tr("Method")])
        self.grp_detail_card.setTitle(self.tr("Selected Segment"))

    def _localize_nav_toolbar(self):
        """把 matplotlib 导航工具栏的按钮名/提示翻成当前界面语言。

        委托给 :func:`dssrr.gui.i18n.localize_nav_toolbar`，中英双向；切换语言
        时由 :meth:`retranslate_ui` 再次调用即可还原成英文。
        """
        try:
            from .i18n import localize_nav_toolbar
            localize_nav_toolbar(getattr(self, "_nav", None))
        except Exception:
            pass


def _qss(dark, theme=None):
    """按深色/浅色生成窗口样式表（跟随 DSSRR 主题令牌）。"""
    theme = theme or {}
    if dark:
        bg = theme.get("theme_bg") or "#1A1A1A"
        fg = theme.get("theme_fg") or "#D4C5A9"
        panel = theme.get("theme_panel") or "#1F1F1F"
        surface = theme.get("theme_surface1") or "#3D3D3D"
        crust = theme.get("theme_crust") or "#0C0C0C"
        accent = theme.get("theme_accent") or "#C14B28"
        return f"""
QWidget {{ background-color: transparent; color: {fg}; }}
QMainWindow {{ background-color: {bg}; }}
QDialog {{ background-color: {bg}; }}
QGroupBox {{
    font-weight: 600;
    border: 1px solid {surface};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    background: {panel};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {fg};
}}
QLabel {{ color: {fg}; }}
QLabel#fieldLabel {{ font-weight: 500; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    padding: 4px 8px;
    border: 1px solid {surface};
    border-radius: 6px;
    background: {crust};
    color: {fg};
    selection-background-color: {accent};
}}
QComboBox QAbstractItemView {{
    background: {panel};
    color: {fg};
    border: 1px solid {surface};
    selection-background-color: {accent};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {accent}; }}
QPushButton {{
    padding: 5px 14px;
    border: 1px solid {surface};
    border-radius: 6px;
    background: {panel};
    color: {fg};
    font-weight: 500;
}}
QPushButton#preset {{
    padding: 3px 8px;
    border-radius: 5px;
}}
QMenu {{ background: {panel}; color: {fg}; border: 1px solid {surface}; }}
QMenu::item {{ padding: 4px 16px; }}
QMenu::item:selected {{ background: {surface}; }}
QPushButton:hover {{ background: {surface}; }}
QPushButton:pressed {{ background: {crust}; }}
QRadioButton {{ spacing: 6px; padding: 2px 4px; color: {fg}; }}
QToolBar {{
    background: {crust};
    border: none;
    border-bottom: 1px solid {surface};
    spacing: 3px;
    padding: 3px;
}}
QToolBar::separator {{ background: {surface}; width: 1px; margin: 4px 2px; }}
QToolButton {{
    color: {fg};
    background: {panel};
    border: 1px solid {surface};
    border-radius: 4px;
    padding: 4px 6px;
    margin: 1px;
}}
QToolButton:hover {{ background: {surface}; }}
QToolButton:pressed {{ background: {accent}; }}
QToolButton:checked {{ background: {surface}; border-color: {accent}; }}
QTableWidget {{ background: {panel}; color: {fg}; gridline-color: {surface}; border: 1px solid {surface}; }}
QTableWidget::item {{ padding: 2px 4px; }}
QHeaderView::section {{ background: {crust}; color: {fg}; border: 1px solid {surface}; padding: 2px 4px; }}
QLabel#detailStats {{ color: {fg}; font-weight: 600; padding: 2px 4px; }}
QStatusBar {{ background: {crust}; color: {fg}; }}
/* 不自定义 SpinBox up/down 子控件，让 Qt 保留系统默认步进按钮。 */
"""
    bg = theme.get("theme_bg") or "#EEEEEE"
    fg = theme.get("theme_fg") or "#111111"
    panel = theme.get("theme_panel") or "#FFFFFF"
    surface = theme.get("theme_surface1") or "#D0D7E2"
    accent = theme.get("theme_accent") or "#2563EB"
    return f"""
QWidget {{ background-color: transparent; color: {fg}; }}
QMainWindow {{ background-color: {bg}; }}
QDialog {{ background-color: {bg}; }}
QGroupBox {{
    font-weight: 600;
    border: 1px solid #c9d1d9;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    background: {panel};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #243b53;
}}
QLabel {{ color: #243b53; }}
QLabel#fieldLabel {{ font-weight: 500; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    padding: 4px 8px;
    border: 1px solid #c9d1d9;
    border-radius: 6px;
    background: #ffffff;
    color: #111111;
    selection-background-color: {accent};
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: #4a90d9; }}
QPushButton {{
    padding: 5px 14px;
    border: 1px solid #b6c2cf;
    border-radius: 6px;
    background: #eef2f6;
    color: #111111;
    font-weight: 500;
}}
QPushButton#preset {{
    padding: 3px 8px;
    border-radius: 5px;
}}
QMenu {{ background: #1e2630; color: #e8edf2; border: 1px solid #374151; }}
QMenu::item {{ padding: 4px 16px; }}
QMenu::item:selected {{ background: #374151; }}
QPushButton:hover {{ background: #e2e8f0; }}
QPushButton:pressed {{ background: #d6dee8; }}
QRadioButton {{ spacing: 6px; padding: 2px 4px; color: #111111; }}
QToolBar {{
    background: #f6f8fa;
    border: none;
    border-bottom: 1px solid #d0d7e2;
    spacing: 3px;
    padding: 3px;
}}
QToolBar::separator {{ background: #d0d7e2; width: 1px; margin: 4px 2px; }}
QToolButton {{
    color: #111111;
    background: #ffffff;
    border: 1px solid #b6c2cf;
    border-radius: 4px;
    padding: 4px 6px;
    margin: 1px;
}}
QToolButton:hover {{ background: #e2e8f0; }}
QToolButton:pressed {{ background: #d6dee8; }}
QToolButton:checked {{ background: #d6dee8; border-color: #4a90d9; }}
QTableWidget {{ background: #ffffff; color: #243b53; gridline-color: #d0d7e2; border: 1px solid #d0d7e2; }}
QTableWidget::item {{ padding: 2px 4px; }}
QHeaderView::section {{ background: #f6f8fa; color: #243b53; border: 1px solid #d0d7e2; padding: 2px 4px; }}
QLabel#detailStats {{ color: #243b53; font-weight: 600; padding: 2px 4px; }}
QStatusBar {{ background: #f6f8fa; color: #243b53; }}
/* 不自定义 SpinBox up/down 子控件，让 Qt 保留系统默认步进按钮。 */
"""


def _toolbar_qss(dark):
    """matplotlib 导航工具栏专用 QSS（直接 setStyleSheet 到工具栏，
    不依赖父窗口 QSS 传播——传播在某些 Qt 版本下对 QToolBar 不生效，
    会出现深色模式工具栏背景发白的问题）。"""
    if dark:
        return (
            "QToolBar { background-color: #0C0C0C; border: none; "
            "border-bottom: 1px solid #3D3D3D; spacing: 3px; padding: 3px; }\n"
            "QToolBar::handle { background-color: #0C0C0C; }\n"
            "QToolBar::separator { background-color: #3D3D3D; width: 1px; margin: 4px 2px; }\n"
            "QToolButton { color: #D4C5A9; background-color: #3D3D3D; "
            "border: 1px solid #5C5C5C; border-radius: 4px; padding: 4px 6px; margin: 1px; }\n"
            "QToolButton:hover { background-color: #5C5C5C; border-color: #D4C5A9; }\n"
            "QToolButton:pressed { background-color: #7A7A7A; }\n"
            "QToolButton:checked { background-color: #5C5C5C; border-color: #D4C5A9; }"
        )
    return (
        "QToolBar { background-color: #f6f8fa; border: none; "
        "border-bottom: 1px solid #d0d7e2; spacing: 3px; padding: 3px; }\n"
        "QToolBar::handle { background-color: #f6f8fa; }\n"
        "QToolBar::separator { background-color: #d0d7e2; width: 1px; margin: 4px 2px; }\n"
        "QToolButton { color: #111111; background-color: #ffffff; "
        "border: 1px solid #b6c2cf; border-radius: 4px; padding: 4px 6px; margin: 1px; }\n"
        "QToolButton:hover { background-color: #e2e8f0; border-color: #4a90d9; }\n"
        "QToolButton:pressed { background-color: #d6dee8; }\n"
        "QToolButton:checked { background-color: #d6dee8; border-color: #4a90d9; }"
    )


# ----------------------------------------------------------------------
# 离线导出入口（无 GUI 环境验证/批量出图）
# ----------------------------------------------------------------------
def export_cli(argv):
    """python db_repair_viewer.py --export <raw_root> <fix_root> <station> <day> <start_min> <win_min> <mode> <out.png>"""
    if len(argv) < 8:
        print("usage: db_repair_viewer.py --export RAW FIX STATION DAY START_MIN WIN_MIN MODE OUT.png")
        return 1
    raw_root, fix_root, station, day, sm, wm, mode, out = argv[:8]
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.switch_backend("Agg")
    _, _, index = scan_days(raw_root)
    raw_path, fix_path, err = resolve_paths(station, day, raw_root, fix_root, index)
    if err == "RAW_MISSING":
        print(f"ERROR raw missing: {station} {day}")
        return 2
    if err == "FIX_MISSING":
        print(f"ERROR fix missing: {station} {day}")
        return 3
    segs = load_segments(_repair_catalog_path())
    fig, info = render_view(station, day, int(sm), int(wm), mode,
                            raw_root, fix_root, segs,
                            raw_path=raw_path, fix_path=fix_path)
    fig.savefig(out, dpi=100)
    print(f"saved {out}  {info}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        sys.exit(export_cli(sys.argv[2:]))
    app = QApplication(sys.argv)
    win = DbRepairViewerWindow()
    win.show()
    sys.exit(app.exec_())


class RepairSettingsDialog(QDialog):
    """去异常参数设置页（查看器 Settings 按钮）。

    三类参数：
    - 异常检测（E7Detector 全部 21 项）：尖峰/阶跃/冻结/RMS 突发/事件
      模板检测阈值与边界处理；
    - 修复分级：连续缺失 ≤lin_max_run 点→线性、≤z_max_run→Z 分数、
      其余→DSSRR；缺失占比 miss_frac_lo/hi 决定"只修缺失 run"还是整段；
    - DSSRR 参考与能量保护：参考段长度（≥L×ref_ratio，[ref_min,ref_max]s）、
      背景窗、振荡区保护比值窗、重建能量上限因子。

    保存后写入 ~/.dssrr_settings.json；手动重处理（ManualRepairDialog
    auto 模式）每次运行读取最新设置。批处理流水线默认不读用户设置
    （保证论文结果可复现），除非显式加 ``--settings``。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dark = bool(getattr(parent, "_dark", False))
        self._theme = dict(getattr(parent, "_theme", {}) or {})
        from dssrr.repair_lib import repair_settings as RS
        self._rs = RS
        self._settings = RS.load_settings()
        self.setWindowTitle(self.tr("Repair Parameters"))
        self.setMinimumSize(640, 500)
        self.resize(720, 700)
        self._build_ui()

    # 参数可读显示名（英文 source 字符串，供界面与 i18n 使用）
    _PARAM_NAMES = {
        "spike_window_sec": "Spike window", "spike_k": "Spike threshold K",
        "step_k": "Step threshold K", "freeze_min_sec": "Min freeze duration",
        "rms_short_sec": "RMS short window", "rms_bg_sec": "RMS background window",
        "rms_ratio": "RMS trigger ratio", "min_burst_sec": "Min burst duration",
        "low_dynamic_rms": "Low-dynamic RMS", "low_dynamic_gain": "Low-dynamic gain",
        "merge_gap_sec": "Merge gap", "short_pad_sec": "Short pad",
        "long_pad_sec": "Long pad", "max_repair_sec": "Max repair duration",
        "event_morph_frac": "Event morph tolerance", "event_jitter_frac": "Event jitter tolerance",
        "event_min_sec": "Min event duration", "event_coda_ratio": "Event coda ratio",
        "event_rms_smooth": "Event RMS smooth", "event_head_mid_ratio": "Head/mid ratio",
        "event_missing_frac": "Event missing frac",
        "lin_max_run": "Linear max run", "z_max_run": "Z-score max run",
        "miss_frac_lo": "Missing frac low", "miss_frac_hi": "Missing frac high",
        "z_thr": "Z-score threshold", "z_bg_sec": "Z-score background",
        "spike_k_sigma": "Spike repair sigma", "ref_ratio": "Reference ratio",
        "ref_min_sec": "Min reference", "ref_max_sec": "Max reference",
        "bkg_sec": "Energy background", "osc_lo": "Oscillation low",
        "osc_hi": "Oscillation high", "std_raw_factor": "Rebuild/raw cap",
        "std_bkg_factor": "Rebuild/bg cap",
        "quantize_to_int": "Quantize to integers",
    }

    # 每个参数的中文含义说明（英文 source，i18n 走 ts；用于参数悬浮提示）
    _PARAM_TIPS = {
        "spike_window_sec": "Window (s) to detect isolated spikes: a sample "
                            "deviating more than spike_k × local std/MAD",
        "spike_k": "Spike detection threshold (std/MAD units); larger = fewer detections",
        "step_k": "Step/offset detection threshold on the first-difference median",
        "freeze_min_sec": "Minimum duration (s) of a constant run treated as "
                          "frozen clock; only its missing samples are repaired",
        "rms_short_sec": "Short RMS window for burst/oscillation energy tracking",
        "rms_bg_sec": "Background RMS window for the long-term energy baseline",
        "rms_ratio": "Burst trigger: short-RMS / background-RMS ≥ this ratio",
        "min_burst_sec": "Minimum duration (s) of a burst/oscillation segment",
        "low_dynamic_rms": "Absolute RMS floor below which the signal is low-dynamic (quiet)",
        "low_dynamic_gain": "Multiplier on low_dynamic_rms to scale detection in quiet segments",
        "merge_gap_sec": "Merge anomalies closer than this (s) into one segment "
                        "(e.g. ≥ 120 s to keep DSSRR reference clear of neighbours)",
        "short_pad_sec": "Boundary padding (s) added on both sides of short anomalies",
        "long_pad_sec": "Boundary padding (s) for long segments so the reference "
                        "is fully outside the anomaly",
        "max_repair_sec": "Upper limit (s) of a single repaired segment; longer segments are split",
        "event_morph_frac": "Event (moonquake) detection: envelope shape tolerance",
        "event_jitter_frac": "Event detection: allowed envelope onset jitter",
        "event_min_sec": "Minimum duration (s) of an event candidate",
        "event_coda_ratio": "Coda length multiplier (× event duration) for the event tail",
        "event_rms_smooth": "Smoothing factor of the event envelope RMS",
        "event_head_mid_ratio": "Head/body amplitude ratio expected for a catalog event",
        "event_missing_frac": "Max missing fraction inside an event candidate before rejection",
        "lin_max_run": "Missing runs ≤ this many samples are linearly interpolated",
        "z_max_run": "Missing runs from 3 to z_max_run samples are Z-score cleaned "
                     "(background stats from neighbours)",
        "miss_frac_lo": "Missing-fraction lower bound for DSSRR (repair if ≥)",
        "miss_frac_hi": "Missing-fraction upper bound; above it the segment is left "
                        "as-is (too little information)",
        "z_thr": "Z-score outlier threshold (σ) for spike removal in the Z-score path",
        "z_bg_sec": "Background window (s) used for Z-score statistics",
        "spike_k_sigma": "σ threshold for repairing isolated spikes before DSSRR",
        "ref_ratio": "Reference length = anomaly length × ref_ratio (each side); "
                     "clamped to [ref_min, ref_max] s (paper default 120–600 s)",
        "ref_min_sec": "Reference segment minimum (s); paper default 120 s",
        "ref_max_sec": "Reference segment maximum (s); paper default 600 s",
        "bkg_sec": "Energy background window (s) for the oscillation guard",
        "osc_lo": "Oscillation-ratio lower bound: inside active oscillation → back off",
        "osc_hi": "Oscillation-ratio upper bound",
        "std_raw_factor": "Cap: rebuilt std ≤ raw-reference std × this factor",
        "std_bkg_factor": "Cap: rebuilt std ≤ background std × this factor",
        "quantize_to_int": "Round repaired values to integers to match the raw "
                           "int32 archives; disable for float precision",
    }

    # ---------------- UI ----------------
    def _build_ui(self):
        RS = self._rs
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(8)

        self._boxes = {}  # (group, key) -> spinbox

        groups = [
            (self.tr("Anomaly detection"), "detect", RS.detect_spec(),
             self.tr("Spike/step/freeze/burst/event detection thresholds "
                     "and boundary padding; applied in Auto pipeline.")),
            (self.tr("Repair grading"), "repair",
             {k: RS.repair_spec()[k] for k in (
                 "lin_max_run", "z_max_run", "miss_frac_lo", "miss_frac_hi",
                 "z_thr", "z_bg_sec")},
             self.tr("Missing-run grading: runs ≤ lin_max_run samples are "
                     "linearly interpolated, ≤ z_max_run are Z-score cleaned, "
                     "longer runs or energy/step segments go to DSSRR.")),
            (self.tr("DSSRR reference & energy guard"), "repair",
             {k: RS.repair_spec()[k] for k in (
                 "spike_k_sigma", "ref_ratio", "ref_min_sec", "ref_max_sec",
                 "bkg_sec", "osc_lo", "osc_hi", "std_raw_factor",
                 "std_bkg_factor")},
             self.tr("DSSRR reference length ≥ L×ref_ratio (clamped to "
                     "[ref_min, ref_max] s); energy guard backs off repair "
                     "when the segment is part of active oscillation "
                     "background (osc_lo..osc_hi) or the rebuilt std exceeds "
                     "raw×std_raw_factor / bg×std_bkg_factor.")),
            (self.tr("Output"), "repair",
             {"quantize_to_int": (True,)},
             self.tr("Quantize repaired values back to integer counts "
                     "(round-half-even) to match the raw int32 archives. "
                     "Raw counts are integers; PSD-rebuilt values are real, "
                     "±0.5 quantization noise is negligible vs DSSRR "
                     "rebuild uncertainty. Disable to keep float precision "
                     "for high-resolution spectral analysis.")),
        ]

        for title, grp, spec, tip in groups:
            box = QGroupBox(title)
            box.setToolTip(tip)
            grid = QGridLayout(box)
            grid.setContentsMargins(8, 6, 8, 6)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(4)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(3, 1)
            keys = list(spec)
            for idx in range(0, len(keys), 2):
                row = idx // 2
                for col in (0, 1):
                    pos = idx + col
                    if pos >= len(keys):
                        break
                    key = keys[pos]
                    dflt = spec[key][0]
                    c = col * 3
                    lab = QLabel(self.tr(self._PARAM_NAMES.get(key, key)))
                    lab.setToolTip(self.tr(self._PARAM_TIPS.get(key, key)))
                    if isinstance(dflt, bool):
                        cb = QCheckBox()
                        cb.setChecked(bool(self._settings[grp][key]))
                        cb.setToolTip(self.tr(self._PARAM_TIPS.get(key, key)))
                        self._boxes[(grp, key)] = cb
                        grid.addWidget(lab, row, c)
                        grid.addWidget(cb, row, c + 1)
                        continue
                    vmin, vmax, dec, unit = spec[key][1], spec[key][2], spec[key][3], spec[key][4]
                    if isinstance(dflt, int):
                        sb = QSpinBox()
                        sb.setRange(int(vmin), int(vmax))
                        sb.setValue(int(self._settings[grp][key]))
                    else:
                        sb = QDoubleSpinBox()
                        sb.setRange(float(vmin), float(vmax))
                        sb.setDecimals(dec)
                        sb.setSingleStep(10 ** (-dec))
                        sb.setValue(float(self._settings[grp][key]))
                    sb.setToolTip(self.tr(self._PARAM_TIPS.get(key, key)))
                    sb.setMinimumWidth(84)
                    sb.setMaximumWidth(110)
                    self._boxes[(grp, key)] = sb
                    grid.addWidget(lab, row, c)
                    grid.addWidget(sb, row, c + 1)
                    if unit:
                        ulab = QLabel(unit)
                        ulab.setObjectName("unit")
                        grid.addWidget(ulab, row, c + 2)
            v.addWidget(box)
        v.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        row_btn = QHBoxLayout()
        btn_help = QPushButton(self.tr("Help"))
        btn_help.setToolTip(self.tr("Open the parameter explanation help"))
        btn_help.clicked.connect(self._open_help)
        btn_reset = QPushButton(self.tr("Reset defaults"))
        btn_reset.clicked.connect(self._reset)
        btn_save = QPushButton(self.tr("Save"))
        btn_save.setObjectName("accent")
        btn_save.clicked.connect(self._save)
        btn_close = QPushButton(self.tr("Close"))
        btn_close.clicked.connect(self.reject)
        row_btn.addWidget(btn_help)
        row_btn.addWidget(btn_reset)
        row_btn.addStretch(1)
        row_btn.addWidget(btn_save)
        row_btn.addWidget(btn_close)
        root.addLayout(row_btn)
        self._apply_style()

    def _open_help(self):
        dlg = RepairHelpDialog(self)
        dlg.exec_()

    # ---------------- actions ----------------
    def _collect(self):
        out = {"detect": {}, "repair": {}}
        for (grp, key), sb in self._boxes.items():
            out[grp][key] = sb.isChecked() if hasattr(sb, "isChecked") \
                and not hasattr(sb, "value") else sb.value()
        return out

    def _reset(self):
        base = self._rs.default_settings()
        for (grp, key), sb in self._boxes.items():
            if hasattr(sb, "isChecked") and not hasattr(sb, "value"):
                sb.setChecked(bool(base[grp][key]))
            else:
                sb.setValue(base[grp][key])

    def _save(self):
        try:
            self._rs.save_settings(self._collect())
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Error"), str(exc))
            return
        QMessageBox.information(
            self, self.tr("Saved"),
            self.tr("Parameters saved and will be applied to the next "
                    "manual re-processing (Auto pipeline)."))
        self.accept()

    def _apply_style(self):
        fg = self._theme.get("theme_fg", "#D4C5A9" if self._dark else "#243B53")
        bg = self._theme.get("theme_bg", "#1A1A1A" if self._dark else "#F5F5F5")
        surface = self._theme.get("theme_surface1", "#3D3D3D" if self._dark else "#D0D7E2")
        accent = self._theme.get("theme_accent", "#C9A227")
        self.setStyleSheet(f"""
QDialog {{ background-color: {bg}; }}
QLabel {{ color: {fg}; font-size: 12px; }}
QLabel#unit {{ color: {fg}; font-size: 10px; }}
QGroupBox {{ color: {fg}; font-size: 13px; font-weight: 600;
  border: 1px solid {surface}; border-radius: 8px;
  margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {fg}; }}
QPushButton {{ background-color: {surface}; color: {fg};
  border: 1px solid {surface}; border-radius: 5px; padding: 5px 18px;
  font-size: 12px; }}
QPushButton:hover {{ border-color: {fg}; }}
QPushButton:disabled {{ color: {surface}; }}
QPushButton#accent {{ background-color: {accent}; color: #1A1A1A;
  border: none; font-weight: 600; }}
QPushButton#accent:hover {{ background-color: #B8921F; }}
QSpinBox, QDoubleSpinBox {{ background-color: {bg}; color: {fg};
  border: 1px solid {surface}; border-radius: 4px; padding: 2px 4px; }}
/* 步进按钮：避免 Windows 默认黑块（主题色按钮 + 浅色箭头） */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
  subcontrol-origin: border; subcontrol-position: center right;
  width: 16px; background-color: {surface}; border-radius: 3px;
}}
QCheckBox {{ color: {fg}; font-size: 12px; spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QScrollArea {{ border: none; background: transparent; }}
""")


class RepairHelpDialog(QDialog):
    """查看器使用帮助：快速上手 + 各参数分组说明 + 图例/方法/验收。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dark = bool(getattr(parent, "_dark", False))
        self._theme = dict(getattr(parent, "_theme", {}) or {})
        self.setWindowTitle(self.tr("Viewer Help"))
        self.setMinimumSize(760, 560)
        self.resize(860, 640)
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)
        tabs = QTabWidget()
        tabs.addTab(self._page_start(), self.tr("Quick Start"))
        tabs.addTab(self._page_detect(), self.tr("Detection Parameters"))
        tabs.addTab(self._page_repair(), self.tr("Repair & Output"))
        tabs.addTab(self._page_dssrr(), self.tr("DSSRR & Energy Guard"))
        tabs.addTab(self._page_legend(), self.tr("Legend & Concepts"))
        root.addWidget(tabs, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        btn_close = QPushButton(self.tr("Close"))
        btn_close.clicked.connect(self.accept)
        row.addWidget(btn_close)
        root.addLayout(row)
        self._apply_style()

    def _scrolled(self, widgets):
        """把一组 QLabel 放入滚动区。"""
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        for lab in widgets:
            lab.setWordWrap(True)
            v.addWidget(lab)
        v.addStretch(1)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QScrollArea.NoFrame)
        sc.setWidget(w)
        return sc

    def _sec(self, text):
        lab = QLabel(text)
        lab.setStyleSheet("font-weight:600; font-size:13px;")
        return lab

    @staticmethod
    def _tr_param(text):
        """参数名/说明的翻译挂在 RepairSettingsDialog 上下文；帮助页调用时
        必须用同一 context，否则会中英混杂（tr 落在 RepairHelpDialog 找不到词条）。"""
        from PyQt5.QtCore import QCoreApplication
        return QCoreApplication.translate("RepairSettingsDialog", text)

    def _param_line(self, key):
        names = RepairSettingsDialog._PARAM_NAMES
        tips = RepairSettingsDialog._PARAM_TIPS
        name = self._tr_param(names.get(key, key))
        tip = self._tr_param(tips.get(key, key))
        return QLabel(f"• <b>{name}</b> — {tip}")

    def _page_start(self):
        steps = [
            self.tr("Step 1 — Point Raw DB and Repaired DB at the two "
                    "database roots (Browse buttons)."),
            self.tr("Step 2 — Pick Stream and Day; the stream list holds every "
                    "trace in the raw DB and the day list comes from "
                    "the database folders automatically."),
            self.tr("Step 3 — Set Start (min) and Window (min); the preset "
                    "buttons 30m/1h/6h/12h/1d change the window in one click "
                    "and refresh immediately."),
            self.tr("Step 4 — Choose a view: Time · Split (raw above, "
                    "repaired below), Time · Overlay, Frequency · PSD, "
                    "Frequency · Amplitude, or Combo 2×2."),
            self.tr("Step 5 — The Repair Details table at the bottom lists "
                    "every anomaly of the day. Click a row to highlight that "
                    "segment on the canvas without changing the zoom."),
            self.tr("Step 6 — Manual Repair re-processes the selected "
                    "segment with a live preview; Apply writes the result "
                    "back to the repaired DB and updates the details."),
            self.tr("Step 7 — Settings opens the parameter editor; changes "
                    "apply to the next manual re-processing. Reset defaults "
                    "restores the paper v5 values."),
            self.tr("Step 8 — Export PNG saves the current canvas."),
        ]
        ws = [self._sec(self.tr("Viewer usage"))]
        ws += [QLabel(s) for s in steps]
        return self._scrolled(ws)

    def _page_detect(self):
        rows = [
            ("spike_window_sec", "spike_k", "step_k", "freeze_min_sec",
             "rms_short_sec", "rms_bg_sec", "rms_ratio", "min_burst_sec",
             "low_dynamic_rms", "low_dynamic_gain", "merge_gap_sec",
             "short_pad_sec", "long_pad_sec", "max_repair_sec",
             "event_morph_frac", "event_jitter_frac", "event_min_sec",
             "event_coda_ratio", "event_rms_smooth",
             "event_head_mid_ratio", "event_missing_frac"),
        ]
        ws = [self._sec(self.tr(
            "Anomaly detection: spike/step/freeze/burst/event detectors and "
            "boundary padding (Auto pipeline). Hover any field in Settings "
            "for the same explanation."))]
        for key in rows[0]:
            ws.append(self._param_line(key))
        return self._scrolled(ws)

    def _page_repair(self):
        keys = ["lin_max_run", "z_max_run", "miss_frac_lo", "miss_frac_hi",
                "z_thr", "z_bg_sec", "spike_k_sigma", "quantize_to_int"]
        ws = [self._sec(self.tr("Repair grading and output"))]
        ws.append(QLabel(self.tr(
            "Missing runs are graded: ≤ lin_max_run samples → linear "
            "interpolation; 3–z_max_run samples → Z-score cleaning; longer "
            "runs and energy/step anomalies → DSSRR rebuild.")))
        for key in keys:
            ws.append(self._param_line(key))
        return self._scrolled(ws)

    def _page_dssrr(self):
        keys = ["ref_ratio", "ref_min_sec", "ref_max_sec", "bkg_sec",
                "osc_lo", "osc_hi", "std_raw_factor", "std_bkg_factor"]
        ws = [self._sec(self.tr("DSSRR reference and energy guard"))]
        ws.append(QLabel(self.tr(
            "DSSRR synthesizes the missing segment from PSD characteristics "
            "of reference segments on both sides. The energy guard backs off "
            "repair when the segment sits inside active oscillation "
            "background or the rebuilt std would exceed the caps.")))
        for key in keys:
            ws.append(self._param_line(key))
        return self._scrolled(ws)

    def _page_legend(self):
        ws = [self._sec(self.tr("Canvas colors"))]
        ws.append(QLabel(self.tr(
            "• repair segments — each colored by its repair method "
            "(see the color legend on the canvas and the method list below)")))
        ws.append(QLabel(self.tr(
            "• <span style='color:#27ae60'>green</span> — event segment "
            "(moonquake, protected by the catalog; not modified)")))
        ws.append(QLabel(self.tr(
            "• <span style='color:#7f8c8d'>gray</span> — gap segment (no "
            "data, kept as-is)")))
        ws.append(QLabel(self.tr(
            "• gold fill with red dashed edges — the segment selected in the "
            "details table")))
        ws.append(self._sec(self.tr("Repair methods")))
        for k, v in [
            ("dssrr", self.tr("PSD-characteristics rebuild from both-side "
                              "references (long gaps / anomalies)")),
            ("dssrr_fb_missing", self.tr("DSSRR with missing-run fallback: "
                                         "leftover zero/NaN points are "
                                         "linearly closed")),
            ("freeze_missing", self.tr("Frozen-clock segment: only the "
                                       "missing samples inside it are repaired")),
            ("missing_runs", self.tr("Short isolated missing runs")),
            ("spike", self.tr("Isolated spike repair")),
            ("zscore", self.tr("Z-score cleaning")),
            ("catalog_protect", self.tr("Moonquake catalog protection "
                                        "(event left untouched)")),
            ("keep", self.tr("Kept as-is (gap / too little information)")),
        ]:
            col = METHOD_COLORS.get(k, DEFAULT_REPAIR_COLOR)
            ws.append(QLabel(
                f"• <span style='color:{col}'>■</span> "
                f"<b>{method_name(k)}</b> — {v}"))
        ws.append(self._sec(self.tr("Verification")))
        ws.append(QLabel(self.tr(
            "Each day is verified after processing: protected moonquake "
            "segments must stay unchanged, modified-sample ratios are "
            "reported, and residual statistics are checked before the day "
            "is accepted.")))
        return self._scrolled(ws)

    def _apply_style(self):
        fg = self._theme.get("theme_fg", "#D4C5A9" if self._dark else "#243B53")
        bg = self._theme.get("theme_bg", "#1A1A1A" if self._dark else "#F5F5F5")
        surface = self._theme.get("theme_surface1", "#3D3D3D" if self._dark else "#D0D7E2")
        self.setStyleSheet(f"""
QDialog {{ background-color: {bg}; }}
QLabel {{ color: {fg}; font-size: 12px; }}
QTabWidget::pane {{ border: 1px solid {surface}; border-radius: 6px; }}
QTabBar::tab {{ background: {surface}; color: {fg}; padding: 6px 14px;
  margin-right: 2px; border-top-left-radius: 5px;
  border-top-right-radius: 5px; }}
QTabBar::tab:selected {{ background: {bg}; border: 1px solid {surface};
  border-bottom: 2px solid #C9A227; font-weight: 600; }}
QPushButton {{ background-color: {surface}; color: {fg};
  border: 1px solid {surface}; border-radius: 5px; padding: 5px 18px;
  font-size: 12px; }}
QPushButton:hover {{ border-color: {fg}; }}
QScrollArea {{ border: none; background: transparent; }}
""")
