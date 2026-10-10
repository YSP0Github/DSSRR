# -*- coding: utf-8 -*-
"""ManualRepairDialog —— 手动重处理对话框（波形可视化 + 鼠标选段版）。

面向「分析师逐段修」的交互式修复界面，与查看器共享当前台站/日期/路径上下文。

交互方式
--------
- **导入**：单个/多个文件，或整个文件夹 → 进入文件下拉列表，选哪个处理哪个。
- **框选**：在波形画布上直接拖鼠标选要处理的位置
  （左键拖 = 替换选区，右键拖 = 添加选区，双击 = 清空选区）。
- **方法**：每种方法在 :data:`METHOD_LABELS` 中都有中英双语标签，可选
  Auto（按规则分级）/ 线性插值 / Z 分数 / 尖峰(MAD) / DSSRR 双侧谱参考 /
  月震保护型尖峰。
- **预览与写回**：实时预览 raw / 当前修复库 / 新结果三态，确认后可写回修复库。

与查看器的分工
--------------
本对话框复用 :mod:`dssrr.gui.viewer` 的加载与样式工具（``SR``、
``MAX_START_MIN``、``load_mhz``、``load_segments``、``_toolbar_qss``），
修复执行则委托 :class:`dssrr.repair_lib.repair_engine.StreamRepair`，
因此 GUI 与批量流水线跑的是同一套算法。

依赖：PyQt5、matplotlib、obspy。
"""
from __future__ import annotations

import os

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QSpinBox, QPushButton, QCheckBox, QMessageBox, QFileDialog,
    QLineEdit, QListWidget, QListWidgetItem, QFrame, QSplitter,
)
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from .viewer import (SR, MAX_START_MIN, load_mhz, load_segments,
                     _toolbar_qss, run_modal_dialog)
from . import nav_hint


# ----------------------------------------------------------------------
# 手动重处理对话框：对当前打开文件（台站/日期）的任意段重新去异常，
# 复用 E7 多级修复函数，实时预览 raw/当前修复库/新结果，可写回修复库
# ----------------------------------------------------------------------
METHOD_LABELS = {
    "auto": ("Auto (by rules)", "自动（按规则分级）"),
    "linear": ("Linear interpolation", "线性插值"),
    "zscore": ("Z-score cleaning", "Z 分数清理"),
    "spike": ("Spike (MAD) repair", "尖峰（MAD）修复"),
    "dssrr": ("DSSRR spectral reconstruction", "DSSRR 双侧谱参考重建"),
    "moon_spike": ("Moonquake-protected spike repair", "月震保护型尖峰修复"),
}


def _parse_file_id(path):
    """从文件名解析 (station, day)。支持：
    - 日文件：XA.{station}.{day}.mseed       （如 XA.S12.19760113.mseed）
    - 片段文件：XA.{station}.{band}.{channel}.{start}-{end}.mseed
      （day 取 start 字段前 8 位）
    解析失败抛 ValueError。
    """
    fn = os.path.basename(path)
    if not (fn.startswith("XA.") and fn.endswith(".mseed")):
        raise ValueError(f"Not a XA.*.mseed file: {fn}")
    parts = fn.split(".")
    if len(parts) < 4:
        raise ValueError(f"Unrecognized file name: {fn}")
    stn = parts[1]
    if len(parts) == 4:
        day = parts[2]
    elif len(parts) >= 6:
        day = parts[4][:8]
    else:
        raise ValueError(f"Unrecognized file name: {fn}")
    if not day.isdigit():
        raise ValueError(f"No date in file name: {fn}")
    return stn, day


class ManualRepairDialog(QDialog):
    """手动重处理当前打开文件的一段数据，实时可视化对比处理结果。

    - 导入文件（单个/多个）或导入文件夹，文件进入下拉列表，选中即加载
    - 波形画布上用鼠标直接选择段（左拖替换/右拖添加/双击清空），
      选区自动同步到 Segment range 输入框
    - 修复方法默认"自动"（按 E7 分级规则），可手动指定
    - 执行后在预览画布上同画布对比 raw / 当前修复库 / 新结果三条曲线
    - "应用到修复库"把新结果写回 database_repaired 对应文件该段（需确认）
    """

    def __init__(self, parent, station, day, raw_path, fix_path,
                 default_seg=None, default_method="auto"):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Manual Repair") + f" — {station} {day}")
        self.resize(1180, 860)
        self._station = station
        self._day = day
        self._raw_path = raw_path
        self._fix_path = fix_path
        self._default_seg = default_seg  # dict 含 s/e 绝对样本索引
        self._default_method = default_method
        self._new_counts = None          # 最近一次预览的新结果（counts 域）
        self._result = None              # (s, e, counts) 或 None
        self._hidden_series = set()      # 图例点击隐藏的曲线（raw/DB/new）
        # 导入文件列表与当前选中
        self._imported = []              # [(path, station, day)]
        # 选区状态（秒），沿用 DSSRR Manual Repair 的交互约定
        self._selections = []            # sorted, non-overlapping (s, e) 秒
        self._selection_patch = None
        self._multi_patches = []
        self._right_drag_start = None
        self._right_drag_patch = None
        self._span_selector = None
        # 主题跟随查看器
        parent = self.parent()
        self._dark = bool(getattr(parent, "_dark", False))
        self._theme = dict(getattr(parent, "_theme", {}) or {})
        self._build_ui()
        self._load_lib()
        self._apply_style()
        # 若调用方已提供 raw 文件，直接导入并加载
        if self._raw_path and os.path.exists(self._raw_path):
            self._add_imported_file(self._raw_path)
            self._load_imported(0)
        elif station and day:
            self.setWindowTitle(
                self.tr("Manual Repair") + f" — {station} {day}")

    # ---------------- UI ----------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)
        root.setContentsMargins(10, 8, 10, 8)

        # ---- 导入行：导入文件/文件夹 + 文件下拉 + 修复库文件 ----
        import_row = QHBoxLayout()
        import_row.setSpacing(6)
        self.btn_import_files = QPushButton(self.tr("Import files…"))
        self.btn_import_files.setToolTip(
            self.tr("Select one or more XA.*.mseed files to add to the list"))
        self.btn_import_files.clicked.connect(self._import_files)
        self.btn_import_folder = QPushButton(self.tr("Import folder…"))
        self.btn_import_folder.setToolTip(
            self.tr("Scan a folder (recursively) for *.mseed files and add "
                    "them all to the list"))
        self.btn_import_folder.clicked.connect(self._import_folder)
        self.cb_files = QComboBox()
        self.cb_files.setMinimumWidth(320)
        self.cb_files.setToolTip(self.tr("Imported files; pick one to work on"))
        self.cb_files.currentIndexChanged.connect(self._on_file_selected)
        self.btn_load_fix = QPushButton(self.tr("Load repaired…"))
        self.btn_load_fix.setToolTip(self.tr(
            "Open the matching repaired DB file (optional; auto-derived "
            "from the raw file when possible)"))
        self.btn_load_fix.clicked.connect(self._load_fix_file)
        import_row.addWidget(self.btn_import_files)
        import_row.addWidget(self.btn_import_folder)
        import_row.addWidget(self.cb_files, 1)
        import_row.addWidget(self.btn_load_fix)
        root.addLayout(import_row)

        # 隐藏的 raw/fix 路径框（保留给旧逻辑读取路径；界面以导入列表为准）
        self.ed_raw = QLineEdit()
        self.ed_raw.setVisible(False)
        self.ed_fix = QLineEdit()
        self.ed_fix.setVisible(False)
        root.addWidget(self.ed_raw)
        root.addWidget(self.ed_fix)

        # ---- 异常事件列表（读取文件后自动检测填充；双击定位到该段）----
        lbl_ev = QLabel(self.tr("Detected events (double-click to locate):"))
        self.list_events = QListWidget()
        self.list_events.setFixedHeight(88)
        self.list_events.itemDoubleClicked.connect(self._on_event_double_click)
        self._events = []            # [(s, e, label)] 绝对样本索引
        ev_box = QVBoxLayout()
        ev_box.setSpacing(2)
        ev_box.addWidget(lbl_ev)
        ev_box.addWidget(self.list_events)
        root.addLayout(ev_box)

        # ---- 修复参数行 ----
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        lbl_range = QLabel(self.tr("Segment range (min)"))
        self.sp_s = QSpinBox()
        self.sp_s.setRange(0, MAX_START_MIN)
        self.sp_e = QSpinBox()
        self.sp_e.setRange(0, MAX_START_MIN)
        if self._default_seg:
            s_min = int(self._default_seg["s"] / SR / 60.0)
            e_min = int(self._default_seg["e"] / SR / 60.0)
            self.sp_s.setValue(s_min)
            self.sp_e.setValue(max(e_min, s_min + 1))
        else:
            self.sp_e.setValue(30)
        lbl_method = QLabel(self.tr("Repair method"))
        self.cb_method = QComboBox()
        for key, (en, zh) in METHOD_LABELS.items():
            self.cb_method.addItem(f"{en} / {zh}", key)
        idx = self.cb_method.findData(self._default_method)
        self.cb_method.setCurrentIndex(max(0, idx))
        form.addWidget(lbl_range, 0, 0)
        form.addWidget(self.sp_s, 0, 1)
        form.addWidget(self.sp_e, 0, 2)
        form.addWidget(lbl_method, 0, 3)
        form.addWidget(self.cb_method, 0, 4)
        # 整数化/实数化切换：默认跟随设置页 Output 组 quantize_to_int，
        # 本次重处理即时生效（预览与写回一致）
        from dssrr.repair_lib.repair_settings import load_settings
        self.cb_quant = QCheckBox(self.tr("Quantize to int"))
        self.cb_quant.setChecked(
            bool(load_settings()["repair"].get("quantize_to_int", True)))
        self.cb_quant.setToolTip(
            self.tr("Round repaired values to integer counts on write-back. "
                    "Uncheck to keep float precision in this run."))
        form.addWidget(self.cb_quant, 0, 5)
        form.setColumnStretch(6, 1)

        self.btn_run = QPushButton(self.tr("Run & Preview"))
        self.btn_run.clicked.connect(self.run_preview)
        self.btn_apply = QPushButton(self.tr("Apply to repaired DB"))
        self.btn_apply.clicked.connect(self.apply_to_db)
        self.btn_apply.setEnabled(False)
        btns = QHBoxLayout()
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_apply)
        btns.addStretch(1)

        root.addLayout(form)
        root.addLayout(btns)

        # ---- 上下分栏：选段波形画布（上） + 预览画布（下） ----
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # 上部：选段波形画布（Segment Selection 交互）
        seg_frame = QFrame()
        seg_box = QVBoxLayout(seg_frame)
        seg_box.setContentsMargins(0, 2, 0, 2)
        seg_box.setSpacing(2)
        seg_hint = QHBoxLayout()
        self.lbl_seg_title = QLabel(self.tr("Segment Selection"))
        self.lbl_seg_title.setStyleSheet("font-weight:600; font-size:13px;")
        self.lbl_selection = QLabel(self.tr("Selection: None"))
        self.lbl_selection.setStyleSheet("font-size:11px; color:#888;")
        seg_hint.addWidget(self.lbl_seg_title)
        seg_hint.addWidget(self.lbl_selection)
        seg_hint.addStretch(1)
        hint = QLabel(self.tr("Left drag: replace. Right drag: add. "
                              "Double-click: clear."))
        self.lbl_seg_hint = hint
        hint.setStyleSheet("font-size:10px; color:#999;")
        seg_hint.addWidget(hint)
        seg_box.addLayout(seg_hint)
        self._seg_figure = Figure(figsize=(11, 2.4), dpi=100)
        self._seg_canvas = FigureCanvas(self._seg_figure)
        self._seg_canvas.setMinimumHeight(170)
        self._seg_nav = NavigationToolbar2QT(self._seg_canvas, seg_frame)
        self._seg_nav.setStyleSheet(_toolbar_qss(self._dark))
        seg_box.addWidget(self._seg_nav)
        seg_box.addWidget(self._seg_canvas, 1)
        splitter.addWidget(seg_frame)

        # 平移/缩放模式下 matplotlib 会抢走 canvas.widgetlock，段选择静默失效；
        # 由这个组件把提示语切成警告（逻辑与 DePulseDialog 共用）。
        self._seg_nav_hint = nav_hint.SegmentNavHint(
            self.lbl_seg_hint, self._seg_nav, self.tr, parent=self,
            muted_color="#999999", is_dark=lambda: self._dark,
        ).install()

        # 下部：预览画布
        self.figure = Figure(figsize=(11, 4.0), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(260)
        self.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self._nav = NavigationToolbar2QT(self.canvas, self)
        self._nav.setStyleSheet(_toolbar_qss(self._dark))
        # matplotlib 工具栏文字是第三方硬编码英文，不走 self.tr()，需手动本地化
        nav_hint.localize_toolbar(self._nav)
        splitter.addWidget(self._nav)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 3)
        splitter.setSizes([240, 30, 420])
        root.addWidget(splitter, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

    def closeEvent(self, event):
        # 停掉段选择提示的兜底轮询，避免对话框关闭后定时器还在打空转
        nav_hint_widget = getattr(self, "_seg_nav_hint", None)
        if nav_hint_widget is not None:
            nav_hint_widget.stop()
        super().closeEvent(event)

    def _load_lib(self):
        """延迟加载 E7 修复函数（独立包 dssrr.repair_lib）。"""
        try:
            from dssrr.repair_lib import e7_repair_build as RB
            self._rb = RB
        except Exception as exc:
            self._rb = None
            self.lbl_status.setText(
                self.tr("Failed to load repair library") + f": {exc}")

    # ---------------- 导入 / 文件切换 ----------------
    def _import_files(self):
        """选择单个或多个原始 .mseed 文件加入下拉列表。"""
        paths, _ = run_modal_dialog(
            self,
            lambda host: QFileDialog.getOpenFileNames(
                host, self.tr("Import raw DB files"), "",
                "MiniSEED (*.mseed);;All files (*)"))
        if not paths:
            return
        added = 0
        first = -1
        for p in paths:
            idx = self._add_imported_file(p)
            if idx >= 0:
                added += 1
                if first < 0:
                    first = idx
        if added:
            self.cb_files.setCurrentIndex(first)
            self.lbl_status.setText(
                self.tr("Imported") + f": {added} file(s)")

    def _import_folder(self):
        """递归扫描文件夹下所有 *.mseed 文件加入下拉列表。"""
        folder = run_modal_dialog(
            self,
            lambda host: QFileDialog.getExistingDirectory(
                host, self.tr("Import folder (recursive *.mseed)")))
        if not folder:
            return
        found = []
        for dirpath, _dirs, files in os.walk(folder):
            for fn in sorted(files):
                if fn.endswith(".mseed"):
                    found.append(os.path.join(dirpath, fn))
        if not found:
            QMessageBox.information(
                self, self.tr("Import folder"),
                self.tr("No *.mseed files found in this folder."))
            return
        added = 0
        first = -1
        for p in found:
            idx = self._add_imported_file(p)
            if idx >= 0:
                added += 1
                if first < 0:
                    first = idx
        if added:
            self.cb_files.setCurrentIndex(first)
            self.lbl_status.setText(
                self.tr("Imported") + f": {added} file(s)")

    def _add_imported_file(self, path):
        """把一个 .mseed 加入导入列表。返回该文件在列表中的索引；
        解析失败或已存在时返回 -1 / 已有索引。"""
        try:
            station, day = _parse_file_id(path)
        except ValueError:
            return -1
        for i, (p, _s, _d) in enumerate(self._imported):
            if p == path:
                return i
        self._imported.append((path, station, day))
        self.cb_files.addItem(
            os.path.basename(path), path)
        self.cb_files.setItemData(
            self.cb_files.count() - 1, path, Qt.ToolTipRole)
        return len(self._imported) - 1

    def _on_file_selected(self, index):
        if 0 <= index < len(self._imported):
            self._load_imported(index)

    def _load_imported(self, index):
        """加载下拉列表中第 index 个文件：解析台站/日期、推导修复库、
        重绘选段波形、检测事件。"""
        if not (0 <= index < len(self._imported)):
            return
        path, station, day = self._imported[index]
        if not os.path.exists(path):
            self.lbl_status.setText(
                self.tr("File missing") + f": {path}")
            return
        self._raw_path = path
        self._station = station
        self._day = day
        self.ed_raw.setText(path)
        fix = self._derive_fix_path(path)
        if fix:
            self._fix_path = fix
            self.ed_fix.setText(fix)
        self.setWindowTitle(
            self.tr("Manual Repair") + f" — {station} {day}")
        self._selections = []
        self._selection_patch = None
        self._multi_patches = []
        self._plot_segment_trace()
        self._populate_events()
        self.btn_apply.setEnabled(
            bool(self._fix_path and os.path.exists(self._fix_path)))
        self.lbl_status.setText(
            f"{self.tr('Loaded')}: {os.path.basename(path)}  "
            f"({station} {day})")

    def _derive_fix_path(self, raw_path):
        """智能推导修复库文件：仅当路径含 database 标识时做
        database→database_repaired 镜像替换；推导不出返回空串。"""
        try:
            if "database" not in raw_path:
                return ""
            cand = raw_path.replace("database_repaired", "database")
            cand = cand.replace("database", "database_repaired")
            if os.path.exists(cand):
                return cand
        except Exception:
            pass
        return ""

    def _load_fix_file(self):
        """选择修复库 .mseed 文件（可选）。"""
        path, _ = run_modal_dialog(
            self,
            lambda host: QFileDialog.getOpenFileName(
                host, self.tr("Open repaired DB file"), "",
                "MiniSEED (*.mseed);;All files (*)"))
        if not path:
            return
        self._fix_path = path
        self.ed_fix.setText(path)
        self.btn_apply.setEnabled(True)
        self.lbl_status.setText(self.tr("Repaired DB") + f": {path}")

    # ---------------- 事件列表 ----------------
    def _populate_events(self):
        """用 E7Detector 检测当前原始文件，把异常段写入事件列表。"""
        self.list_events.clear()
        self._events = []
        if not self._raw_path or not os.path.exists(self._raw_path):
            return
        try:
            from dssrr.repair_lib.repair_settings import load_settings
            from dssrr.repair_lib.e7_detector import E7Detector
            raw, _ = load_mhz(self._raw_path)
            d = raw.copy()
            det = E7Detector(sr=SR, **load_settings()["detect"])
            rep, ev, spikes, info = det.detect(d)
            ev = sorted(ev)
            for s, e in ev:
                sm, em = s / SR / 60.0, e / SR / 60.0
                self._events.append((s, e))
                self.list_events.addItem(
                    f"[{sm:.2f} - {em:.2f} min]  ({e - s + 1} samples)")
            self.lbl_status.setText(
                self.tr("Detected") + f": {len(ev)} event segment(s)")
        except Exception as exc:
            self.lbl_status.setText(f"{self.tr('Detect failed')}: {exc}")

    def _on_event_double_click(self, item):
        """双击事件条目：把段范围定位到该事件并运行预览。"""
        row = self.list_events.row(item)
        if not (0 <= row < len(self._events)):
            return
        s, e = self._events[row]
        self.sp_s.setValue(max(0, int(s / SR / 60.0)))
        self.sp_e.setValue(max(int(s / SR / 60.0) + 1,
                               int(np.ceil((e + 1) / SR / 60.0))))
        self.run_preview()

    # ---------------- 选段波形画布（Segment Selection） ----------------
    def _seg_colors(self):
        if self._dark:
            return "#1F1F1F", "#D4C5A9", "#94A3B8"
        return "white", "#333333", "#111827"

    def _plot_segment_trace(self):
        """绘制当前文件的 MHZ 波形并挂接鼠标选段交互。"""
        fig = self._seg_figure
        fig.clear()
        bg, fg, wave = self._seg_colors()
        fig.patch.set_facecolor(bg)
        fig.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)
        if not self._raw_path or not os.path.exists(self._raw_path):
            ax.text(0.5, 0.5, self.tr("Import a raw file to visualize"),
                    ha="center", va="center", color=fg, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._seg_canvas.draw()
            return
        try:
            raw, _ = load_mhz(self._raw_path)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                    color=fg, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self._seg_canvas.draw()
            return
        self._current_sr = SR
        times = np.arange(len(raw)) / SR
        ax.plot(times, raw, color=wave, linewidth=0.8)
        ax.set_xlabel(self.tr("Seconds since file start"))
        ax.set_ylabel(self.tr("counts"))
        ax.margins(x=0.005, y=0.05)
        # 鲁棒 y 范围：按 1%/99% 分位数裁剪，避免尖峰把正常波形压成一条线
        if len(raw) > 10:
            lo, hi = np.percentile(raw, [1.0, 99.0])
            span = max(float(hi - lo), 1.0)
            ax.set_ylim(lo - 0.12 * span, hi + 0.12 * span)
        if self._dark:
            for spine in ax.spines.values():
                spine.set_color("#5C5C5C")
            ax.tick_params(colors="#D4C5A9")
            ax.xaxis.label.set_color("#D4C5A9")
            ax.yaxis.label.set_color("#D4C5A9")
        self._seg_figure.subplots_adjust(
            left=0.06, right=0.98, top=0.95, bottom=0.18)
        self._seg_canvas.draw()

        # 挂接交互：左拖替换 / 右拖添加 / 双击清空
        if self._span_selector is not None:
            try:
                self._span_selector.disconnect_events()
            except Exception:
                pass
        self._span_selector = SpanSelector(
            ax, self._on_segment_span_selected, "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="#93c5fd"),
            interactive=True, button=1)
        self._right_drag_start = None
        self._right_drag_patch = None
        canvas = self._seg_canvas
        for cid in getattr(canvas, "_dssrr_seg_cids", []):
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        cids = []

        def on_mouse_press(event):
            if event.inaxes != ax:
                return
            if event.button == 3:
                self._right_drag_start = event.xdata
                if self._right_drag_patch:
                    try:
                        self._right_drag_patch.remove()
                    except Exception:
                        pass
                    self._right_drag_patch = None

        def on_mouse_move(event):
            if self._right_drag_start is None or event.inaxes != ax:
                return
            if self._right_drag_patch:
                try:
                    self._right_drag_patch.remove()
                except Exception:
                    pass
            s0, s1 = sorted((self._right_drag_start, event.xdata))
            self._right_drag_patch = ax.axvspan(
                s0, s1, color="#FBBF24", alpha=0.3)
            canvas.draw_idle()

        def on_mouse_release(event):
            if event.inaxes != ax:
                return
            if event.button == 3 and self._right_drag_start is not None:
                s0, s1 = sorted((self._right_drag_start, event.xdata))
                if s1 - s0 > 0.01:
                    self._merge_selections((s0, s1))
                    self._redraw_selections()
                self._right_drag_start = None
                if self._right_drag_patch:
                    try:
                        self._right_drag_patch.remove()
                    except Exception:
                        pass
                    self._right_drag_patch = None
                canvas.draw_idle()

        def on_double_click(event):
            if event.inaxes != ax:
                return
            self._clear_selections()

        cids.append(canvas.mpl_connect("button_press_event", on_mouse_press))
        cids.append(canvas.mpl_connect("button_release_event", on_mouse_release))
        cids.append(canvas.mpl_connect("motion_notify_event", on_mouse_move))
        cids.append(canvas.mpl_connect(
            "button_press_event",
            lambda e: on_double_click(e) if e.button == 1 and e.dblclick else None))
        canvas._dssrr_seg_cids = cids

        # 恢复现有选区高亮
        self._selection_patch = None
        self._multi_patches = []
        self._redraw_selections()

    def _on_segment_span_selected(self, xmin, xmax):
        """左键拖拽：替换全部选区。"""
        if xmin is None or xmax is None or xmin == xmax:
            return
        s0, s1 = sorted((float(xmin), float(xmax)))
        self._selections = []
        self._merge_selections((s0, s1))
        self._redraw_selections()
        self._sync_range_from_selections()

    def _merge_selections(self, new_range):
        """插入选区并合并重叠区间（秒）。"""
        s, e = new_range
        all_ranges = list(self._selections) + [(s, e)]
        all_ranges.sort()
        merged = []
        for start, end in all_ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self._selections = merged

    def _clear_selections(self):
        self._selections = []
        self._redraw_selections()
        self._sync_range_from_selections()

    def _redraw_selections(self):
        """重绘所有选区高亮条。"""
        if self._selection_patch:
            try:
                self._selection_patch.remove()
            except Exception:
                pass
            self._selection_patch = None
        for p in self._multi_patches:
            try:
                p.remove()
            except Exception:
                pass
        self._multi_patches = []
        ax = self._seg_canvas.figure.axes[0] if self._seg_canvas.figure.axes else None
        if ax is not None:
            for i, (s, e) in enumerate(self._selections):
                color = "#93C5FD" if i == 0 else "#A5B4FC"
                alpha = 0.3 if i == 0 else 0.2
                patch = ax.axvspan(s, e, color=color, alpha=alpha)
                if i == 0:
                    self._selection_patch = patch
                else:
                    self._multi_patches.append(patch)
            self._seg_canvas.draw_idle()
        self._update_selection_label()

    def _update_selection_label(self):
        if not self._selections:
            self.lbl_selection.setText(self.tr("Selection: None"))
        elif len(self._selections) == 1:
            s, e = self._selections[0]
            self.lbl_selection.setText(
                self.tr("Selection: {s:.2f}s — {e:.2f}s").format(s=s, e=e))
        else:
            self.lbl_selection.setText(
                self.tr("Selection: {n} regions").format(
                    n=len(self._selections)))

    def _sync_range_from_selections(self):
        """把首段选区（秒）同步到 Segment range（分钟）输入框。"""
        if not self._selections:
            return
        s_sec, e_sec = self._selections[0]
        self.sp_s.setValue(max(0, int(s_sec / 60.0)))
        self.sp_e.setValue(max(int(s_sec / 60.0) + 1,
                               int(np.ceil(e_sec / 60.0))))

    # ---------------- 执行 ----------------
    def run_preview(self):
        if not self._raw_path or not os.path.exists(self._raw_path):
            self.lbl_status.setText(
                self.tr("Import a raw file first (Import files… / "
                        "Import folder…)"))
            return
        if self._rb is None:
            self._load_lib()
        if self._rb is None:
            return
        try:
            s = int(self.sp_s.value() * 60 * SR)
            e = int(self.sp_e.value() * 60 * SR)
            raw, _ = load_mhz(self._raw_path)
            if self._fix_path and os.path.exists(self._fix_path):
                fix_cur, _ = load_mhz(self._fix_path)
            else:
                fix_cur = raw.copy()
            if len(fix_cur) < len(raw):
                fix_cur = np.pad(fix_cur, (0, len(raw) - len(fix_cur)))
            else:
                fix_cur = fix_cur[:len(raw)]
            e = min(e, len(raw) - 1)
            if e <= s:
                raise ValueError(self.tr("Invalid range: end <= start"))
            # 原始域工作副本（与 e7 流水线一致：去 DC）
            good = raw[(raw > -1) & np.isfinite(raw)]
            dc = float(np.median(good)) if len(good) else 0.0
            x = raw.copy()
            x[~np.isfinite(x)] = 0.0
            x[x <= -1] = 0.0
            x = x - dc
            d = raw.copy()
            method = self.cb_method.currentData()
            if method == "auto":
                # auto = 对选中区间（乃至全天）重新跑 E7 完整流水线：
                # 检测异常子段 → 目录核验 → 参考重叠合并 → 逐子段分级修复
                # （线性/Z 分数/DSSRR/尖峰各司其职），正常数据不动。
                # 与全库批处理 build_repaired_data 同一实现，口径完全一致，
                # 并非把整个区间丢给单一方法（更不是整段 DSSRR 合成）。
                # 参数来自去异常参数设置页（查看器 Settings 按钮）。
                rb = self._rb
                from dssrr.repair_lib.repair_settings import load_settings
                st = load_settings()
                det = getattr(self, "_det", None)
                if det is None:
                    from dssrr.repair_lib.e7_detector import E7Detector
                    det = E7Detector(sr=SR, **st["detect"])
                    self._det = det
                rep, ev, gaps, info = det.detect(d)
                rep, ev, spikes = rb.verify_catalog(rep, ev, self._day)
                extra = []
                for a, b in ev:
                    for rs, re in rb.missing_runs_in(d, a, b):
                        extra.append((rs, re, "missing"))
                if extra:
                    spikes = rb._merge_simple(sorted(spikes + extra))
                rp = dict(st["repair"])
                rp["quantize_to_int"] = self.cb_quant.isChecked()
                rep = rb.merge_ref_overlap(
                    rep, SR,
                    ref_ratio=rp["ref_ratio"],
                    ref_min_sec=rp["ref_min_sec"],
                    ref_max_sec=rp["ref_max_sec"])
                new_counts, cnt = rb.build_repaired_data(
                    d, rep, ev, spikes, gaps, info, sr=SR, params=rp)
                self._new_counts = new_counts
                self._result = (s, e, new_counts)
                self._preview(raw, fix_cur, new_counts, s, e, dc)
                self.btn_apply.setEnabled(True)
                self.lbl_status.setText(
                    f"{self.tr('Pipeline done')}  "
                    f"{self.tr('segments')}={cnt['rep']}  "
                    f"lin={cnt['lin']}  z={cnt['z']}  "
                    f"spike={cnt['spike']}  dssrr={cnt['dssrr']}  "
                    f"fb={cnt['fb']}  "
                    f"{self.tr('preview')} [{self.sp_s.value()}-"
                    f"{self.sp_e.value()} min]")
                return
            # 指定单一方法（用户主动选择）：只对该区间应用该方法。
            # DSSRR 对超长段合成极慢且参考窗(≤600s)无法代表整段，保留
            # 180 min 防呆；linear/zscore/spike/moon_spike 无参考窗问题。
            events = self._event_walls()
            if method == "dssrr" and (e - s + 1) > 180 * 60 * SR:
                self.lbl_status.setText(
                    self.tr("DSSRR segment too long (max 180 min). "
                            "Pick 'Auto' for pipeline repair of a long "
                            "range, or narrow the range."))
                return
            if method == "dssrr":
                nref = self._count_dssrr_refs(s, e, len(raw), x)
                if nref < 1:
                    self.lbl_status.setText(
                        self.tr("DSSRR needs at least one healthy reference "
                                "segment: the range touches the data boundary "
                                "or its reference windows are all "
                                "missing/anomalous. Narrow the range or pick "
                                "another method."))
                    return
            # 按方法修复
            new_x = self._repair(x, d, s, e, method, events, dc)
            new_counts = new_x + dc
            if self.cb_quant.isChecked():
                new_counts = np.round(new_counts)
            self._new_counts = new_counts
            self._result = (s, e, new_counts)
            self._preview(raw, fix_cur, new_counts, s, e, dc)
            self.btn_apply.setEnabled(True)
            lbl = self.cb_method.itemText(self.cb_method.currentIndex())
            self.lbl_status.setText(
                f"{self.tr('Repaired')} [{self.sp_s.value()}-{self.sp_e.value()} min]  "
                f"{self.tr('method')}: {lbl}   "
                f"{self.tr('segment')}: {s/SR:.1f}-{e/SR:.1f} s")
        except Exception as exc:
            self.lbl_status.setText(f"{self.tr('Error')}: {exc}")

    def _event_walls(self):
        """该日 event 段（绝对样本索引）作为 DSSRR 参考保护墙。"""
        segs = getattr(self.parent(), "_segments", {})
        bucket = segs.get((self._station, self._day), {})
        return [(sg["s"], sg["e"]) for sg in bucket.get("event", [])]

    def _count_dssrr_refs(self, s, e, n, x):
        """预检 DSSRR 参考段可用性：与 ReferenceSpectrumReplacer._extract_reference
        同口径（gap 0.5s、长度 ref_sec_for、有限样本 ≥4）。ref 参数遵循
        去异常参数设置页（ref_ratio/ref_min/max_sec）。"""
        rb = self._rb
        from dssrr.repair_lib.repair_settings import load_settings
        rp = load_settings()["repair"]
        L_sec = (e - s + 1) / SR
        ref = float(np.clip(max(rp["ref_min_sec"], L_sec * rp["ref_ratio"]),
                            rp["ref_min_sec"], rp["ref_max_sec"]))
        # 安全间隔 G：与去异常参数设置页保持一致（默认 0.5 s）
        gap = int(rp.get("reference_gap_sec", 0.5) * SR)
        rn = int(ref * SR)
        ok = 0
        rb_end = s - gap
        rb_start = max(0, rb_end - rn)
        if rb_end > rb_start:
            r = x[rb_start:rb_end]
            r = r[np.isfinite(r)]
            if len(r) >= 4:
                ok += 1
        ra_start = e + 1 + gap
        ra_end = min(n, ra_start + rn)
        if ra_end > ra_start:
            r = x[ra_start:ra_end]
            r = r[np.isfinite(r)]
            if len(r) >= 4:
                ok += 1
        return ok

    def _repair(self, x, d, s, e, method, events, dc):
        rb = self._rb
        if method == "linear":
            rb.linear_fill_seg(x, s, e, d)
        elif method == "zscore":
            rb.zscore_clean_seg(x, s, e, d)
        elif method == "spike":
            rb.spike_fill_seg(x, s, e, d)
        elif method == "moon_spike":
            rb.moonquake_spike_repair(x, s, e)
        else:  # dssrr
            try:
                from dssrr.repair_lib.repair_settings import load_settings
                rp = load_settings()["repair"]
                ref_kw = {k: rp[k] for k in (
                    "ref_ratio", "ref_min_sec", "ref_max_sec", "bkg_sec",
                    "osc_lo", "osc_hi", "std_raw_factor", "std_bkg_factor")}
                x, _ok = rb.dssrr_repair_with_guard(
                    x, d, s, e, events, [], SR, dc, **ref_kw)
            except ValueError as exc:
                if "healthy reference segment" in str(exc):
                    raise ValueError(
                        self.tr("DSSRR needs at least one healthy reference "
                                "segment: the range touches the data boundary "
                                "or its reference windows are all "
                                "missing/anomalous. Narrow the range or pick "
                                "another method."))
                raise
        return x

    def _preview(self, raw, fix_cur, new_counts, s, e, dc):
        fig = self.figure
        fig.clear()
        fig.patch.set_facecolor("#1F1F1F" if self._dark else "white")
        fig.set_facecolor("#1F1F1F" if self._dark else "white")
        ax = fig.add_subplot(111)
        t = np.arange(e - s + 1) / SR / 60.0
        colors = {
            "raw": "#9CA3AF" if self._dark else "#888888",
            "fix": "#60A5FA" if self._dark else "#1f77b4",
            "new": "#FBBF24" if self._dark else "#E07B00",
        }
        ax.plot(t, raw[s:e + 1], lw=0.7, color=colors["raw"],
                label=self.tr("raw"))
        ax.plot(t, fix_cur[s:e + 1], lw=0.7, color=colors["fix"],
                label=self.tr("current repaired DB"))
        ax.plot(t, new_counts[s:e + 1], lw=0.9, color=colors["new"],
                label=self.tr("new result"))
        ax.legend(loc="upper right")
        self._apply_series_visibility()
        ax.set_xlabel("min")
        ax.set_ylabel("counts")
        ax.set_title(f"{self._station} {self._day}  "
                     f"[{s/SR/60.0:.1f}-{e/SR/60.0:.1f} min]  "
                     f"{self.tr('manual repair preview')}")
        if self._dark:
            for spine in ax.spines.values():
                spine.set_color("#5C5C5C")
            ax.tick_params(colors="#D4C5A9")
            ax.xaxis.label.set_color("#D4C5A9")
            ax.yaxis.label.set_color("#D4C5A9")
            ax.title.set_color("#D4C5A9")
        fig.tight_layout()
        self.canvas.draw()

    # ---------------- 图例点击切换曲线 ----------------
    def _apply_series_visibility(self):
        """渲染后应用图例隐藏状态：为图例曲线启用点击(pick)，并按
        self._hidden_series 隐藏对应数据曲线。"""
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
        """点击图例曲线：切换该曲线的显示/隐藏（仅重绘，不重建）。"""
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

    # ---------------- 写回 ----------------
    def apply_to_db(self):
        if self._result is None:
            return
        s, e, new_counts = self._result
        if not self._fix_path or not os.path.exists(self._fix_path):
            QMessageBox.warning(
                self, self.tr("Apply to repaired DB"),
                self.tr("Load a repaired DB file first (Load repaired…)."))
            return
        ret = QMessageBox.question(
            self, self.tr("Apply to repaired DB"),
            self.tr("This overwrites the segment in the repaired DB file:\n\n"
                    "{path}\n\n[{s}-{e}] s = [{sm}-{em}] min\n\n"
                    "Continue?").format(
                        path=self._fix_path, s=s, e=e,
                        sm=s / SR / 60.0, em=e / SR / 60.0),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        try:
            self._write_back(s, e, new_counts)
            self._refresh_catalog_after_apply(s, e)
            self.lbl_status.setText(
                self.tr("Applied. Repaired DB updated; click Refresh in viewer "
                        "to see the new data."))
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Error"),
                                 f"{self.tr('Write failed')}: {exc}")

    def _refresh_catalog_after_apply(self, s, e):
        """B：手动 apply 后刷新该日明细 CSV（被覆盖段标记 manual_apply）。

        只重载 self._segments（表格在用户点 Refresh 时随画布一起更新，
        状态栏已提示）。
        """
        try:
            rb = self._rb
            if rb is None:
                return
            det = getattr(self, "_det", None)
            if det is None:
                from dssrr.repair_lib.e7_detector import E7Detector
                from dssrr.repair_lib.repair_settings import load_settings
                det = E7Detector(sr=SR, **load_settings()["detect"])
                self._det = det
            res = rb.collect_day_rows(self._raw_path, det)
            if not res:
                return
            stn, day, rows = res
            # 根据日期找对应月份的 catalog CSV（E7_catalog_YYYYMM.csv），
            # 优先修复库 catalogs/ 目录，回退已加载的路径列表。
            day_str = str(day).replace("-", "")
            ym = day_str[:6]
            cat = ""
            fix_root = self.ed_fix.text().strip()
            if fix_root:
                cand = os.path.join(fix_root, "catalogs",
                                    f"E7_catalog_{ym}.csv")
                if os.path.exists(cand):
                    cat = cand
            if not cat:
                for cp in getattr(self, "_catalog_paths", []):
                    if f"E7_catalog_{ym}" in cp:
                        cat = cp
                        break
            if cat:
                rb.update_catalog_day(cat, stn, day, rows, manual=(s, e))
                self._segments = load_segments(
                    getattr(self, "_catalog_paths", cat))
        except Exception:
            # 明细刷新失败不影响写回结果本身
            pass

    def _write_back(self, s, e, new_counts):
        from obspy import read, Stream
        st = read(self._fix_path)
        # 写回 dtype 跟随整数化开关：勾选 → 原始整数 dtype（int32），
        # 取消 → float64（保留实数精度）
        tr2 = None
        for tr in st:
            if tr.stats.channel == "MHZ":
                out = tr.data.astype(float).copy()
                out[s:e + 1] = new_counts[s:e + 1]
                tr2 = tr.copy()
                tr2.data = out.astype(
                    tr.data.dtype if self.cb_quant.isChecked()
                    else np.float64)
                break
        if tr2 is None:
            raise RuntimeError(self.tr("No MHZ channel in repaired file"))
        st_out = Stream()
        for tr in st:
            if tr.stats.channel == "MHZ":
                st_out.append(tr2)
            else:
                st_out.append(tr.copy())
        st_out.write(self._fix_path, format="MSEED")

    def _apply_style(self):
        fg = self._theme.get("theme_fg", "#D4C5A9" if self._dark else "#243B53")
        bg = self._theme.get("theme_bg", "#1A1A1A" if self._dark else "#F5F5F5")
        surface = self._theme.get("theme_surface1", "#3D3D3D" if self._dark else "#D0D7E2")
        self.setStyleSheet(f"""
QDialog {{ background-color: {bg}; }}
QLabel {{ color: {fg}; }}
QPushButton {{ background-color: {surface}; color: {fg};
  border: 1px solid {surface}; border-radius: 4px; padding: 4px 12px; }}
QPushButton:hover {{ border-color: {fg}; }}
QPushButton:disabled {{ color: {surface}; }}
QSpinBox, QComboBox, QLineEdit {{ background-color: {bg}; color: {fg};
  border: 1px solid {surface}; border-radius: 3px; }}
QCheckBox {{ color: {fg}; spacing: 6px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; }}
QListWidget {{ background-color: {bg}; color: {fg};
  border: 1px solid {surface}; border-radius: 4px; }}
""")
        # 画布背景与工具栏随主题
        seg_bg, seg_fg, _w = self._seg_colors()
        self._seg_figure.patch.set_facecolor(seg_bg)
        self._seg_figure.set_facecolor(seg_bg)
        self._seg_canvas.setStyleSheet(f"background-color: {seg_bg};")
        self.figure.patch.set_facecolor(seg_bg)
        self.figure.set_facecolor(seg_bg)
        self.canvas.setStyleSheet(f"background-color: {seg_bg};")
        self._seg_nav.setStyleSheet(_toolbar_qss(self._dark))
        self._nav.setStyleSheet(_toolbar_qss(self._dark))
        # 段选择提示语是 inline 样式（写死颜色），换主题后要重设，
        # 否则平移/缩放激活时的警告色会停留在旧主题的配色上
        nav_hint_widget = getattr(self, "_seg_nav_hint", None)
        if nav_hint_widget is not None:
            nav_hint_widget.refresh()
