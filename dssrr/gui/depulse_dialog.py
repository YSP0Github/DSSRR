# -*- coding: utf-8 -*-
"""DePulseDialog —— DSSRR Manual Repair 面板（手动去脉冲 / 多方法修复）。

一个功能完整的波形修复面板：左侧选段与参数，右侧波形与频谱预览，
底部可一键导出修复结果。

主要能力
--------
- **段选择**：Trace 列表 + 导入（文件/文件夹）入口，可对任意段重处理。
- **多方法叠加**：阈值截断 / MAD / 滑动均值 / z 分数 / 月震保护型尖峰 /
  Isolation Forest / LOF / 神经网络修补等，可多选并按顺序叠加执行。
- **实时预览**：修复前 / 修复后波形与频谱对照，支持对数/线性谱切换。
- **导出报告**：内置导出按钮，接入
  :class:`~dssrr.gui.export_report_dialog.ExportSettingsDialog` 生成对比图。
- **保存**：点 OK 后弹出保存界面，导出 SAC / MiniSEED / TXT / ASCII。

与 DSSRR 其他模块的关系
----------------------
- 修复执行：:class:`dssrr.repair_lib.repair_engine.StreamRepair`
- 主题取色：:func:`get_theme_colors`（本模块导出，供图文字编辑器复用）
- 报告绘图：:class:`dssrr.repair_lib.report_plot.RepairReportPlotter`

说明：本面板不再包含「Anomaly Diagnostic（异常诊断）」子面板——诊断能力已
移到 :class:`dssrr.repair_lib.depulse_advanced.AnomalyDiagnostic`，由需要它的
调用方按需使用。

依赖：PyQt5、matplotlib、obspy。
"""

import io
import os

import numpy as np
from obspy import Stream, Trace, read as obspy_read
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFrame, QLabel, QPushButton, QCheckBox,
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
    QScrollArea, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QToolButton, QMenu, QProgressBar, QSplitter, QTextBrowser,
    QAbstractItemView, QSizePolicy,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from dssrr.repair_lib.repair_engine import StreamRepair

from . import nav_hint


class MplCanvas(FigureCanvasQTAgg):
    """Lightweight matplotlib canvas with only the API the dialog needs."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.fig = self.figure
        if parent is not None:
            self.setParent(parent)

    def reset_axes(self, bg_color=None, fg_color=None):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._apply_colors(bg_color, fg_color)
        return self.axes

    def set_canvas_colors(self, bg_color, fg_color):
        self._apply_colors(bg_color, fg_color)
        self.draw_idle()

    def _apply_colors(self, bg_color, fg_color):
        if bg_color:
            self.figure.set_facecolor(bg_color)
            self.axes.set_facecolor(bg_color)
        if fg_color:
            self.axes.tick_params(colors=fg_color)
            for spine in self.axes.spines.values():
                spine.set_color(fg_color)
            self.axes.xaxis.label.set_color(fg_color)
            self.axes.yaxis.label.set_color(fg_color)
            self.axes.title.set_color(fg_color)


def _is_dark():
    app = QApplication.instance()
    if app is None:
        return False
    for widget in app.topLevelWidgets():
        bg = getattr(widget, "theme_bg", None)
        if bg is not None:
            return QColor(bg).lightness() < 128
    return False


def _nav_toolbar_qss(dark):
    """matplotlib 导航工具栏专用 QSS。

    必须直接 setStyleSheet 到工具栏本身：NavigationToolbar2QT 内部的按钮是
    QToolButton，父窗口的 QSS 在某些 Qt 版本下不会传播到 QToolBar 的
    QToolButton，会出现"浅色主题下工具栏一条白底"的观感问题。
    与 viewer._toolbar_qss 保持同一色板。
    """
    if dark:
        return (
            "QToolBar { background-color: #252525; border: none; "
            "border-bottom: 1px solid #3D3D3D; spacing: 3px; padding: 3px; }\n"
            "QToolBar::handle { background-color: #252525; }\n"
            "QToolBar::separator { background-color: #3D3D3D; width: 1px; margin: 4px 2px; }\n"
            "QToolButton { color: #D4C5A9; background-color: #3D3D3D; "
            "border: 1px solid #5C5C5C; border-radius: 4px; padding: 3px 5px; margin: 1px; }\n"
            "QToolButton:hover { background-color: #5C5C5C; border-color: #D4C5A9; }\n"
            "QToolButton:pressed { background-color: #7A7A7A; }\n"
            "QToolButton:checked { background-color: #5C5C5C; border-color: #D4C5A9; }"
        )
    return (
        "QToolBar { background-color: #E9EEF5; border: none; "
        "border-bottom: 1px solid #D0D7E2; spacing: 3px; padding: 3px; }\n"
        "QToolBar::handle { background-color: #E9EEF5; }\n"
        "QToolBar::separator { background-color: #D0D7E2; width: 1px; margin: 4px 2px; }\n"
        "QToolButton { color: #111111; background-color: #FFFFFF; "
        "border: 1px solid #B6C2CF; border-radius: 4px; padding: 3px 5px; margin: 1px; }\n"
        "QToolButton:hover { background-color: #E2E8F0; border-color: #4A90D9; }\n"
        "QToolButton:pressed { background-color: #D6DEE8; }\n"
        "QToolButton:checked { background-color: #D6DEE8; border-color: #4A90D9; }"
    )


def get_theme_colors():
    if _is_dark():
        return {
            "bg": "#1A1A1A", "surface": "#1F1F1F", "surface2": "#252525",
            "surface3": "#2D2D2D", "border": "#3D3D3D",
            "border_light": "#4A4A4A", "text": "#D4C5A9",
            "text_secondary": "#8B7D6B", "text_muted": "#6B6054",
            "accent": "#C14B28", "accent_hover": "#D45A35",
            "accent_bg": "rgba(193,75,40,0.12)", "success": "#4A7C59",
            "warning": "#C9A227", "error": "#B22234",
            "selected_bg": "#C14B28", "selected_text": "#FFFFFF",
            "btn_bg": "#252525", "btn_hover": "#333333",
            "btn_border": "#3D3D3D", "btn_primary_bg": "#C14B28",
            "btn_primary_text": "#FFFFFF", "btn_disabled": "#2A2A2A",
            "btn_disabled_text": "#5A5A5A", "input_bg": "#161616",
            "input_border": "#4A4A4A", "input_focus": "#C14B28",
            "group_bg": "#1F1F1F", "group_border": "#3D3D3D",
            "group_title": "#C14B28", "tab_bg": "#252525",
            "tab_selected": "#1F1F1F", "tab_text": "#8B7D6B",
            "tab_selected_text": "#C14B28", "scrollbar_bg": "#252525",
            "scrollbar_handle": "#3D3D3D", "scrollbar_hover": "#555555",
            "tooltip_bg": "#2D2D2D", "tooltip_border": "#3D3D3D",
        }
    return {
        "bg": "#f6f8fc", "surface": "#ffffff", "surface2": "#f1f5f9",
        "surface3": "#e2e8f0", "border": "#d8e0ef",
        "border_light": "#cbd5e1", "text": "#1f2937",
        "text_secondary": "#475569", "text_muted": "#94a3b8",
        "accent": "#2563eb", "accent_hover": "#1d4ed8",
        "success": "#16a34a", "warning": "#d97706", "error": "#dc2626",
        "accent_bg": "rgba(37,99,235,0.08)", "selected_bg": "#dbeafe",
        "selected_text": "#0f172a", "btn_bg": "#ffffff",
        "btn_hover": "#eff6ff", "btn_border": "#cbd5e1",
        "btn_primary_bg": "#2563eb", "btn_primary_text": "#ffffff",
        "btn_disabled": "#e2e8f0", "btn_disabled_text": "#94a3b8",
        "input_bg": "#ffffff", "input_border": "#cbd5e1",
        "input_focus": "#3b82f6", "group_bg": "#ffffff",
        "group_border": "#d8e0ef", "group_title": "#2563eb",
        "tab_bg": "#eef2f7", "tab_selected": "#ffffff",
        "tab_text": "#334155", "tab_selected_text": "#1d4ed8",
        "scrollbar_bg": "#eef2f7", "scrollbar_handle": "#cbd5e1",
        "scrollbar_hover": "#94a3b8", "tooltip_bg": "#FFFBEA",
        "tooltip_border": "#d8e0ef",
    }


class _DePulsePreviewWorker(QThread):
    """Background worker for all de-pulse preview operations with cancellation."""
    progress = pyqtSignal(float, float, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, data, configs, base_stream=None, segment_info=None):
        super().__init__()
        self._data = data
        self._configs = configs
        self._base_stream = base_stream.copy() if base_stream is not None else None
        self._segment_info = dict(segment_info) if segment_info else None
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._base_stream is None or len(self._base_stream) == 0:
                raise ValueError("No waveform data available for preview.")
            total_methods = max(1, len(self._configs))
            worker_configs = []
            for index, config in enumerate(self._configs):
                params = dict(config.get("params", {}))
                if config.get("id") == "nn_inpaint":
                    params["progress_callback"] = (
                        lambda current, total, index=index, config=config:
                        self._emit_nn_progress(index, total_methods, config, current, total))
                worker_configs.append({**config, "params": params})

            self.progress.emit(0, total_methods, "")
            if self._segment_info and self._segment_info.get("time_range"):
                processed_stream = StreamRepair._apply_depulse_segment(
                    self._base_stream, worker_configs, self._segment_info)
            else:
                processed_stream = StreamRepair._apply_depulse_methods(
                    self._base_stream,
                    worker_configs,
                    progress_callback=lambda current, total, config: self.progress.emit(
                        current, total, str(config.get("label", ""))),
                    cancel_callback=lambda: self._cancelled,
                )
            if self._cancelled:
                return
            if processed_stream is None or len(processed_stream) == 0:
                raise RuntimeError("De-pulse preview produced no data.")
            self.progress.emit(total_methods, total_methods, "")
            self.finished.emit(processed_stream)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_nn_progress(self, index, total_methods, config, current, total):
        if self._cancelled:
            return
        fraction = min(1.0, max(0.0, current / max(1, total)))
        self.progress.emit(index + fraction, total_methods,
                           str(config.get("label", "")))



class DePulseDialog(QDialog):
    """Dialog to configure de-pulse (outlier removal) methods."""

    def __init__(self, translator, parent=None, preview_callback=None,
                 base_stream=None, signal_window=None, segment_mode=False,
                 enable_segment_selection=False, accept_callback=None):
        super().__init__(parent)
        self._tr = translator
        self.preview_callback = preview_callback
        self._accept_callback = accept_callback
        # Keep the dialog isolated from the processor until the final accept.
        self.base_stream = base_stream.copy() if base_stream is not None else None
        self._working_stream = self.base_stream.copy() if self.base_stream is not None else None
        self._repair_history = []
        self.signal_window = signal_window
        self._segment_mode_enabled = segment_mode
        self._segment_selection_available = segment_mode or enable_segment_selection
        self._save_mode = None
        # Segment selection state
        self._current_trace_index = 0
        self._selected_range = None
        self._current_sr = None
        self._selection_patch = None
        self._span_selector = None
        self.setWindowTitle(self._tr("Anomaly Repair"))
        self.setWindowFlags(
            Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(820)
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        self._available_screen_height = available.height() if available is not None else 900
        self._available_screen_width = available.width() if available is not None else 1280
        responsive_min_width = min(820, max(640, self._available_screen_width - 24))
        self.setMinimumWidth(responsive_min_width)
        if self._segment_selection_available:
            self.setMinimumHeight(min(700, self.maximumHeight()))
        self._theme_colors = get_theme_colors()
        _c = self._theme_colors
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {_c['bg']};
                color: {_c['text']};
            }}
            QLabel {{ background: transparent; }}
            #heroFrame, #previewFrame, #methodCard {{
                border: 1px solid {_c['border']};
                border-radius: 12px;
                background-color: {_c['surface']};
            }}
            #heroFrame {{
                background-color: {_c['surface']};
                border-color: {_c['border']};
            }}
            #previewFrame {{
                background-color: {_c['surface']};
            }}
            QLabel {{ color: {_c['text']}; }}
            QCheckBox {{ font-weight: 600; color: {_c['text']}; }}
            QDoubleSpinBox, QSpinBox, QLineEdit {{
                background-color: {_c['input_bg']};
                color: {_c['text']};
                border: 1px solid {_c['input_border']};
                padding: 6px;
                border-radius: 6px;
            }}
            QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled {{
                background-color: {_c['btn_disabled']};
                color: {_c['btn_disabled_text']};
                border: 1px solid {_c['input_border']};
            }}
            QLabel:disabled {{
                color: {_c['btn_disabled_text']};
            }}
            QLineEdit::placeholder {{
                color: {_c['text_muted']};
            }}
            QPushButton {{
                border-radius: 6px;
                padding: 6px 18px;
                font-weight: 600;
            }}
            QPushButton#primaryButton {{
                background-color: {_c['accent']};
                color: {_c['btn_primary_text']};
            }}
            QPushButton#primaryButton:disabled {{
                background-color: {_c['btn_disabled']};
                color: {_c['btn_disabled_text']};
            }}
            QPushButton#compactActionButton {{
                min-width: 96px;
                max-width: 132px;
                min-height: 22px;
                padding: 4px 10px;
                font-size: 12px;
                background-color: {_c['surface2']};
                color: {_c['text']};
                border: 1px solid {_c['border']};
            }}
            QPushButton#compactActionButton[action="save"] {{
                background-color: {_c['accent_bg']};
                color: {_c['accent']};
                border-color: {_c['accent']};
            }}
            QPushButton#compactActionButton:disabled {{
                background-color: {_c['btn_disabled']};
                color: {_c['btn_disabled_text']};
                border-color: {_c['border']};
            }}
            QScrollArea {{
                border: none;
                background-color: {_c['surface']};
            }}
            QScrollArea > QWidget#qt_scrollarea_viewport {{
                background-color: {_c['surface']};
            }}
            QComboBox {{
                background-color: {_c['input_bg']};
                color: {_c['text']};
                border: 1px solid {_c['input_border']};
                padding: 6px;
                border-radius: 6px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {_c['surface2']};
                color: {_c['text']};
                selection-background-color: {_c['accent']};
                selection-color: {_c['btn_primary_text']};
            }}
            QProgressBar {{
                background-color: {_c['input_bg']};
                color: {_c['text']};
                border: 1px solid {_c['input_border']};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {_c['accent']};
            }}
        """)
        self._instruction_text = self._tr("Choose one method to suppress spikes/outliers. Use Preview to compare before and after.")
        self._preview_label = self._tr("Preview")
        # 嵌入到 DSSRR 主窗口时，本 dialog 本身不可见；弹文件框/消息框若以
        # 它为父，关闭后焦点会落回不可见 dialog，主窗口被压到桌面最底层。
        # 统一改用可见的宿主窗口作父。
        self._host_window = parent
        self._is_preview_busy = False
        self._preview_result_stream = None
        self._preview_result_configs = None
        self._preview_result_segment_info = None
        self._preview_requested_configs = None
        self._preview_requested_segment_info = None
        self._preview_source_stream = None
        self._preview_worker = None
        self.preview_button = None
        self.preview_status = None
        self._build_ui()

    def _host(self):
        """返回真正可见的宿主窗口（主窗口），供弹窗作父窗口使用。

        独立运行时返回本 dialog 自身；嵌入到 DSSRRMainWindow 时返回主窗口，
        避免以不可见 dialog 为父导致关闭弹窗后主窗口掉到桌面最底层。
        """
        w = getattr(self, "_host_window", None)
        if w is None or not w.isVisible():
            return self
        return w

    def _build_ui(self):
        self._content_widget = QWidget(self)
        layout = QVBoxLayout(self._content_widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        hero_frame = QFrame()
        hero_frame.setObjectName("heroFrame")
        hero_layout = QVBoxLayout(hero_frame)
        hero_layout.setSpacing(4)

        # Title row with help button
        title_row = QHBoxLayout()
        title = QLabel(self.windowTitle())
        self._hero_title = title
        title.setStyleSheet("font-size:18px; font-weight:600;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        help_btn = QPushButton(self._tr("Help"))
        self._help_btn = help_btn
        help_btn.setToolTip(self._tr("Open usage guide"))
        help_btn.clicked.connect(self._show_help)
        title_row.addWidget(help_btn)
        hero_layout.addLayout(title_row)

        # Refresh instruction text for current language
        self._instruction_text = self._tr("Choose one method to suppress spikes/outliers. Use Preview to compare before and after.")
        self._preview_label = self._tr("Preview")
        intro = QLabel(self._instruction_text)
        intro.setWordWrap(True)
        hero_layout.addWidget(intro)
        layout.addWidget(hero_frame)

        # ---- Segment mode panel ----
        if self._segment_selection_available:
            self._segment_frame = self._build_segment_panel()

            # Vertical splitter: segment panel on top, content below
            self._v_splitter = QSplitter(Qt.Vertical, self)
            self._v_splitter.setChildrenCollapsible(False)
            self._v_splitter.setHandleWidth(8)
            self._v_splitter.addWidget(self._segment_frame)

        _c = get_theme_colors()

        self.content_splitter = QSplitter(Qt.Horizontal, self)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(8)
        if self._segment_selection_available:
            self._v_splitter.addWidget(self.content_splitter)
            self._v_splitter.setStretchFactor(0, 1)  # segment panel
            self._v_splitter.setStretchFactor(1, 3)  # content splitter
            # 段选择面板默认高度 300，避免文字/波形被压缩显示不全
            self._v_splitter.setSizes([300, 500])
            layout.addWidget(self._v_splitter, 1)
        else:
            layout.addWidget(self.content_splitter, 1)

        method_scroll = QScrollArea()
        method_scroll.setWidgetResizable(True)
        method_scroll.setMinimumWidth(260 if self._available_screen_width < 1000 else 300)
        method_widget = QWidget()
        self._method_widget = method_widget
        method_widget.setStyleSheet(f"background-color: {_c['surface']};")
        method_layout = QVBoxLayout(method_widget)
        method_layout.setContentsMargins(0, 0, 0, 0)
        method_layout.setSpacing(12)

        # Threshold truncation
        self.threshold_check = QCheckBox(self._tr("Threshold truncation"))
        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText(self._tr("3, >3, or <-3"))
        thresh_card, thresh_layout = self._create_method_card(self.threshold_check)
        thresh_form = QFormLayout()
        thresh_form.setContentsMargins(0, 0, 0, 0)
        thresh_form.setSpacing(8)
        thresh_form.addRow(QLabel(self._tr("Amplitude threshold:")), self.threshold_input)
        self._add_method_form(thresh_layout, self.threshold_check, thresh_form)
        method_layout.addWidget(thresh_card)

        # MAD cleaning
        self.mad_check = QCheckBox(self._tr("MAD-based cleaning"))
        self.mad_spin = QDoubleSpinBox()
        self.mad_spin.setRange(0.1, 20.0)
        self.mad_spin.setSingleStep(0.1)
        self.mad_spin.setValue(3.0)
        mad_card, mad_layout = self._create_method_card(self.mad_check)
        mad_form = QFormLayout()
        mad_form.setContentsMargins(0, 0, 0, 0)
        mad_form.setSpacing(8)
        mad_form.addRow(QLabel(self._tr("MAD threshold:")), self.mad_spin)
        self._add_method_form(mad_layout, self.mad_check, mad_form)
        method_layout.addWidget(mad_card)

        # Moving average cleaning
        self.ma_check = QCheckBox(self._tr("Moving-average cleaning"))
        self.ma_window_spin = QSpinBox()
        self.ma_window_spin.setRange(1, 10000)
        self.ma_window_spin.setValue(5)
        self.ma_amp_spin = QDoubleSpinBox()
        self.ma_amp_spin.setRange(0.1, 1e6)
        self.ma_amp_spin.setValue(3.0)
        ma_card, ma_layout = self._create_method_card(self.ma_check)
        ma_form = QFormLayout()
        ma_form.setContentsMargins(0, 0, 0, 0)
        ma_form.setSpacing(8)
        ma_form.addRow(QLabel(self._tr("Window size:")), self.ma_window_spin)
        ma_form.addRow(QLabel(self._tr("Amplitude threshold:")), self.ma_amp_spin)
        self._add_method_form(ma_layout, self.ma_check, ma_form)
        method_layout.addWidget(ma_card)

        # Conservative local repair for isolated spikes mixed into moonquake events.
        self.moonquake_spike_check = QCheckBox(
            self._tr("Moonquake-protected local spike repair"))
        self.moonquake_spike_threshold_spin = QDoubleSpinBox()
        self.moonquake_spike_threshold_spin.setRange(3.0, 20.0)
        self.moonquake_spike_threshold_spin.setSingleStep(0.5)
        self.moonquake_spike_threshold_spin.setValue(8.0)
        self.moonquake_spike_global_spin = QDoubleSpinBox()
        self.moonquake_spike_global_spin.setRange(1.0, 20.0)
        self.moonquake_spike_global_spin.setSingleStep(0.5)
        self.moonquake_spike_global_spin.setValue(4.0)
        self.moonquake_spike_detection_window_spin = QDoubleSpinBox()
        self.moonquake_spike_detection_window_spin.setRange(0.15, 5.0)
        self.moonquake_spike_detection_window_spin.setDecimals(2)
        self.moonquake_spike_detection_window_spin.setSingleStep(0.05)
        self.moonquake_spike_detection_window_spin.setValue(0.75)
        self.moonquake_spike_scale_window_spin = QDoubleSpinBox()
        self.moonquake_spike_scale_window_spin.setRange(1.0, 60.0)
        self.moonquake_spike_scale_window_spin.setSingleStep(0.5)
        self.moonquake_spike_scale_window_spin.setValue(4.0)
        self.moonquake_spike_width_spin = QSpinBox()
        self.moonquake_spike_width_spin.setRange(1, 8)
        self.moonquake_spike_width_spin.setValue(2)
        self.moonquake_spike_return_ratio_spin = QDoubleSpinBox()
        self.moonquake_spike_return_ratio_spin.setRange(0.05, 1.0)
        self.moonquake_spike_return_ratio_spin.setDecimals(2)
        self.moonquake_spike_return_ratio_spin.setSingleStep(0.05)
        self.moonquake_spike_return_ratio_spin.setValue(0.35)
        self.moonquake_spike_rescue_check = QCheckBox(
            self._tr("Enable strong-peak rescue"))
        self.moonquake_spike_rescue_check.setChecked(True)
        self.moonquake_spike_rescue_sigma_spin = QDoubleSpinBox()
        self.moonquake_spike_rescue_sigma_spin.setRange(8.0, 100.0)
        self.moonquake_spike_rescue_sigma_spin.setSingleStep(1.0)
        self.moonquake_spike_rescue_sigma_spin.setValue(16.0)
        self.moonquake_spike_rescue_global_spin = QDoubleSpinBox()
        self.moonquake_spike_rescue_global_spin.setRange(4.0, 30.0)
        self.moonquake_spike_rescue_global_spin.setSingleStep(1.0)
        self.moonquake_spike_rescue_global_spin.setValue(6.0)
        self.moonquake_spike_rescue_ratio_spin = QDoubleSpinBox()
        self.moonquake_spike_rescue_ratio_spin.setRange(1.0, 4.0)
        self.moonquake_spike_rescue_ratio_spin.setDecimals(2)
        self.moonquake_spike_rescue_ratio_spin.setSingleStep(0.05)
        self.moonquake_spike_rescue_ratio_spin.setValue(1.25)
        moonquake_card, moonquake_layout = self._create_method_card(
            self.moonquake_spike_check)
        moonquake_form = QFormLayout()
        moonquake_form.setContentsMargins(0, 0, 0, 0)
        moonquake_form.setSpacing(8)
        moonquake_form.addRow(
            QLabel(self._tr("Robust threshold (sigma):")),
            self.moonquake_spike_threshold_spin)
        moonquake_form.addRow(
            QLabel(self._tr("Minimum global amplitude (sigma):")),
            self.moonquake_spike_global_spin)
        moonquake_form.addRow(
            QLabel(self._tr("Detection window (s):")),
            self.moonquake_spike_detection_window_spin)
        moonquake_form.addRow(
            QLabel(self._tr("Scale window (s):")),
            self.moonquake_spike_scale_window_spin)
        moonquake_form.addRow(
            QLabel(self._tr("Maximum spike width (samples):")),
            self.moonquake_spike_width_spin)
        moonquake_form.addRow(
            QLabel(self._tr("Neighbour return ratio:")),
            self.moonquake_spike_return_ratio_spin)
        moonquake_form.addRow(
            self.moonquake_spike_rescue_check)
        moonquake_form.addRow(
            QLabel(self._tr("Strong-peak local threshold (sigma):")),
            self.moonquake_spike_rescue_sigma_spin)
        moonquake_form.addRow(
            QLabel(self._tr("Strong-peak global floor (sigma):")),
            self.moonquake_spike_rescue_global_spin)
        moonquake_form.addRow(
            QLabel(self._tr("Strong-peak neighbour ratio:")),
            self.moonquake_spike_rescue_ratio_spin)
        self._add_method_form(
            moonquake_layout, self.moonquake_spike_check, moonquake_form)
        method_layout.addWidget(moonquake_card)

        # Z-score cleaning
        self.z_check = QCheckBox(self._tr("Z-score cleaning"))
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0.1, 20.0)
        self.z_spin.setSingleStep(0.1)
        self.z_spin.setValue(3.0)
        z_card, z_layout = self._create_method_card(self.z_check)
        z_form = QFormLayout()
        z_form.setContentsMargins(0, 0, 0, 0)
        z_form.setSpacing(8)
        z_form.addRow(QLabel(self._tr("Z threshold:")), self.z_spin)
        self._add_method_form(z_layout, self.z_check, z_form)
        method_layout.addWidget(z_card)

        # Reuse DSSRR's existing inpainting routine for explicitly selected
        # missing/anomalous intervals.
        self.missing_value_interpolation_check = QCheckBox(
            self._tr("Missing-value interpolation"))
        self.missing_value_interpolation_combo = QComboBox()
        for label, method in [
            ("Linear", "linear"),
            ("Nearest", "nearest"),
            ("Quadratic", "quadratic"),
            ("Cubic", "cubic"),
            ("PCHIP", "pchip"),
        ]:
            self.missing_value_interpolation_combo.addItem(self._tr(label), method)
        missing_card, missing_layout = self._create_method_card(
            self.missing_value_interpolation_check)
        missing_form = QFormLayout()
        missing_form.setContentsMargins(0, 0, 0, 0)
        missing_form.setSpacing(8)
        missing_form.addRow(
            QLabel(self._tr("Interpolation method:")),
            self.missing_value_interpolation_combo)
        self._add_method_form(
            missing_layout, self.missing_value_interpolation_check, missing_form)
        method_layout.addWidget(missing_card)

        # Two-sided reference-spectrum repair for a manually selected anomaly.
        self.reference_spectrum_check = QCheckBox(
            self._tr("Two-sided reference spectrum repair"))
        # Reference-length defaults follow the DSSRR paper protocol:
        # L = clip(anomaly/2, 120 s, 600 s). The 120 s floor is the published
        # recommended default (8.3 mHz resolution at 6.625 Hz); shorter windows
        # are spectrally unstable. The spin boxes stay permissive so a user can
        # still override for a specific archive, but they now start at 120 s.
        self.reference_before_spin = QDoubleSpinBox()
        self.reference_before_spin.setRange(0.5, 600.0)
        self.reference_before_spin.setSingleStep(50.0)
        self.reference_before_spin.setValue(120.0)
        self.reference_after_spin = QDoubleSpinBox()
        self.reference_after_spin.setRange(0.5, 600.0)
        self.reference_after_spin.setSingleStep(50.0)
        self.reference_after_spin.setValue(120.0)
        self.reference_gap_spin = QDoubleSpinBox()
        self.reference_gap_spin.setRange(0.0, 60.0)
        self.reference_gap_spin.setSingleStep(0.1)
        self.reference_gap_spin.setValue(0.5)
        self.reference_auto_check = QCheckBox(
            self._tr("Auto reference length (from anomaly)"))
        self.reference_auto_check.setToolTip(self._tr(
            "When enabled, reference length = clip(anomaly/2, min, 600) s. "
            "Manual before/after fields are disabled."))
        self.reference_auto_check.setChecked(True)
        self.reference_min_spin = QDoubleSpinBox()
        self.reference_min_spin.setRange(30.0, 300.0)
        self.reference_min_spin.setSingleStep(10.0)
        self.reference_min_spin.setValue(120.0)
        self.reference_min_spin.setToolTip(self._tr(
            "Minimum auto reference length (s). Keep >=120 for a stable Welch "
            "PSD; the paper default range is 120--600 s."))
        self.reference_seed_spin = QSpinBox()
        self.reference_seed_spin.setRange(-1, 2147483647)
        self.reference_seed_spin.setValue(0)
        reference_card, reference_layout = self._create_method_card(
            self.reference_spectrum_check)
        reference_form = QFormLayout()
        reference_form.setContentsMargins(0, 0, 0, 0)
        reference_form.setSpacing(8)
        reference_form.addRow(self.reference_auto_check)
        self.reference_min_label = QLabel(self._tr("Auto min reference (s):"))
        reference_form.addRow(
            self.reference_min_label,
            self.reference_min_spin)
        self.reference_before_label = QLabel(self._tr("Reference before (s):"))
        self.reference_after_label = QLabel(self._tr("Reference after (s):"))
        reference_form.addRow(
            self.reference_before_label,
            self.reference_before_spin)
        reference_form.addRow(
            self.reference_after_label,
            self.reference_after_spin)
        reference_form.addRow(
            QLabel(self._tr("Safety gap (s):")),
            self.reference_gap_spin)
        reference_form.addRow(
            QLabel(self._tr("Random seed (-1 = random):")),
            self.reference_seed_spin)
        # 整数化开关：月震原始计数为整数值，参考频谱重建输出为实数；
        # 勾选后恢复数据四舍五入回整数（默认跟随去异常参数设置页
        # quantize_to_int，与修复库对比查看器手动重处理同口径）。
        self.reference_quantize_check = QCheckBox(
            self._tr("Quantize to int"))
        self.reference_quantize_check.setToolTip(
            self._tr("Round repaired values to integer counts on write-back. "
                     "Uncheck to keep float precision in this run."))
        from dssrr.repair_lib.repair_settings import load_settings
        self.reference_quantize_check.setChecked(
            bool(load_settings()["repair"].get("quantize_to_int", True)))
        reference_form.addRow(self.reference_quantize_check)
        self.reference_auto_check.toggled.connect(self._on_reference_auto_toggled)
        self._on_reference_auto_toggled(self.reference_auto_check.isChecked())
        self._add_method_form(
            reference_layout, self.reference_spectrum_check, reference_form)
        method_layout.addWidget(reference_card)

        self.layered_reference_spectrum_check = QCheckBox(
            self._tr("Layered reference spectrum repair"))
        self.layered_focus_mode_combo = QComboBox()
        self.layered_focus_mode_combo.addItem(
            self._tr("Ultra-low-frequency priority"), "ultra_low")
        self.layered_focus_mode_combo.addItem(
            self._tr("High-frequency priority"), "high")
        self.layered_focus_mode_combo.addItem(
            self._tr("Custom band"), "custom")
        self.layered_focus_mode_combo.currentIndexChanged.connect(
            self._on_layered_focus_mode_changed)
        self.layered_band_min_spin = QDoubleSpinBox()
        self.layered_band_min_spin.setRange(0.0, 100.0)
        self.layered_band_min_spin.setDecimals(4)
        self.layered_band_min_spin.setSingleStep(0.001)
        self.layered_band_min_spin.setValue(0.001)
        self.layered_band_max_spin = QDoubleSpinBox()
        self.layered_band_max_spin.setRange(0.0, 100.0)
        self.layered_band_max_spin.setDecimals(4)
        self.layered_band_max_spin.setSingleStep(0.001)
        self.layered_band_max_spin.setValue(0.08)
        self.layered_transition_spin = QDoubleSpinBox()
        self.layered_transition_spin.setRange(0.0, 20.0)
        self.layered_transition_spin.setDecimals(4)
        self.layered_transition_spin.setSingleStep(0.001)
        self.layered_transition_spin.setValue(0.01)
        self.layered_reference_before_spin = QDoubleSpinBox()
        self.layered_reference_before_spin.setRange(0.5, 600.0)
        self.layered_reference_before_spin.setSingleStep(50.0)
        self.layered_reference_before_spin.setValue(120.0)
        self.layered_reference_after_spin = QDoubleSpinBox()
        self.layered_reference_after_spin.setRange(0.5, 600.0)
        self.layered_reference_after_spin.setSingleStep(50.0)
        self.layered_reference_after_spin.setValue(120.0)
        self.layered_reference_gap_spin = QDoubleSpinBox()
        self.layered_reference_gap_spin.setRange(0.0, 60.0)
        self.layered_reference_gap_spin.setSingleStep(0.1)
        self.layered_reference_gap_spin.setValue(0.5)
        self.layered_reference_seed_spin = QSpinBox()
        self.layered_reference_seed_spin.setRange(-1, 2147483647)
        self.layered_reference_seed_spin.setValue(0)
        self._on_layered_focus_mode_changed()
        layered_card, layered_layout = self._create_method_card(
            self.layered_reference_spectrum_check)
        layered_form = QFormLayout()
        layered_form.setContentsMargins(0, 0, 0, 0)
        layered_form.setSpacing(8)
        layered_form.addRow(
            QLabel(self._tr("Frequency focus:")),
            self.layered_focus_mode_combo)
        layered_form.addRow(
            QLabel(self._tr("Band minimum (Hz):")),
            self.layered_band_min_spin)
        layered_form.addRow(
            QLabel(self._tr("Band maximum (Hz):")),
            self.layered_band_max_spin)
        layered_form.addRow(
            QLabel(self._tr("Transition width (Hz):")),
            self.layered_transition_spin)
        layered_form.addRow(
            QLabel(self._tr("Reference before (s):")),
            self.layered_reference_before_spin)
        layered_form.addRow(
            QLabel(self._tr("Reference after (s):")),
            self.layered_reference_after_spin)
        layered_form.addRow(
            QLabel(self._tr("Safety gap (s):")),
            self.layered_reference_gap_spin)
        layered_form.addRow(
            QLabel(self._tr("Random seed (-1 = random):")),
            self.layered_reference_seed_spin)
        # 整数化开关：月震原始计数为整数值，参考频谱重建输出为实数；
        # 勾选后恢复数据四舍五入回整数（默认跟随去异常参数设置页
        # quantize_to_int，与修复库对比查看器手动重处理同口径）。
        self.layered_quantize_check = QCheckBox(
            self._tr("Quantize to int"))
        self.layered_quantize_check.setToolTip(
            self._tr("Round repaired values to integer counts on write-back. "
                     "Uncheck to keep float precision in this run."))
        from dssrr.repair_lib.repair_settings import load_settings
        self.layered_quantize_check.setChecked(
            bool(load_settings()["repair"].get("quantize_to_int", True)))
        layered_form.addRow(self.layered_quantize_check)
        self._add_method_form(
            layered_layout,
            self.layered_reference_spectrum_check,
            layered_form)
        method_layout.addWidget(layered_card)

        self._method_checks = [
            self.threshold_check,
            self.mad_check,
            self.ma_check,
            self.moonquake_spike_check,
            self.z_check,
            self.missing_value_interpolation_check,
            self.reference_spectrum_check,
            self.layered_reference_spectrum_check,
        ]
        self._method_exclusive_guard = False
        for method_check in self._method_checks:
            method_check.toggled.connect(
                lambda checked, cb=method_check: self._on_method_toggled(cb, checked))

        method_layout.addStretch(1)

        method_scroll.setWidget(method_widget)

        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")
        preview_frame.setMinimumWidth(360 if self._available_screen_width < 1000 else 480)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_layout.setSpacing(10)

        preview_title = QLabel(self._tr("De-pulse Preview"))
        self._preview_title = preview_title
        preview_title.setStyleSheet("font-size:16px; font-weight:600;")
        preview_layout.addWidget(preview_title)

        self.preview_canvas = MplCanvas(preview_frame, width=6, height=3, dpi=100)
        self.preview_canvas.setMinimumHeight(180)
        self.preview_canvas.set_canvas_colors(bg_color=_c['surface'], fg_color=_c['text'])
        # 图例点击切换 Before/After 曲线显示（与查看器同机制）
        if not hasattr(self, "_preview_hidden_series"):
            self._preview_hidden_series = set()
        self.preview_canvas.mpl_connect("pick_event", self._on_preview_legend_pick)
        preview_layout.addWidget(self.preview_canvas, 1)

        self.preview_toolbar = NavigationToolbar(self.preview_canvas, preview_frame)
        self.preview_toolbar.setIconSize(QSize(18, 18))
        self.preview_toolbar.setStyleSheet(_nav_toolbar_qss(_is_dark()))
        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.addWidget(self.preview_toolbar)
        toolbar_layout.addStretch(1)
        preview_layout.addLayout(toolbar_layout)

        self.preview_status = QLabel()
        self.preview_status.setWordWrap(True)
        preview_layout.addWidget(self.preview_status)

        from PyQt5.QtWidgets import QProgressBar
        self.nn_progress = QProgressBar()
        self.nn_progress.setRange(0, 100)
        self.nn_progress.setValue(0)
        self.nn_progress.setVisible(False)
        self.nn_progress.setFormat(self._tr("Processing: %p%"))
        preview_layout.addWidget(self.nn_progress)

        self.content_splitter.addWidget(method_scroll)
        self.content_splitter.addWidget(preview_frame)
        self.content_splitter.setStretchFactor(0, 0)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([360, 840])

        # Bottom bar: Preview | Export | Screenshot | Save | Undo | stretch | Cancel | OK
        bottom_bar = QHBoxLayout()
        bottom_bar.setSpacing(8)
        self.preview_button = QPushButton(self._preview_label)
        self.preview_button.setObjectName("primaryButton")
        self.preview_button.clicked.connect(self._handle_preview)
        bottom_bar.addWidget(self.preview_button)

        self.cancel_preview_btn = QPushButton(self._tr("Stop"))
        self.cancel_preview_btn.setObjectName("primaryButton")
        self.cancel_preview_btn.setVisible(False)
        self.cancel_preview_btn.clicked.connect(self._cancel_preview)
        bottom_bar.addWidget(self.cancel_preview_btn)

        # Screenshot button - copy to clipboard only
        self.screenshot_btn = QPushButton(self._tr("Screenshot"))
        self.screenshot_btn.setObjectName("compactActionButton")
        self.screenshot_btn.setToolTip(self._tr("Copy preview to clipboard"))
        self.screenshot_btn.clicked.connect(self._screenshot_to_clipboard)
        bottom_bar.addWidget(self.screenshot_btn)

        self.save_current_btn = QPushButton(self._tr("Save current repair"))
        self.save_current_btn.setObjectName("compactActionButton")
        self.save_current_btn.setProperty("action", "save")
        self.save_current_btn.setToolTip(self._tr(
            "Commit the current preview inside this dialog. It will not change the main window."))
        self.save_current_btn.clicked.connect(self._save_current_repair)
        bottom_bar.addWidget(self.save_current_btn)

        self.undo_current_btn = QPushButton(self._tr("Undo last repair"))
        self.undo_current_btn.setObjectName("compactActionButton")
        self.undo_current_btn.setToolTip(self._tr(
            "Undo the latest repair saved inside this dialog."))
        self.undo_current_btn.clicked.connect(self._undo_current_repair)
        bottom_bar.addWidget(self.undo_current_btn)

        bottom_bar.addStretch(1)

        # 导出报告：图形化修复报告（可设标题/尺寸/频段，带实时预览）
        self.export_report_btn = QPushButton(self._tr("Export Report..."))
        self.export_report_btn.setObjectName("compactActionButton")
        self.export_report_btn.setToolTip(self._tr(
            "Export a graphical repair report: configure title, size and "
            "frequency range, preview it, then save as PNG/PDF/SVG."))
        self.export_report_btn.clicked.connect(self._export_report)
        bottom_bar.addWidget(self.export_report_btn)

        # Cancel and OK/Save buttons
        self.cancel_btn = QPushButton(self._tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        bottom_bar.addWidget(self.cancel_btn)

        # 原先的 Save / Export As... 两个按钮都走同一个 _export_result
        # （同一个文件对话框，仅 _save_mode 字符串不同），功能完全重复。
        # 合并为一个导出按钮，语义明确：导出修复后的波形数据文件。
        if self._segment_mode_enabled:
            self.save_as_btn = QPushButton(self._tr("Export Waveform..."))
            self.save_as_btn.setObjectName("primaryButton")
            self.save_as_btn.setToolTip(self._tr(
                "Write the repaired waveform to a file (MiniSEED/SAC/TXT/ASCII)."))
            self.save_as_btn.clicked.connect(lambda: self._set_save_mode("save_as"))
            bottom_bar.addWidget(self.save_as_btn)
        else:
            self.ok_btn = QPushButton(self._tr("OK"))
            self.ok_btn.setObjectName("primaryButton")
            self.ok_btn.clicked.connect(lambda: self._set_save_mode("replace"))
            bottom_bar.addWidget(self.ok_btn)

        layout.addLayout(bottom_bar)

        self._draw_preview_placeholder()
        self._update_repair_action_state()

        outer_layout = self.layout()
        if outer_layout is None:
            outer_layout = QVBoxLayout(self)
            outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self._content_widget)

    def _apply_theme_style(self):
        """Apply the dialog-level theme stylesheet after a theme change."""
        colors = self._theme_colors
        self.setStyleSheet(f"""
            QDialog {{ background-color: {colors['bg']}; color: {colors['text']}; }}
            QLabel {{ background: transparent; color: {colors['text']}; }}
            #heroFrame, #previewFrame, #methodCard {{
                border: 1px solid {colors['border']};
                border-radius: 12px;
                background-color: {colors['surface']};
            }}
            QCheckBox {{ color: {colors['text']}; font-weight: 600; }}
            QCheckBox::indicator {{ width: 14px; height: 14px; }}
            QCheckBox::indicator:unchecked {{
                background-color: {colors['input_bg']};
                border: 1px solid {colors['input_border']};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['accent']};
                border: 1px solid {colors['accent']};
                border-radius: 3px;
            }}
            QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{
                background-color: {colors['input_bg']};
                color: {colors['text']};
                border: 1px solid {colors['input_border']};
                padding: 6px;
                border-radius: 6px;
            }}
            QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled, QComboBox:disabled {{
                background-color: {colors['btn_disabled']};
                color: {colors['btn_disabled_text']};
                border: 1px solid {colors['input_border']};
            }}
            QLabel:disabled {{
                color: {colors['btn_disabled_text']};
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors['surface2']}; color: {colors['text']};
                selection-background-color: {colors['accent']};
                selection-color: {colors['btn_primary_text']};
            }}
            QScrollArea, QScrollArea > QWidget#qt_scrollarea_viewport {{
                border: none; background-color: {colors['surface']};
            }}
            QProgressBar {{
                background-color: {colors['input_bg']}; color: {colors['text']};
                border: 1px solid {colors['input_border']}; border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{ background-color: {colors['accent']}; }}
            QPushButton {{ color: {colors['text']}; }}
            QPushButton#primaryButton {{
                background-color: {colors['accent']}; color: {colors['btn_primary_text']};
            }}
            QPushButton#compactActionButton {{
                min-width: 96px; max-width: 132px; min-height: 22px;
                padding: 4px 10px; font-size: 12px;
                background-color: {colors['surface2']}; color: {colors['text']};
                border: 1px solid {colors['border']};
            }}
            QPushButton#compactActionButton[action="save"] {{
                background-color: {colors['accent_bg']}; color: {colors['accent']};
                border-color: {colors['accent']};
            }}
            QPushButton#compactActionButton:disabled {{
                background-color: {colors['btn_disabled']};
                color: {colors['btn_disabled_text']};
                border-color: {colors['border']};
            }}
        """)

    def _refresh_inline_theme(self):
        """重刷写死颜色的 inline 样式（换主题时这些不会自动跟随）。

        _apply_theme_style() 只覆盖 objectName 规则；直接 setStyleSheet 写在
        控件上的颜色（标题、状态文本、停止按钮、提示语等）需要单独重设，
        否则切换主题后这些元素会停留在旧配色上。
        """
        c = self._theme_colors
        if getattr(self, "_hero_title", None) is not None:
            self._hero_title.setStyleSheet(
                f"font-size:18px; font-weight:600; color: {c['text']};")
        if getattr(self, "_help_btn", None) is not None:
            self._help_btn.setStyleSheet(
                f"QPushButton {{ background: {c['accent_bg']}; color: {c['accent']}; "
                f"border-radius: 4px; padding: 3px 12px; font-weight: 600; font-size: 12px; }}"
                f"QPushButton:hover {{ background: {c['accent']}; color: {c['btn_primary_text']}; }}")
        if getattr(self, "_method_widget", None) is not None:
            self._method_widget.setStyleSheet(f"background-color: {c['surface']};")
        if getattr(self, "_preview_title", None) is not None:
            self._preview_title.setStyleSheet(
                f"font-size:16px; font-weight:600; color: {c['text']};")
        if getattr(self, "preview_status", None) is not None:
            self.preview_status.setStyleSheet(f"color: {c['text_secondary']};")
        if getattr(self, "cancel_preview_btn", None) is not None:
            self.cancel_preview_btn.setStyleSheet(
                f"QPushButton#primaryButton {{ background-color: {c['error']}; color: #ffffff; }}"
                f"QPushButton#primaryButton:disabled {{ background-color: {c['btn_disabled']}; color: {c['btn_disabled_text']}; }}")
        if getattr(self, "_seg_title", None) is not None:
            self._seg_title.setStyleSheet(
                f"font-size:14px; font-weight:600; color: {c['text']};")
        if getattr(self, "_seg_selection_label", None) is not None:
            self._seg_selection_label.setStyleSheet(
                f"color: {c['text_secondary']}; font-size: 11px; padding-left: 4px;")
        if getattr(self, "_seg_hint", None) is not None:
            # 不能直接写死"弱化色"：平移/缩放激活时提示语要显示成警告色，
            # 交给 _refresh_seg_hint() 按当前导航模式决定文案与配色。
            self._refresh_seg_hint()
        # 两个 matplotlib 画布跟随主题
        for canvas_name in ("preview_canvas", "_seg_canvas"):
            canvas = getattr(self, canvas_name, None)
            if canvas is not None:
                canvas.set_canvas_colors(c['surface'], c['text'])
        # matplotlib 导航工具栏：QToolBar/QToolButton 不吃父级 QSS 传播，
        # 必须重新 setStyleSheet，否则换肤后仍是白色底。
        tb_qss = _nav_toolbar_qss(_is_dark())
        for tb_name in ("preview_toolbar", "_seg_toolbar"):
            tb = getattr(self, tb_name, None)
            if tb is not None:
                tb.setStyleSheet(tb_qss)

    def set_tool_theme(self, *_args, **_kwargs):
        """Refresh this already-open dialog when the main theme changes."""
        self._theme_colors = get_theme_colors()
        self._apply_theme_style()
        self._refresh_inline_theme()
        self.retranslate_ui()

    def _create_method_card(self, checkbox):
        card = QFrame()
        card.setObjectName("methodCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(10)
        header = QHBoxLayout()
        header.addWidget(checkbox)
        header.addStretch(1)
        card_layout.addLayout(header)
        return card, card_layout

    @staticmethod
    def _add_method_form(card_layout, checkbox, form_layout):
        """Show method parameters only while the corresponding method is active."""
        container = QWidget()
        container.setLayout(form_layout)
        container.setVisible(checkbox.isChecked())
        checkbox.toggled.connect(container.setVisible)
        card_layout.addWidget(container)

    def _on_method_toggled(self, selected_checkbox, checked):
        """Keep the de-pulse method list mutually exclusive in the UI."""
        if self._method_exclusive_guard or not checked:
            return
        self._method_exclusive_guard = True
        try:
            for checkbox in self._method_checks:
                if checkbox is not selected_checkbox and checkbox.isChecked():
                    checkbox.setChecked(False)
        finally:
            self._method_exclusive_guard = False

    def _draw_preview_placeholder(self, message=None):
        if message is None:
            message = self._instruction_text
        if hasattr(self, "preview_canvas") and self.preview_canvas:
            if hasattr(self.preview_canvas, "reset_axes"):
                ax = self.preview_canvas.reset_axes()
            else:
                self.preview_canvas.fig.clear()
                ax = self.preview_canvas.fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, message, ha='center', va='center',
                    transform=ax.transAxes, color="#6b7280", fontsize=10, wrap=True)
            self.preview_canvas.draw()
        if self.preview_status:
            self.preview_status.setText(message)

    def showEvent(self, event):
        super().showEvent(event)
        # Dynamic segment content needs one layout pass after the dialog has
        # been shown; otherwise the first paint can use stale splitter sizes.
        QTimer.singleShot(0, self._refresh_initial_layout)
        QTimer.singleShot(100, self._refresh_initial_layout)

    def _refresh_initial_layout(self):
        if not self.isVisible():
            return
        layout = self.layout()
        if layout is not None:
            layout.activate()
        self.updateGeometry()
        if hasattr(self, "content_splitter"):
            available_width = max(1, self.content_splitter.width())
            left_width = min(320, max(260, int(available_width * 0.34)))
            self.content_splitter.setSizes([left_width, max(1, available_width - left_width)])
        if hasattr(self, "_seg_canvas"):
            self._seg_canvas.updateGeometry()
            self._seg_canvas.draw_idle()

    def _set_preview_busy(self, busy: bool):
        if not self.preview_button:
            return
        if busy and not self._is_preview_busy:
            self._is_preview_busy = True
            self.preview_button.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
        elif not busy and self._is_preview_busy:
            self._is_preview_busy = False
            self.preview_button.setEnabled(True)
            QApplication.restoreOverrideCursor()
        self._update_repair_action_state()

    def _handle_accept(self):
        if not self.get_selected_methods():
            QMessageBox.information(self, self._tr("Information"), self._tr("Please select at least one method."))
            return
        if not self._validate_reference_spectrum_selection():
            return
        self.accept()

    def _preview_matches_current_settings(self):
        if self._preview_result_stream is None:
            return False
        return self.get_preview_result(self.get_selected_methods()) is not None

    def _update_repair_action_state(self):
        """Keep the compact workflow actions aligned with the current dialog state."""
        save_button = getattr(self, "save_current_btn", None)
        undo_button = getattr(self, "undo_current_btn", None)
        if save_button is not None:
            save_button.setEnabled(
                not self._is_preview_busy and self._preview_matches_current_settings())
        if undo_button is not None:
            undo_button.setEnabled(bool(self._repair_history) and not self._is_preview_busy)

    def _screenshot_to_clipboard(self):
        """Copy the preview figure directly to clipboard."""
        if not self.preview_canvas or not self.preview_canvas.fig:
            QMessageBox.warning(self, self._tr("Warning"),
                                self._tr("No preview available. Run de-pulse first."))
            return
        try:
            buf = io.BytesIO()
            self.preview_canvas.fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = QImage.fromData(buf.read())
            QApplication.clipboard().setImage(img)
            QMessageBox.information(self, self._tr("Success"),
                                    self._tr("Preview copied to clipboard."))
        except Exception as e:
            QMessageBox.warning(self, self._tr("Error"),
                                self._tr("Failed to copy to clipboard: {}").format(e))

    def _save_current_repair(self):
        """Commit the latest preview inside this dialog without touching the main window."""
        if self._is_preview_busy:
            return
        if not self._preview_matches_current_settings():
            if self.preview_status:
                self.preview_status.setText(self._tr("Preview a repair before saving it."))
            self._update_repair_action_state()
            return

        self._repair_history.append(self._working_stream.copy())
        self._working_stream = self._preview_result_stream.copy()
        self._preview_result_stream = None
        self._preview_result_configs = None
        self._preview_result_segment_info = None
        self._preview_requested_configs = None
        self._preview_requested_segment_info = None
        self._preview_source_stream = None
        if self.preview_status:
            self.preview_status.setText(self._tr("Current repair saved."))
        self._update_repair_action_state()

    def _undo_current_repair(self):
        """Undo the latest repair committed inside this dialog."""
        if self._is_preview_busy:
            return
        if not self._repair_history:
            if self.preview_status:
                self.preview_status.setText(self._tr("Nothing to undo."))
            self._update_repair_action_state()
            return

        self._working_stream = self._repair_history.pop().copy()
        self._preview_result_stream = None
        self._preview_result_configs = None
        self._preview_result_segment_info = None
        self._preview_requested_configs = None
        self._preview_requested_segment_info = None
        self._preview_source_stream = None
        if self.preview_status:
            self.preview_status.setText(self._tr("Last saved repair undone."))
        self._update_repair_action_state()

    def _handle_preview(self):
        configs = self.get_selected_methods()
        if not configs:
            QMessageBox.information(self, self._tr("Information"), self._tr("Please select at least one method."))
            self._draw_preview_placeholder(self._tr("Please select at least one method."))
            return
        if not self._validate_reference_spectrum_selection():
            return
        # DSSRR standalone: preview is always available.

        # Clean up any previous worker
        self._cleanup_worker()
        self._preview_result_stream = None
        self._preview_result_configs = None
        self._preview_result_segment_info = None
        self._preview_requested_configs = configs
        self._preview_requested_segment_info = self.get_segment_info()

        preview_source = (
            self._working_stream
            if self._working_stream is not None
            else getattr(self, "base_stream", None)
        )
        if preview_source is None or len(preview_source) == 0:
            return
        self._preview_source_stream = preview_source.copy()
        base_stream = self._preview_source_stream
        # 预览针对"当前选中的 trace"，不是永远第一个。多文件导入时
        # _current_trace_index 指向下拉框当前条目，绘制与 worker 入参都必须
        # 跟随它，否则切到第二条数据仍显示第一条。
        cur = min(max(0, self._current_trace_index), len(base_stream) - 1)
        data = np.asarray(base_stream[cur].data, dtype=float)

        self._set_preview_busy(True)
        if self.cancel_preview_btn:
            self.cancel_preview_btn.setVisible(True)
        self.nn_progress.setVisible(True)
        self.nn_progress.setValue(0)
        self.nn_progress.setFormat(self._tr("Processing: %p%"))
        if self.preview_status:
            self.preview_status.setText(self._tr("Processing..."))

        self._preview_worker = _DePulsePreviewWorker(
            data, configs, base_stream, self._preview_requested_segment_info)
        self._preview_worker.progress.connect(self._on_preview_progress)
        self._preview_worker.finished.connect(self._on_preview_done)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.start()

    def _on_preview_progress(self, current, total, status):
        if hasattr(self, 'nn_progress') and self.nn_progress:
            progress_percent = int(round(current / max(1, total) * 100))
            self.nn_progress.setValue(min(100, max(0, progress_percent)))
            if status:
                self.nn_progress.setFormat(
                    self._tr("Processing {method}: %p%").format(method=status))
            else:
                self.nn_progress.setFormat(self._tr("Processing: %p%"))
        if self.preview_status:
            self.preview_status.setText(
                self._tr("Processing {method}").format(method=status)
                if status else self._tr("Processing..."))

    def _on_preview_done(self, processed_stream):
        self._set_preview_busy(False)
        if self.cancel_preview_btn:
            self.cancel_preview_btn.setVisible(False)
        if hasattr(self, 'nn_progress') and self.nn_progress:
            self.nn_progress.setVisible(False)
        try:
            base_stream = self._preview_source_stream or self.base_stream
            if base_stream and len(base_stream) > 0 and processed_stream and len(processed_stream) > 0:
                self._preview_result_stream = processed_stream.copy()
                self._preview_result_configs = (
                    self._preview_requested_configs
                    if self._preview_requested_configs is not None
                    else self.get_selected_methods())
                self._preview_result_segment_info = self._preview_requested_segment_info
                # 与 _handle_preview 一致：按当前选中的 trace 绘制，
                # 避免多文件导入时预览区永远显示第一条。
                cur = min(max(0, self._current_trace_index),
                          len(base_stream) - 1)
                orig_trace = base_stream[cur]
                result = np.asarray(processed_stream[cur].data, dtype=float)
                npts = min(len(orig_trace.data), len(result))
                sampling_rate = orig_trace.stats.sampling_rate or 1.0
                times = np.arange(npts) / sampling_rate
                orig_data = np.asarray(orig_trace.data[:npts], dtype=float)
                proc_data = result[:npts]
                if hasattr(self.preview_canvas, "reset_axes"):
                    ax = self.preview_canvas.reset_axes(
                        bg_color=self._theme_colors['surface'],
                        fg_color=self._theme_colors['text'])
                else:
                    self.preview_canvas.fig.clear()
                    ax = self.preview_canvas.fig.add_subplot(111)
                ax.plot(times, orig_data, color='#94a3b8', linewidth=1.2, label=self._tr("Before"))
                ax.plot(times, proc_data, color='#2563eb', linewidth=1.0, label=self._tr("After"))
                changed = np.abs(proc_data - orig_data) > np.finfo(float).eps
                if changed.any():
                    lower = min(np.nanmin(orig_data), np.nanmin(proc_data))
                    upper = max(np.nanmax(orig_data), np.nanmax(proc_data))
                    ax.fill_between(times, lower, upper, where=changed,
                                    color='#f59e0b', alpha=0.12, linewidth=0)
                if self.preview_status:
                    self.preview_status.setText(
                        self._tr("Modified samples: {count:,} ({ratio:.3f}%)").format(
                            count=int(np.count_nonzero(changed)),
                            ratio=100 * np.count_nonzero(changed) / max(1, npts)))
                self._update_repair_action_state()
                ax.set_title(self._tr("De-pulse Preview"))
                ax.set_xlabel(self._tr("Time (s)"))
                ax.set_ylabel(self._tr("Amplitude"))
                ax.legend()
                self._apply_preview_series_visibility()
                ax.grid(True, linestyle='--', alpha=0.5, color='#d5dbe8')
                self.preview_canvas.draw()
        except Exception as exc:
            QMessageBox.warning(self, self._tr("Preview Failed"), str(exc))
            self._draw_preview_placeholder(self._tr("Preview Failed"))
            self._update_repair_action_state()

    def _on_preview_failed(self, error):
        self._set_preview_busy(False)
        if self.cancel_preview_btn:
            self.cancel_preview_btn.setVisible(False)
        if hasattr(self, 'nn_progress') and self.nn_progress:
            self.nn_progress.setVisible(False)
        QMessageBox.warning(self, self._tr("Preview Failed"), error)
        self._draw_preview_placeholder(self._tr("Preview Failed"))
        self._update_repair_action_state()

    # ---------------- 图例点击切换曲线（Before/After） ----------------
    def _apply_preview_series_visibility(self):
        """渲染后应用图例隐藏状态：为图例曲线启用点击(pick)，并按
        self._preview_hidden_series 隐藏对应数据曲线。"""
        for ax in self.preview_canvas.figure.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            for ln in leg.get_lines():
                ln.set_picker(5)
            for ln in ax.get_lines():
                if ln.get_label() in self._preview_hidden_series:
                    ln.set_visible(False)

    def _on_preview_legend_pick(self, event):
        """点击图例曲线：切换该曲线的显示/隐藏（仅重绘，不重建）。"""
        artist = getattr(event, "artist", None)
        if artist is None:
            return
        label = artist.get_label()
        if not label or label.startswith("_"):
            return
        if label in self._preview_hidden_series:
            self._preview_hidden_series.discard(label)
        else:
            self._preview_hidden_series.add(label)
        vis = label not in self._preview_hidden_series
        for ax in self.preview_canvas.figure.axes:
            for ln in ax.get_lines():
                if ln.get_label() == label:
                    ln.set_visible(vis)
        self.preview_canvas.draw_idle()

    def _cancel_preview(self):
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.cancel()
            if self.preview_status:
                self.preview_status.setText(self._tr("Cancelling..."))

    def _validate_reference_spectrum_selection(self):
        """Two-sided reference spectrum repair requires a manually selected core."""
        selected = self.get_segment_info()
        uses_selected_regions = (
            getattr(self, "missing_value_interpolation_check", None) is not None
            and self.missing_value_interpolation_check.isChecked()
        ) or (
            (
                getattr(self, "reference_spectrum_check", None) is not None
                and self.reference_spectrum_check.isChecked()
            )
            or (
                getattr(self, "layered_reference_spectrum_check", None) is not None
                and self.layered_reference_spectrum_check.isChecked()
            )
        )
        if uses_selected_regions and not (
            selected and selected.get("enabled") and selected.get("time_range")
        ):
            QMessageBox.information(
                self._host(),
                self._tr("Information"),
                self._tr(
                    "This repair method requires a selected segment."
                ),
            )
            return False
        return True

    def _cleanup_worker(self):
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.cancel()
            self._preview_worker.wait(5000)
        self._preview_worker = None

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _show_help(self):
        """Show the usage guide dialog."""
        from PyQt5.QtWidgets import QTextBrowser, QDialog, QVBoxLayout

        help_dialog = QDialog(self)
        help_dialog.setWindowTitle(self._tr("Usage Guide"))
        help_dialog.setMinimumSize(700, 560)
        help_dialog.resize(900, 720)

        layout = QVBoxLayout(help_dialog)
        layout.setContentsMargins(16, 16, 16, 16)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(self._help_html())
        layout.addWidget(browser)

        close_btn = QPushButton(self._tr("Close"))
        close_btn.clicked.connect(help_dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        help_dialog.exec_()

    def _legacy_help_html(self):
        tr = self._tr
        return f"""<div style="font-size:13px; line-height:1.7; color:#333;">
<h2 style="color:#1e40af;">{tr('Basic Steps')}</h2>
<ol>
<li><b>{tr('Step 1: Select segment')}</b> — {tr('Enable segment selection, drag left-click to select the anomaly region. Right-click to add more regions. Double-click to clear.')}</li>
<li><b>{tr('Step 2: Choose method')}</b> — {tr('Check one method on the left. Selecting another method automatically clears the previous one.')}</li>
<li><b>{tr('Step 3: Preview')}</b> — {tr('Click "Preview" to see before/after comparison. The blue line shows the result.')}</li>
<li><b>{tr('Step 4: Confirm')}</b> — {tr('Click "OK" to apply. The result replaces the original data. You can undo from the main window.')}</li>
</ol>

<h2 style="color:#1e40af; margin-top:16px;">{tr('Methods Overview')}</h2>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%; font-size:12px;">
<tr style="background:#e0e7ff;"><th style="text-align:left;">{tr('Method')}</th><th style="text-align:left;">{tr('Effect')}</th><th style="text-align:left;">{tr('Recommended Settings')}</th></tr>

<tr><td><b>{tr('Threshold truncation')}</b></td>
<td>{tr('Clips values exceeding the threshold to the threshold level. Good for hard saturation (e.g., ADC max).')}</td>
<td>{tr('Threshold: set to instrument max (e.g., 1023 for 10-bit ADC).')}</td></tr>

<tr><td><b>{tr('MAD-based cleaning')}</b></td>
<td>{tr('Detects outliers using Median Absolute Deviation. Robust to non-Gaussian noise.')}</td>
<td>{tr('MAD threshold: 3.0 (default) catches ~0.3% outliers. Increase to 5.0 for fewer false positives.')}</td></tr>

<tr><td><b>{tr('Z-score cleaning')}</b></td>
<td>{tr('Detects outliers using Z-score (standard deviations from mean). Fast but sensitive to extreme values.')}</td>
<td>{tr('Z threshold: 3.0 catches ~0.3%. Use 4.0–5.0 for conservative cleaning.')}</td></tr>

<tr><td><b>{tr('Moving-average cleaning')}</b></td>
<td>{tr('Compares each point to its local moving average. Deviations above threshold are flagged.')}</td>
<td>{tr('Window: 5–20 samples. Amplitude threshold: 2.0–5.0.')}</td></tr>

<tr><td><b>{tr('Reference-spectrum replacement')}</b></td>
<td>{tr('Synthesizes replacement signal from spectral features of healthy segments before/after the anomaly. Best for complex anomalies with specific frequency content.')}</td>
<td>{tr('Before/After reference: 50–200s. Safety gap: 0.5–5s. Random seed: use fixed value for debugging, -1 for production.')}</td></tr>
</table>

<h2 style="color:#1e40af; margin-top:16px;">{tr('Parameter Reference')}</h2>

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; width:100%; font-size:12px;">
<tr style="background:#e0e7ff;"><th style="text-align:left;">{tr('Parameter')}</th><th style="text-align:left;">{tr('Description')}</th><th style="text-align:left;">{tr('Range')}</th></tr>
<tr><td>{tr('Amplitude threshold')}</td><td>{tr('Maximum allowed absolute value. Points above are clipped.')}</td><td>0 – ∞</td></tr>
<tr><td>{tr('MAD threshold')}</td><td>{tr('Outlier threshold in MAD units (1 MAD ≈ 0.6745 σ).')}</td><td>1.0 – 10.0 (default 3.0)</td></tr>
<tr><td>{tr('Z threshold')}</td><td>{tr('Outlier threshold in standard deviations.')}</td><td>1.0 – 10.0 (default 3.0)</td></tr>
<tr><td>{tr('Window size')}</td><td>{tr('Moving average window length in samples.')}</td><td>3 – 100 (default 5)</td></tr>
<tr><td>{tr('Before reference (s)')}</td><td>{tr('Length of reference segment before anomaly for spectral estimation.')}</td><td>0.5 – 600 (default 100)</td></tr>
<tr><td>{tr('After reference (s)')}</td><td>{tr('Length of reference segment after anomaly.')}</td><td>0.5 – 600 (default 100)</td></tr>
<tr><td>{tr('Safety gap (s)')}</td><td>{tr('Minimum distance between reference segment and anomaly boundary.')}</td><td>0 – 30 (default 0.5)</td></tr>
<tr><td>{tr('Random seed')}</td><td>{tr('Controls reproducibility of synthesized signal. Fixed value = reproducible. -1 = random each time.')}</td><td>-1, 0 – 9999</td></tr>
</table>

<h2 style="color:#1e40af; margin-top:16px;">{tr('Tips')}</h2>
<ul>
<li>{tr('Choose one method, preview, then switch methods if the result is not satisfactory.')}</li>
<li>{tr('For saturation spikes: Threshold truncation is fastest. Reference-spectrum gives smoother results.')}</li>
<li>{tr('For complex waveforms: reference-spectrum replacement preserves signal character better than simple interpolation.')}</li>
<li>{tr('Always preview before confirming. The preview shows exactly what will change.')}</li>
<li>{tr('Results can be undone from the main window toolbar (Undo button).')}</li>
</ul>
</div>"""

    def _help_css(self):
        """Use readable colors for the help browser in both themes."""
        colors = self._theme_colors
        dark = QColor(colors["bg"]).lightness() < 128
        if dark:
            surface = colors["surface"]
            text = "#F3F4F6"
            muted = "#CBD5E1"
            border = "#718096"
            header = "#3A4658"
            callout = "#263449"
            code_bg = colors["input_bg"]
            warning = "#4A321C"
            accent = "#76A9FA"
        else:
            surface = "#ffffff"
            text = "#243247"
            muted = "#475569"
            border = "#b8c4d6"
            header = "#dbe8ff"
            callout = "#eef5ff"
            code_bg = "#f1f5f9"
            warning = "#fff7ed"
            accent = colors["accent"]
        return """<style>
body {{ font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif; background:{surface}; color:{text}; font-size:13px; }}
h2 {{ color:{accent}; margin:8px 0 6px; font-size:20px; }}
h3 {{ color:{accent}; margin:16px 0 5px; font-size:15px; }}
p, li {{ color:{text}; line-height:1.55; }}
.lead {{ background:{callout}; border-left:4px solid {accent}; padding:10px 12px; }}
.warn {{ background:{warning}; border-left:4px solid {warn}; padding:10px 12px; }}
table {{ border-collapse:collapse; width:100%; margin:6px 0 12px; }}
th {{ background:{header}; color:{text}; text-align:left; }}
th, td {{ border:1px solid {border}; color:{text}; padding:7px; vertical-align:top; }}
code {{ background:{code_bg}; color:{text}; padding:1px 3px; }}
</style>""".format(
            surface=surface, text=text, accent=accent, muted=muted,
            border=border, header=header, callout=callout, code_bg=code_bg,
            warning=warning, warn=colors["warning"])

    def _help_html(self):
        """Return a complete guide in the active language."""
        # The guide is intentionally rendered as one language.  Translating
        # only headings leaves long explanatory cells in English when a source
        # entry is missing from the .ts file.
        zh = self._tr("Anomaly Repair") == "异常修复工具"
        if zh:
            return self._help_css() + """
<h2>去异常：快速使用指南</h2>
<p class="lead"><b>目标：</b>只处理确定的异常样本，同时尽量保留真实地震信号。检测结果只是候选，不代表这些样本一定应该删除。</p>
<h3>30 秒上手</h3>
<ol><li><b>需要局部修复时先框选。</b>勾选“启用分段选择”，在波形上左键拖动选择异常核心区；右键可追加区域，双击可清空。</li><li><b>只选择一种方法。</b>先按下方建议选择方法和参数，不确定时优先使用 MAD 或插值，不要直接使用强重建方法。</li><li><b>必须先预览。</b>确认时域曲线、接缝和频谱没有异常放大，再点击“确定”或“保存”。应用后可从主界面撤销。</li></ol>
<h3>先按现象选方法</h3>
<p class="lead"><b>分层参考频谱修复适用于：</b>异常段较长、需要人工框选异常核心，并且你只关心某个研究频带的情况。它在双侧参考频谱修复基础上，进一步优先保留超低频、高频或自定义频带；它不是自动异常检测方法。</p>
<table><tr><th>你看到的现象</th><th>首选方法</th><th>原因与注意事项</th></tr>
<tr><td>混杂在月震事件中的孤立尖峰或短饱和平台</td><td>月震保护型局部尖峰修复</td><td>常规路径修复前后快速回归的短尖峰；强尖峰救援路径只处理远高于稳健背景的峰顶，以及短的重复满量程平台，不替换整个月震事件。</td></tr>
<tr><td>单个尖峰、短脉冲</td><td>MAD 清理 + 线性/三次填补</td><td>稳健、快速。阈值过低会误伤真实峰值。</td></tr>
<tr><td>接近仪器满量程的平台或削顶</td><td>阈值截断或双侧参考频谱修复</td><td>阈值应接近实际量程；先确认不是正常的大振幅事件。</td></tr>
<tr><td>长段污染、希望保持频带特征</td><td>双侧参考频谱修复</td><td>必须框选异常区，并确保前后参考段是健康且相对平稳的。</td></tr></table>
<h3>每种方法怎么理解</h3>
<p class="lead"><b>分层参考频谱修复：</b>先用异常前后的健康参考段生成替代波形，再对指定频带施加参考段引导层。选择“超低频优先”可保留长周期/月震低频趋势；选择“高频优先”可保留短周期纹理；“自定义频带”用于明确的研究频段。频带外仍保留部分背景纹理。</p>
<table><tr><th>方法</th><th>它做什么</th><th>建议起点</th></tr><tr><td><b>分层参考频谱修复</b></td><td>在双侧参考频谱修复结果上进一步强化指定研究频带，同时保留部分频带外背景纹理。</td><td>超低频优先 <code>0～0.08 Hz</code>；高频优先 <code>0.25 Hz</code> 到 Nyquist；有明确目标时使用自定义频带。</td></tr></table>
<table><tr><th>方法</th><th>它做什么</th><th>建议起点</th></tr>
<tr><td><b>阈值截断</b></td><td>将超过幅值上限的样本压回阈值。</td><td>阈值填仪器真实最大值；只适合明确的饱和问题。</td></tr>
<tr><td><b>MAD 清理</b></td><td>用中位数和 MAD 估计稳健尺度，识别偏离过大的样本。</td><td>MAD 阈值 3.0；误报多时提高到 4.0～5.0。</td></tr>
<tr><td><b>Z 分数清理</b></td><td>按均值和标准差识别偏离样本。</td><td>Z 阈值 4.0～5.0 更保守；强事件或漂移数据慎用。</td></tr>
<tr><td><b>移动平均清理</b></td><td>把局部移动平均作为背景，替换偏离过大的点。</td><td>窗口 5～20 个样本；窗口过大可能抹平真实波形。</td></tr>
<tr><td><b>月震保护型局部尖峰修复</b></td><td>用滚动中位数和局部 MAD 识别尖峰；遇到月震高活动段时，强尖峰救援会检查极端孤立峰和重复满量程平台，只对峰顶做 PCHIP/线性插值。</td><td>常规路径：局部 8 sigma、全局 4 sigma、最大宽度 2 个采样点。救援路径默认 16 sigma、邻点比 1.25、最多 4 个采样点。红色箭头仍指向极端峰时，优先保留救援并先预览。</td></tr>
<tr><td><b>双侧参考频谱修复</b></td><td>从异常前后健康数据估计频谱，合成修复段并做端点融合。</td><td>前后参考段 50～200 秒，安全间隔 0.5～5 秒；必须人工确认。</td></tr></table>
<h3>参数只看这几项</h3>
<p class="lead"><b>强尖峰救援：</b>针对高活动月震中仍明显脱离邻点的极端峰，以及重复达到观测上限的短平台；默认局部阈值为 16 sigma、邻点比 1.25、最多救援 4 个采样点。它只处理峰顶，不替换整段月震。</p>
<p class="lead"><b>分层参考频谱修复：</b>必须先启用分段选择并框选异常核心。超低频优先默认关注 <code>0～0.08 Hz</code>，高频优先默认从 <code>0.25 Hz</code> 到 Nyquist，自定义模式需要手动确认频带上下限和过渡带宽。该方法是统计重建，不能恢复异常段原始瞬时相位。</p>
<table><tr><th>参数</th><th>含义</th><th>调整建议</th></tr>
<tr><td>幅值阈值</td><td>允许的最大绝对幅值。</td><td>填写仪器量程，不要凭波形最高点猜测。</td></tr>
<tr><td>MAD / Z 阈值</td><td>判定偏离程度的严格度。</td><td>数值越大越保守、误报越少，但可能漏掉异常。</td></tr>
<tr><td>月震保护型尖峰参数</td><td>常规参数控制局部异常；“启用强尖峰救援”专门处理高活动段中仍明显脱离邻点的极端峰，并识别短满量程平台。</td><td>先保持救援开启。仍有漏检时降低强尖峰阈值（例如 16 → 12）；误修复真实峰时提高该值或关闭救援。不要把最大救援宽度调得过大。</td></tr>
<tr><td>参考段前后长度</td><td>用于频谱和局部统计估计的健康数据长度。</td><td>太短不稳定，太长可能混入非平稳事件。</td></tr>
<tr><td>安全间隔</td><td>异常边界与参考段之间保留的距离。</td><td>避免把异常边缘带入参考段；边界不确定时适当增大。</td></tr>
<tr><td>随机种子</td><td>控制频谱合成的可重复性。</td><td>调试用固定值；比较实验必须固定；生产探索可设为 -1。</td></tr></table>
<p class="warn"><b>不要自动修复：</b>异常区很长、前后波形明显变化、存在完整起始—振荡—衰减包络、参考段也含尖峰，或你无法判断它是否为真实事件。此时只预览并保留原始数据。</p>
<h3>预览不满意时怎么排查</h3><ul><li>接缝跳变：缩小异常核心区、增大安全间隔，或改用参考频谱方法。</li><li>修复后能量变大：检查参考段是否被污染，降低重建范围，不要用异常原段估计频谱。</li><li>真实事件被抹平：提高 MAD/Z 阈值，改用手动框选，只处理尖峰核心。</li><li>方法不可用：检查 scikit-learn 或 PyTorch 是否安装；优先回退到线性/三次填补。</li></ul>"""
        return self._help_css() + """
<h2>De-pulse: Quick Start Guide</h2>
<p class="lead"><b>Goal:</b> remove confirmed abnormal samples while preserving real seismic signals. A detection result is only a candidate, not proof that the samples should be removed.</p>
<h3>Start in 30 seconds</h3>
<ol><li><b>Select the region when local repair is needed.</b> Enable segment selection and left-drag over the anomaly core. Right-drag adds a region; double-click clears selections.</li><li><b>Choose one method.</b> Start with MAD or interpolation when uncertain. Do not apply a strong reconstruction method without confirmation.</li><li><b>Preview before applying.</b> Check the waveform, joins, and spectrum, then click OK or Save. The main window can undo the applied result.</li></ol>
<h3>Choose by what you see</h3>
<p class="lead"><b>Use layered reference spectrum repair when:</b> the contaminated interval is long, you can select its core manually, and one research frequency band matters more than broadband texture. It starts from the two-sided repair and prioritizes an ultra-low, high, or custom band; it does not detect anomalies automatically.</p>
<table><tr><th>Observed pattern</th><th>First choice</th><th>Reason and caution</th></tr><tr><td>Single or double-sample spikes mixed into a moonquake</td><td>Moonquake-protected local spike repair</td><td>Repairs only short isolated spikes whose neighbours return quickly; it does not reconstruct the event core.</td></tr><tr><td>Single spike or short pulse</td><td>MAD + linear/cubic fill</td><td>Fast and robust. A low threshold can remove real peaks.</td></tr><tr><td>Clipping or an instrument full-scale plateau</td><td>Threshold clipping or reference-spectrum replacement</td><td>Use the real instrument limit and confirm it is not a genuine event.</td></tr><tr><td>Long contamination with known band character</td><td>Reference-spectrum replacement</td><td>Requires a manually selected core and healthy, reasonably stationary references.</td></tr></table>
<h3>What each method does</h3>
<p class="lead"><b>Layered reference spectrum repair:</b> first synthesizes a replacement from healthy before/after references, then adds a reference-guided layer in the selected band. Ultra-low priority preserves long-period lunar trends, high priority preserves short-period texture, and custom band targets a defined research range. Supporting texture outside the band is retained.</p>
<table><tr><th>Method</th><th>What it does</th><th>Good starting point</th></tr><tr><td><b>Layered reference spectrum repair</b></td><td>Further emphasizes a selected research band on top of the two-sided repair while retaining supporting texture outside that band.</td><td>Ultra-low <code>0–0.08 Hz</code>; high <code>0.25 Hz</code> to Nyquist; use custom band when the target range is known.</td></tr></table>
<table><tr><th>Method</th><th>What it does</th><th>Good starting point</th></tr><tr><td><b>Threshold truncation</b></td><td>Clips values beyond an amplitude limit.</td><td>Use the real instrument maximum; only for confirmed saturation.</td></tr><tr><td><b>MAD cleaning</b></td><td>Uses the median and median absolute deviation to find robust outliers.</td><td>MAD threshold 3.0; use 4.0–5.0 for fewer false positives.</td></tr><tr><td><b>Z-score cleaning</b></td><td>Flags samples by standard-deviation distance from the mean.</td><td>Use 4.0–5.0 for conservative cleaning; avoid strong events and drift.</td></tr><tr><td><b>Moving-average cleaning</b></td><td>Compares each point with its local moving-average background.</td><td>Window 5–20 samples; large windows can flatten real waveforms.</td></tr><tr><td><b>Moonquake-protected local spike repair</b></td><td>Uses a rolling median, local robust scale, short-run limit, and neighbour-return check before interpolating only isolated spikes.</td><td>Local threshold 8 sigma, minimum global amplitude 4 sigma, detection window 0.75 s, and maximum width 2 samples. Use this for narrow spikes mixed into a moonquake.</td></tr><tr><td><b>Reference-spectrum replacement</b></td><td>Estimates healthy before/after spectra, synthesizes a replacement, and crossfades the joins.</td><td>50–200 s references and 0.5–5 s safety gap; always confirm manually.</td></tr></table>
<h3>Parameters that matter</h3>
<p class="lead"><b>Strong-peak rescue:</b> handles only extreme peaks that remain clearly above their active neighbours and short repeated full-scale plateaus. Defaults are a local 16 sigma threshold, a 1.25 neighbour ratio, and at most 4 samples. It repairs the peak top, not the whole moonquake.</p>
<p class="lead"><b>Layered reference spectrum repair:</b> first enable segment selection and select the anomaly core. Ultra-low priority defaults to <code>0–0.08 Hz</code>, high priority to <code>0.25 Hz–Nyquist</code>, and custom mode requires justified band limits and transition width. This is statistical reconstruction and cannot restore the original instantaneous phase.</p>
<table><tr><th>Parameter</th><th>Meaning</th><th>How to tune</th></tr><tr><td>Amplitude threshold</td><td>Maximum allowed absolute amplitude.</td><td>Use the instrument range, not the highest value visible in one trace.</td></tr><tr><td>MAD / Z threshold</td><td>Strictness of outlier detection.</td><td>Larger values are more conservative and may miss anomalies.</td></tr><tr><td>Moonquake spike parameters</td><td>Local threshold, minimum global amplitude, detection/scale windows, and maximum spike width decide whether only isolated spikes are repaired.</td><td>Start at local 8 sigma, global 4 sigma, and maximum width 2 samples.</td></tr><tr><td>Before/after reference</td><td>Healthy data used for spectral and local-statistic estimates.</td><td>Too short is unstable; too long may include a non-stationary event.</td></tr><tr><td>Safety gap</td><td>Distance between the anomaly boundary and reference data.</td><td>Increase it when the anomaly edge is uncertain.</td></tr><tr><td>Random seed</td><td>Reproducibility of synthesized signals.</td><td>Use a fixed value for debugging and comparisons; use -1 for fresh randomness.</td></tr></table>
<p class="warn"><b>Do not auto-repair:</b> long regions, strong before/after changes, a complete onset–oscillation–decay envelope, contaminated references, or uncertainty about whether the signal is real. Preview only and preserve the original.</p>
<h3>If the preview looks wrong</h3><ul><li>Join jumps: reduce the anomaly core, increase the safety gap, or use reference-spectrum repair.</li><li>Energy grows: clean the references, reduce the repair range, and never estimate the target spectrum from the contaminated core.</li><li>A real event is flattened: raise MAD/Z thresholds, use manual selection, and repair only the spike core.</li><li>A method is disabled: install scikit-learn or PyTorch, or fall back to linear/cubic filling.</li></ul>""".replace("Reference-spectrum replacement", "Two-sided reference spectrum repair")

    def retranslate_ui(self, *_args, **_kwargs):
        """Rebuild only the content widget with the current translation."""
        state = self._capture_ui_state()
        self._cleanup_worker()
        old_content = getattr(self, "_content_widget", None)
        outer_layout = self.layout()
        if old_content is not None and outer_layout is not None:
            # Close any open popup (e.g. combo dropdown) before tearing down widgets,
            # otherwise the popup can linger as a floating top-level window.
            popup = QApplication.activePopupWidget()
            if popup is not None:
                popup.close()
            # Hide the old widget tree BEFORE removeWidget+deleteLater so that
            # children like the trace combo cannot remain visible as floating
            # ghosts while deletion is pending on the event loop.
            old_content.hide()
            outer_layout.removeWidget(old_content)
            old_content.deleteLater()
        # Update window title BEFORE building UI so _build_ui() sees the new title
        self.setWindowTitle(self._tr("Anomaly Repair"))
        self._build_ui()
        self._restore_ui_state(state)
        self._update_repair_action_state()

    set_tool_language = retranslate_ui

    def _capture_ui_state(self):
        state = {"methods": [], "values": {}, "combos": {}, "segment": None}
        for name in ("threshold_input", "mad_spin", "ma_window_spin", "ma_amp_spin",
                     "moonquake_spike_threshold_spin", "moonquake_spike_global_spin",
                     "moonquake_spike_detection_window_spin",
                     "moonquake_spike_scale_window_spin", "moonquake_spike_width_spin",
                     "moonquake_spike_return_ratio_spin",
                     "moonquake_spike_rescue_sigma_spin",
                     "moonquake_spike_rescue_global_spin",
                     "moonquake_spike_rescue_ratio_spin",
                     "z_spin", "reference_before_spin", "reference_after_spin",
                     "reference_gap_spin", "reference_seed_spin", "layered_band_min_spin",
                     "layered_band_max_spin", "layered_transition_spin",
                     "layered_reference_before_spin", "layered_reference_after_spin",
                     "layered_reference_gap_spin", "layered_reference_seed_spin"):
            widget = getattr(self, name, None)
            if widget is not None:
                state["values"][name] = widget.text() if isinstance(widget, QLineEdit) else widget.value()
        for name in ("layered_focus_mode_combo",):
            widget = getattr(self, name, None)
            if widget is not None:
                state["combos"][name] = widget.currentData()
        for checkbox in getattr(self, "_method_checks", []):
            state["methods"].append(checkbox.isChecked())
        state["moonquake_rescue"] = self.moonquake_spike_rescue_check.isChecked()
        if getattr(self, "_segment_selection_available", False):
            state["segment"] = {
                "enabled": self._seg_enabled_check.isChecked(),
                "trace": self._current_trace_index,
                "selections": list(getattr(self, "_selections", [])),
                "apply_all": self._seg_apply_all_check.isChecked(),
            }
        return state

    def _restore_ui_state(self, state):
        for name, value in state.get("values", {}).items():
            widget = getattr(self, name, None)
            if widget is None:
                continue
            if isinstance(widget, QLineEdit):
                widget.setText(value)
            else:
                widget.setValue(value)
        for name, value in state.get("combos", {}).items():
            widget = getattr(self, name, None)
            if widget is None:
                continue
            index = widget.findData(value)
            if index >= 0:
                widget.setCurrentIndex(index)
        for checkbox, checked in zip(getattr(self, "_method_checks", []), state.get("methods", [])):
            if checked:
                checkbox.setChecked(True)
        if hasattr(self, "moonquake_spike_rescue_check"):
            self.moonquake_spike_rescue_check.setChecked(
                state.get("moonquake_rescue", True))
        segment = state.get("segment")
        if segment and getattr(self, "_segment_selection_available", False):
            self._seg_enabled_check.setChecked(segment["enabled"])
            self._seg_apply_all_check.setChecked(segment["apply_all"])
            max_index = max(0, self._seg_trace_combo.count() - 1)
            self._current_trace_index = min(segment["trace"], max_index)
            self._seg_trace_combo.setCurrentIndex(self._current_trace_index)
            self._selections = list(segment["selections"])
            self._plot_segment_trace()
            self._redraw_selections()

    def closeEvent(self, event):
        self._stop_seg_nav_timer()
        self._cleanup_worker()
        super().closeEvent(event)

    def reject(self):
        self._cleanup_worker()
        # 嵌入式时 accept()/reject() 都作用于不可见的 dialog，没有可见效果。
        # 主窗口场景下 Cancel 应表现为"返回查看器页"，否则点了没有任何反馈。
        if self._is_embedded():
            try:
                self._preview_result_stream = None
                self._preview_source_stream = None
                hw = getattr(self, "_host_window", None)
                if hw is not None and hasattr(hw, "_switch_page"):
                    hw._switch_page(0)
            except Exception:
                pass
            return
        super().reject()

    def get_selected_methods(self):
        methods = []
        if self.threshold_check.isChecked():
            methods.append({
                "id": "threshold",
                "label": self._tr("Threshold truncation"),
                "params": {"threshold": self.threshold_input.text().strip() or "3"}
            })
        if self.mad_check.isChecked():
            methods.append({
                "id": "mad",
                "label": self._tr("MAD-based cleaning"),
                "params": {"mad_threshold": self.mad_spin.value()}
            })
        if self.ma_check.isChecked():
            methods.append({
                "id": "moving_avg",
                "label": self._tr("Moving-average cleaning"),
                "params": {
                    "window_size": self.ma_window_spin.value(),
                    "amplitude_threshold": self.ma_amp_spin.value()
                }
            })
        if self.moonquake_spike_check.isChecked():
            methods.append({
                "id": "moonquake_protected_spike",
                "label": self._tr("Moonquake-protected local spike repair"),
                "params": {
                    "threshold_sigma": self.moonquake_spike_threshold_spin.value(),
                    "minimum_global_sigma": self.moonquake_spike_global_spin.value(),
                    "detection_window_sec": self.moonquake_spike_detection_window_spin.value(),
                    "scale_window_sec": self.moonquake_spike_scale_window_spin.value(),
                    "max_width_samples": self.moonquake_spike_width_spin.value(),
                    "neighbour_return_ratio": self.moonquake_spike_return_ratio_spin.value(),
                    "strong_peak_rescue": self.moonquake_spike_rescue_check.isChecked(),
                    "strong_peak_sigma": self.moonquake_spike_rescue_sigma_spin.value(),
                    "strong_peak_global_sigma": self.moonquake_spike_rescue_global_spin.value(),
                    "strong_peak_ratio": self.moonquake_spike_rescue_ratio_spin.value(),
                    "rescue_max_width_samples": 4,
                    "interpolation_radius_samples": 2,
                },
            })
        if self.z_check.isChecked():
            methods.append({
                "id": "z_score",
                "label": self._tr("Z-score cleaning"),
                "params": {"z_threshold": self.z_spin.value()}
            })
        if self.missing_value_interpolation_check.isChecked():
            methods.append({
                "id": "missing_value_interpolation",
                "label": self._tr("Missing-value interpolation"),
                "params": {
                    "method": (
                        self.missing_value_interpolation_combo.currentData()
                        or "linear"
                    ),
                },
            })
        if (getattr(self, "reference_spectrum_check", None)
                and self.reference_spectrum_check.isChecked()):
            seed = self.reference_seed_spin.value()
            methods.append({
                "id": "reference_spectrum",
                "label": self._tr("Two-sided reference spectrum repair"),
                "params": {
                    "reference_auto": self.reference_auto_check.isChecked(),
                    "reference_min_sec": self.reference_min_spin.value(),
                    "reference_before_sec": self.reference_before_spin.value(),
                    "reference_after_sec": self.reference_after_spin.value(),
                    "reference_gap_sec": self.reference_gap_spin.value(),
                    "random_seed": None if seed < 0 else seed,
                    "quantize_to_int": self.reference_quantize_check.isChecked(),
                },
            })
        if (getattr(self, "layered_reference_spectrum_check", None)
                and self.layered_reference_spectrum_check.isChecked()):
            seed = self.layered_reference_seed_spin.value()
            focus_mode = (
                self.layered_focus_mode_combo.currentData()
                or "ultra_low"
            )
            band_min_hz = self.layered_band_min_spin.value()
            band_max_hz = self.layered_band_max_spin.value()
            if focus_mode == "ultra_low":
                band_min_hz = None
            elif focus_mode == "high" and band_max_hz <= band_min_hz:
                band_max_hz = None
            methods.append({
                "id": "layered_reference_spectrum",
                "label": self._tr("Layered reference spectrum repair"),
                "params": {
                    "reference_before_sec": self.layered_reference_before_spin.value(),
                    "reference_after_sec": self.layered_reference_after_spin.value(),
                    "reference_gap_sec": self.layered_reference_gap_spin.value(),
                    "random_seed": None if seed < 0 else seed,
                    "focus_mode": focus_mode,
                    "band_min_hz": band_min_hz,
                    "band_max_hz": band_max_hz,
                    "transition_hz": self.layered_transition_spin.value(),
                    "quantize_to_int": self.layered_quantize_check.isChecked(),
                },
            })
        return methods


    def _on_reference_auto_toggled(self, checked: bool = False) -> None:
        """Auto mode disables manual before/after reference fields."""
        manual_enabled = not checked
        self.reference_before_spin.setEnabled(manual_enabled)
        self.reference_after_spin.setEnabled(manual_enabled)
        self.reference_before_label.setEnabled(manual_enabled)
        self.reference_after_label.setEnabled(manual_enabled)

    def _on_layered_focus_mode_changed(self):
        if not hasattr(self, "layered_focus_mode_combo"):
            return
        mode = self.layered_focus_mode_combo.currentData() or "ultra_low"
        if mode == "ultra_low":
            self.layered_band_min_spin.setValue(0.0)
            self.layered_band_min_spin.setEnabled(False)
            self.layered_band_max_spin.setValue(0.08)
            self.layered_transition_spin.setValue(0.01)
        elif mode == "high":
            self.layered_band_min_spin.setEnabled(True)
            self.layered_band_min_spin.setValue(0.25)
            self.layered_band_max_spin.setValue(5.0)
            self.layered_transition_spin.setValue(0.05)
        else:
            self.layered_band_min_spin.setEnabled(True)
            self.layered_band_min_spin.setValue(0.001)
            self.layered_band_max_spin.setValue(0.08)
            self.layered_transition_spin.setValue(0.01)

    def get_preview_result(self, method_configs):
        """Return a copy of the latest preview when settings are unchanged."""
        import json

        if self._preview_result_stream is None:
            return None

        # Use JSON serialization for reliable nested-dict comparison
        try:
            configs_match = (
                json.dumps(self._preview_result_configs, sort_keys=True, default=str)
                == json.dumps(method_configs, sort_keys=True, default=str)
            )
        except Exception:
            configs_match = (self._preview_result_configs == method_configs)

        try:
            seg_match = (
                json.dumps(self._preview_result_segment_info, sort_keys=True, default=str)
                == json.dumps(self.get_segment_info(), sort_keys=True, default=str)
            )
        except Exception:
            seg_match = (self._preview_result_segment_info == self.get_segment_info())

        if not configs_match or not seg_match:
            return None
        return self._preview_result_stream.copy()

    def get_current_result(self):
        """Return the dialog result without applying anything to the main window."""
        preview = self.get_preview_result(self.get_selected_methods())
        if preview is not None:
            return preview
        # A working stream is only a committed dialog result while history is
        # non-empty. After undoing the first save, the dialog is back at its
        # original state and the normal accept fallback should run.
        if self._working_stream is not None and self._repair_history:
            return self._working_stream.copy()
        return None

    def _build_segment_panel(self):
        from matplotlib.widgets import SpanSelector

        frame = QFrame()
        frame.setObjectName("heroFrame")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setSpacing(6)
        frame_layout.setContentsMargins(14, 10, 14, 10)

        colors = self._theme_colors

        # ── Header row: title + enable + trace combo + apply-all ──
        header = QHBoxLayout()
        header.setSpacing(6)
        seg_title = QLabel(self._tr("Segment Selection"))
        self._seg_title = seg_title
        seg_title.setStyleSheet("font-size:14px; font-weight:600;")
        header.addWidget(seg_title)

        self._seg_enabled_check = QCheckBox(self._tr("Enable Segment Selection"))
        self._seg_enabled_check.setChecked(self._segment_mode_enabled)
        self._seg_enabled_check.toggled.connect(self._on_segment_toggle)
        header.addWidget(self._seg_enabled_check)

        header.addWidget(QLabel(self._tr("Trace:")))
        # Give the combo an explicit owner.  Without this, the popup view can
        # outlive the dynamically rebuilt segment panel and keep a stale
        # top-level position in the main window.
        self._seg_trace_combo = QComboBox(frame)
        self._seg_trace_combo.setMinimumWidth(260)
        if self.base_stream:
            for idx, trace in enumerate(self.base_stream):
                label = f"{trace.stats.network}.{trace.stats.station}.{trace.stats.channel}"
                self._seg_trace_combo.addItem(label, idx)
        self._seg_trace_combo.currentIndexChanged.connect(self._on_segment_trace_changed)
        header.addWidget(self._seg_trace_combo, 1)

        self._seg_apply_all_check = QCheckBox(self._tr("Apply to all traces"))
        header.addWidget(self._seg_apply_all_check)

        # Import button: files (single/multi) or a whole folder.
        self._seg_import_btn = QToolButton()
        self._seg_import_btn.setText(self._tr("Import"))
        self._seg_import_btn.setToolTip(
            self._tr("Import one or more waveform files."))
        self._seg_import_btn.setPopupMode(QToolButton.InstantPopup)
        import_menu = QMenu(self._seg_import_btn)
        act_files = import_menu.addAction(self._tr("Import files..."))
        act_folder = import_menu.addAction(self._tr("Import folder..."))
        act_files.triggered.connect(self._import_files)
        act_folder.triggered.connect(self._import_folder)
        self._seg_import_btn.setMenu(import_menu)
        header.addWidget(self._seg_import_btn)

        header.addStretch(1)
        frame_layout.addLayout(header)

        self._seg_content = QFrame()
        # 注意：不能在此处 setVisible(True) —— _seg_content 此刻还没有父窗口
        # （直到下面 frame_layout.addWidget 才 reparent 进 frame）。无父控件
        # setVisible(True) 会变成**可见的顶层窗口**，空标题被窗口管理器显示为
        # applicationDisplayName "DSSRR"，随后 reparent 又消失，表现为切换语言
        # （或重建面板）时闪过一个 "DSSRR" 弹窗。可见性在 addWidget 之后再设。
        seg_content_layout = QVBoxLayout(self._seg_content)
        seg_content_layout.setContentsMargins(0, 0, 0, 0)
        seg_content_layout.setSpacing(4)

        # ── Canvas ──
        # 画布默认高度调大（2.2→3.2 英寸），并提高最小高度，保证 x 轴
        # 标签（"距道起始的秒数"）与刻度在默认布局下完整显示。
        self._seg_canvas = MplCanvas(frame, width=7, height=3.2, dpi=100)
        self._seg_canvas.setMinimumHeight(170)
        # 竖向 Expanding + 布局里 stretch=1：把面板的**全部剩余高度**交给画布。
        # 不这么做的话，FigureCanvas 的高度会卡在 figsize 换算出的 320 px，
        # 而下面那个工具栏行（QHBoxLayout 的 expandingDirections() 含 Vertical）
        # 会把剩余高度全部吃掉 —— 放大窗口后中间就空出一条无内容的横带，画布
        # 却不见长高（预览面板一直是对的：那里 canvas 就带了 stretch=1）。
        self._seg_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._seg_canvas.set_canvas_colors(bg_color=colors['surface'], fg_color=colors['text'])
        seg_content_layout.addWidget(self._seg_canvas, 1)

        # ── Toolbar row: nav toolbar | hint label ──
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(4)

        self._seg_toolbar = NavigationToolbar(self._seg_canvas, frame)
        self._seg_toolbar.setStyleSheet(_nav_toolbar_qss(_is_dark()))
        toolbar_row.addWidget(self._seg_toolbar)

        self._seg_selection_label = QLabel(self._tr("Selection: None"))
        toolbar_row.addWidget(self._seg_selection_label)
        toolbar_row.addStretch(1)

        hint = QLabel(self._tr("Left drag: replace. Right drag: add. Double-click: clear."))
        self._seg_hint = hint
        toolbar_row.addWidget(hint)

        # stretch=0：工具栏行保持自身高度，不参与分配剩余竖向空间（剩余空间
        # 归上面的画布）。这样放大窗口时只有画布变高，工具栏紧贴画布下沿。
        seg_content_layout.addLayout(toolbar_row, 0)

        # 导航工具栏的按钮名/提示跟随界面语言；同时挂上"平移/缩放模式"监听，
        # 因为这两种模式下 SpanSelector 会被 matplotlib 的 widgetlock 挡住，
        # 框选静默失效——必须给用户一句明确的提示（见 _refresh_seg_hint）。
        self._install_seg_nav_hooks()

        frame_layout.addWidget(self._seg_content)
        # reparent 完成后再按勾选状态设置可见性（此时它是 frame 的子控件，
        # 不会成为顶层窗口）。
        self._seg_content.setVisible(self._seg_enabled_check.isChecked())

        # Internal state — all selections stored as merged list of (start, end)
        self._selections = []  # sorted, non-overlapping
        self._right_drag_start = None

        QTimer.singleShot(200, self._plot_segment_trace)
        return frame

    def _merge_selections(self, new_range):
        """Insert new_range into self._selections, merging overlaps."""
        s, e = new_range
        merged = []
        inserted = False
        for existing in merged:
            pass  # placeholder
        # Build list with new range included
        all_ranges = list(self._selections) + [(s, e)]
        all_ranges.sort()
        # Merge overlapping
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

    def _redraw_selections(self):
        """Redraw all selection patches on the canvas."""
        # Remove old patches
        if self._selection_patch:
            try:
                self._selection_patch.remove()
            except Exception:
                pass
            self._selection_patch = None
        if hasattr(self, '_multi_patches'):
            for p in self._multi_patches:
                try:
                    p.remove()
                except Exception:
                    pass
            self._multi_patches = []
        # Draw current selections
        ax = self._seg_canvas.axes
        for i, (s, e) in enumerate(self._selections):
            color = "#93c5fd" if i == 0 else "#a5b4fc"
            alpha = 0.3 if i == 0 else 0.2
            patch = ax.axvspan(s, e, color=color, alpha=alpha)
            if i == 0:
                self._selection_patch = patch
            else:
                if not hasattr(self, '_multi_patches'):
                    self._multi_patches = []
                self._multi_patches.append(patch)
        self._seg_canvas.draw_idle()
        self._update_selection_label()

    def _update_selection_label(self):
        if not self._selections:
            self._seg_selection_label.setText(self._tr("Selection: None"))
        elif len(self._selections) == 1:
            s, e = self._selections[0]
            self._seg_selection_label.setText(
                self._tr("Selection: {s:.2f}s — {e:.2f}s").format(s=s, e=e))
        else:
            self._seg_selection_label.setText(
                self._tr("Selection: {n} regions").format(n=len(self._selections)))

    def _get_all_selections(self):
        """Return all selected regions as list of (start, end) tuples."""
        return list(self._selections)

    def _on_segment_toggle(self, checked):
        self._seg_content.setVisible(checked)

    # ------------------------------------------------------------------
    # 段选择：导航工具栏（平移/缩放）状态 → 提示语
    # ------------------------------------------------------------------
    # matplotlib 的平移/缩放模式会抢占 ``canvas.widgetlock``，SpanSelector 的
    # 按下事件被直接丢弃，表现为"怎么拖都选不中，而且没有任何反馈"。工具栏按钮
    # 虽然会显示为按下态，但不显眼。这里在工具栏右侧常驻一句提示，并在模式
    # 激活时切换成警告文案，把原因说清楚。
    #
    # 具体逻辑封装在 :class:`dssrr.gui.nav_hint.SegmentNavHint`（与
    # ManualRepairDialog 共用）；下面几个方法只是薄委托，保留名字是为了让
    # 外部代码与测试不必关心内部实现。
    #: 空闲时的操作说明（英文原文，与词条表 key 逐字符一致）。
    _SEG_HINT_IDLE = nav_hint.IDLE_TEXT
    #: 平移/缩放模式激活时的警告文案。
    _SEG_HINT_NAV = nav_hint.NAV_TEXT

    def _install_seg_nav_hooks(self):
        """本地化导航工具栏，并挂上平移/缩放模式监听。"""
        toolbar = getattr(self, "_seg_toolbar", None)
        if toolbar is None or getattr(self, "_seg_hint", None) is None:
            return
        self._seg_nav_hint = nav_hint.SegmentNavHint(
            self._seg_hint, toolbar, self._tr, parent=self,
            muted_color=lambda: self._theme_colors.get("text_muted", "#94A3B8"),
            is_dark=_is_dark,
        ).install()

    def _seg_nav_active(self):
        """导航工具栏当前是否处于平移/缩放模式。"""
        hint = getattr(self, "_seg_nav_hint", None)
        if hint is not None:
            return hint.nav_active()
        return nav_hint.nav_mode_active(getattr(self, "_seg_toolbar", None))

    def _refresh_seg_hint(self, *_args):
        """按导航模式刷新提示语：空闲=操作说明，平移/缩放=警告。"""
        hint = getattr(self, "_seg_nav_hint", None)
        if hint is not None:
            hint.refresh()
            return
        # 面板还没装好（或组件已被销毁）时的兜底：直接按英文原文设置
        label = getattr(self, "_seg_hint", None)
        if label is None:
            return
        try:
            label.setText(self._tr(self._SEG_HINT_IDLE))
            label.setStyleSheet(
                f"color: {self._theme_colors['text_muted']}; font-size: 10px;")
        except RuntimeError:
            pass

    def _stop_seg_nav_timer(self):
        hint = getattr(self, "_seg_nav_hint", None)
        if hint is not None:
            hint.stop()

    def _plot_segment_trace(self):
        if not self.base_stream or len(self.base_stream) == 0:
            return
        trace = self.base_stream[self._current_trace_index]
        data = trace.data
        if data is None or len(data) == 0:
            return
        sampling_rate = trace.stats.sampling_rate or 1.0
        self._current_sr = float(sampling_rate)
        times = np.arange(len(data)) / self._current_sr
        ax = self._seg_canvas.reset_axes(
            bg_color=self._theme_colors['surface'], fg_color=self._theme_colors['text'])
        ax.plot(times, data,
                color=("#111827" if not _is_dark()
                       else self._theme_colors["text"]),
                linewidth=0.8)
        ax.set_xlabel(self._tr("Seconds since trace start"))
        ax.set_ylabel(self._tr("Amplitude"))
        ax.margins(x=0.005, y=0.05)
        self._seg_canvas.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.25)
        self._seg_canvas.draw()

        # Disconnect old handlers
        if self._span_selector:
            self._span_selector.disconnect_events()

        # Left-click SpanSelector (replace selection)
        from matplotlib.widgets import SpanSelector
        self._span_selector = SpanSelector(
            ax,
            self._on_segment_span_selected,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="#93c5fd"),
            interactive=True,
            button=1,  # left button only — right button handled separately
        )

        # Right-click drag handler (add to selection)
        self._right_drag_start = None
        self._right_drag_patch = None
        canvas = self._seg_canvas

        def on_mouse_press(event):
            if event.inaxes != ax:
                return
            if event.button == 3:  # right-click
                self._right_drag_start = event.xdata
                # Remove previous temp patch
                if self._right_drag_patch:
                    try:
                        self._right_drag_patch.remove()
                    except Exception:
                        pass
                    self._right_drag_patch = None

        def on_mouse_move(event):
            """Draw temporary patch during right-click drag."""
            if self._right_drag_start is None or event.inaxes != ax:
                return
            if self._right_drag_patch:
                try:
                    self._right_drag_patch.remove()
                except Exception:
                    pass
            start = min(self._right_drag_start, event.xdata)
            end = max(self._right_drag_start, event.xdata)
            self._right_drag_patch = ax.axvspan(
                start, end, color="#fbbf24", alpha=0.3)
            canvas.draw_idle()

        def on_mouse_release(event):
            if event.inaxes != ax:
                return
            if event.button == 3 and self._right_drag_start is not None:
                start = min(self._right_drag_start, event.xdata)
                end = max(self._right_drag_start, event.xdata)
                if end - start > 0.01:  # minimum drag distance
                    self._merge_selections((start, end))
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

        # Disconnect old canvas handlers to avoid stacking
        canvas.mpl_disconnect(canvas._button_press_id) if hasattr(canvas, '_button_press_id') else None
        canvas.mpl_disconnect(canvas._button_release_id) if hasattr(canvas, '_button_release_id') else None
        canvas.mpl_disconnect(canvas._double_click_id) if hasattr(canvas, '_double_click_id') else None
        canvas.mpl_disconnect(canvas._motion_id) if hasattr(canvas, '_motion_id') else None

        canvas._button_press_id = canvas.mpl_connect('button_press_event', on_mouse_press)
        canvas._button_release_id = canvas.mpl_connect('button_release_event', on_mouse_release)
        canvas._motion_id = canvas.mpl_connect('motion_notify_event', on_mouse_move)
        canvas._double_click_id = canvas.mpl_connect('button_press_event',
            lambda e: on_double_click(e) if e.button == 1 and e.dblclick else None)

        # Restore existing selections
        self._selections = []
        self._selection_patch = None
        self._multi_patches = []
        self._update_selection_label()

    def _on_segment_span_selected(self, xmin, xmax):
        """Left-drag: replace selection. Right-drag: add to selection."""
        if xmin == xmax:
            return
        start, end = sorted([xmin, xmax])
        # Left-drag replaces all selections
        self._selections = []
        self._merge_selections((start, end))
        self._redraw_selections()

    def _on_segment_trace_changed(self, index):
        previous = self._current_trace_index
        self._current_trace_index = index
        self._plot_segment_trace()
        # 切换到另一条数据后，上一事件的预览结果不再对应当前 trace：
        # 必须清空，否则预览区仍显示旧事件，且此时点"保存当前修复/导出"
        # 会把旧事件的处理结果带进新事件的语境。
        if index != previous and self._preview_result_stream is not None:
            self._reset_preview_for_trace_change()

    def _reset_preview_for_trace_change(self):
        """切 trace 后清空预览产物并复位预览区与选择。"""
        self._cleanup_worker()
        self._preview_result_stream = None
        self._preview_result_configs = None
        self._preview_result_segment_info = None
        self._preview_requested_configs = None
        self._preview_requested_segment_info = None
        self._preview_source_stream = None
        self._selections = []
        self._draw_preview_placeholder(
            self._tr("Switched to another trace. Run Preview again."))
        self._update_repair_action_state()

    def _apply_custom_window(self):
        """Re-activate the span selector for manual dragging."""
        if not self.base_stream or len(self.base_stream) == 0:
            return
        self._plot_segment_trace()
        self._seg_selection_label.setText(
            self._tr("Drag across the waveform to select a range."))

    def get_segment_info(self):
        if (not self._segment_selection_available
                or not self._seg_enabled_check.isChecked()):
            return None
        all_selections = self._get_all_selections()
        time_range = all_selections[0] if all_selections else None
        return {
            "enabled": True,
            "trace_index": self._current_trace_index,
            "time_range": time_range,
            "all_selections": all_selections if all_selections else None,
            "apply_all": self._seg_apply_all_check.isChecked(),
            "apply_savgol": False,
            "savgol_window": 0,
        }

    # ------------------------------------------------------------------
    # Import waveforms
    # ------------------------------------------------------------------

    def _import_files(self):
        # 以可见宿主窗口为父，避免关闭文件框后主窗口掉到桌面最底层
        host = self._host()
        paths, _ = QFileDialog.getOpenFileNames(
            host, self._tr("Import waveform files"), "",
            "Waveforms (*.mhz *.mseed *.miniseed *.sac *.ascii *.txt);;"
            "All files (*.*)")
        if host is not None:
            host.raise_()
            host.activateWindow()
        if paths:
            self._load_paths(paths)

    def _import_folder(self):
        host = self._host()
        folder = QFileDialog.getExistingDirectory(
            host, self._tr("Import folder"))
        if host is not None:
            host.raise_()
            host.activateWindow()
        if not folder:
            return
        wanted = {'.mhz', '.mseed', '.miniseed', '.sac', '.ascii', '.txt'}
        paths = []
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            if (os.path.isfile(full)
                    and os.path.splitext(name)[1].lower() in wanted):
                paths.append(full)
        if not paths:
            QMessageBox.information(
                self, self._tr("Information"),
                self._tr("No supported waveform files found in this folder."))
            return
        self._load_paths(paths)

    def _load_paths(self, paths):
        added = 0
        for path in paths:
            try:
                st = obspy_read(path)
            except Exception as exc:
                QMessageBox.warning(
                    self, self._tr("Warning"),
                    self._tr("Could not read {name}: {error}").format(
                        name=os.path.basename(path), error=exc))
                continue
            for tr in st:
                self._append_trace(tr, path)
                added += 1
        if added:
            self._seg_enabled_check.setChecked(True)
            self._seg_content.setVisible(True)
            self._seg_trace_combo.setCurrentIndex(
                self._seg_trace_combo.count() - 1)
            if self.preview_status:
                self.preview_status.setText(
                    self._tr("Imported {count} trace(s).").format(count=added))

    def _append_trace(self, tr, source_path):
        if self.base_stream is None:
            self.base_stream = Stream()
            self._working_stream = Stream()
        self.base_stream.append(tr)
        self._working_stream.append(tr.copy())
        name = os.path.basename(str(source_path))
        label = (f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}"
                 f" - {name}")
        self._seg_trace_combo.addItem(label, len(self.base_stream) - 1)

    def add_stream_from_viewer(self, stream, source="viewer"):
        """Convenience entry: add the viewer's current Stream to the trace list."""
        added = 0
        for tr in stream:
            self._append_trace(tr, source)
            added += 1
        if added:
            self._seg_enabled_check.setChecked(True)
            self._seg_content.setVisible(True)
            self._seg_trace_combo.setCurrentIndex(
                self._seg_trace_combo.count() - 1)

    # ------------------------------------------------------------------
    # Export repaired data
    # ------------------------------------------------------------------

    def _export_report(self):
        """导出图形化修复报告：先弹导出设置（含实时预览），确认后再落盘。

        数据来源为本对话框自己的预览流（_preview_source_stream /
        _preview_result_stream），绘图实现委托
        :class:`dssrr.repair_lib.report_plot.RepairReportPlotter`。
        """
        host = self._host()
        if self._is_preview_busy:
            QMessageBox.information(
                host, self._tr("Information"),
                self._tr("Please wait for the preview to finish."))
            return
        if self._preview_result_stream is None:
            QMessageBox.information(
                host, self._tr("Information"),
                self._tr("Preview a repair first, then export the report."))
            return
        original_stream = self._preview_source_stream
        repaired_stream = self._preview_result_stream
        if original_stream is None or len(original_stream) == 0 or len(repaired_stream) == 0:
            QMessageBox.warning(host, self._tr("Warning"),
                                self._tr("No data available for export."))
            return
        try:
            from .export_report_dialog import ExportSettingsDialog
            from ..repair_lib.report_plot import RepairReportPlotter
        except ImportError as exc:
            QMessageBox.critical(
                host, self._tr("Error"),
                self._tr("Failed to import report module: {}").format(exc))
            return

        # 与预览区一致：报告只针对当前选中的事件。
        cur = min(max(0, self._current_trace_index),
                  min(len(original_stream), len(repaired_stream)) - 1)
        original_data = np.asarray(original_stream[cur].data)
        repaired_data = np.asarray(repaired_stream[cur].data)
        sr = float(original_stream[cur].stats.sampling_rate) or 1.0
        # 报告语言：DSSRR 未加载 Qt 翻译，tr() 恒为英文原文，故按系统区域判定，
        # 中文系统出中文报告，其余出英文。
        language = self._report_language()

        # 尽量保留全部选中修复区间；缺失时退回中段默认区间。
        segment_info = self._preview_result_segment_info or self.get_segment_info()
        if segment_info and segment_info.get("enabled") and segment_info.get("time_range"):
            time_ranges = segment_info.get("all_selections") or [segment_info["time_range"]]
            anomaly_regions = [
                (int(start * sr), int(end * sr)) for start, end in time_ranges]
            anomaly_start, anomaly_end = anomaly_regions[0]
        else:
            npts = len(original_data)
            anomaly_start = int(npts * 0.45)
            anomaly_end = int(npts * 0.55)
            anomaly_regions = [(anomaly_start, anomaly_end)]

        dlg = ExportSettingsDialog(
            parent=host, lang=language, sr=sr,
            original_data=original_data, repaired_data=repaired_data,
            anomaly_start=anomaly_start, anomaly_end=anomaly_end,
            anomaly_regions=anomaly_regions)
        accepted = dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()
        if host is not None:
            host.raise_()
            host.activateWindow()
        if accepted != QDialog.Accepted:
            return

        settings = dlg.get_settings()
        fmt = settings["format"]
        format_filters = {
            "png": "PNG Images (*.png)",
            "pdf": "PDF Files (*.pdf)",
            "svg": "SVG Files (*.svg)",
        }
        file_filter = format_filters.get(fmt, "All Files (*)")
        default_name = self._default_report_name(original_stream[cur], fmt)
        path, _ = QFileDialog.getSaveFileName(
            host, self._tr("Save Report Image"),
            os.path.join(os.path.expanduser("~"), default_name),
            "{0};;All Files (*)".format(file_filter))
        if host is not None:
            host.raise_()
            host.activateWindow()
        if not path:
            return
        try:
            plotter = RepairReportPlotter(sr=sr, lang=language)
            fig = plotter.plot_report(
                original=original_data, repaired=repaired_data,
                anomaly_start=anomaly_start, anomaly_end=anomaly_end,
                anomaly_regions=anomaly_regions,
                save_path=path, dpi=settings["dpi"], format=fmt,
                figsize=settings["figsize"],
                custom_title=settings["title"] if settings["title"] else None,
                freq_range=settings["freq_range"],
                show_time_domain=settings["show_time_domain"],
                show_psd=settings["show_psd"],
                show_linear_spectrum=settings["show_linear_spectrum"],
                show_quality_metrics=settings["show_quality_metrics"],
                show_reference_segments=settings["show_reference_segments"],
                show_inset_zoom=settings["show_inset_zoom"])
            if fig is not None:
                try:
                    from matplotlib import pyplot as _plt
                    _plt.close(fig)
                except Exception:
                    pass
            QMessageBox.information(
                host, self._tr("Success"),
                self._tr("Report saved to:\n{path}").format(path=path))
        except Exception as exc:
            QMessageBox.critical(
                host, self._tr("Error"),
                self._tr("Failed to save report: {}").format(exc))

    @staticmethod
    def _report_language():
        """返回导出报告 / 导出设置对话框使用的语言（``"zh"`` / ``"en"``）。

        实现委托给 :func:`dssrr.i18n.resolve_language`——**默认英文**，与
        未加载 Qt 翻译的英文界面一致；想出中文报告设 ``DSSRR_LANG=zh`` 即可。

        历史实现是"按系统区域猜"（中文 Windows 就出中文），结果开源版 GUI 的
        导出设置与报告在中文系统上整篇中文、与界面语言对不上，已弃用该策略。
        """
        from ..i18n import resolve_language
        return resolve_language()

    @staticmethod
    def _default_report_name(tr, fmt):
        net = tr.stats.network or "XA"
        sta = tr.stats.station or "STN"
        return "{0}.{1}_repair_report.{2}".format(net, sta, fmt)

    def _export_result(self, stream):
        # 用可见的宿主窗口作父：以不可见 dialog 为父时，关闭文件框后焦点
        # 会落回 dialog，主窗口被压到桌面最底层（用户反馈的"跑到最底层"）。
        host = self._host()
        if host is not None:
            host.raise_()
            host.activateWindow()
        path, selected_filter = QFileDialog.getSaveFileName(
            host, self._tr("Save repaired data"),
            self._default_save_name(stream),
            "MiniSEED (*.mseed);;SAC (*.sac);;Text (*.txt);;"
            "ASCII (*.ascii);;All files (*.*)")
        # 文件框关闭后立刻把宿主窗口拉回前台（含取消/关闭的情况）
        if host is not None:
            host.raise_()
            host.activateWindow()
        if not path:
            return False
        fmt = self._format_from_selection(path, selected_filter)
        try:
            self._write_stream(stream, path, fmt)
        except Exception as exc:
            QMessageBox.critical(
                host, self._tr("Error"),
                self._tr("Failed to save: {error}").format(error=exc))
            return False
        QMessageBox.information(
            host, self._tr("Success"),
            self._tr("Repaired data saved to:\n{path}").format(path=path))
        return True

    @staticmethod
    def _format_from_selection(path, selected_filter):
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        if ext in {'mseed', 'sac', 'txt', 'ascii'}:
            return ext
        if 'SAC' in selected_filter:
            return 'sac'
        if 'ASCII' in selected_filter:
            return 'ascii'
        if 'Text' in selected_filter:
            return 'txt'
        return 'mseed'

    @staticmethod
    def _default_save_name(stream):
        tr = stream[0]
        net = tr.stats.network or 'XA'
        sta = tr.stats.station or 'STN'
        chan = tr.stats.channel or 'MHZ'
        return f"{net}.{sta}.{chan}_repaired.mseed"

    def _write_stream(self, stream, path, fmt):
        if fmt == 'sac':
            self._split_gaps(Stream([stream[0]])).write(path, format='SAC')
        elif fmt == 'mseed':
            self._split_gaps(stream).write(path, format='MSEED')
        elif fmt == 'ascii':
            self._write_ascii(stream[0], path)
        else:
            self._write_txt(stream[0], path)

    @staticmethod
    def _split_gaps(stream):
        out = Stream()
        for tr in stream:
            data = tr.data
            if np.ma.isMaskedArray(data):
                out.extend(tr.split())
            elif data.dtype.kind == 'f' and np.isnan(data).any():
                t = tr.copy()
                t.data = np.ma.masked_invalid(data)
                out.extend(t.split())
            else:
                out.append(tr)
        return out

    @staticmethod
    def _write_ascii(tr, path):
        times = tr.times()
        with open(path, 'w', encoding='utf-8') as f:
            for t, v in zip(times, np.asarray(tr.data)):
                f.write(f"{round(t, 6)} {v}\n")

    @staticmethod
    def _write_txt(tr, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write("Stats Information:\n")
            for key, value in tr.stats.__dict__.items():
                f.write(f"{key}: {value}\n")
            f.write("--------\nTrace Data:\n")
            f.write(", ".join(map(str, np.asarray(tr.data))) + "\n")
            f.write("--------\n")

    def _is_embedded(self):
        """是否被 reparent 进主窗口（_content_widget 已不在本 dialog 内）。"""
        content = getattr(self, "_content_widget", None)
        if content is None:
            return False
        return content.window() is not self

    def _current_trace_of(self, stream):
        """从整条 stream 中取出当前下拉框选中的那条 trace，返回单道 Stream。

        多文件导入时 base_stream 里有多个事件，但预览区、段选择图都只画
        当前选中的那条。导出必须与之对齐，否则会把所有事件写进同一个文件。
        """
        if stream is None or len(stream) == 0:
            return stream
        idx = min(max(0, self._current_trace_index), len(stream) - 1)
        return Stream([stream[idx].copy()])

    def _set_save_mode(self, mode):
        # 嵌入式（DSSRR 主窗口 Manual Repair 页）下本 dialog 从未 show()，
        # accept()/reject() 语义失效：accept() 会去关闭一个不可见的窗口，
        # 且会误触发主窗口的收尾逻辑。此时导出成功只需给用户反馈。
        embedded = self._is_embedded()
        host = self._host()
        if not self._validate_reference_spectrum_selection():
            return
        result = self.get_current_result()
        if result is None:
            QMessageBox.information(
                host, self._tr("Information"),
                self._tr("Preview a repair first, then confirm to export."))
            return
        # 只导出当前选中的事件，与预览区所见一致。
        result = self._current_trace_of(result)
        if not self._export_result(result):
            return
        self._save_mode = mode
        if not embedded:
            self.accept()

    def get_save_mode(self):
        return self._save_mode

    def __getstate__(self):
        """
        Exclude GUI parent objects from deepcopy/pickling so worker threads can clone configs.
        """
        state = self.__dict__.copy()
        state["parent"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
