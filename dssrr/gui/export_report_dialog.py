"""
修复报告导出设置对话框
======================
:class:`ExportSettingsDialog` 让用户在导出对比图之前**先调参、再预览、后保存**：
左侧是一组可折叠的设置项，右侧是实时刷新的 matplotlib 预览画布。

可调项（右侧预览即时生效）
--------------------------
- 标题、DPI、输出格式（png / pdf / svg …）、画布宽高
- 频域显示范围（freq min / max）
- 各面板开关：时域对比、PSD、线性谱等
- 图内文字：通过 :func:`dssrr.gui.plot_text_editor.install_plot_text_editor_action`
  接入的批编辑器，可改标题/轴标签的字体与颜色

调用方（手动修复面板 / 查看器）通过 :meth:`ExportSettingsDialog.get_settings`
取回最终设置字典，交给 :class:`~dssrr.repair_lib.report_plot.RepairReportPlotter`
出图。

设置持久化使用 Qt ``QSettings``（组织名/产品名取自 :mod:`dssrr.paths` 的
``APP_ORG`` / ``APP_NAME``），因此与 DSSRR 其余设置落在同一处。

依赖：PyQt5、matplotlib。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QSettings, QSize, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..i18n import resolve_language
from .plot_text_editor import install_plot_text_editor_action


# 设置键名常量
_SETTINGS_PREFIX = "AnomalyRepair/Export"
_KEY_TITLE = f"{_SETTINGS_PREFIX}/Title"
_KEY_DPI = f"{_SETTINGS_PREFIX}/DPI"
_KEY_FORMAT = f"{_SETTINGS_PREFIX}/Format"
_KEY_WIDTH = f"{_SETTINGS_PREFIX}/Width"
_KEY_HEIGHT = f"{_SETTINGS_PREFIX}/Height"
_KEY_FREQ_MIN = f"{_SETTINGS_PREFIX}/FreqMin"
_KEY_FREQ_MAX = f"{_SETTINGS_PREFIX}/FreqMax"
_KEY_SHOW_TIME = f"{_SETTINGS_PREFIX}/ShowTimeDomain"
_KEY_SHOW_PSD = f"{_SETTINGS_PREFIX}/ShowPSD"
_KEY_SHOW_LINEAR = f"{_SETTINGS_PREFIX}/ShowLinearSpectrum"
_KEY_SHOW_QUALITY = f"{_SETTINGS_PREFIX}/ShowQualityMetrics"
_KEY_SHOW_REF = f"{_SETTINGS_PREFIX}/ShowReferenceSegments"
_KEY_SHOW_INSET = f"{_SETTINGS_PREFIX}/ShowInsetZoom"


class ExportSettingsDialog(QDialog):
    """
    导出设置对话框。

    允许用户自定义异常修复报告导出的各项参数，包括：
    - 基本设置：标题、DPI、格式、尺寸
    - 频带设置：频率范围
    - 显示选项：各子图的显示/隐藏
    - 实时预览功能

    Parameters
    ----------
    parent : 父窗口
    lang : 语言代码。``None`` / 省略 = 按 :func:`dssrr.i18n.resolve_language`
        解析（默认**英文**，与 DSSRR 界面一致；``DSSRR_LANG=zh`` 可切中文）
    sr : 采样率，用于自动设置频率上限
    original_data : 原始波形数据 (可选，用于预览)
    repaired_data : 修复后波形数据 (可选，用于预览)
    anomaly_start : 异常起始位置 (可选，用于预览)
    anomaly_end : 异常结束位置 (可选，用于预览)
    """

    def __init__(
        self,
        parent=None,
        lang: Optional[str] = None,
        sr: float = 100.0,
        original_data: Optional[np.ndarray] = None,
        repaired_data: Optional[np.ndarray] = None,
        anomaly_start: Optional[int] = None,
        anomaly_end: Optional[int] = None,
        anomaly_regions: Optional[list[tuple[int, int]]] = None,
    ):
        super().__init__(parent)
        # 语言不再由调用方"猜"系统区域，统一交给 dssrr.i18n 解析：
        # 未显式指定时默认英文，与未加载 Qt 翻译的英文界面保持一致。
        self.lang = resolve_language(lang)
        self.sr = sr
        # 导出对话框的“上次使用设置”存进原生注册表/plist。组织名与产品名统一
        # 取 dssrr.paths 中的 APP_ORG / APP_NAME，保证与 DSSRR 其余设置同处存放。
        from ..paths import APP_NAME, APP_ORG
        self._settings = QSettings(APP_ORG, APP_NAME)

        # 预览数据
        self._original_data = original_data
        self._repaired_data = repaired_data
        self._anomaly_start = anomaly_start
        self._anomaly_end = anomaly_end
        self._anomaly_regions = anomaly_regions
        self._preview_fig = None
        # 视口变化后重新适配预览的防抖定时器
        self._preview_resize_timer = QTimer(self)
        self._preview_resize_timer.setSingleShot(True)
        self._preview_resize_timer.setInterval(180)
        self._preview_resize_timer.timeout.connect(self._reflow_preview)

        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        """初始化界面。"""
        self.setWindowTitle(self._tr("Export Settings"))
        self.setMinimumWidth(900)
        self.setMinimumHeight(600)
        self.resize(1100, 700)

        # 设置窗口标志：最小化 + 最大化 + 关闭按钮
        flags = self.windowFlags()
        flags &= ~Qt.WindowContextHelpButtonHint
        flags |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
        self.setWindowFlags(flags)

        # 主垂直布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)

        # 内容区域（左右分栏，中间分界线可拖动调节）
        self._content_splitter = QSplitter(Qt.Horizontal)
        self._content_splitter.setHandleWidth(6)

        # 左侧：设置面板
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        # 基本设置组
        settings_layout.addWidget(self._create_basic_group())

        # 频带设置组
        settings_layout.addWidget(self._create_freq_group())

        # 显示选项组
        settings_layout.addWidget(self._create_display_group())

        # 预览按钮
        self.preview_btn = QPushButton(self._tr("Preview"))
        self.preview_btn.setObjectName("primaryButton")
        self.preview_btn.clicked.connect(self._update_preview)
        settings_layout.addWidget(self.preview_btn)

        settings_layout.addStretch()

        self._content_splitter.addWidget(settings_widget)
        self._content_splitter.setStretchFactor(0, 2)

        # 右侧：预览面板
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_label = QLabel(self._tr("Preview"))
        preview_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        preview_layout.addWidget(preview_label)

        # 预览图显示区域 - matplotlib 交互式画布（可缩放/平移/编辑文字/保存）
        self.preview_canvas = FigureCanvas(
            Figure(figsize=(8, 5), facecolor="white"))
        self._nav = NavigationToolbar(self.preview_canvas, preview_widget)
        self._localize_nav_toolbar()
        install_plot_text_editor_action(
            self._nav, self.preview_canvas, parent=self, translator=self._tr)
        preview_layout.addWidget(self._nav)

        # 初始占位提示（plot_report 复用 figure 时会 clear 掉）
        _ax = self.preview_canvas.figure.add_subplot(111)
        _ax.text(0.5, 0.5, self._tr("Click 'Preview' to generate"),
                 ha="center", va="center", transform=_ax.transAxes,
                 color="#888", fontsize=12)
        _ax.set_axis_off()
        self.preview_canvas.draw()

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)  # 画布按实际尺寸滚动查看
        self.preview_scroll.setMinimumWidth(550)
        self.preview_scroll.setStyleSheet(
            "QScrollArea { background-color: white; border: 1px solid #ccc; }"
        )
        self.preview_scroll.setWidget(self.preview_canvas)
        preview_layout.addWidget(self.preview_scroll, 1)

        self._content_splitter.addWidget(preview_widget)
        self._content_splitter.setStretchFactor(1, 5)
        # 初始比例保持原 2:5
        self._content_splitter.setSizes([300, 700])

        main_layout.addWidget(self._content_splitter, 1)

        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton(self._tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.export_btn = QPushButton(self._tr("Export"))
        self.export_btn.setObjectName("primaryButton")
        self.export_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.export_btn)

        main_layout.addLayout(button_layout)

    def _localize_nav_toolbar(self):
        """将 matplotlib 导航工具栏按钮名称与提示本地化。

        委托给 :func:`dssrr.gui.i18n.localize_nav_toolbar`（中英双向），避免与
        查看器 / 手动修复面板各维护一份重复的映射表。
        """
        try:
            from .i18n import localize_nav_toolbar
            localize_nav_toolbar(getattr(self, "_nav", None), self.lang)
        except Exception:
            pass

    def _create_basic_group(self) -> QGroupBox:
        """创建基本设置组。"""
        group = QGroupBox(self._tr("Basic Settings"))
        layout = QFormLayout(group)
        layout.setSpacing(8)

        # 自定义标题
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText(self._tr("Default title"))
        self.title_edit.setToolTip(self._tr(
            "Leave empty to use the default report title."))
        layout.addRow(self._tr("Title:"), self.title_edit)

        # DPI
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(200)
        self.dpi_spin.setSuffix(" dpi")
        layout.addRow(self._tr("Resolution:"), self.dpi_spin)

        # 格式
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "PDF", "SVG"])
        layout.addRow(self._tr("Format:"), self.format_combo)

        # 尺寸
        size_layout = QHBoxLayout()
        self.width_spin = QSpinBox()
        self.width_spin.setRange(8, 32)
        self.width_spin.setValue(16)
        self.width_spin.setSuffix(" cm")

        self.height_spin = QSpinBox()
        self.height_spin.setRange(6, 24)
        self.height_spin.setValue(11)
        self.height_spin.setSuffix(" cm")

        size_layout.addWidget(self.width_spin)
        size_layout.addWidget(QLabel("×"))
        size_layout.addWidget(self.height_spin)
        size_layout.addStretch()
        layout.addRow(self._tr("Size:"), size_layout)

        return group

    def _create_freq_group(self) -> QGroupBox:
        """创建频带设置组。"""
        group = QGroupBox(self._tr("Frequency Range"))
        layout = QFormLayout(group)
        layout.setSpacing(8)

        # 最小频率
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0.0, 1000.0)
        self.freq_min_spin.setValue(0.0)
        self.freq_min_spin.setDecimals(2)
        self.freq_min_spin.setSuffix(" Hz")
        layout.addRow(self._tr("Min frequency:"), self.freq_min_spin)

        # 最大频率
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(0.0, 1000.0)
        self.freq_max_spin.setValue(0.0)
        self.freq_max_spin.setDecimals(2)
        self.freq_max_spin.setSuffix(" Hz")
        self.freq_max_spin.setToolTip(self._tr(
            "0 = Auto (Nyquist × 0.9)"))
        layout.addRow(self._tr("Max frequency:"), self.freq_max_spin)

        return group

    def _create_display_group(self) -> QGroupBox:
        """创建显示选项组。"""
        group = QGroupBox(self._tr("Display Options"))
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # 第一行
        row1 = QHBoxLayout()
        self.show_time_cb = QCheckBox(self._tr("Time domain"))
        self.show_time_cb.setChecked(True)
        self.show_psd_cb = QCheckBox(self._tr("PSD spectrum"))
        self.show_psd_cb.setChecked(True)
        row1.addWidget(self.show_time_cb)
        row1.addWidget(self.show_psd_cb)
        row1.addStretch()
        layout.addLayout(row1)

        # 第二行
        row2 = QHBoxLayout()
        self.show_linear_cb = QCheckBox(self._tr("Linear spectrum"))
        self.show_linear_cb.setChecked(True)
        self.show_quality_cb = QCheckBox(self._tr("Quality metrics"))
        self.show_quality_cb.setChecked(True)
        row2.addWidget(self.show_linear_cb)
        row2.addWidget(self.show_quality_cb)
        row2.addStretch()
        layout.addLayout(row2)

        # 第三行
        row3 = QHBoxLayout()
        self.show_ref_cb = QCheckBox(self._tr("Reference segments"))
        self.show_ref_cb.setChecked(True)
        self.show_inset_cb = QCheckBox(self._tr("Inset zoom"))
        self.show_inset_cb.setChecked(True)
        row3.addWidget(self.show_ref_cb)
        row3.addWidget(self.show_inset_cb)
        row3.addStretch()
        layout.addLayout(row3)

        return group

    def _load_settings(self):
        """从 QSettings 加载上次的设置。"""
        # 基本设置
        self.title_edit.setText(self._settings.value(_KEY_TITLE, "", str))
        self.dpi_spin.setValue(int(self._settings.value(_KEY_DPI, 200)))
        format_idx = self._settings.value(_KEY_FORMAT, 0, int)
        self.format_combo.setCurrentIndex(format_idx)
        self.width_spin.setValue(int(self._settings.value(_KEY_WIDTH, 16)))
        self.height_spin.setValue(int(self._settings.value(_KEY_HEIGHT, 11)))

        # 频带设置
        self.freq_min_spin.setValue(float(self._settings.value(_KEY_FREQ_MIN, 0.0)))
        self.freq_max_spin.setValue(float(self._settings.value(_KEY_FREQ_MAX, 0.0)))

        # 显示选项
        self.show_time_cb.setChecked(
            self._settings.value(_KEY_SHOW_TIME, True, bool))
        self.show_psd_cb.setChecked(
            self._settings.value(_KEY_SHOW_PSD, True, bool))
        self.show_linear_cb.setChecked(
            self._settings.value(_KEY_SHOW_LINEAR, True, bool))
        self.show_quality_cb.setChecked(
            self._settings.value(_KEY_SHOW_QUALITY, True, bool))
        self.show_ref_cb.setChecked(
            self._settings.value(_KEY_SHOW_REF, True, bool))
        self.show_inset_cb.setChecked(
            self._settings.value(_KEY_SHOW_INSET, True, bool))

    def _save_settings(self):
        """保存设置到 QSettings。"""
        # 基本设置
        self._settings.setValue(_KEY_TITLE, self.title_edit.text())
        self._settings.setValue(_KEY_DPI, self.dpi_spin.value())
        self._settings.setValue(_KEY_FORMAT, self.format_combo.currentIndex())
        self._settings.setValue(_KEY_WIDTH, self.width_spin.value())
        self._settings.setValue(_KEY_HEIGHT, self.height_spin.value())

        # 频带设置
        self._settings.setValue(_KEY_FREQ_MIN, self.freq_min_spin.value())
        self._settings.setValue(_KEY_FREQ_MAX, self.freq_max_spin.value())

        # 显示选项
        self._settings.setValue(_KEY_SHOW_TIME, self.show_time_cb.isChecked())
        self._settings.setValue(_KEY_SHOW_PSD, self.show_psd_cb.isChecked())
        self._settings.setValue(_KEY_SHOW_LINEAR, self.show_linear_cb.isChecked())
        self._settings.setValue(_KEY_SHOW_QUALITY, self.show_quality_cb.isChecked())
        self._settings.setValue(_KEY_SHOW_REF, self.show_ref_cb.isChecked())
        self._settings.setValue(_KEY_SHOW_INSET, self.show_inset_cb.isChecked())

    def resizeEvent(self, event):
        """窗口尺寸变化时，防抖重新适配预览图宽度。"""
        super().resizeEvent(event)
        if self._preview_fig is not None:
            self._preview_resize_timer.start()

    def _reflow_preview(self):
        """视口变化后重绘预览，使画布始终适配预览区宽度。"""
        if self._preview_fig is None:
            return
        self._update_preview()

    def _update_preview(self):
        """生成并更新预览图。"""
        if self._original_data is None or self._repaired_data is None:
            QMessageBox.warning(self, self._tr("Warning"),
                                self._tr("No data available for preview."))
            return

        try:
            from ..repair_lib.report_plot import RepairReportPlotter
        except ImportError as e:
            QMessageBox.critical(self, self._tr("Error"),
                                 self._tr("Failed to import report module: {}").format(e))
            return

        try:
            # 获取设置
            settings = self.get_settings()

            # 创建绘图器
            plotter = RepairReportPlotter(sr=self.sr, lang=self.lang)

            # 确定异常区域
            anomaly_start = self._anomaly_start
            anomaly_end = self._anomaly_end
            if anomaly_start is None or anomaly_end is None:
                npts = len(self._original_data)
                anomaly_start = int(npts * 0.45)
                anomaly_end = int(npts * 0.55)

            # 预览画布尺寸策略：
            # 保持用户设定的宽高比，但保证每个子图有足够的纵向空间——
            # 若直接按视口宽度等比缩放，5 个子图堆叠时会显得非常扁。
            # 因此这里以"每行最小高度"为约束，取宽度自适应值与最小高度中的较大者，
            # 超出视口的部分由滚动区提供滚动查看。
            base_w, base_h = settings["figsize"]
            aspect = (float(base_h) / float(base_w)) if base_w else (11.0 / 16.0)
            avail_w = self.preview_scroll.viewport().width()
            if avail_w < 200:
                avail_w = 900  # 对话框尚未显示时的兜底宽度
            preview_dpi = 100

            # 统计纵向子图数量，用于估算所需最小高度
            # （PSD / 线性各自只占一图，修复前与修复后同图对比）
            n_stacked_rows = 1 if settings["show_time_domain"] else 0
            if settings["show_psd"]:
                n_stacked_rows += 1
            if settings["show_linear_spectrum"]:
                n_stacked_rows += 1
            if settings["show_quality_metrics"]:
                n_stacked_rows += 1
            n_stacked_rows = max(n_stacked_rows, 1)
            avail_h = self.preview_scroll.viewport().height()
            if avail_h < 150:
                avail_h = 600  # 兜底
            # 每个子图所需的最小像素高度（含标题与轴标签），保证不扁
            min_row_px = 120
            min_canvas_h_px = n_stacked_rows * min_row_px

            # 候选一：宽度自适应视口，按宽高比得高度
            preview_w_in = max(6.0, avail_w / float(preview_dpi))
            preview_h_in = preview_w_in * aspect
            # 若由此得到的每行高度不足，则改由最小高度反推宽度（保持宽高比），
            # 宁可出现横向滚动，也不把子图压扁。
            if preview_h_in * preview_dpi < min_canvas_h_px:
                preview_h_in = min_canvas_h_px / float(preview_dpi)
                preview_w_in = preview_h_in / aspect if aspect > 0 else preview_w_in
            # 若高度已超出视口仍有富余，也不无限放大，取二者上限约束
            max_h_px = max(avail_h, min_canvas_h_px)
            if preview_h_in * preview_dpi > max_h_px * 1.6:
                preview_h_in = max_h_px * 1.6 / float(preview_dpi)
                preview_w_in = preview_h_in / aspect if aspect > 0 else preview_w_in

            # 复用交互式画布的 figure（report_plot 传入 fig 时清空并重绘）
            fig = self.preview_canvas.figure

            # 生成预览图（直接渲染到画布，可交互缩放/平移/编辑）
            self._preview_fig = plotter.plot_report(
                original=self._original_data,
                repaired=self._repaired_data,
                anomaly_start=anomaly_start,
                anomaly_end=anomaly_end,
                anomaly_regions=self._anomaly_regions,
                figsize=(preview_w_in, preview_h_in),
                dpi=preview_dpi,  # 预览使用较低分辨率
                custom_title=settings["title"] if settings["title"] else None,
                freq_range=settings["freq_range"],
                show_time_domain=settings["show_time_domain"],
                show_psd=settings["show_psd"],
                show_linear_spectrum=settings["show_linear_spectrum"],
                show_quality_metrics=settings["show_quality_metrics"],
                show_reference_segments=settings["show_reference_segments"],
                show_inset_zoom=settings["show_inset_zoom"],
                fig=fig,
            )

            # 画布尺寸显式跟随 figure 的英寸尺寸与 dpi，
            # 否则 get_width_height() 返回的是旧 widget 尺寸，
            # 导致 figure 放大后画布未同步、图缩在左上角留出大片空白。
            fig_w, fig_h = fig.get_size_inches()
            canvas_dpi = fig.get_dpi()
            target_w = max(1, int(round(fig_w * canvas_dpi)))
            target_h = max(1, int(round(fig_h * canvas_dpi)))
            self.preview_canvas.setMinimumSize(target_w, target_h)
            self.preview_canvas.resize(target_w, target_h)
            self.preview_canvas.updateGeometry()
            self.preview_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, self._tr("Error"),
                                 self._tr("Failed to generate preview: {}").format(e))

    def accept(self):
        """确认并保存设置。"""
        self._save_settings()
        super().accept()

    def reject(self):
        """取消并关闭。"""
        super().reject()

    def get_settings(self) -> dict:
        """
        获取用户配置的设置。

        Returns
        -------
        dict : 包含所有导出设置的字典
        """
        # 频率范围
        freq_min = self.freq_min_spin.value()
        freq_max = self.freq_max_spin.value()
        if freq_max <= freq_min:
            freq_max = self.sr / 2 * 0.9  # 自动值

        return {
            # 基本设置
            "title": self.title_edit.text().strip(),
            "dpi": self.dpi_spin.value(),
            "format": self.format_combo.currentText().lower(),
            "figsize": (self.width_spin.value(), self.height_spin.value()),

            # 频带设置
            "freq_range": (freq_min, freq_max),

            # 显示选项
            "show_time_domain": self.show_time_cb.isChecked(),
            "show_psd": self.show_psd_cb.isChecked(),
            "show_linear_spectrum": self.show_linear_cb.isChecked(),
            "show_quality_metrics": self.show_quality_cb.isChecked(),
            "show_reference_segments": self.show_ref_cb.isChecked(),
            "show_inset_zoom": self.show_inset_cb.isChecked(),
        }

    def _tr(self, text: str) -> str:
        """查表翻译；表里没有或语言不认识时**回退英文原文**。

        注意兜底必须是 ``translations["en"]`` 而不是 ``["zh"]``：开源发行版的
        界面是英文的，若这里回退中文，任何一个漏翻的键都会在英文界面上冒出
        一句中文。
        """
        translations = {
            "zh": {
                "Export Settings": "导出设置",
                "Basic Settings": "基本设置",
                "Title:": "标题:",
                "Default title": "使用默认标题",
                "Leave empty to use the default report title.":
                    "留空则使用默认报告标题。",
                "Resolution:": "分辨率:",
                "Format:": "格式:",
                "Size:": "尺寸:",
                "Frequency Range": "频带范围",
                "Min frequency:": "最小频率:",
                "Max frequency:": "最大频率:",
                "0 = Auto (Nyquist × 0.9)":
                    "0 = 自动（奈奎斯特频率 × 0.9）",
                "Display Options": "显示选项",
                "Time domain": "时域对比图",
                "PSD spectrum": "功率谱密度图",
                "Linear spectrum": "线性频谱图",
                "Quality metrics": "质量指标表格",
                "Reference segments": "参考段",
                "Inset zoom": "放大插图",
                "Preview": "预览",
                "Export": "导出",
                "Cancel": "取消",
                "Edit Text": "编辑文字",
                "Batch edit titles, axis labels, and legends":
                    "批量编辑标题、轴标签和图例",
                "Click 'Preview' to generate": "点击「预览」生成报告图",
                "No data available for preview.":
                    "没有可用于预览的数据。",
                "Failed to import report module: {}":
                    "导入报告模块失败：{}",
                "Failed to generate preview: {}":
                    "生成预览失败：{}",
            },
            "en": {
                "Export Settings": "Export Settings",
                "Basic Settings": "Basic Settings",
                "Title:": "Title:",
                "Default title": "Default title",
                "Leave empty to use the default report title.":
                    "Leave empty to use the default report title.",
                "Resolution:": "Resolution:",
                "Format:": "Format:",
                "Size:": "Size:",
                "Frequency Range": "Frequency Range",
                "Min frequency:": "Min frequency:",
                "Max frequency:": "Max frequency:",
                "0 = Auto (Nyquist × 0.9)":
                    "0 = Auto (Nyquist × 0.9)",
                "Display Options": "Display Options",
                "Time domain": "Time domain",
                "PSD spectrum": "PSD spectrum",
                "Linear spectrum": "Linear spectrum",
                "Quality metrics": "Quality metrics",
                "Reference segments": "Reference segments",
                "Inset zoom": "Inset zoom",
                "Preview": "Preview",
                "Export": "Export",
                "Cancel": "Cancel",
                "Edit Text": "Edit Text",
                "Batch edit titles, axis labels, and legends":
                    "Batch edit titles, axis labels, and legends",
                "Click 'Preview' to generate": "Click 'Preview' to generate",
                "No data available for preview.":
                    "No data available for preview.",
                "Failed to import report module: {}":
                    "Failed to import report module: {}",
                "Failed to generate preview: {}":
                    "Failed to generate preview: {}",
            },
        }
        lang_dict = translations.get(self.lang, translations["en"])
        return lang_dict.get(text, text)
