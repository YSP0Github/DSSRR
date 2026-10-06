"""DSSRR 版 Matplotlib 图内文字批编辑器。

让用户在导出报告图之前，直接在界面上修改图内文字：标题、坐标轴标签、
图例等的**内容、字体、字号与颜色**。它通过给 matplotlib 工具栏挂一个
``QAction`` 的方式接入任意 FigureCanvas，因此可复用在不同对话框里。

对外接口
--------
- :func:`install_plot_text_editor_action` —— 给 ``NavigationToolbar2QT``
  装一个「Edit Text」按钮，点击即打开编辑器。
- :func:`show_plot_text_editor` —— 直接打开编辑器对话框。
- :class:`EditTextDialog` —— 编辑器本体。

CJK 字体处理：编辑器会检查所选字体是否真正包含目标文字的字形
（``_font_supports_text``），中文会自动回退到支持 CJK 的字体
（``_safe_font_family``），避免导出图里出现「豆腐块」。

本模块完全自包含：只依赖 PyQt5 与 matplotlib，主题取色复用
:func:`dssrr.gui.depulse_dialog.get_theme_colors`，不依赖任何外部主程序。
"""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPalette
from PyQt5.QtWidgets import (
    QColorDialog, QComboBox, QDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QVBoxLayout, QWidget, QAction,
)

from matplotlib import font_manager
from matplotlib.ft2font import FT2Font


_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _contains_cjk(text: str) -> bool:
    return any(
        "\u2e80" <= char <= "\u9fff"
        or "\u3400" <= char <= "\u4dbf"
        or "\uf900" <= char <= "\ufaff"
        for char in str(text or "")
    )


def _font_supports_text(family: str, text: str) -> bool:
    """Return whether a Matplotlib font contains all CJK glyphs in text."""
    if not _contains_cjk(text):
        return True
    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
        charmap = FT2Font(path).get_charmap()
        return all(ord(char) in charmap for char in str(text or "") if _contains_cjk(char))
    except (OSError, RuntimeError, ValueError):
        return False


def _safe_font_family(family: str, text: str) -> str:
    """Keep the selected font for Latin text and choose a CJK fallback safely."""
    if not family or _font_supports_text(family, text):
        return family
    for fallback in ("Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        if _font_supports_text(fallback, text):
            return fallback
    return family


def _apply_layout_after_text_edit(figure) -> None:
    """文字编辑后重新适配版式。

    ``tight_layout`` 与 ``constrained_layout`` 互斥：在已启用布局引擎
    （constrained / compressed）的 figure 上调用它会触发 matplotlib 告警
    "This figure includes Axes that are not compatible with tight_layout"，
    并可能把引擎从 constrained 切回 tight，破坏已算好的间距。

    由于修复报告图（report_plot.py）现在默认使用 ``constrained_layout``，
    这里必须先判断是否已有布局引擎：有则交给引擎自动维持版式，直接返回；
    仅对旧式（无引擎）figure 保留原来的 ``tight_layout`` 行为。
    """
    import warnings

    layout_engine = getattr(figure, "get_layout_engine", None)
    if callable(layout_engine):
        try:
            if figure.get_layout_engine() is not None:
                return
        except Exception:
            pass
    get_constrained = getattr(figure, "get_constrained_layout", None)
    if callable(get_constrained):
        try:
            if get_constrained():
                return
        except Exception:
            pass

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
        figure.tight_layout()


def _edit_text_icon_path():
    """Return the plot-text editor icon path (optional; missing file is fine)."""
    return os.path.join(_SCRIPT_DIR, "icons", "edit_text.svg")


def _theme_colors():
    """复用 DSSRR 自带的主题色（与 DePulseDialog 同一套）。"""
    try:
        from .depulse_dialog import get_theme_colors
        return get_theme_colors()
    except Exception:
        return {
            "bg": "#f6f8fc", "surface": "#ffffff", "surface2": "#f1f5f9",
            "border": "#d8e0ef", "text": "#1f2937", "text_secondary": "#475569",
            "accent": "#2563eb", "input_bg": "#ffffff", "input_border": "#cbd5e1",
        }


class EditTextDialog(QDialog):
    """Edit titles, axis labels, and legend titles for every axes."""

    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self.figure = figure
        self._rows = []
        self.setWindowTitle(self.tr("Edit Text"))
        self.setMinimumSize(760, 520)
        flags = self.windowFlags()
        flags &= ~Qt.WindowContextHelpButtonHint
        flags |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowModality(Qt.NonModal)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(10)

        axes_group = QGroupBox(self.tr("Axes Text"))
        axes_layout = QVBoxLayout(axes_group)
        axes_layout.setContentsMargins(10, 14, 10, 10)
        figure_title_row = QHBoxLayout()
        figure_title_row.addWidget(QLabel(self.tr("Figure Title:")))
        self.figure_title_edit = QLineEdit(
            self.figure._suptitle.get_text() if getattr(self.figure, "_suptitle", None) else ""
        )
        self.figure_title_edit.setClearButtonEnabled(True)
        self.figure_title_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.figure_title_edit.setPlaceholderText(self.tr("Optional figure-level title"))
        figure_title_row.addWidget(self.figure_title_edit, 1)
        axes_layout.addLayout(figure_title_row)
        scroll = QScrollArea()
        scroll.setObjectName("plotTextAxesScroll")
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(220)
        table_widget = QWidget()
        table_widget.setObjectName("plotTextAxesTable")
        table_layout = QGridLayout(table_widget)
        table_layout.setContentsMargins(6, 6, 6, 6)
        table_layout.setHorizontalSpacing(10)
        table_layout.setVerticalSpacing(8)
        headers = [
            self.tr("Axes"), self.tr("Title"), self.tr("X Label"),
            self.tr("Y Label"), self.tr("Legend Title"),
        ]
        for col, header in enumerate(headers):
            label = QLabel(header)
            label.setObjectName("plotTextTableHeader")
            label.setStyleSheet("font-weight: 600;")
            table_layout.addWidget(label, 0, col, alignment=Qt.AlignTop)

        row_index = 1
        for idx, ax in enumerate(self.figure.axes):
            if not hasattr(ax, "get_title"):
                continue
            legend = ax.get_legend()
            legend_title = legend.get_title().get_text() if legend else ""
            title_edit = QLineEdit(ax.get_title())
            xlabel_edit = QLineEdit(ax.get_xlabel())
            ylabel_edit = QLineEdit(ax.get_ylabel())
            legend_edit = QLineEdit(legend_title)
            for edit in (title_edit, xlabel_edit, ylabel_edit, legend_edit):
                edit.setMinimumWidth(135)
                edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                edit.setClearButtonEnabled(True)
            for col in range(len(headers)):
                table_layout.setRowStretch(row_index, 0)
                _widget = QLabel(f"{idx + 1}") if col == 0 else (
                    title_edit if col == 1 else (
                        xlabel_edit if col == 2 else (
                            ylabel_edit if col == 3 else legend_edit
                        )
                    )
                )
                _align = (Qt.AlignTop | Qt.AlignLeft) if col == 0 else Qt.AlignTop
                table_layout.addWidget(_widget, row_index, col, alignment=_align)
            self._rows.append({
                "axes": ax,
                "title": title_edit,
                "xlabel": xlabel_edit,
                "ylabel": ylabel_edit,
                "legend": legend_edit,
            })
            row_index += 1
        if not self._rows:
            table_layout.addWidget(QLabel(self.tr("No axes found")), 1, 0, 1, len(headers))
        table_layout.setColumnMinimumWidth(0, 32)
        table_layout.setColumnStretch(1, 2)
        table_layout.setColumnStretch(2, 1)
        table_layout.setColumnStretch(3, 1)
        table_layout.setColumnStretch(4, 1)
        table_layout.setRowMinimumHeight(0, 28)
        for r in range(row_index):
            table_layout.setRowStretch(r, 0)
        table_layout.setRowStretch(row_index, 1)
        scroll.setWidget(table_widget)
        axes_layout.addWidget(scroll)
        main_layout.addWidget(axes_group, 1)

        font_group = QGroupBox(self.tr("Font"))
        font_layout = QGridLayout(font_group)
        font_layout.setContentsMargins(10, 12, 10, 10)
        font_layout.setHorizontalSpacing(12)
        font_layout.setVerticalSpacing(8)
        font_layout.addWidget(QLabel(self.tr("Font Family:")), 0, 0)
        self.font_family_combo = QComboBox()
        self.font_family_combo.setEditable(True)
        self.font_family_combo.setInsertPolicy(QComboBox.NoInsert)
        self.font_family_combo.addItem(self.tr("Keep current"), "")
        for family in ("DejaVu Sans", "Arial", "Times New Roman", "Microsoft YaHei", "SimSun"):
            self.font_family_combo.addItem(family, family)
        self.font_family_combo.setToolTip(self.tr("Choose the font family for edited text."))
        font_layout.addWidget(self.font_family_combo, 0, 1)

        font_layout.addWidget(QLabel(self.tr("Font Size:")), 0, 2)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(0, 72)
        self.font_size_spin.setValue(0)
        self.font_size_spin.setSpecialValueText(self.tr("Keep current"))
        self.font_size_spin.setToolTip(self.tr("Set the font size in points; zero keeps the current size."))
        font_layout.addWidget(self.font_size_spin, 0, 3)

        font_layout.addWidget(QLabel(self.tr("Font Weight:")), 0, 4)
        self.font_weight_combo = QComboBox()
        self.font_weight_combo.addItem(self.tr("Keep current"), "")
        for label, value in (
            (self.tr("Normal"), "normal"), (self.tr("Light"), "light"),
            (self.tr("Medium"), "medium"), (self.tr("Semibold"), "semibold"),
            (self.tr("Bold"), "bold"), (self.tr("Heavy"), "heavy"),
        ):
            self.font_weight_combo.addItem(label, value)
        self.font_weight_combo.setToolTip(self.tr("Set the weight for edited text."))
        font_layout.addWidget(self.font_weight_combo, 0, 5)

        font_layout.addWidget(QLabel(self.tr("Text Color:")), 1, 0)
        self.text_color = None
        self.text_color_btn = QPushButton(self.tr("Keep current"))
        self.text_color_btn.setToolTip(self.tr("Choose a color for edited text."))
        self.text_color_btn.clicked.connect(self._choose_text_color)
        font_layout.addWidget(self.text_color_btn, 1, 1, 1, 2)
        main_layout.addWidget(font_group)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        self.refresh_btn = QPushButton(self.tr("Refresh"))
        self.refresh_btn.setToolTip(self.tr("Reload text from the current figure."))
        self.refresh_btn.clicked.connect(self._refresh_from_figure)
        self.apply_btn = QPushButton(self.tr("Apply"))
        self.apply_btn.setToolTip(self.tr("Apply the text and style changes."))
        self.apply_btn.clicked.connect(self._apply_changes)
        self.close_btn = QPushButton(self.tr("Close"))
        self.close_btn.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.close_btn)
        main_layout.addLayout(button_layout)
        self._apply_editor_theme()

    def _apply_editor_theme(self):
        """Theme the scroll viewport and axes editor table explicitly."""
        c = _theme_colors()
        self.setStyleSheet(f"""
QDialog {{ background-color: {c['bg']}; color: {c['text']}; }}
QDialog QLabel {{ background: transparent; color: {c['text']}; }}
QDialog QGroupBox {{
    margin-top: 14px; padding: 12px 10px 10px 10px;
    border: 1px solid {c['border']}; border-radius: 10px;
    background-color: {c['surface']}; font-weight: 600;
}}
QDialog QLineEdit {{
    min-height: 24px; padding: 5px 8px;
    border: 1px solid {c['input_border']}; border-radius: 7px;
    background-color: {c['input_bg']}; color: {c['text']};
}}
QDialog QComboBox, QDialog QSpinBox {{
    min-height: 24px; padding: 4px 8px;
    border: 1px solid {c['input_border']}; border-radius: 7px;
    background-color: {c['input_bg']}; color: {c['text']};
}}
QDialog QPushButton {{
    padding: 5px 14px; border: 1px solid {c['border']};
    border-radius: 6px; background-color: {c['surface']}; color: {c['text']};
}}
QScrollArea#plotTextAxesScroll {{
    background-color: {c['surface2']};
    border: 1px solid {c['border']};
    border-radius: 7px;
}}
QScrollArea#plotTextAxesScroll > QWidget > QWidget,
QWidget#plotTextAxesTable {{
    background-color: {c['surface2']};
    color: {c['text']};
}}
QLabel#plotTextTableHeader {{
    color: {c['text_secondary']};
    background-color: {c['surface']};
    border-bottom: 1px solid {c['border']};
    padding: 4px 6px;
}}
""")
        for widget in (
            self.findChild(QScrollArea, "plotTextAxesScroll"),
            self.findChild(QScrollArea, "plotTextAxesScroll").viewport(),
            self.findChild(QWidget, "plotTextAxesTable"),
        ):
            palette = widget.palette()
            palette.setColor(QPalette.Window, QColor(c["surface2"]))
            palette.setColor(QPalette.Base, QColor(c["surface2"]))
            widget.setPalette(palette)
            widget.setAutoFillBackground(True)

    def _choose_text_color(self):
        color = QColorDialog.getColor(parent=self, title=self.tr("Choose Text Color"))
        if not color.isValid():
            return
        self.text_color = color.name()
        self.text_color_btn.setText(self.text_color)
        self.text_color_btn.setStyleSheet(
            f"background-color: {self.text_color}; color: {'white' if color.lightness() < 128 else 'black'};"
        )

    def _refresh_from_figure(self):
        """Reload editable fields from the current figure."""
        for row in self._rows:
            ax = row["axes"]
            legend = ax.get_legend()
            row["title"].setText(ax.get_title())
            row["xlabel"].setText(ax.get_xlabel())
            row["ylabel"].setText(ax.get_ylabel())
            row["legend"].setText(legend.get_title().get_text() if legend else "")
        suptitle = getattr(self.figure, "_suptitle", None)
        self.figure_title_edit.setText(suptitle.get_text() if suptitle else "")

    def _apply_changes(self):
        """Apply edits while keeping this dialog open."""
        font_size = self.font_size_spin.value()
        font_family = self.font_family_combo.currentData()
        if self.font_family_combo.currentIndex() == 0:
            font_family = ""
        elif not font_family:
            font_family = self.font_family_combo.currentText().strip()
        font_weight = self.font_weight_combo.currentData()

        figure_title = self.figure_title_edit.text().strip()
        if figure_title:
            self.figure.suptitle(figure_title)
        elif getattr(self.figure, "_suptitle", None) is not None:
            self.figure._suptitle.set_text("")
        suptitle = getattr(self.figure, "_suptitle", None)
        if suptitle is not None:
            if font_size > 0:
                suptitle.set_fontsize(font_size)
            if font_family:
                suptitle.set_fontfamily(_safe_font_family(font_family, suptitle.get_text()))
            if font_weight:
                suptitle.set_fontweight(font_weight)
            if self.text_color:
                suptitle.set_color(self.text_color)

        for row in self._rows:
            ax = row["axes"]
            if not hasattr(ax, "set_title"):
                continue
            ax.set_title(row["title"].text())
            ax.set_xlabel(row["xlabel"].text())
            ax.set_ylabel(row["ylabel"].text())
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(row["legend"].text())

            if font_size > 0:
                ax.title.set_fontsize(font_size)
                ax.xaxis.label.set_fontsize(font_size)
                ax.yaxis.label.set_fontsize(font_size)
            if font_family:
                ax.title.set_fontfamily(_safe_font_family(font_family, ax.title.get_text()))
                ax.xaxis.label.set_fontfamily(_safe_font_family(font_family, ax.xaxis.label.get_text()))
                ax.yaxis.label.set_fontfamily(_safe_font_family(font_family, ax.yaxis.label.get_text()))
            if font_weight:
                ax.title.set_fontweight(font_weight)
                ax.xaxis.label.set_fontweight(font_weight)
                ax.yaxis.label.set_fontweight(font_weight)
            if self.text_color:
                ax.title.set_color(self.text_color)
                ax.xaxis.label.set_color(self.text_color)
                ax.yaxis.label.set_color(self.text_color)
            if legend is not None:
                legend_title = legend.get_title()
                legend_texts = list(legend.get_texts())
                if font_size > 0:
                    legend_title.set_fontsize(font_size)
                    for text in legend_texts:
                        text.set_fontsize(font_size)
                if font_family:
                    legend_title.set_fontfamily(_safe_font_family(font_family, legend_title.get_text()))
                    for text in legend_texts:
                        text.set_fontfamily(_safe_font_family(font_family, text.get_text()))
                if font_weight:
                    legend_title.set_fontweight(font_weight)
                    for text in legend_texts:
                        text.set_fontweight(font_weight)
                if self.text_color:
                    legend_title.set_color(self.text_color)
                    for text in legend_texts:
                        text.set_color(self.text_color)

        canvas = getattr(self.figure, "canvas", None)
        if canvas is not None:
            try:
                _apply_layout_after_text_edit(self.figure)
            except Exception:
                pass
            canvas.draw_idle()


def install_plot_text_editor_action(toolbar, canvas, parent=None, translator=None):
    """Install the shared batch text editor action on a Matplotlib toolbar."""
    if toolbar is None or canvas is None:
        return None
    action = getattr(toolbar, "_dssrr_text_editor_action", None)
    if action is not None:
        return action

    def tr(text):
        if callable(translator):
            return translator(text)
        parent_translator = getattr(parent, "tr", None)
        return parent_translator(text) if callable(parent_translator) else text

    icon_path = _edit_text_icon_path()
    icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
    action = QAction(icon, tr("Edit Text"), toolbar)
    action.setToolTip(tr("Batch edit titles, axis labels, and legends"))
    action.setStatusTip(tr("Batch edit titles, axis labels, and legends"))

    def open_text_editor():
        show_plot_text_editor(toolbar, canvas, parent)

    action.triggered.connect(open_text_editor)
    toolbar.addSeparator()
    toolbar.addAction(action)
    toolbar._dssrr_text_editor_action = action
    return action


def show_plot_text_editor(toolbar, canvas, parent=None):
    """Show one non-modal text editor next to the owning plot window."""
    if toolbar is None or canvas is None:
        return None
    figure = getattr(canvas, "figure", None)
    if figure is None:
        return None
    dialog = getattr(toolbar, "_dssrr_text_editor_dialog", None)
    if dialog is not None and dialog.isVisible():
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    dialog_parent = parent
    try:
        toolbar_window = toolbar.window()
        if toolbar_window is not None and toolbar_window is not parent:
            dialog_parent = toolbar_window
    except Exception:
        pass

    dialog = EditTextDialog(figure, dialog_parent)
    toolbar._dssrr_text_editor_dialog = dialog

    def clear_dialog(*_):
        toolbar._dssrr_text_editor_dialog = None

    dialog.destroyed.connect(clear_dialog)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
