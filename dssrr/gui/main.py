# -*- coding: utf-8 -*-
"""DSSRR GUI 统一入口（集成式主窗口）。

把 DSSRR 的桌面工具整合进**一个主窗口**（侧边栏导航 + 双页面）：

- **Repaired DB Viewer** —— 双库对比查看器（已内置 Manual / Batch / Settings / Help）
- **Manual Repair** —— 手动重处理页面（与查看器共享当前台站/日期/路径上下文，
  从查看器点 Manual Repair 也直接跳转到本页面）

Batch Repair 已集成在查看器内部（Batch Repair 按钮），不再单独列入口。
主题支持一键明/暗切换，查看器画布与两个工具页面同步换肤。

结构
----
- :func:`_make_app` —— 创建/复用 ``QApplication``，并**在任何 QSettings 之前**
  设置组织名/产品名（取自 :mod:`dssrr.paths` 的 ``APP_ORG`` / ``APP_NAME``），
  同时设置任务栏窗口图标。顺序很关键：``QSettings`` 的落盘位置由这两个名字
  决定，图标又必须在 ``QApplication`` 存在后才能创建。
- :class:`DSSRRMainWindow` —— 集成式主窗口。以 reparent 的方式把查看器与
  手动修复页塞进同一个 ``QStackedWidget``，两个工具自身的逻辑保持不变；
  由本窗口统一接管 QSS。
- :func:`launch_gui` / :func:`main` —— 对外入口，供
  ``python -m dssrr.gui`` 与 console script ``dssrr-gui`` 调用。

入口方式：
- ``python -m dssrr.gui``
- console script ``dssrr-gui``（见 pyproject.toml [project.scripts]）

依赖：PyQt5、matplotlib、obspy、pandas（``pip install "dssrr[gui]"``）。
"""
from __future__ import annotations

import os
import sys

from PyQt5.QtCore import Qt, QSize, QPointF
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QButtonGroup, QFrame, QSizePolicy,
)

from .icons import app_icon
from . import i18n as gui_i18n
from ..paths import APP_NAME, APP_ORG


def _make_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # 组织名 / 产品名决定 QSettings 的落盘位置，必须在任何 QSettings 之前设置
    app.setOrganizationName(APP_ORG)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName("DSSRR")
    # 任务栏 / 窗口管理器图标（Linux 下部分 WM 只认 QApplication 级图标）
    app.setWindowIcon(app_icon())
    # 界面语言：装一个查字典的 QTranslator，让全项目 580+ 处 self.tr() 直接生效。
    # 必须在主窗口构造之前调用——窗口构造时就会用 self.tr() 取词，装晚了首屏
    # 会是英文。语言取自 QSettings 的 UI/Language（首次运行默认英文）。
    gui_i18n.install(app)
    return app


# ----------------------------------------------------------------------
# 集成式主窗口：侧边栏导航 + 双页面（查看器 / 手动重处理）
# ----------------------------------------------------------------------
class DSSRRMainWindow(QMainWindow):
    """把 Repaired DB Viewer 与 Manual Repair 整合进同一个主窗口。

    嵌入方式（保持两个工具自身逻辑不变）：
    - 查看器：创建 DbRepairViewerWindow 后取其 centralWidget 与 matplotlib
      工具栏 reparent 到查看器页面；查看器自身窗口级 QSS 因控件已 reparent
      不再生效，由本窗口统一 QSS 接管；画布/工具栏主题仍由查看器维护。
    - 手动重处理：创建 DePulseDialog（段选择 + 方法 + 预览），
      将其 _content_widget reparent 到手动页；点确认后弹保存界面导出文件。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("DSSRR — DB Repair Suite"))
        self.setWindowIcon(app_icon())
        self.resize(1440, 900)
        # 最小尺寸**故意声明得比布局自身的最小值（minimumSizeHint）更小**：
        # Qt 实际生效的最小尺寸 = max(setMinimumSize, minimumSizeHint)，
        # 若这里写死一个更大的值（例如 1100x720），它就会盖过布局的真实下限，
        # 使 _place_on_screen 里"收缩到屏幕内"的钳制失效——窗口一旦放不下屏幕，
        # 关掉模态对话框后就会被窗口管理器挪到屏幕底部。这里只留一个宽松下限，
        # 真正的下限交给布局自己算。
        self.setMinimumSize(960, 640)
        self._dark = False
        # 主题令牌必须先于 _embed_viewer 设置：查看器 __init__ 会从 parent
        # 读取 theme_* 属性（_collect_theme）
        self._apply_theme_tokens()
        self._build_chrome()
        self._install_theme_holder()
        self._embed_viewer()
        self._embed_manual()
        self._apply_style()
        self._nav_viewer.setChecked(True)
        self.pages.setCurrentIndex(0)
        # 只让当前页参与最小尺寸计算（见 _sync_page_size_policy 的说明）
        self._sync_page_size_policy(0)
        self._place_on_screen()
        self.statusBar().showMessage(self.tr("Ready"))

    # ---------------- 窗口摆放 ----------------
    def _place_on_screen(self):
        """把窗口摆在可用屏幕区域内（居中 + 必要时收缩）。

        显式定位有两个好处：
        - 首次打开就在屏幕中央，而不是 Qt 的默认位置；
        - 万一布局的最小尺寸仍大于屏幕（例如用户屏幕很小），窗口会被收缩到
          可用区域内，而不是伸到屏幕外——窗口超出屏幕正是它在关闭模态对话框
          后被窗口管理器"挪到屏幕底部"的诱因。
        """
        try:
            scr = self.screen() or QApplication.primaryScreen()
            if scr is None:
                return
            avail = scr.availableGeometry()
            if not avail.isValid():
                return
            w = min(self.width(), avail.width())
            h = min(self.height(), avail.height())
            w = max(w, self.minimumSizeHint().width())
            h = max(h, self.minimumSizeHint().height())
            w, h = min(w, avail.width()), min(h, avail.height())
            self.resize(w, h)
            self.move(avail.x() + (avail.width() - w) // 2,
                      avail.y() + (avail.height() - h) // 2)
        except Exception:
            pass

    # ---------------- 主题令牌（查看器经 parent 属性跟随） ----------------
    def _apply_theme_tokens(self):
        if self._dark:
            self.theme_bg = "#111827"
            self.theme_fg = "#E5E7EB"
            self.theme_surface1 = "#374151"
            self.theme_surface2 = "#2A3444"
            self.theme_accent = "#C14B28"
            self.theme_mantle = "#0B1220"
            self.theme_crust = "#0B1220"
            self.theme_panel = "#1F2937"
        else:
            self.theme_bg = "#F1F4F9"
            self.theme_fg = "#1F2937"
            self.theme_surface1 = "#D8E0EA"
            self.theme_surface2 = "#E9EEF5"
            self.theme_accent = "#2563EB"
            self.theme_mantle = "#E9EEF5"
            self.theme_crust = "#E9EEF5"
            self.theme_panel = "#FFFFFF"
        self._theme = {k: getattr(self, "theme_" + k) for k in (
            "bg", "fg", "surface1", "surface2", "accent",
            "mantle", "crust", "panel")}

    # ---------------- 品牌标记 ----------------
    def _apply_brand_logo(self):
        """侧边栏品牌标记：优先使用应用图标，资源缺失时退回字母 "D"。

        图标自带深色圆角底板，故 QSS 中 #brandLogo 背景置为透明；只有降级
        为字母时才内联补上强调色底（内联样式优先级高于全局 QSS）。
        """
        pixmap = app_icon().pixmap(34, 34)
        if pixmap.isNull():
            self.lbl_logo.setText("D")
            self.lbl_logo.setStyleSheet(
                f"background-color: {self.theme_accent}; color: #FFFFFF;"
                "border-radius: 9px; font-size: 18px; font-weight: 800;")
        else:
            self.lbl_logo.setText("")
            self.lbl_logo.setPixmap(pixmap)

    # ---------------- 导航图标（QPainter 矢量绘制，避免字体依赖） ----------------
    def _nav_icon(self, kind):
        color = QColor("#475569" if not self._dark else "#E2E8F0")
        pm = QPixmap(18, 18)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        if kind == "db":
            # 数据库圆柱：顶/底椭圆 + 两侧边线
            pen = QPen(color, 1.6)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(9.0, 4.0), 6.0, 2.4)
            p.drawEllipse(QPointF(9.0, 14.0), 6.0, 2.4)
            p.drawLine(QPointF(3.0, 4.0), QPointF(3.0, 14.0))
            p.drawLine(QPointF(15.0, 4.0), QPointF(15.0, 14.0))
        else:  # manual：十字修复标识
            pen = QPen(color, 2.2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawLine(9, 3, 9, 15)
            p.drawLine(3, 9, 15, 9)
        p.end()
        return QIcon(pm)

    # ---------------- 界面骨架 ----------------
    def _build_chrome(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- 左侧固定侧边栏（品牌区 + 工具导航 + 主题/退出） ----
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(226)
        sb = QVBoxLayout(self.sidebar)
        sb.setContentsMargins(14, 18, 14, 14)
        sb.setSpacing(4)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        # 品牌标记：直接使用应用图标（比字母 "D" 更完整；图标缺失时降级为字母）
        self.lbl_logo = QLabel()
        self.lbl_logo.setObjectName("brandLogo")
        self.lbl_logo.setAlignment(Qt.AlignCenter)
        self.lbl_logo.setFixedSize(34, 34)
        self._apply_brand_logo()
        brand.addWidget(self.lbl_logo)
        tbox = QVBoxLayout()
        tbox.setSpacing(0)
        self.lbl_title = QLabel("DSSRR")
        self.lbl_title.setObjectName("brandTitle")
        self.lbl_sub = QLabel(self.tr("DB Repair Suite"))
        self.lbl_sub.setObjectName("brandSub")
        tbox.addWidget(self.lbl_title)
        tbox.addWidget(self.lbl_sub)
        brand.addLayout(tbox)
        brand.addStretch(1)
        sb.addLayout(brand)
        sb.addSpacing(14)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_viewer = self._make_nav_button(
            self.tr("Repaired DB Viewer"), "db", 0)
        self._nav_manual = self._make_nav_button(
            self.tr("Manual Repair"), "manual", 1)
        sb.addWidget(self._nav_viewer)
        sb.addWidget(self._nav_manual)

        sb.addStretch(1)

        self.btn_theme = QPushButton(self.tr("Dark mode"))
        self.btn_theme.setObjectName("footerButton")
        self.btn_theme.setCursor(Qt.PointingHandCursor)
        self.btn_theme.clicked.connect(self._toggle_theme)
        sb.addWidget(self.btn_theme)

        # 中英切换：按钮上显示的是**目标语言**的自称（当前英文 → 显示"中文"），
        # 点击后立即重刷全部已打开面板。选择记在 QSettings(UI/Language)。
        self.btn_lang = QPushButton(self._language_button_text())
        self.btn_lang.setObjectName("footerButton")
        self.btn_lang.setCursor(Qt.PointingHandCursor)
        self.btn_lang.setToolTip(self.tr("Switch interface language (English / 中文)"))
        self.btn_lang.clicked.connect(self._toggle_language)
        sb.addWidget(self.btn_lang)

        self.btn_exit = QPushButton(self.tr("Exit"))
        self.btn_exit.setObjectName("footerExit")
        self.btn_exit.setCursor(Qt.PointingHandCursor)
        self.btn_exit.clicked.connect(self.close)
        sb.addWidget(self.btn_exit)

        root.addWidget(self.sidebar)

        # ---- 右侧页面堆叠（布局在嵌入时安装） ----
        self.pages = QStackedWidget()
        self.pages.setObjectName("pages")
        self.page_viewer = QWidget()
        self.page_manual = QWidget()
        self.pages.addWidget(self.page_viewer)
        self.pages.addWidget(self.page_manual)
        root.addWidget(self.pages, 1)

    def _make_nav_button(self, text, icon_kind, page_idx):
        btn = QPushButton(text)
        btn.setObjectName("navButton")
        btn.setCheckable(True)
        btn.setIcon(self._nav_icon(icon_kind))
        btn.setIconSize(QSize(18, 18))
        btn.setCursor(Qt.PointingHandCursor)
        self._nav_group.addButton(btn, page_idx)
        btn.clicked.connect(lambda: self._switch_page(page_idx))
        return btn

    # ---------------- 嵌入工具 ----------------
    def _embed_viewer(self):
        from .viewer import DbRepairViewerWindow, _toolbar_qss
        self._viewer = DbRepairViewerWindow(parent=self)
        cw = self._viewer.centralWidget()
        nav = self._viewer._nav

        v0 = QVBoxLayout(self.page_viewer)
        v0.setContentsMargins(0, 0, 0, 0)
        v0.setSpacing(0)
        nav.setParent(self.page_viewer)
        nav.setStyleSheet(_toolbar_qss(self._dark))
        cw.setParent(self.page_viewer)
        # 关键：查看器中央控件的本地 QSS（仅设背景色）会摊平其所有子控件
        # 的背景（Qt 层叠：就近样式表优先），这正是旧界面按钮全部"一个色"
        # 的原因。嵌入后清掉它，由主窗口统一 QSS 接管全部控件样式。
        cw.setStyleSheet("")
        v0.addWidget(nav)
        v0.addWidget(cw, 1)

        # 查看器状态栏消息转发到主窗口状态栏（查看器窗口本身不可见）
        try:
            self._viewer.statusBar().messageChanged.connect(
                self.statusBar().showMessage)
        except Exception:
            pass

        # 防止后续主题同步再次写入会摊平子控件的中央控件本地 QSS
        orig_canvas = self._viewer._apply_canvas_theme

        def _patched_canvas():
            orig_canvas()
            cwc = self._viewer.centralWidget()
            if cwc is not None:
                cwc.setStyleSheet("")

        self._viewer._apply_canvas_theme = _patched_canvas

        # 集成模式下：查看器里的 Manual Repair 按钮改为跳转到手动重处理页
        try:
            self._viewer.btn_manual.clicked.disconnect()
        except Exception:
            pass
        self._viewer.btn_manual.clicked.connect(lambda: self._switch_page(1))

    def _embed_manual(self):
        from .depulse_dialog import DePulseDialog
        # 新版 Manual Repair：DePulseDialog（段选择 + 方法 + 预览）
        self._manual = DePulseDialog(
            self.tr, parent=self, segment_mode=True)
        # 页面布局只建一次并缓存：Qt 不允许给已有布局的 QWidget 再 setLayout
        # （新布局会被忽略并打印警告），所以语言切换重挂控件时必须复用它。
        v = QVBoxLayout(self.page_manual)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self._manual_layout = v
        self._mount_manual_content()

    def _mount_manual_content(self):
        """把 ``DePulseDialog._content_widget`` 挂进 Manual Repair 页。

        与 :meth:`_embed_manual` 拆开是因为**切换界面语言时**会调用
        ``_manual.retranslate_ui()`` 重建 content widget（旧控件被 deleteLater），
        此时必须重新挂载一次，否则页面会变成空白。
        """
        layout = getattr(self, "_manual_layout", None)
        if layout is None:
            layout = QVBoxLayout(self.page_manual)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            self._manual_layout = layout
        # 摘掉上一次挂载的控件（换语言时它已被重建，旧的那个正在等 deleteLater）。
        # 注意：不要用 setParent(None) —— 那会把旧面板变成**无父顶层窗口**
        # （未设标题时窗口管理器显示 applicationDisplayName "DSSRR"，内容仍是
        # Manual Repair 页），随后随 deleteLater 销毁，表现为切换语言时
        # 屏幕上闪过一个 "DSSRR" 弹窗。旧控件在 DePulseDialog.retranslate_ui()
        # 里已经 hide() + deleteLater()，这里只需从布局摘除并保持隐藏。
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
        content = self._manual._content_widget
        content.setParent(self.page_manual)
        # 与查看器同策略：清掉 dialog 的窗口级样式，由主窗口统一 QSS 接管。
        # 否则 Manual Repair 页会一直沿用 dialog 内置的 #1A1A1A 色系，
        # 与侧边栏/查看器页的 #111827 / #F1F4F9 色系不一致（切换主题后
        # 半深半浅）。
        content.setStyleSheet("")
        layout.addWidget(content, 1)
        # 让已 reparent 的控件按新父窗口重新解析 QSS（Qt 缓存了旧样式）
        self._repolish(content)
        # 重解析 QSS 不会重设写死颜色的 inline 样式，主题相关的提示语再刷一遍
        try:
            self._manual._refresh_inline_theme()
        except Exception:
            pass

    # ---------------- 页面切换 ----------------
    def _sync_page_size_policy(self, current):
        """让 QStackedWidget 只按**当前页**计算最小尺寸。

        QStackedWidget 默认取所有页（**含隐藏页**）最小宽度的最大值。隐藏的
        Manual Repair 页内部有横向分割条，最小宽度约 1630 px，于是整个主窗口
        的最小宽度被顶到 1850 px 以上 —— 比常见屏幕还宽。窗口一旦宽过屏幕，
        Windows 就会在每次模态对话框（Browse）关闭后重新"摆放"它，表现为
        主窗口莫名跳到屏幕底部。

        把非当前页的尺寸策略设为 Ignored，它就不再参与最小尺寸计算；切回来
        时恢复，页面自身布局不受影响。
        """
        for i in range(self.pages.count()):
            pg = self.pages.widget(i)
            if pg is None:
                continue
            if i == current:
                pg.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            else:
                pg.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def _switch_page(self, idx):
        self.pages.setCurrentIndex(idx)
        self._sync_page_size_policy(idx)
        if idx == 1:
            self._sync_manual_from_viewer()
        # 同步导航高亮（查看器内按钮等直接调用 _switch_page 时也选中对应导航）
        btn = self._nav_group.button(idx)
        if btn is not None:
            btn.setChecked(True)

    def _sync_manual_from_viewer(self):
        """把查看器当前选中的原始文件导入 Manual Repair 的 Trace 列表：
        已导入则直接选中，否则新增并显示波形。"""
        try:
            v = self._viewer
            raw_path, _fix_path, err = v._paths()
            if err or not raw_path or not os.path.exists(raw_path):
                self.statusBar().showMessage(
                    self.tr("Select raw/repaired DB and station/day in the "
                            "viewer first"))
                return
            m = self._manual
            key = os.path.basename(raw_path)
            for i in range(m._seg_trace_combo.count()):
                if key in m._seg_trace_combo.itemText(i):
                    m._seg_trace_combo.setCurrentIndex(i)
                    self.statusBar().showMessage(
                        self.tr("Manual Repair synced with viewer") +
                        f": {key}")
                    return
            m._load_paths([raw_path])
            self.statusBar().showMessage(
                self.tr("Manual Repair synced with viewer") + f": {key}")
        except Exception:
            pass

    # ---------------- 主题切换 ----------------
    @staticmethod
    def _repolish(widget):
        """强制 Qt 重新解析已 reparent 控件的 QSS。

        QSS 在 polish 阶段解析并缓存，reparent 后不会自动重算，导致嵌入
        页面沿用旧父窗口（dialog）的配色。unpolish + polish 可强制刷新。
        """
        if widget is None:
            return
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
        for child in widget.findChildren(QWidget):
            style.unpolish(child)
            style.polish(child)
            child.update()

    def _install_theme_holder(self):
        """在状态栏右侧放一个永久控件显示当前主题，避免被 viewer 状态栏消息覆盖。"""
        self._theme_holder = QLabel()
        self._theme_holder.setObjectName("themeHolder")
        self._theme_holder.setMargin(2)
        self._refresh_theme_holder()
        self.statusBar().addPermanentWidget(self._theme_holder)

    def _refresh_theme_holder(self):
        if not hasattr(self, "_theme_holder"):
            return
        accent = self.theme_accent
        # dot 前缀：深色=☾ 浅色=☀（避免依赖图标文件）
        dot = "●"
        label = self.tr("Dark theme") if self._dark else self.tr("Light theme")
        self._theme_holder.setText(f"  {dot} {label}  ")
        self._theme_holder.setStyleSheet(
            f"QLabel#themeHolder {{ color: {accent}; font-weight: 600; "
            f"padding: 0 8px; }}")

    def _toggle_theme(self):
        from .viewer import _toolbar_qss
        self._dark = not self._dark
        self._apply_theme_tokens()
        self._refresh_theme_holder()
        # 侧边栏图标随主题重绘
        self._nav_viewer.setIcon(self._nav_icon("db"))
        self._nav_manual.setIcon(self._nav_icon("manual"))
        self._apply_style()
        self.btn_theme.setText(
            self.tr("Light mode") if self._dark else self.tr("Dark mode"))
        try:
            self._viewer._apply_style()
        except Exception:
            pass
        try:
            from .depulse_dialog import get_theme_colors
            m = self._manual
            # 不能调 set_tool_theme()/retranslate_ui()，那会销毁嵌入在
            # page_manual 上的 content 并在不可见的 dialog 重建；这里只做
            # 「颜色令牌 + 写死的 inline 样式 + 强制重解析 QSS」三步。
            m._theme_colors = get_theme_colors()
            # 重刷 dialog 内置样式（方法卡/输入框/按钮等 objectName 规则）
            m._apply_theme_style()
            # 重刷写死颜色的 inline 样式（标题/状态文本/停止按钮/提示等）
            m._refresh_inline_theme()
            # content 已 reparent 到主窗口：强制 Qt 按新 QSS 重新解析
            self._repolish(m._content_widget)
            saved_sel = list(m._selections)
            # 重画段波形（会清空选区），随后恢复已有选区
            m._plot_segment_trace()
            m._selections = saved_sel
            m._redraw_selections()
            # 预览画布背景跟随主题（保留已绘对比曲线）
            c = m._theme_colors
            m.preview_canvas.set_canvas_colors(c['surface'], c['text'])
        except Exception:
            pass
        # 临时提示（2.5s 后自动消失），避免被 viewer 后续 refresh 永远盖掉
        self.statusBar().showMessage(
            self.tr("Dark theme") if self._dark else self.tr("Light theme"),
            2500)

    # ---------------- 界面语言切换（English / 中文） ----------------
    def _language_button_text(self):
        """侧边栏语言按钮的文字：显示**目标语言**的自称。

        英文界面下显示 "Language: 中文"，中文界面下显示 "语言: English"，
        用户一眼就知道点下去会变成什么语言。
        """
        return "{}: {}".format(
            self.tr("Language"),
            gui_i18n.language_label(gui_i18n.target_language()))

    def _toggle_language(self):
        """在英文 / 中文之间切换，并让所有已打开的面板立刻重刷文字。

        ``retranslate=False``：本窗口的 :meth:`retranslate_ui` 会做一遍完整重刷
        （含把重建后的 Manual Repair 面板重新挂载回页面），让 i18n 层再遍历一次
        只会把这个重面板重复重建，所以这里自己来。
        """
        code = gui_i18n.toggle_language(retranslate=False)
        self.retranslate_ui()
        self.statusBar().showMessage(
            "{}: {}".format(self.tr("Interface language"),
                            gui_i18n.language_label(code)), 2500)

    def retranslate_ui(self):
        """按当前语言重设本窗口及其内嵌面板的**静态**文字。

        说明
        ----
        - ``gui_i18n.apply_language()`` 内部已经遍历顶层控件并调用它们的
          ``retranslate_ui()``（见 :func:`dssrr.gui.i18n._retranslate_widgets`），
          但本方法**必须**显式再做一遍：本窗口与两个内嵌面板都是 reparent 过来
          的，``topLevelWidgets()`` 里拿不到内嵌的 Manual Repair 内容控件。
        - Manual Repair 的 ``retranslate_ui()`` 会**重建** content widget
          （旧控件 deleteLater），重建后必须重新挂载回 ``page_manual``，
          否则页面会空白。
        """
        self.setWindowTitle(self.tr("DSSRR — DB Repair Suite"))
        self.lbl_sub.setText(self.tr("DB Repair Suite"))
        self._nav_viewer.setText(self.tr("Repaired DB Viewer"))
        self._nav_manual.setText(self.tr("Manual Repair"))
        self.btn_theme.setText(
            self.tr("Light mode") if self._dark else self.tr("Dark mode"))
        self.btn_lang.setText(self._language_button_text())
        self.btn_lang.setToolTip(
            self.tr("Switch interface language (English / 中文)"))
        self.btn_exit.setText(self.tr("Exit"))
        self._refresh_theme_holder()
        # 查看器页（静态文字 + matplotlib 导航工具栏）
        try:
            self._viewer.retranslate_ui()
        except Exception:
            pass
        # 手动修复页：重建 content widget 后必须重新挂载回本页。
        # 两步分开 try：即使重建中途出错，也要把当前 content 挂回页面，
        # 否则 Manual Repair 页会变成空白。
        try:
            self._manual.retranslate_ui()
        except Exception:
            pass
        try:
            self._mount_manual_content()
        except Exception:
            pass

    # ---------------- 统一样式 ----------------
    def _apply_style(self):
        self.setStyleSheet(self._build_qss())
        self._refresh_theme_holder()

    def _build_qss(self):
        bg = self.theme_bg
        fg = self.theme_fg
        surface = self.theme_surface1
        panel = self.theme_panel
        crust = self.theme_crust
        accent = self.theme_accent
        # 侧边栏配色随主题：浅色模式用浅色侧边栏（深字+蓝高亮），
        # 深色模式保持深色导航
        if self._dark:
            sb_bg = "#0F172A"
            sb_border = "#1E293B"
            sb_title = "#F8FAFC"
            sb_sub = "#94A3B8"
            sb_item = "#CBD5E1"
            sb_hover = "#1E293B"
            sb_hover_text = "#F8FAFC"
            sb_btn = "#1E293B"
            sb_btn_border = "#334155"
            sb_btn_hover = "#273449"
            sb_exit = "#FCA5A5"
            sb_exit_border = "#7F1D1D"
            sb_exit_hover_bg = "#7F1D1D"
        else:
            sb_bg = "#E9EEF5"
            sb_border = "#D8E0EA"
            sb_title = "#1F2937"
            sb_sub = "#64748B"
            sb_item = "#334155"
            sb_hover = "#D8E0EA"
            sb_hover_text = "#0F172A"
            sb_btn = "#FFFFFF"
            sb_btn_border = "#D8E0EA"
            sb_btn_hover = "#F1F4F9"
            sb_exit = "#B91C1C"
            sb_exit_border = "#E5B8B8"
            sb_exit_hover_bg = "#B91C1C"
        return f"""
/* 子控件默认透明：窗口底色只由 QMainWindow / QDialog 提供。
   若这里写成 background-color: {bg}，所有没有显式背景的子控件（尤其是
   作为分区标题的 QLabel#fieldLabel）都会各自刷上一层窗口底色，看起来就是
   标题后面拖着一条灰色长条。 */
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
QLabel {{ color: {fg}; font-size: 12px; }}
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
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border-color: {accent};
}}
QPushButton {{
    padding: 5px 14px;
    border: 1px solid {surface};
    border-radius: 6px;
    background: {panel};
    color: {fg};
    font-weight: 500;
    font-size: 12px;
}}
QPushButton:hover {{ background: {surface}; }}
QPushButton:pressed {{ background: {crust}; }}
QPushButton:disabled {{
    color: #9AA5B1;
    background: {crust};
    border-color: {surface};
}}
QPushButton#preset {{ padding: 3px 8px; border-radius: 5px; }}
QRadioButton {{ spacing: 6px; padding: 2px 4px; color: {fg}; }}
QCheckBox {{ color: {fg}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {surface};
    border-radius: 4px;
    background: {crust};
}}
QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
}}
QTableWidget {{
    background: {panel};
    color: {fg};
    gridline-color: {surface};
    border: 1px solid {surface};
}}
QTableWidget::item {{ padding: 2px 4px; }}
QTableWidget::item:selected {{ background: {accent}; color: #FFFFFF; }}
QHeaderView::section {{
    background: {crust};
    color: {fg};
    border: 1px solid {surface};
    padding: 3px 6px;
}}
QLabel#detailStats {{ color: {fg}; font-weight: 600; padding: 2px 4px; }}
QListWidget {{
    background: {panel};
    color: {fg};
    border: 1px solid {surface};
    border-radius: 6px;
}}
QListWidget::item {{ padding: 3px 6px; }}
QListWidget::item:selected {{ background: {accent}; color: #FFFFFF; }}
QTabWidget::pane {{
    border: 1px solid {surface};
    border-radius: 6px;
    background: {panel};
}}
QTabBar::tab {{
    background: {surface};
    color: {fg};
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:selected {{
    background: {panel};
    border: 1px solid {surface};
    border-bottom: 2px solid {accent};
    font-weight: 600;
}}
QToolBar {{
    background: {crust};
    border: none;
    border-bottom: 1px solid {surface};
    spacing: 3px;
    padding: 3px;
}}
QToolButton {{
    color: {fg};
    background: {panel};
    border: 1px solid {surface};
    border-radius: 4px;
    padding: 4px 6px;
    margin: 1px;
}}
QToolButton:hover {{ background: {surface}; }}
QToolButton:checked {{ background: {surface}; border-color: {accent}; }}
QStatusBar {{ background: {crust}; color: {fg}; }}
QScrollArea {{ border: none; background: transparent; }}
QSplitter::handle {{ background: {surface}; }}
QSplitter::handle:vertical {{ height: 2px; }}
QMenu {{ background: {panel}; color: {fg}; border: 1px solid {surface}; }}
QMenu::item {{ padding: 4px 16px; }}
QMenu::item:selected {{ background: {surface}; }}

/* ---- 侧边栏（配色随主题：浅色模式浅色导航，深色模式深色导航） ---- */
#sidebar {{
    background-color: {sb_bg};
    border: none;
    border-right: 1px solid {sb_border};
}}
#brandLogo {{
    background-color: transparent;
    border-radius: 9px;
}}
#brandTitle {{ color: {sb_title}; font-size: 17px; font-weight: 700; }}
#brandSub {{ color: {sb_sub}; font-size: 11px; }}
#navButton {{
    color: {sb_item};
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
}}
#navButton:hover {{ background: {sb_hover}; color: {sb_hover_text}; }}
#navButton:checked {{
    background: {accent};
    color: #FFFFFF;
    font-weight: 600;
}}
#footerButton {{
    color: {sb_item};
    background: {sb_btn};
    border: 1px solid {sb_btn_border};
    border-radius: 8px;
    padding: 9px 12px;
    text-align: left;
    font-size: 12px;
}}
#footerButton:hover {{ background: {sb_btn_hover}; color: {sb_hover_text}; }}
#footerExit {{
    color: {sb_exit};
    background: transparent;
    border: 1px solid {sb_exit_border};
    border-radius: 8px;
    padding: 9px 12px;
    text-align: left;
    font-size: 12px;
}}
#footerExit:hover {{ background: {sb_exit_hover_bg}; color: #FFFFFF; }}
"""


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main():
    """启动 DSSRR GUI（QApplication + 集成式主窗口）。"""
    app = _make_app()
    win = DSSRRMainWindow()
    win.show()
    return sys.exit(app.exec_())


def launch_gui():
    """别名：启动 GUI（供 dssrr/__init__.py 导出）。"""
    main()


if __name__ == "__main__":
    main()
