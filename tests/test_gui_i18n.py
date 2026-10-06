# -*- coding: utf-8 -*-
"""界面双语支持（English / 中文）与"缩放时无法框选异常段"提示的回归测试。

覆盖三块：
1. :class:`dssrr.gui.i18n.DssrrTranslator` 的**回退语义** —— 查不到时必须返回
   ``None``（null QString）而不是 ``""``，否则英文界面所有文字会变空白。
2. 语言切换：侧边栏文字、查看器/手动修复面板、matplotlib 导航工具栏都要跟着
   切，并且切回英文后**不能残留中文**（语言按钮上的"中文"除外，那是目标语言
   的自称）。
3. 段选择提示：导航工具栏进入平移/缩放模式时提示语要变成警告，退出后恢复，
   且换主题不会把警告色刷掉。

需要 PyQt5（headless 下用 ``QT_QPA_PLATFORM=offscreen``）。
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QCoreApplication, QSettings  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from dssrr.paths import APP_NAME, APP_ORG  # noqa: E402
from dssrr.i18n import set_language as core_set_language  # noqa: E402
from dssrr.gui import i18n as gi  # noqa: E402
from dssrr.gui.translations_zh import ZH  # noqa: E402

CJK_RANGE = ("\u4e00", "\u9fff")


def cjk(text):
    """返回 ``text`` 里所有汉字（用于断言"没有残留中文"）。"""
    lo, hi = CJK_RANGE
    return "".join(ch for ch in text if lo <= ch <= hi)


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    a.setOrganizationName(APP_ORG)
    a.setApplicationName(APP_NAME)
    return a


@pytest.fixture(autouse=True)
def _restore_language(app):
    """每个用例结束后把语言恢复成英文，并还原 QSettings 里保存的值。

    这里刻意用 ``retranslate=False``：只需要把词条表复位，不需要再遍历一遍
    上个用例留下的（可能已在析构中的）窗口。最后再把**进程级语言**清掉，
    否则会污染依赖 ``DSSRR_LANG`` 环境变量的 test_language_defaults.py。
    """
    settings = QSettings(APP_ORG, APP_NAME)
    saved = settings.value(gi.SETTINGS_KEY, None)
    yield
    gi.apply_language("en", save=False, retranslate=False)
    core_set_language(None)
    if saved is None:
        settings.remove(gi.SETTINGS_KEY)
    else:
        settings.setValue(gi.SETTINGS_KEY, saved)


# ---------------------------------------------------------------------------
# 1. 翻译器回退语义
# ---------------------------------------------------------------------------
def test_translator_returns_translation_for_known_entry():
    tr = gi.DssrrTranslator({"Segment Selection": "分段选择"})
    assert tr.translate("AnyContext", "Segment Selection") == "分段选择"


def test_translator_returns_none_for_missing_entry():
    """核心回归点：查不到必须是 None（→ null QString），Qt 才会用英文原文。"""
    tr = gi.DssrrTranslator({"Segment Selection": "分段选择"})
    assert tr.translate("AnyContext", "Not In Table") is None


def test_qt_falls_back_to_source_text_when_entry_missing(app):
    """把上面那条语义放到 Qt 的真实调用链上验证。

    历史 bug：``translate()`` 返回 ``""``，Qt 认为"翻译结果就是空串"直接用掉，
    英文界面所有文字变空白。
    """
    tr = gi.DssrrTranslator({})
    app.installTranslator(tr)
    try:
        assert QCoreApplication.translate("Ctx", "Hello World") == "Hello World"
    finally:
        app.removeTranslator(tr)

def test_chinese_table_has_no_empty_values():
    """空译文会让对应文字变空白（同上面那个坑），一律不允许。"""
    empty = [k for k, v in ZH.items() if not str(v).strip()]
    assert empty == [], "中文词条表存在空译文: %r" % (empty[:5],)


# ---------------------------------------------------------------------------
# 2. 语言切换
# ---------------------------------------------------------------------------
def test_language_label_and_target(app):
    gi.apply_language("en", save=False)
    assert gi.language_label("en") == "English"
    assert gi.language_label("zh") == "中文"
    assert gi.language_label("zh_CN") == "中文"
    assert gi.target_language() == "zh"
    gi.apply_language("zh", save=False)
    assert gi.target_language() == "en"


def test_apply_language_persists_to_qsettings(app):
    gi.apply_language("zh", save=True)
    assert QSettings(APP_ORG, APP_NAME).value(gi.SETTINGS_KEY, "", str) == "zh"
    gi.apply_language("en", save=True)
    assert QSettings(APP_ORG, APP_NAME).value(gi.SETTINGS_KEY, "", str) == "en"


def test_localize_nav_toolbar_round_trip(app):
    """导航工具栏必须**双向**本地化：切回英文时要还原成 matplotlib 原文。"""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
    from matplotlib.figure import Figure
    from PyQt5.QtWidgets import QMainWindow

    host = QMainWindow()
    canvas = FigureCanvasQTAgg(Figure())
    host.setCentralWidget(canvas)
    tb = NavigationToolbar2QT(canvas, host)

    gi.apply_language("zh", save=False)
    gi.localize_nav_toolbar(tb)
    texts = [a.text() for a in tb.actions() if a.text()]
    assert "缩放" in texts and "平移" in texts and "Zoom" not in texts

    gi.apply_language("en", save=False)
    gi.localize_nav_toolbar(tb)
    texts = [a.text() for a in tb.actions() if a.text()]
    assert "Zoom" in texts and "Pan" in texts
    assert cjk("".join(texts)) == ""
    host.close()


def test_nav_toolbar_tooltips_are_localised(app):
    """按钮的悬浮提示（例如 "Zoom to rectangle"）也要跟着切。"""
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
    from matplotlib.figure import Figure
    from PyQt5.QtWidgets import QMainWindow

    host = QMainWindow()
    canvas = FigureCanvasQTAgg(Figure())
    host.setCentralWidget(canvas)
    tb = NavigationToolbar2QT(canvas, host)
    zoom = [a for a in tb.actions() if a.text() == "Zoom"][0]
    assert "Zoom to rectangle" in zoom.toolTip()

    gi.apply_language("zh", save=False)
    gi.localize_nav_toolbar(tb)
    assert zoom.toolTip().startswith("缩放到矩形区域")
    host.close()


def test_main_window_toggle_switches_whole_shell(app):
    """点一次语言按钮：侧边栏、状态栏主题标签、查看器静态文字全部切中文。"""
    from dssrr.gui.main import DSSRRMainWindow, _make_app

    _make_app()
    gi.apply_language("en", save=False)
    win = DSSRRMainWindow()
    try:
        assert win.btn_lang.text() == "Language: 中文"
        assert win._nav_viewer.text() == "Repaired DB Viewer"
        assert win._viewer.grp_view.title() == "View"

        win.btn_lang.click()
        assert gi.current_language() == "zh"
        assert win.btn_lang.text() == "语言: English"
        assert win._nav_viewer.text() == "修复库查看器"
        assert win.btn_exit.text() == "退出"
        assert win._viewer.grp_view.title() == "视图"
        assert cjk(win._theme_holder.text()) != ""

        win.btn_lang.click()
        assert gi.current_language() == "en"
        assert win._nav_viewer.text() == "Repaired DB Viewer"
        assert win._viewer.grp_view.title() == "View"
    finally:
        win.close()


def test_manual_page_stays_mounted_after_language_switch(app):
    """Manual Repair 面板的 retranslate_ui() 会重建 content widget，
    重建后必须重新挂载回 page_manual，否则页面空白。"""
    from dssrr.gui.main import DSSRRMainWindow, _make_app

    _make_app()
    gi.apply_language("en", save=False)
    win = DSSRRMainWindow()
    try:
        assert win._manual._content_widget.parent() is win.page_manual
        win.btn_lang.click()                       # -> 中文
        assert win._manual._content_widget.parent() is win.page_manual
        assert win.page_manual.layout().count() == 1
        assert win._manual._seg_title.text() == "分段选择"
        win.btn_lang.click()                       # -> English
        assert win._manual._content_widget.parent() is win.page_manual
        assert win._manual._seg_title.text() == "Segment Selection"
    finally:
        win.close()


def test_no_cjk_left_after_switching_back_to_english(app):
    """中→英来回切一轮后，主界面不应残留中文（语言按钮上的"中文"是设计如此）。"""
    from dssrr.gui.main import DSSRRMainWindow, _make_app

    _make_app()
    gi.apply_language("en", save=False)
    win = DSSRRMainWindow()
    try:
        win.btn_lang.click()
        win.btn_lang.click()
        for name in ("_nav_viewer", "_nav_manual", "lbl_sub", "btn_theme",
                     "btn_exit", "_theme_holder"):
            text = getattr(win, name).text()
            assert cjk(text) == "", "%s 残留中文: %r" % (name, text)
        assert cjk(win._viewer.grp_view.title()) == ""
        assert cjk(win._viewer.btn_refresh.text()) == ""
        assert cjk(win._manual._seg_title.text()) == ""
        assert cjk(win._manual._seg_hint.text()) == ""
        navs = ([a.text() for a in win._viewer._nav.actions()]
                + [a.text() for a in win._manual._seg_toolbar.actions()])
        assert cjk("".join(t for t in navs if t)) == ""
    finally:
        win.close()


# ---------------------------------------------------------------------------
# 3. "缩放/平移时无法框选异常段"提示
# ---------------------------------------------------------------------------
@pytest.fixture
def manual(app):
    """一个 segment_mode 的 DePulseDialog（不显示，仅用于取控件状态）。"""
    from dssrr.gui.depulse_dialog import DePulseDialog

    gi.apply_language("en", save=False)
    dlg = DePulseDialog(gi.translate, segment_mode=True)
    yield dlg
    try:
        dlg._stop_seg_nav_timer()
    except Exception:
        pass
    dlg.close()


def test_seg_hint_idle_text(manual):
    assert manual._seg_nav_active() is False
    assert manual._seg_hint.text() == manual._SEG_HINT_IDLE


def test_seg_hint_warns_while_zoom_active(manual):
    manual._seg_toolbar.zoom()
    assert manual._seg_nav_active() is True
    assert manual._seg_hint.text() == manual._SEG_HINT_NAV
    # 警告配色要跟空闲态不同（否则提示语看不出"出问题了"）
    assert "#B45309" in manual._seg_hint.styleSheet()
    assert "font-weight: 600" in manual._seg_hint.styleSheet()


def test_seg_hint_restores_after_leaving_nav_mode(manual):
    manual._seg_toolbar.pan()
    assert manual._seg_hint.text() == manual._SEG_HINT_NAV
    manual._seg_toolbar.pan()
    assert manual._seg_nav_active() is False
    assert manual._seg_hint.text() == manual._SEG_HINT_IDLE


def test_seg_hint_warning_survives_theme_refresh(manual):
    """换主题会重刷 inline 样式，不能把警告色刷回弱化色。"""
    manual._seg_toolbar.zoom()
    before = manual._seg_hint.styleSheet()
    manual._refresh_inline_theme()
    assert manual._seg_hint.styleSheet() == before


def test_seg_hint_is_translated(manual):
    manual._seg_toolbar.zoom()
    zh_hint = ZH[manual._SEG_HINT_NAV]
    assert cjk(zh_hint) != ""
    # retranslate=False：这里只想验证"提示语会跟着语言变"，不想触发面板重建
    # （重建会连工具栏一起换新，导航模式被复位，提示语自然回到空闲态）
    gi.apply_language("zh", save=False, retranslate=False)
    manual._refresh_seg_hint()
    assert manual._seg_hint.text() == zh_hint
    gi.apply_language("en", save=False, retranslate=False)
    manual._refresh_seg_hint()
    assert manual._seg_hint.text() == manual._SEG_HINT_NAV


def test_seg_nav_timer_is_running(manual):
    """兜底轮询要真的跑起来（个别 matplotlib 版本不回写按钮勾选状态）。"""
    assert manual._seg_nav_hint.is_running()
    manual._stop_seg_nav_timer()
    assert not manual._seg_nav_hint.is_running()


def test_seg_nav_hint_component_is_shared(app):
    """DePulseDialog 与 ManualRepairDialog 必须用同一份提示逻辑，别各写一遍。"""
    import io

    from dssrr.gui import nav_hint
    from dssrr.gui.depulse_dialog import DePulseDialog

    # DePulseDialog 的类常量就是共享组件里的英文原文（同一份 key）
    assert DePulseDialog._SEG_HINT_IDLE == nav_hint.IDLE_TEXT
    assert DePulseDialog._SEG_HINT_NAV == nav_hint.NAV_TEXT

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("depulse_dialog.py", "manual_repair.py"):
        src = io.open(os.path.join(root, "dssrr", "gui", name),
                      encoding="utf-8").read()
        assert "nav_hint.SegmentNavHint(" in src, name
    # 提示语本身必须都在中文词条表里
    assert nav_hint.IDLE_TEXT in ZH
    assert nav_hint.NAV_TEXT in ZH


def test_zoom_hint_text_mentions_turning_zoom_off():
    """提示语必须把"怎么恢复"讲清楚，不能只说"不可用"。"""
    hint = ("Zoom/pan is active — segment selection is disabled. "
            "Click Zoom or Pan again to turn it off, then drag to select.")
    assert hint in ZH
    assert "退出该模式" in ZH[hint]


# ---------------------------------------------------------------------------
# 4. 重刷遍历：内嵌面板由外层负责，不要被重复重建
# ---------------------------------------------------------------------------
def test_has_handled_ancestor(app):
    from PyQt5.QtWidgets import QWidget

    from dssrr.gui.i18n import _has_handled_ancestor

    outer = QWidget()
    mid = QWidget(outer)
    leaf = QWidget(mid)
    assert _has_handled_ancestor(leaf, {id(outer)}) is True
    assert _has_handled_ancestor(leaf, {id(mid)}) is True
    assert _has_handled_ancestor(leaf, {id(leaf)}) is False
    assert _has_handled_ancestor(outer, set()) is False
    outer.deleteLater()


def test_retranslate_widgets_prefers_outermost_handler(app):
    """外层面板处理后，其内嵌子控件不再被单独调用（否则手动修复页会空白）。"""
    from PyQt5.QtWidgets import QWidget

    from dssrr.gui import i18n as gi_mod

    calls = []

    class Outer(QWidget):
        def retranslate_ui(self):
            calls.append("outer")

    class Inner(QWidget):
        def retranslate_ui(self):
            calls.append("inner")

    outer = Outer()
    inner = Inner(outer)
    try:
        gi_mod._retranslate_widgets(app)
        assert "outer" in calls
        assert "inner" not in calls
    finally:
        outer.close()
        outer.deleteLater()
