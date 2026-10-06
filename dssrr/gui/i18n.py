# -*- coding: utf-8 -*-
"""界面语言运行时：装一个查字典的 ``QTranslator``，让 ``self.tr()`` 直接生效。

设计要点
--------
**为什么是 QTranslator 子类**
    DSSRR 的界面代码里已经有 580 多处 ``self.tr("...")`` / ``QCoreApplication.
    translate(...)``。只要把一个自定义 ``QTranslator`` 装到 ``QApplication`` 上，
    这些调用点**一行都不用改**就会去问我们的 :meth:`DssrrTranslator.translate`。
    比起把每处调用改成 ``i18n.tr(...)``，这个方案改动面小得多，也不会漏掉
    以后新写的 ``self.tr()``。

**查不到怎么办**
    ``translate()`` 返回**空字符串**表示"我没有这条翻译"，Qt 会自动回退到源码
    里的英文原文。所以词条表可以只覆盖已翻译的部分，缺条目不会让界面变成空白。

**和 :mod:`dssrr.i18n` 的关系**
    :mod:`dssrr.i18n` 管"当前语言是什么"（库层，不依赖 Qt）；本模块管"界面文字
    怎么翻译"（GUI 层）。切换语言时本模块会同步调用 ``dssrr.i18n.set_language``，
    于是导出报告、图标题、诊断文本也跟着切。

**持久化**
    选择记在 ``QSettings(APP_ORG, APP_NAME)`` 的 ``UI/Language`` 键下，与 DSSRR
    其余设置同处存放，下次启动自动恢复。
"""

from __future__ import annotations

from typing import Dict, Optional

from PyQt5.QtCore import QCoreApplication, QEvent, QObject, QSettings, QTranslator
from PyQt5.QtWidgets import QApplication, QWidget

from ..i18n import (DEFAULT_LANGUAGE, normalize_language, resolve_language,
                    set_language as _set_core_language)
from ..paths import APP_NAME, APP_ORG
from .translations_zh import ZH

#: QSettings 里保存界面语言的键名。
SETTINGS_KEY = "UI/Language"

#: 语言代码 → 词条表。
TABLES: Dict[str, Dict[str, str]] = {
    "zh": ZH,
    # 英文是源码原文，不需要表（空表 = 全部回退原文）
    "en": {},
}

#: 语言代码 → 在切换按钮上显示的"目标语言"名字。
LANGUAGE_LABELS = {"en": "English", "zh": "中文"}


def language_label(code: Optional[str]) -> str:
    """把语言代码转成按钮上显示的**语言自称**（``"en"`` → ``"English"``）。"""
    return LANGUAGE_LABELS.get(normalize_language(code) or "", str(code))


def target_language() -> str:
    """返回"点一下切换按钮会切到哪个语言"（当前是中文就返回 ``"en"``）。"""
    return "zh" if _current == "en" else "en"


class DssrrTranslator(QTranslator):
    """按"英文原文 → 译文"字典查表的翻译器。

    Qt 会拿 ``(context, sourceText, disambiguation, n)`` 来问；我们**忽略
    context**，只用 ``sourceText`` 查表。这样同一个英文串在不同类里也只需要
    维护一条译文。

    .. warning::
       查不到时**必须返回 ``None``**（Python 的 None → Qt 的 null QString），
       Qt 才会回退到源码里的英文原文。返回 ``""`` 是一个**非 null 的空
       QString**，Qt 会当成"翻译结果就是空串"直接用掉 —— 表现为英文界面所有
       文字全部变空白。实测：返回 ``''`` → ``translate()`` 得到 ``''``；
       返回 ``None`` → 得到原文。
    """

    def __init__(self, table: Optional[Dict[str, str]] = None, parent=None):
        super().__init__(parent)
        self._table: Dict[str, str] = dict(table or {})

    def set_table(self, table: Optional[Dict[str, str]]) -> None:
        self._table = dict(table or {})

    # ---- QTranslator 接口 ----
    def translate(self, context, source, disambiguation=None, n=-1):  # noqa: N802
        # 返回 None = 没有这条翻译，Qt 回退到 source 原文（见类文档）
        return self._table.get(source)

    def isEmpty(self):  # noqa: N802
        return not self._table


# ---------------------------------------------------------------------------
# 模块级状态
# ---------------------------------------------------------------------------
_app: Optional[QApplication] = None
_translator: Optional[DssrrTranslator] = None
_qt_translator: Optional[QTranslator] = None      # Qt 自带控件的翻译（可选）
_current: str = DEFAULT_LANGUAGE


def current_language() -> str:
    """返回当前界面语言代码（``"en"`` / ``"zh"``）。"""
    return _current


def translate(text: str, lang: Optional[str] = None) -> str:
    """把一句英文原文翻成目标语言（``lang=None`` = 当前界面语言）。

    给"拿不到 ``self.tr``"的地方用（例如模块级函数、数据模型里的显示名）。
    """
    code = normalize_language(lang) or _current
    return TABLES.get(code, {}).get(text, text)


# ---------------------------------------------------------------------------
# 安装 / 切换
# ---------------------------------------------------------------------------
def _read_saved_language() -> Optional[str]:
    """读取用户上次用切换按钮选的语言；**没选过返回 ``None``**。

    刻意不在这里兜底成英文：``None`` 表示"用户没有偏好"，让 :func:`install`
    继续去看环境变量 ``DSSRR_LANG``。如果这里就返回 ``"en"``，启动时会把进程
    语言钉死成英文，``DSSRR_LANG=zh`` 就永远失效了。
    """
    try:
        saved = QSettings(APP_ORG, APP_NAME).value(SETTINGS_KEY, "", str)
    except Exception:  # noqa: BLE001 - 设置后端不可用时不该拖垮启动
        saved = ""
    return normalize_language(saved)


def _save_language(code: str) -> None:
    try:
        QSettings(APP_ORG, APP_NAME).setValue(SETTINGS_KEY, code)
    except Exception:  # noqa: BLE001
        pass


def install(app=None, lang: Optional[str] = None) -> str:
    """在 ``QApplication`` 上装好翻译器并应用语言。

    Parameters
    ----------
    app : QApplication or None
        默认取 ``QApplication.instance()``。
    lang : str or None
        目标语言；``None`` = 依次尝试「用户上次保存的选择 → 环境变量
        ``DSSRR_LANG`` → 英文」。**用户用切换按钮选过的语言优先于环境变量**，
        但环境变量仍然能在首次启动时决定界面语言。

    Returns
    -------
    str
        实际生效的语言代码。
    """
    global _app, _translator, _qt_translator
    app = app or QApplication.instance()
    if app is None:
        # 没有 QApplication 时只更新库层语言，避免调用方崩溃
        _current = _set_core_language(lang or DEFAULT_LANGUAGE) or DEFAULT_LANGUAGE
        return _current

    # 幂等：换 QApplication（测试里常见）或重复调用时，先把旧的翻译器摘掉，
    # 否则它们会一直堆在 app 上，多个翻译器互相抢答。
    if _app is not None:
        for old in (_translator, _qt_translator):
            if old is None:
                continue
            try:
                _app.removeTranslator(old)
            except Exception:  # noqa: BLE001
                pass
    _translator = _qt_translator = None
    _app = app

    _translator = DssrrTranslator(TABLES.get("en", {}), _app)
    _app.installTranslator(_translator)
    # Qt 自带对话框（QMessageBox / QFileDialog / QColorDialog…）的按钮文字来自
    # Qt 自己的 .qm；PyQt5 轮子里通常带了 qtbase_zh_CN，能加载就一并装上。
    _qt_translator = QTranslator(_app)
    _app.installTranslator(_qt_translator)

    target = (normalize_language(lang)
              or _read_saved_language()
              or resolve_language())
    return apply_language(target, save=False)


def apply_language(code: Optional[str], save: bool = True,
                   retranslate: bool = True) -> str:
    """切换到 ``code``，并让所有已存在的界面重新取词。

    顺序很重要：**先**换词条表、**再**让控件重建文字，否则控件会拿到旧语言。

    Parameters
    ----------
    code : str or None
        目标语言代码。
    save : bool
        是否写入 ``QSettings``。
    retranslate : bool
        是否自动遍历控件重刷。当调用方**自己**会做一遍完整重刷时（例如
        :class:`dssrr.gui.main.DSSRRMainWindow` 需要额外把重建后的 Manual
        Repair 面板重新挂载回页面）传 ``False``，避免同一面板被重建两次。
    """
    global _current
    target = normalize_language(code) or DEFAULT_LANGUAGE
    if _translator is not None:
        _translator.set_table(TABLES.get(target, {}))
    # 同步库层语言：导出报告 / 图标题 / 诊断文本跟着界面走
    _set_core_language(target)
    _load_qt_translation(target)
    _current = target
    if save:
        _save_language(target)
    if retranslate:
        _retranslate_widgets()
    return target


def toggle_language(retranslate: bool = True) -> str:
    """在英文 / 中文之间来回切，返回切换后的语言。"""
    return apply_language("zh" if _current == "en" else "en",
                          retranslate=retranslate)


def _load_qt_translation(code: str) -> None:
    """尽量加载 Qt 自带的中文翻译（失败静默，不影响自研词条）。"""
    if _qt_translator is None:
        return
    if code == "zh":
        try:
            from PyQt5.QtCore import QLibraryInfo
            base = QLibraryInfo.location(QLibraryInfo.TranslationsPath)
            for name in ("qtbase_zh_CN", "qt_zh_CN"):
                if _qt_translator.load(name, base):
                    return
        except Exception:  # noqa: BLE001
            pass
    # 切回英文：清掉之前加载的（load 失败时 isEmpty 仍为 True）
    _qt_translator.load("", "")


# ---------------------------------------------------------------------------
# 让已有控件重新取词
# ---------------------------------------------------------------------------
#: 控件上"重新设置文字"的方法名，按优先级尝试。
_RETRANSLATE_METHODS = ("retranslate_ui", "retranslateUi", "set_tool_language")


def _widget_alive(widget) -> bool:
    """控件底层的 C++ 对象是否还在。

    ``retranslate_ui()`` 会 ``deleteLater()`` 掉旧面板，而被删掉的控件可能还挂在
    遍历队列里（Qt 的延迟删除在处理事件时才真正析构）。对这些控件调用任何方法
    都会抛 ``RuntimeError``，必须先探活。
    """
    try:
        widget.objectName()
    except RuntimeError:
        return False
    return True


def _has_handled_ancestor(widget, handled) -> bool:
    """widget 的某个祖先是否已经负责重刷过（避免重复重建内嵌面板）。"""
    try:
        parent = widget.parentWidget()
        while parent is not None:
            if id(parent) in handled:
                return True
            parent = parent.parentWidget()
    except RuntimeError:
        # 祖先链上有控件已被销毁，按"没有已处理的祖先"处理
        return False
    return False


def _retranslate_widgets(app=None) -> int:
    """遍历现有窗口，调用它们各自的"重新取词"方法。

    遍历顺序是**祖先优先**的广度优先：主窗口（``DSSRRMainWindow``）先被处理，
    它在自己的 ``retranslate_ui()`` 里负责重刷 reparent 过来的查看器页与手动
    修复页；于是这些被内嵌的子面板会被 :func:`_has_handled_ancestor` 跳过。

    这个"跳过"不是优化而是**正确性要求**：``DePulseDialog.retranslate_ui()``
    会重建 ``_content_widget``（旧控件 deleteLater）。如果它被单独调用而没人
    把新控件重新挂回 ``page_manual``，Manual Repair 页就会变成空白。

    瞬时对话框（批量修复、设置、帮助、导出）都是**每次打开时新建**的，天然会
    用上新语言，不需要在这里处理。

    整个过程对"控件已被销毁"是容错的：重建面板会产生一批待删除控件，而某些
    ``retranslate_ui()`` 实现（例如查看器的 ``refresh()``）会跑事件循环，把
    它们真正析构掉。

    Returns
    -------
    int
        实际被要求重刷的控件数量（调试用）。
    """
    app = app or _app or QApplication.instance()
    if app is None:
        return 0
    count = 0
    handled = set()
    seen = set()
    queue = list(app.topLevelWidgets())
    while queue:
        widget = queue.pop(0)
        if id(widget) in seen:
            continue
        seen.add(id(widget))
        if not _widget_alive(widget):
            continue
        # 先把子控件入队（保证仍是祖先优先的顺序），再决定本控件要不要重刷
        try:
            queue.extend(widget.findChildren(QWidget))
        except RuntimeError:
            continue
        if not _has_handled_ancestor(widget, handled):
            for name in _RETRANSLATE_METHODS:
                fn = getattr(widget, name, None)
                if not callable(fn):
                    continue
                try:
                    fn()
                except TypeError:
                    # 形如 set_tool_language(lang_code, persist=False) 需要参数 → 跳过
                    continue
                except Exception:  # noqa: BLE001 - 单个面板重刷失败不该中断切换
                    continue
                handled.add(id(widget))
                count += 1
                break
        # 通知 Qt 自身重新翻译标准控件（QDialogButtonBox 等）
        try:
            app.sendEvent(widget, QEvent(QEvent.LanguageChange))
        except Exception:  # noqa: BLE001
            pass
    return count


# ---------------------------------------------------------------------------
# matplotlib 导航工具栏
# ---------------------------------------------------------------------------
#: matplotlib 自带按钮的英文 → 中文。工具栏是第三方控件，文字不走 self.tr()，
#: 只能拿到 QAction 后按原文替换。
_NAV_ZH = {
    "Home": "主页",
    "Back": "后退",
    "Forward": "前进",
    "Pan": "平移",
    "Zoom": "缩放",
    "Subplots": "子图",
    "Customize": "定制",
    "Save": "保存",
    "Reset original view": "重置原始视图",
    "Back to previous view": "返回上一个视图",
    "Forward to next view": "前进到下一个视图",
    "Left button pans, Right button zooms\nx/y fixes axis, CTRL fixes aspect":
        "左键平移，右键缩放\nx/y 锁定坐标轴，CTRL 锁定纵横比",
    "Zoom to rectangle\nx/y fixes axis": "缩放到矩形区域\nx/y 锁定坐标轴",
    "Configure subplots": "配置子图",
    "Edit axis, curve and image parameters": "编辑坐标轴、曲线和图像参数",
    "Save the figure": "保存图像",
}

#: 反向表：中文 → 英文。切回英文界面时用它把按钮文字还原成 matplotlib 原文。
#: 没有反向表的话，中→英切换后工具栏会永远停在中文（因为英文表是空表，
#: 原实现遇到空表就直接 return 了）。
_NAV_EN = {zh: en for en, zh in _NAV_ZH.items()}


def localize_nav_toolbar(toolbar, lang: Optional[str] = None) -> None:
    """把 matplotlib 导航工具栏的按钮名/提示翻成当前界面语言。

    中英文**双向**：中文界面下英文 → 中文，切回英文界面时中文 → 英文。
    这样语言来回切换不会把工具栏"卡"在中文。表里没有的条目原样保留。
    """
    if toolbar is None:
        return
    code = normalize_language(lang) or _current
    table = _NAV_ZH if code == "zh" else _NAV_EN
    try:
        actions = toolbar.actions()
    except Exception:  # noqa: BLE001
        return
    for action in actions:
        text = action.text()
        if text in table:
            action.setText(table[text])
        tip = action.toolTip()
        if tip in table:
            action.setToolTip(table[tip])


__all__ = [
    "DssrrTranslator",
    "LANGUAGE_LABELS",
    "SETTINGS_KEY",
    "TABLES",
    "apply_language",
    "current_language",
    "install",
    "language_label",
    "localize_nav_toolbar",
    "target_language",
    "toggle_language",
    "translate",
]
