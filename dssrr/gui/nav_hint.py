# -*- coding: utf-8 -*-
"""段选择画布旁"平移/缩放模式下无法框选"的提示组件。

为什么需要
----------
matplotlib 的 **Pan / Zoom** 模式会抢占 ``canvas.widgetlock``，``SpanSelector``
的按下事件被直接丢弃。用户看到的现象是"在波形上怎么拖都选不中异常段，而且没有
任何反馈"——导航工具栏的按钮虽然会显示为按下态，但不显眼，很容易被当成 bug。

本模块把这段逻辑封装成一个可复用的小组件，供 Manual Repair 的两个面板共用：

- :class:`dssrr.gui.depulse_dialog.DePulseDialog`（集成主窗口的 Manual Repair 页）
- :class:`dssrr.gui.manual_repair.ManualRepairDialog`（查看器的独立手动重处理对话框）

它是怎么知道模式变了的
----------------------
1. **主路径**：监听工具栏上**可勾选**的 ``QAction`` 的 ``toggled`` 信号。
   matplotlib 的 ``NavigationToolbar2QT._update_buttons_checked()`` 会把当前
   模式回写到按钮的勾选状态，键盘快捷键（``p`` / ``o``）走的也是这条路，所以
   这个信号足够可靠。
2. **兜底**：一个低频 ``QTimer`` 轮询 ``toolbar.mode``。个别 matplotlib 版本若
   不回写勾选状态，250 ms 内也会被纠正。

模式判断为什么用"字符串非空"
----------------------------
``NavigationToolbar2.mode`` 在新版本里是 ``_Mode`` 枚举：``_Mode.NONE`` 的字符串
值就是空串，``_Mode.PAN`` 是 ``"pan/zoom"``，``_Mode.ZOOM`` 是 ``"zoom rect"``；
老版本里它就是个普通字符串 ``""``。两种情况都能用"``str(mode)`` 非空"统一判断。

文案与翻译
----------
英文原文就是模块常量 :data:`IDLE_TEXT` / :data:`NAV_TEXT`，中文译文登记在
:mod:`dssrr.gui.translations_zh`（key 必须与常量**逐字符一致**）。
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from PyQt5.QtCore import QObject, QTimer

#: 空闲时的操作说明（英文原文，也是中文词条表的 key）。
IDLE_TEXT = "Left drag: replace. Right drag: add. Double-click: clear."

#: 平移/缩放模式激活时的警告文案（英文原文，也是中文词条表的 key）。
NAV_TEXT = ("Zoom/pan is active — segment selection is disabled. "
            "Click Zoom or Pan again to turn it off, then drag to select.")

#: 警告配色：浅色主题用深琥珀，深色主题用亮琥珀（都要在底色上有足够对比度）。
_WARN_COLOR = {False: "#B45309", True: "#FBBF24"}


def nav_mode_active(toolbar) -> bool:
    """工具栏当前是否处于平移/缩放模式（见模块文档的"字符串非空"说明）。"""
    if toolbar is None:
        return False
    mode = getattr(toolbar, "mode", "") or ""
    return str(mode).strip() != ""


def localize_toolbar(toolbar, lang: Optional[str] = None) -> None:
    """本地化 matplotlib 导航工具栏（延迟导入，避免循环依赖）。"""
    try:
        from .i18n import localize_nav_toolbar
    except Exception:  # pragma: no cover - 只在 Qt 不可用时发生
        return
    localize_nav_toolbar(toolbar, lang)


class SegmentNavHint(QObject):
    """把一个 ``QLabel`` 变成"随导航模式切换的提示语"。

    Parameters
    ----------
    label : QLabel
        要切换文字与配色的提示标签。
    toolbar : NavigationToolbar2QT or None
        matplotlib 导航工具栏。
    translator : callable or None
        形如 ``self.tr`` 的翻译函数；``None`` = 原样返回英文。
    muted_color : str or callable
        空闲态的"弱化"颜色；可以是常量，也可以是返回颜色串的 callable
        （面板换主题后颜色会变，所以推荐传 callable）。
    is_dark : callable or None
        返回当前是否深色主题，用于挑警告配色。
    interval : int
        兜底轮询周期（毫秒）。
    """

    def __init__(self, label, toolbar, translator: Optional[Callable] = None,
                 parent: Optional[QObject] = None,
                 muted_color: Union[str, Callable[[], str]] = "#94A3B8",
                 is_dark: Optional[Callable[[], bool]] = None,
                 interval: int = 250):
        super().__init__(parent)
        self._label = label
        self._toolbar = toolbar
        self._tr = translator or (lambda text: text)
        self._muted_color = muted_color
        self._is_dark = is_dark
        self._timer = QTimer(self)
        self._timer.setInterval(interval)
        self._timer.timeout.connect(self.refresh)

    # ---------------- 装配 ----------------
    def install(self, localize: bool = True) -> "SegmentNavHint":
        """本地化工具栏、挂上模式监听、启动兜底轮询，并立刻刷新一次。"""
        if localize:
            localize_toolbar(self._toolbar)
        try:
            for action in self._toolbar.actions():
                if not action.isCheckable():
                    continue
                if getattr(action, "_dssrr_nav_hooked", False):
                    continue
                action.toggled.connect(self.refresh)
                action._dssrr_nav_hooked = True
        except Exception:  # noqa: BLE001 - 工具栏结构异常不该拖垮面板
            pass
        self._timer.start()
        self.refresh()
        return self

    def stop(self) -> None:
        """停止兜底轮询（面板关闭 / 销毁时调用）。"""
        try:
            self._timer.stop()
        except Exception:  # noqa: BLE001
            pass

    def is_running(self) -> bool:
        return bool(self._timer.isActive())

    # ---------------- 状态 ----------------
    def nav_active(self) -> bool:
        return nav_mode_active(self._toolbar)

    def _muted(self) -> str:
        color = self._muted_color
        if callable(color):
            try:
                color = color()
            except Exception:  # noqa: BLE001
                color = "#94A3B8"
        return str(color)

    def _warn(self) -> str:
        dark = False
        if callable(self._is_dark):
            try:
                dark = bool(self._is_dark())
            except Exception:  # noqa: BLE001
                dark = False
        return _WARN_COLOR[dark]

    # ---------------- 刷新 ----------------
    def refresh(self, *_args) -> None:
        """按当前导航模式刷新提示语与配色。"""
        label = self._label
        if label is None:
            return
        try:
            if self.nav_active():
                label.setText(self._tr(NAV_TEXT))
                label.setStyleSheet(
                    f"color: {self._warn()}; font-size: 10px; font-weight: 600;")
            else:
                label.setText(self._tr(IDLE_TEXT))
                label.setStyleSheet(
                    f"color: {self._muted()}; font-size: 10px;")
        except RuntimeError:
            # 面板重建期间底层 C++ 对象可能已被销毁，静默忽略
            pass


__all__ = [
    "IDLE_TEXT",
    "NAV_TEXT",
    "SegmentNavHint",
    "localize_toolbar",
    "nav_mode_active",
]
