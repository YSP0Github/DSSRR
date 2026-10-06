# -*- coding: utf-8 -*-
"""DSSRR 应用图标资源包。

- ``dssrr.svg`` / ``dssrr.png`` / ``dssrr.ico`` 及多尺寸 PNG 全部由
  :mod:`dssrr.gui.icons.make_icon` 统一生成（几何同源，矢量与位图不会漂移）。
- GUI 侧通过 :func:`app_icon` 取用，避免各处硬编码文件路径。

重新生成图标::

    python -m dssrr.gui.icons.make_icon
"""
from __future__ import annotations

import os

ICON_DIR = os.path.dirname(os.path.abspath(__file__))


def icon_path(name: str = "dssrr.png") -> str:
    """返回图标资源的绝对路径（不校验存在性）。"""
    return os.path.join(ICON_DIR, name)


def app_icon():
    """构造应用 QIcon（多尺寸 PNG + ICO 兜底）。

    文件缺失时返回空 QIcon，由调用方自行降级，不抛异常。
    """
    from PyQt5.QtGui import QIcon

    icon = QIcon()
    for name in ("dssrr.png", "dssrr.ico"):
        path = icon_path(name)
        if os.path.exists(path):
            icon.addFile(path)
    return icon


__all__ = ["ICON_DIR", "icon_path", "app_icon"]
