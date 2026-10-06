# -*- coding: utf-8 -*-
"""``python -m dssrr.gui`` 的运行入口。

这个文件让 DSSRR 图形界面可以用「模块方式」启动，无需先安装 console script::

    python -m dssrr.gui

它只做一件事：把 :func:`dssrr.gui.main.main` 转发为模块级执行。真正的
应用装配（QApplication 创建、组织名/产品名、窗口图标、主窗口）全部在
:mod:`dssrr.gui.main` 里完成，因此本文件不需要也不应该包含任何 GUI 逻辑。

等价入口：
- ``python -m dssrr.gui``            （本文件）
- ``dssrr-gui``                      （console script，安装后可用）
- ``from dssrr.gui import main; main()``
"""

from .main import main

if __name__ == "__main__":
    main()
