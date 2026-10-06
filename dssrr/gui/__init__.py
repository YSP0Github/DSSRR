# -*- coding: utf-8 -*-
"""DSSRR 桌面图形界面子包（PyQt5）。

本子包把 DSSRR 的命令行能力包装成一套桌面工具，全部入口最终汇聚到
:func:`dssrr.gui.main.main`：

- :mod:`dssrr.gui.main`           集成式主窗口（侧边栏导航 + 双页面），GUI 唯一入口
- :mod:`dssrr.gui.viewer`         双库对比查看器（原始库 vs 修复库）
- :mod:`dssrr.gui.batch_repair`   批量修复页面（在查看器内以按钮弹出）
- :mod:`dssrr.gui.manual_repair`  手动重处理对话框（鼠标框选 + 多方法修复）
- :mod:`dssrr.gui.depulse_dialog` 手动去脉冲面板（DSSRR Manual Repair 面板）
- :mod:`dssrr.gui.export_report_dialog` 修复报告导出设置对话框（带实时预览）
- :mod:`dssrr.gui.report_plot`    报告绘图实现（见 :mod:`dssrr.repair_lib.report_plot`）
- :mod:`dssrr.gui.plot_text_editor` 图内文字批编辑器（标题/轴标签字体与颜色）
- :mod:`dssrr.gui.icons`          程序图标与品牌标识（SVG 内联，无需外部资源）

依赖说明：GUI 相关模块需要可选的 ``gui`` 依赖组（PyQt5、matplotlib、
obspy、pandas）。安装方式::

    pip install "dssrr[gui]"

启动方式::

    python -m dssrr.gui        # 等价于 dssrr.gui.main.main()
    dssrr-gui                  # console script，见 pyproject.toml [project.scripts]

注意：本子包在 import 时不创建任何 QApplication，也不会触发 Qt 平台插件加载；
真正的窗口创建发生在 :func:`main` 被调用之后。
"""

from .main import main, launch_gui

__all__ = ["main", "launch_gui"]
