# -*- coding: utf-8 -*-
"""DSSRR 的 Apollo 专用检测 / 修复流水线（应用层）。

这一层把通用的 DSSRR 算法（:mod:`dssrr.core`）接到真实的 Apollo 月震归档上：
先判定哪些时间段是仪器伪迹、哪些是必须保留的真实月震，再按段选择最合适的
修复手段，最后写回修复库并做结构验收。

模块一览
--------
- :mod:`~dssrr.repair_lib.e7_detector`
  E7 检测器：MISSING / FREEZE / SAT / SPIKE / STEP / BURST 六类伪迹 +
  月震事件保护 + 长缺口标记。依赖随包分发的月震目录。
- :mod:`~dssrr.repair_lib.e7_repair_build`
  单文件多级修复与验收：线性插值 → Z 分数 → 局部 MAD 尖峰 →
  DSSRR 谱重建 → 月震保护；含 ``repair_one_file`` / ``verify_file`` /
  ``write_catalog_csv``。
- :mod:`~dssrr.repair_lib.repair_engine`
  ObsPy ``Stream``/``Trace`` 胶水层（:class:`StreamRepair`），供 GUI 手动修复
  与批量流程共用。
- :mod:`~dssrr.repair_lib.depulse_advanced`
  可选的高级检测（Isolation Forest / LOF）与神经网络修补，缺依赖时优雅降级。
- :mod:`~dssrr.repair_lib.depulse_nn`
  一维卷积自编码器（需要 torch）。
- :mod:`~dssrr.repair_lib.report_plot`
  修复报告对比图。
- :mod:`~dssrr.repair_lib.repair_settings`
  用户设置文件的读写（默认 ``~/.dssrr_settings.json``）。
- :mod:`~dssrr.repair_lib.anomaly_repair`
  三个可直接调用的 Replacer（双侧参考 / 分层参考 / 月震保护型尖峰）。

设计说明
--------
本 ``__init__`` **故意保持为空**：``repair_lib`` 下的模块依赖较重（obspy /
pandas / PyQt5 / 可选 torch），而 ``dssrr`` 顶层只依赖 numpy + scipy。让导入
``dssrr.repair_lib`` 不产生任何副作用，调用方才能按需 import 具体模块。
"""

__all__ = []
