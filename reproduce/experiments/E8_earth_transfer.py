# -*- coding: utf-8 -*-
"""E8 兼容入口 —— **这个脚本没有被删，是被搬走了。**

E8 的完整实验管线（下载/预处理 IU.ANMO.00.BHZ、自动选窗、注入平顶 dropout、
六方法修复、写 ``results_earth/earth_instances.csv`` + ``earth_summary.csv``）
已经与出图合并，迁到：

    G:\\SeisY\\docs\\dssrr_paper\\submission_srl\\image_code\\fig08_earth_transfer.py

迁移证据（两处独立）：
1. ``fig08_earth_transfer.py`` 顶部注释原文——
   "实验公共库（指标 / 结果读写）仍位于 experiments 目录，**本脚本移入 image_code 后需补路径**"；
2. ``image_code/README.md`` 把它登记为
   "Fig 8 Earth transfer | ``fig08_earth_transfer.py`` | **实验+出图**（慢）"。

时间线：原 ``E8_earth_transfer.py`` 源码最后修改 2026-10-04 00:48:50（16 134 字节），
2026-10-04 00:53 编译出 ``__pycache__/E8_earth_transfer.cpython-311.pyc``；
10-05 建立 ``submission_srl/image_code/`` 时脚本被移入并改名为 fig08。
该文件**从未被 git 跟踪**（``git log --all -- *E8_earth_transfer*`` 为空），
所以 git 里恢复不出来 —— 但也不需要恢复，实现在 fig08 里。

本文件**只做转发，不复制第二份实现**，以免重蹈"同名不同实现"的覆辙。
历史脚本（``E9_dl_synthetic.py``、``_smoke_e9.py``、``_test_long_gap.py``、
``_test_noiseburst.py``）的 ``from E8_earth_transfer import ...`` 继续可用。

直接运行本文件 = 跑 E8 全流程（较慢，含 torch）。
"""
from __future__ import annotations

import os
import sys

# 必须在 import fig08 之前设置：fig08 依赖 srl_style -> matplotlib，
# 无显示环境下用默认 qtagg 后端会静默 SIGTERM（exit 127、无报错）。
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_CODE = os.path.normpath(
    os.path.join(HERE, "..", "submission_srl", "image_code"))
if IMAGE_CODE not in sys.path:
    sys.path.insert(0, IMAGE_CODE)

from fig08_earth_transfer import (  # noqa: E402,F401
    CASE,
    FFTInterpolation,
    GAP_LENGTHS,
    GAP_REPS,
    METHOD_ORDER,
    OUT_DIR,
    RAW,
    SEED,
    STATION,
    TARGET_FS,
    find_adaptive_boundary,
    load_preprocessed,
    main,
    make_figure,
    repair_dssrr,
    rms_profile,
    select_trials,
)

__all__ = [
    "FFTInterpolation", "repair_dssrr", "load_preprocessed",
    "find_adaptive_boundary", "rms_profile", "select_trials",
    "make_figure", "main",
    "RAW", "OUT_DIR", "TARGET_FS", "SEED", "CASE", "STATION",
    "GAP_LENGTHS", "GAP_REPS", "METHOD_ORDER",
]

if __name__ == "__main__":
    main()
