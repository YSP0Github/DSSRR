# -*- coding: utf-8 -*-
"""一键按依赖顺序重出全部论文图件（正文 Fig 1-10 + 补充 Fig S1）。

用 Agg 后端子进程逐个运行脚本，避免本机 qtagg 后端在无显示 / 后台环境被静默
SIGTERM 杀掉；合成图（fig02 / fig07）在其面板脚本之后运行，保证依赖顺序。

用法：
    python build_all_figures.py            # 重出全部图
    python build_all_figures.py --quick    # 跳过最慢的 fig08（地球数据迁移重实验）

注意：
    - figS1 需要 FDSN 网络访问（obspy Client 下载 Apollo 台站数据）；
    - fig08 会重跑地球数据实验并输出到 experiments/results_earth，耗时较长；
    - figS2（SeisY GUI）为手工截图，无对应脚本。
"""
import os
import sys
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# 出图所用 Python（须装 numpy/scipy/matplotlib/obspy）；默认 miniconda python_3.11，
# 可用环境变量 SRL_PYTHON 覆盖。不能用 sys.executable：本脚本可能被无包的 python 启动。
PYTHON = os.environ.get("SRL_PYTHON", r"G:\miniconda3\envs\python_3.11\python.exe")

# (脚本, 说明) —— 顺序即出图依赖顺序：面板 -> 合成
STEPS = [
    ("fig01_anomaly_types.py",   "Fig 1  anomaly types"),
    ("fig02a_workflow_panel.py", "Fig 2  panel a (workflow)"),
    ("fig02bc_repair_panel.py",  "Fig 2  panel b/c (repair before/after)"),
    ("fig02_method_overview.py", "Fig 2  merge panels -> overview"),
    ("fig03_method_comparison.py", "Fig 3  method comparison bars"),
    ("fig04_psd_comparison.py",  "Fig 4  PSD comparison"),
    ("fig05_length_effect.py",   "Fig 5  gap-length effect"),
    ("fig06_stalta_downstream.py", "Fig 6  STA/LTA downstream"),
    ("fig07ab_ablation_panel.py", "Fig 7  panel a/b (ablation)"),
    ("fig07cd_fusion_panel.py",  "Fig 7  panel c/d (reference fusion)"),
    ("fig07_ablation_dual_sided.py", "Fig 7  merge panels -> dual-sided"),
    ("fig08_earth_transfer.py",  "Fig 8  Earth transfer (slow, reruns experiment)"),
    ("fig09_archival_repair.py", "Fig 9  archival repair"),
    ("fig10_failure_mode.py",    "Fig 10 failure mode"),
    ("figS1_background.py",      "Fig S1 background (needs FDSN network)"),
]
SLOW = {"fig08_earth_transfer.py"}


def main():
    quick = "--quick" in sys.argv
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    t_all = time.time()
    for script, desc in STEPS:
        if quick and script in SLOW:
            print("[skip] %s (--quick)" % script)
            continue
        t0 = time.time()
        print("[run ] %-30s %s" % (script, desc))
        r = subprocess.run([PYTHON, script], cwd=HERE, env=env)
        dt = time.time() - t0
        if r.returncode != 0:
            print("[FAIL] %s  rc=%d  after %.0f s" % (script, r.returncode, dt))
            sys.exit(1)
        print("[ ok ] %-30s %.0f s" % (script, dt))
    print("all figures done in %.0f s" % (time.time() - t_all))


if __name__ == "__main__":
    main()
