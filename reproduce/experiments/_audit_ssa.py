# -*- coding: utf-8 -*-
"""SSA 超参敏感性审计（只读）：验证论文"SSA 无需人工调参"的说法是否站得住。

论文用 seisy SSARepair(max_iter=30)，其中 L = clip(gap/4, 32, 200)、K = 10。
这里扫描 L 与 K，看有没有明显更好的设置被"漏掉"。

用法：G:/miniconda3/envs/seisy/python.exe _audit_ssa.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from obspy import read

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _metrics_lib import all_metrics                        # noqa: E402
from seisy.core.anomaly_repair.comparison_methods import SSARepair  # noqa: E402

V6 = r"G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6"
CLEAN_DIR = r"G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h"
FS = 6.625

CASES = [
    ("S12_19760113_070152_stationary", "dropout", "medium"),
    ("S12_19760113_070152_stationary", "dropout", "large"),
]


def clean_for(case):
    p = case.split("_")
    for f in os.listdir(CLEAN_DIR):
        if f.startswith(f"XA.{p[0]}.") and p[1] in f:
            return os.path.join(CLEAN_DIR, f)
    return None


def main():
    for case, anom, level in CASES:
        cdir = os.path.join(V6, case, anom, level)
        cf = clean_for(case)
        if cf is None:
            continue
        truth = read(cf)[0].data.astype(float)
        cor = read(os.path.join(cdir, "corrupted.mseed"))[0].data.astype(float)
        n = len(cor)
        truth = truth[:n]
        s, e = json.load(open(os.path.join(cdir, "intervals.json"),
                              encoding="utf-8"))["intervals"][0]
        seg = np.arange(s, e + 1)
        gap = e - s + 1
        L_auto = int(max(10, min(max(gap // 4, 32), 200)))
        print(f"\n### {case}/{anom}/{level}  gap={gap} samples "
              f"({gap/FS:.0f} s)   论文 L_auto={L_auto}, K=10")

        def run(L, K, tag):
            r = SSARepair(window_length=L, n_components=K, max_iter=30)
            rep = r.repair(cor, s, e)
            d = all_metrics(truth[seg], rep[seg], FS, 200)
            print(f"    L={str(L):>4} K={str(K):>3}  W1={d['wasserstein']:8.4f}"
                  f"  ACF={d['acf_l2']:7.4f}  RMSE={d['rmse_anom']:8.4f}   {tag}")

        print("  -- 扫描 L（K=10）")
        for L in (32, 64, 128, 200, 300):
            run(L, 10, "<-- 论文自动值" if L == L_auto else "")
        print("  -- 扫描 K（L=L_auto）")
        for K in (2, 5, 10, 20, 50):
            run(L_auto, K, "<-- 论文所用" if K == 10 else "")


if __name__ == "__main__":
    main()
