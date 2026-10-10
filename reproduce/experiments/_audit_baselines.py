# -*- coding: utf-8 -*-
"""对照组实现一致性审计（只读，不改任何实验产物）。

目的：核对论文各图/表所用的同名方法是否真的是同一套实现。
    - 表 1（E-RealInj v6）走 E2r_v5_batch.METHODS -> seisy.core.anomaly_repair
    - 图 5 / 图 3 / 图 6 走 experiments.baselines.make_methods
两套模块里 SSA / UNet 是**独立实现**，本脚本量化差异。

用法：G:/miniconda3/envs/seisy/python.exe _audit_baselines.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from obspy import read

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _metrics_lib import all_metrics                      # noqa: E402
from docs.dssrr_paper.experiments import baselines as B   # noqa: E402
from seisy.core.anomaly_repair.comparison_methods import (  # noqa: E402
    SSARepair as SeisySSA,
    UNetRepair as SeisyUNet,
)

V6 = r"G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6"
CLEAN_DIR = r"G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h"
MODEL = r"G:\SeisY\docs\dssrr_paper\experiments\models\unet_synth.pt"
FS = 6.625
ACF_MAX_LAG = 200

CASES = [
    ("S12_19760113_070152_stationary", "dropout", "medium"),
    ("S12_19760113_070152_stationary", "dropout", "large"),
    ("S12_19760113_070152_stationary", "spike", "medium"),
    ("S12_19761114_230000_stationary", "dropout", "large"),
]


def clean_for(case_dir_name):
    """按案例名找对应 12h 干净记录。"""
    parts = case_dir_name.split("_")
    net, sta = "XA", parts[0]
    day = parts[1]
    for f in os.listdir(CLEAN_DIR):
        if f.startswith(f"{net}.{sta}.") and day in f:
            return os.path.join(CLEAN_DIR, f)
    return None


def main():
    seisy_ssa = SeisySSA(max_iter=30)
    base_ssa = B.SSARepair(sr=FS)
    seisy_unet = SeisyUNet(patch_length=1536, n_epochs=10, model_path=MODEL)
    base_unet = B.UNetRepair(sr=FS)

    print(f"{'case':<46}{'method':<8}{'stored':>9}{'seisy':>9}{'baseline':>10}"
          f"{'|seisy-stored|':>15}{'|base-stored|':>14}")
    print("-" * 111)

    for case, anom, level in CASES:
        cdir = os.path.join(V6, case, anom, level)
        if not os.path.isdir(cdir):
            print(f"[skip] {cdir} 不存在")
            continue
        cf = clean_for(case)
        if cf is None:
            print(f"[skip] 找不到 {case} 的干净记录")
            continue

        truth = read(cf)[0].data.astype(float)
        cor = read(os.path.join(cdir, "corrupted.mseed"))[0].data.astype(float)
        n = len(cor)
        truth = truth[:n]
        with open(os.path.join(cdir, "intervals.json"), encoding="utf-8") as fh:
            intervals = json.load(fh)["intervals"]
        idx = np.concatenate([np.arange(s, e + 1) for s, e in intervals])

        for name, stored_file, mk_seisy, mk_base in [
            ("SSA", "repaired_SSA.mseed", lambda: seisy_ssa.repair(cor, intervals[0][0], intervals[0][1]),
             lambda: base_ssa.repair(cor, intervals[0][0], intervals[0][1])),
            ("UNet", "repaired_UNet.mseed", lambda: seisy_unet.repair(cor, intervals[0][0], intervals[0][1]),
             lambda: base_unet.repair(cor, intervals[0][0], intervals[0][1])),
        ]:
            sf = os.path.join(cdir, stored_file)
            if not os.path.isfile(sf):
                print(f"[skip] {sf} 不存在")
                continue
            stored = read(sf)[0].data.astype(float)[:n]

            try:
                r_seisy = mk_seisy()
            except Exception as exc:      # noqa: BLE001
                print(f"[err ] seisy {name}: {type(exc).__name__}: {exc}")
                continue
            try:
                r_base = mk_base()
            except Exception as exc:      # noqa: BLE001
                print(f"[err ] base  {name}: {type(exc).__name__}: {exc}")
                continue

            def w(x):
                return float(np.sqrt(np.mean((x[idx] - truth[idx]) ** 2)))

            print(f"{case+'/'+anom+'/'+level:<46}{name:<8}"
                  f"{w(stored):>9.4f}{w(r_seisy):>9.4f}{w(r_base):>10.4f}"
                  f"{abs(w(r_seisy)-w(stored)):>15.4f}{abs(w(r_base)-w(stored)):>14.4f}")

            m_s = all_metrics(truth[idx], stored[idx], FS, ACF_MAX_LAG)
            m_a = all_metrics(truth[idx], r_seisy[idx], FS, ACF_MAX_LAG)
            m_b = all_metrics(truth[idx], r_base[idx], FS, ACF_MAX_LAG)
            print(f"{'':<46}{'  W1':<8}{m_s['wasserstein']:>9.4f}"
                  f"{m_a['wasserstein']:>9.4f}{m_b['wasserstein']:>10.4f}")
            print(f"{'':<46}{'  ACF':<8}{m_s['acf_l2']:>9.4f}"
                  f"{m_a['acf_l2']:>9.4f}{m_b['acf_l2']:>10.4f}")
        print()


if __name__ == "__main__":
    main()
