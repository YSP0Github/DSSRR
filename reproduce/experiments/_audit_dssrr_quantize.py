# -*- coding: utf-8 -*-
"""验证：给 baselines.DSSRRRepair 传 quantize=True 能否精确还原 10-04 归档。

背景
----
2026-10-05 把 ``ReferenceSpectrumReplacer.replace()`` 里**无条件取整**改成了
``quantize`` 开关（默认 False）。``E2r_v5/v6`` 自己在写回时 ``np.round``，所以
Table 1 不受影响；但 ``baselines.DSSRRRepair`` 依赖了 replacer 的隐式取整，
于是 10-04 归档（取整）与当前代码（不取整）不一致。

本脚本在长度扫描的 80 个案例上对比三种输出与归档 ``repaired_DSSRR.mseed``：
    q_true : 显式 quantize=True
    q_false: 当前默认（不取整）
    raw    : 直接调 replacer，不传 quantize

用法：G:/miniconda3/envs/seisy/python.exe -u _audit_dssrr_quantize.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from obspy import read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, HERE)

ROOT = os.path.join(HERE, "results_length_scan")
BAK = os.path.join(HERE, "results_length_scan_pre_unify_bak")
FS = 6.625


def main():
    from seisy.core.anomaly_repair.reference_spectrum import ReferenceSpectrumReplacer
    from docs.dssrr_paper.experiments.baselines import DSSRRRepair

    events = sorted(d for d in os.listdir(BAK) if os.path.isdir(os.path.join(BAK, d)))
    res = {"q_true": [], "q_false": []}

    for ev in events:
        drop = os.path.join(BAK, ev, "dropout")
        if not os.path.isdir(drop):
            continue
        for lv in sorted(os.listdir(drop)):
            cdir = os.path.join(BAK, ev, "dropout", lv)
            cf = os.path.join(cdir, "corrupted.mseed")
            af = os.path.join(cdir, "repaired_DSSRR.mseed")
            jf = os.path.join(cdir, "intervals.json")
            if not all(os.path.isfile(p) for p in (cf, af, jf)):
                continue
            corrupted = read(cf)[0].data.astype(float)
            arch = read(af)[0].data.astype(float)
            with open(jf, encoding="utf-8") as fh:
                s, e = json.load(fh)["intervals"][0]

            # q_true / q_false 通过显式 replacer 调用（复刻 baselines 的参数映射）
            anom_sec = (e - s + 1) / FS
            ref_len = float(np.clip(anom_sec / 2.0, 120.0, 600.0))
            out = {}
            for tag, q in (("q_true", True), ("q_false", False)):
                rep = ReferenceSpectrumReplacer(FS)
                seg, _ = rep.replace(
                    corrupted.copy(), s, e,
                    reference_before_sec=ref_len, reference_after_sec=ref_len,
                    reference_gap_sec=1.0, random_seed=42, quantize=q)
                r = corrupted.copy()
                r[s:e + 1] = seg[s:e + 1]
                out[tag] = r
                d = float(np.max(np.abs(r[s:e + 1] - arch[s:e + 1])))
                res[tag].append(d)

            print(f"[{ev} {lv}] dgap q_true={res['q_true'][-1]:9.4f} "
                  f"q_false={res['q_false'][-1]:9.4f}", flush=True)

    print("\n=== 汇总（80 案例，与归档 repaired_DSSRR.mseed 的缺口段最大差）===")
    for tag in ("q_true", "q_false"):
        v = np.array(res[tag])
        print(f"{tag:>8}: max={v.max():.4f}  median={np.median(v):.4f}  "
              f"mean={v.mean():.4f}  |  完全一致(<1e-6)的案例数 = {(v < 1e-6).sum()}/{len(v)}")


if __name__ == "__main__":
    main()
