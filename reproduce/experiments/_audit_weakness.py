# -*- coding: utf-8 -*-
"""对照组"是否被削弱"敏感性审计（只读）。

回答两个最容易被审稿人质疑的问题：
  Q1  论文写 "cubic spline interpolation"，代码实际用 PCHIP —— 换成真正的
      自然三次样条，结论会不会变？（即：现在的 PCHIP 是不是把基线做弱了？）
  Q2  论文写 "FFT spectral extrapolation"，代码只保留最低 1/10 的 rfft 频带 ——
      这个 1/10 是不是拍脑袋的、故意削弱基线？改成更宽的频带会怎样？

用法：G:/miniconda3/envs/seisy/python.exe _audit_weakness.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from obspy import read
from scipy import interpolate

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _metrics_lib import all_metrics                      # noqa: E402
from docs.dssrr_paper.experiments import baselines as B   # noqa: E402

V6 = r"G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6"
CLEAN_DIR = r"G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h"
FS = 6.625
ACF_MAX_LAG = 200
CASES = [
    ("S12_19760113_070152_stationary", "dropout", "medium"),
    ("S12_19760113_070152_stationary", "dropout", "large"),
    ("S12_19761114_230000_stationary", "dropout", "large"),
]


def clean_for(case):
    p = case.split("_")
    for f in os.listdir(CLEAN_DIR):
        if f.startswith(f"XA.{p[0]}.") and p[1] in f:
            return os.path.join(CLEAN_DIR, f)
    return None


def repair_linear(data, s, e):
    r = data.copy()
    r[s:e + 1] = np.linspace(data[s - 1], data[e + 1], e - s + 1)
    return r


def repair_spline(data, s, e, kind, n_pts=50):
    r = data.copy()
    npts = min(n_pts, s, len(data) - e - 1)
    xi = np.concatenate([np.arange(s - npts, s), np.arange(e + 1, e + 1 + npts)]).astype(float)
    yi = np.concatenate([data[s - npts:s], data[e + 1:e + 1 + npts]])
    if kind == "pchip":
        fn = interpolate.PchipInterpolator(xi, yi)
    elif kind == "cubic_natural":
        fn = interpolate.CubicSpline(xi, yi, bc_type="natural")
    elif kind == "cubic_notaknot":
        fn = interpolate.CubicSpline(xi, yi, bc_type="not-a-knot")
    r[s:e + 1] = fn(np.arange(s, e + 1, dtype=float))
    return r


def repair_fft(data, s, e, keep_frac):
    r = data.copy()
    n = len(data)
    known = np.concatenate([data[:s], data[e + 1:]])
    F = np.fft.rfft(known)
    n_low = max(1, int(len(F) * keep_frac))
    Fs = np.zeros_like(F)
    Fs[:n_low] = F[:n_low]
    full = np.fft.irfft(Fs, n=n)
    lm = np.mean(data[max(0, s - 100):s])
    rm = np.mean(data[e + 1:min(n, e + 101)])
    r[s:e + 1] = full[s:e + 1] + (lm + rm) / 2 - np.mean(full[s:e + 1])
    return r


def main():
    for case, anom, level in CASES:
        cdir = os.path.join(V6, case, anom, level)
        cf = clean_for(case)
        if cf is None or not os.path.isdir(cdir):
            print(f"[skip] {case}/{anom}/{level}")
            continue
        truth = read(cf)[0].data.astype(float)
        cor = read(os.path.join(cdir, "corrupted.mseed"))[0].data.astype(float)
        n = len(cor)
        truth = truth[:n]
        s, e = json.load(open(os.path.join(cdir, "intervals.json"),
                              encoding="utf-8"))["intervals"][0]
        seg = np.arange(s, e + 1)

        def m(rep, tag):
            d = all_metrics(truth[seg], rep[seg], FS, ACF_MAX_LAG)
            print(f"    {tag:<34} W1={d['wasserstein']:8.4f}  ACF={d['acf_l2']:7.4f}"
                  f"  RMSE={d['rmse_anom']:8.4f}")

        print(f"\n### {case}/{anom}/{level}   gap={e-s+1} samples "
              f"({(e-s+1)/FS:.0f} s)")

        print("  [Q1] 插值类")
        m(repair_linear(cor, s, e), "Linear (论文口径)")
        m(B.CubicSplineInterpolation().repair(cor, s, e), "PCHIP  <-- 论文实际所用")
        m(repair_spline(cor, s, e, "cubic_natural"), "CubicSpline natural")
        m(repair_spline(cor, s, e, "cubic_notaknot"), "CubicSpline not-a-knot")
        m(repair_spline(cor, s, e, "pchip", 200), "PCHIP, 每侧 200 点")

        print("  [Q2] FFT 频带保留比例")
        for frac in (0.1, 0.25, 0.5, 0.9, 1.0):
            m(repair_fft(cor, s, e, frac), f"FFT keep {frac:>4.2f} of band"
                                           + ("   <-- 论文所用" if frac == 0.1 else ""))


if __name__ == "__main__":
    main()
