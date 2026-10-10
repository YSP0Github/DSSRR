# -*- coding: utf-8 -*-
"""SSA 实现统一对 Figure 5 的影响 —— 对归档结果做增量比对（只读）。

要点
----
``results_length_scan/`` 归档了每个 (事件, 长度) 案例的
``corrupted.mseed`` + ``intervals.json`` + ``repaired_SSA.mseed``，
而 ``stats_summary.csv`` 里就有**旧 baselines 实现**跑出的 SSA 指标行。

真值（truth）不在归档里，但可按 ``run_length_scan.py`` 的口径重建：
从 12h 干净记录取 ``[1 h, 5 h]`` 窗口，即为 clean_window；corrupted 只是
把中间 [s, e] 置零。先校验重建的 truth 与 corrupted 在缺口外逐点一致。

三个判据（都直接决定图 5 是否要重绘）：
  1. dW1 / dACF  —— 新实现 vs 归档（旧实现）在同一真值下的指标差；
  2. dgap        —— 两个修复段逐点最大差；
  3. W1(new, arch) —— 两个修复结果之间的 Wasserstein 距离。

用法：G:/miniconda3/envs/seisy/python.exe -u _audit_ssa_unify_fig05b.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from obspy import read
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, HERE)

ROOT = os.path.join(HERE, "results_length_scan")
CLEAN_DIR = os.path.join(HERE, "clean_data_12h")
FS = 6.625

# 与 run_length_scan.py 完全一致
CLEAN_FILES = {
    "S12_19760113_070152": "XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed",
    "S12_19761114_230000": "XA.S12.01.MHZ.19761114_230000-19761115_115957.mseed",
    "S15_19760125_160000": "XA.S15.01.MHZ.19760125_160000-19760126_045957.mseed",
    "S16_19761114_230000": "XA.S16.01.MHZ.19761114_230000-19761115_115957.mseed",
}


def stats_metrics(truth_seg, repaired_seg):
    w = wasserstein_distance(truth_seg, repaired_seg)

    def acf(x):
        x = x - np.mean(x)
        n = len(x)
        full = np.correlate(x, x, mode="full")[n - 1:]
        full = full / (full[0] + 1e-12)
        return full[np.arange(min(200, n))]

    a_t, a_r = acf(truth_seg), acf(repaired_seg)
    L = min(len(a_t), len(a_r))
    return float(w), float(np.sqrt(np.mean((a_t[:L] - a_r[:L]) ** 2)))


def main():
    from docs.dssrr_paper.experiments.baselines import SSARepair as NewSSA
    ssa = NewSSA(sr=FS)
    print(f"[cfg] new impl -> {type(ssa._impl).__module__}."
          f"{type(ssa._impl).__name__}(max_iter={ssa._impl.max_iter})", flush=True)

    summary = pd.read_csv(os.path.join(ROOT, "stats_summary.csv"))
    arch_tbl = summary[summary["method"] == "SSA"].set_index(
        ["event", "level"])[["wasserstein", "acf_l2"]]

    truth_cache = {}
    rows = []
    events = sorted(d for d in os.listdir(ROOT)
                    if os.path.isdir(os.path.join(ROOT, d)) and d in CLEAN_FILES)
    for ev in events:
        st = read(os.path.join(CLEAN_DIR, CLEAN_FILES[ev]))
        data = st[0].data.astype(float)
        truth_cache[ev] = data[int(1 * 3600 * FS):int(5 * 3600 * FS)]

    for ev in events:
        truth_full = truth_cache[ev]
        drop = os.path.join(ROOT, ev, "dropout")
        for lv in sorted(os.listdir(drop)):
            cdir = os.path.join(drop, lv)
            cf = os.path.join(cdir, "corrupted.mseed")
            jf = os.path.join(cdir, "intervals.json")
            af = os.path.join(cdir, "repaired_SSA.mseed")
            if not all(os.path.isfile(p) for p in (cf, jf, af)):
                continue
            corrupted = read(cf)[0].data.astype(float)
            arch = read(af)[0].data.astype(float)
            with open(jf, encoding="utf-8") as fh:
                s, e = json.load(fh)["intervals"][0]

            # --- 校验重建的真值 ---
            outside = np.ones(len(corrupted), dtype=bool)
            outside[s:e + 1] = False
            n = min(len(corrupted), len(truth_full))
            outside &= np.arange(len(corrupted)) < n
            ok = np.allclose(corrupted[outside], truth_full[:n][outside],
                             atol=1e-6)

            truth = truth_full[:len(corrupted)]
            new = ssa.repair(corrupted, s, e)

            w_new, a_new = stats_metrics(truth[s:e + 1], new[s:e + 1])
            w_arc, a_arc = stats_metrics(truth[s:e + 1], arch[s:e + 1])
            dgap = float(np.max(np.abs(new[s:e + 1] - arch[s:e + 1])))
            w_na = float(wasserstein_distance(new[s:e + 1], arch[s:e + 1]))

            rows.append(dict(event=ev, level=lv, length_min=float(lv[:5]),
                             truth_ok=ok, w_new=w_new, w_arch=w_arc,
                             a_new=a_new, a_arch=a_arc,
                             dw=w_new - w_arc, da=a_new - a_arc,
                             dgap_max=dgap, w_new_vs_arch=w_na))
            print(f"[{ev} {lv}] truth_ok={ok} | W1 new={w_new:7.4f} "
                  f"arch={w_arc:7.4f} d={w_new-w_arc:+7.4f} | "
                  f"ACF d={a_new-a_arc:+7.4f} | dgap={dgap:9.3f} "
                  f"W1(n,a)={w_na:7.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results_stats", "ssa_unify_fig05_compare.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n=== 汇总（n = {len(df)}）===", flush=True)
    print(f"真值重建校验通过率 : {df['truth_ok'].mean() * 100:.1f}%", flush=True)
    for col, nm in [("dw", "ΔW₁"), ("da", "ΔACF"),
                    ("dgap_max", "逐点最大差"), ("w_new_vs_arch", "W₁(新,旧)")]:
        v = df[col].abs()
        print(f"{nm:<12}: max={v.max():.4f}  median={v.median():.4f}  "
              f"mean={v.mean():.4f}", flush=True)

    print("\n=== 按长度做图 5 口径的截尾平均（4 事件）===", flush=True)

    def trimmed(v):
        v = np.sort(np.asarray(v, float))
        return float(np.mean(v[1:-1])) if len(v) >= 4 else float(np.mean(v))

    g = df.groupby("length_min").agg(w_new=("w_new", trimmed),
                                     w_arch=("w_arch", trimmed),
                                     a_new=("a_new", trimmed),
                                     a_arch=("a_arch", trimmed)).reset_index()
    g["dW1"] = g["w_new"] - g["w_arch"]
    g["dACF"] = g["a_new"] - g["a_arch"]
    print(g.round(4).to_string(index=False), flush=True)
    print(f"\n截尾平均后 |ΔW₁| max = {g['dW1'].abs().max():.4f}  "
          f"|ΔACF| max = {g['dACF'].abs().max():.4f}", flush=True)
    print(f"\n明细写入 {out}", flush=True)


if __name__ == "__main__":
    main()
