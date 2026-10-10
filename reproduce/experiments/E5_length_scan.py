"""
E5: 大中小异常段长度扫描（异常段 50 / 500 / 5000 点 ≈ 7.5 / 75.5 / 754.7 s）

背景（用户反馈 2026-09-07）：论文方法主要面向"异常段修复"，需按
异常段规模分档验证；长时程异常（缺失）是方法相对插值类方法的核心场景，
尤其针对低频、平稳数据（如月震 6.625 Hz 背景）的预处理。

设计：
- E5A 真实 Apollo：9 段 12h 记录取干净平稳窗，注入 dropout/step 两类
  长段异常，长度分档 50/500/5000 点，8 方法统一评估。
- E5B 合成低频平稳：两种低频平稳信号（低通 AR / 多正弦低频混合），
  同样 2 异常 × 3 长度 × 20 种子。
- 指标：与 E1/E2 完全一致（evaluate_fair）。

输出：E5_real_instances.csv / E5_synth_instances.csv + 汇总 + 长度趋势图。
"""

from __future__ import annotations

import csv
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from obspy import read
from scipy import signal as sp_signal

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, r"G:\SeisY\docs\dssrr_paper")

from docs.dssrr_paper.experiments.protocol import evaluate_fair
from docs.dssrr_paper.experiments.baselines import make_methods, METHOD_ORDER
from docs.dssrr_paper.experiments.E2_real import load_data_files, select_clean_windows

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SR = 6.625
LENGTHS = [50, 500, 5000]
N_WINDOWS = 2          # E5A 每文件取 2 个干净窗
N_SEEDS_REAL = 8       # E5A 每窗每长度种子数
N_SEEDS_SYN = 20       # E5B 每族每长度种子数
N_SYN = 40000          # 合成信号长度（~100 min）

METRIC_COLS = ["rmse", "mae", "snr", "psd_cosine", "log_spec_dist",
               "boundary_jump", "autocorr_decay_diff", "energy_ratio"]


# ----------------------------------------------------------------------
# 低频平稳合成信号（模拟月震/低频仪器的平稳背景）
# ----------------------------------------------------------------------

def gen_lowfreq_ar(seed: int, n: int = N_SYN, sr: float = SR) -> np.ndarray:
    """白噪声 -> 4 阶 Butterworth 低通 0.3 Hz（平稳低频背景，类似月震噪声底）。"""
    rng = np.random.default_rng(seed)
    sos = sp_signal.butter(4, 0.3, btype="lowpass", fs=sr, output="sos")
    x = sp_signal.sosfilt(sos, rng.standard_normal(n))
    x = x / (np.std(x) + 1e-12)
    return x


def gen_lowfreq_mix(seed: int, n: int = N_SYN, sr: float = SR) -> np.ndarray:
    """多个亚 Hz 正弦 + 低通噪声（低频仪器数据常见形态）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    sig = (1.0 * np.sin(2 * np.pi * 0.02 * t + rng.uniform(0, 2 * np.pi))
           + 0.6 * np.sin(2 * np.pi * 0.006 * t + rng.uniform(0, 2 * np.pi))
           + 0.3 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 2 * np.pi)))
    sos = sp_signal.butter(4, 0.5, btype="lowpass", fs=sr, output="sos")
    noise = sp_signal.sosfilt(sos, rng.standard_normal(n))
    return sig + 0.5 * noise / (np.std(noise) + 1e-12)


SYN_FAMILIES = {"lowfreq_ar": gen_lowfreq_ar, "lowfreq_mix": gen_lowfreq_mix}


# ----------------------------------------------------------------------
# 注入：dropout（置 0）/ step（加偏移），长度=异常段长
# ----------------------------------------------------------------------

def inject_dropout(x: np.ndarray, seed: int, length: int) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    lo = len(x) // 4
    hi = len(x) // 4 * 3 - length
    if hi <= lo:
        hi = lo + 1
    s = int(rng.integers(lo, max(lo + 1, hi)))
    corr = x.copy()
    corr[s:s + length] = 0.0
    return corr, s, s + length - 1


def inject_step(x: np.ndarray, seed: int, length: int) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    lo = len(x) // 4
    hi = len(x) // 4 * 3 - length
    if hi <= lo:
        hi = lo + 1
    s = int(rng.integers(lo, max(lo + 1, hi)))
    ctx = x[max(0, s - int(10 * SR)):s + length + int(10 * SR)]
    off = rng.uniform(2, 5) * (np.std(ctx) + 1e-9) * (1.0 if rng.random() < 0.5 else -1.0)
    corr = x.copy()
    corr[s:s + length] += off
    return corr, s, s + length - 1


INJECTORS = {"dropout": inject_dropout, "step": inject_step}


# ----------------------------------------------------------------------
# worker（模块级，Windows spawn 可 pickle）
# ----------------------------------------------------------------------

def _eval_one(truth: np.ndarray, corr: np.ndarray, s: int, e: int,
              sr: float) -> dict:
    methods = make_methods(sr)
    out = {}
    for mname in METHOD_ORDER:
        try:
            repaired = methods[mname].repair(corr, s, e)
            metrics = evaluate_fair(truth, repaired, s, e, sr)
            if not metrics["valid"] or metrics["rmse"] > 1e6:
                raise ValueError("unstable/failed")
            out[mname] = {k: metrics[k] for k in METRIC_COLS}
        except Exception:
            out[mname] = None  # 失败实例
    return out


def _task_real(key: tuple) -> dict:
    fname, wi, anom, length, seed_idx, sub = key
    sr = SR
    truth = sub.copy()
    seed = int(zlib.crc32(f"{fname}|{wi}|{anom}|{length}|{seed_idx}".encode()))
    corr, s, e = INJECTORS[anom](truth, seed, length)
    res = _eval_one(truth, corr, s, e, sr)
    return {"dataset": "real", "file": fname, "window": wi, "anomaly": anom,
            "length": length, "seed": seed_idx, "start": s, "end": e, "res": res}


def _task_synth(key: tuple) -> dict:
    fam, anom, length, seed_idx = key
    sr = SR
    seed = int(zlib.crc32(f"{fam}|{anom}|{length}|{seed_idx}".encode()))
    truth = SYN_FAMILIES[fam](seed, N_SYN, sr)
    corr, s, e = INJECTORS[anom](truth, seed, length)
    res = _eval_one(truth, corr, s, e, sr)
    return {"dataset": "synth", "family": fam, "anomaly": anom,
            "length": length, "seed": seed_idx, "start": s, "end": e, "res": res}


# ----------------------------------------------------------------------
# 运行
# ----------------------------------------------------------------------

def run(workers: int = 10) -> tuple[str, str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # E5A 真实数据：预取平稳窗子段
    real_tasks = []
    for fname, data, sr in load_data_files():
        windows = select_clean_windows(data, sr, n_windows=N_WINDOWS)
        for wi, (w0, w1) in enumerate(windows):
            sub = data[w0:w1].copy()
            for anom in INJECTORS:
                for length in LENGTHS:
                    for seed_idx in range(N_SEEDS_REAL):
                        real_tasks.append((fname, wi, anom, length, seed_idx, sub))
    print(f"[E5] 真实任务数 = {len(real_tasks)}")

    # E5B 合成
    synth_tasks = []
    for fam in SYN_FAMILIES:
        for anom in INJECTORS:
            for length in LENGTHS:
                for seed_idx in range(N_SEEDS_SYN):
                    synth_tasks.append((fam, anom, length, seed_idx))
    print(f"[E5] 合成任务数 = {len(synth_tasks)}")

    rows = []
    failed = {m: 0 for m in METHOD_ORDER}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_task_real, real_tasks, chunksize=8):
            for mname in METHOD_ORDER:
                if r["res"][mname] is None:
                    failed[mname] += 1
                    continue
                rows.append({
                    "dataset": "real", "file": r["file"], "window": r["window"],
                    "family": "", "anomaly": r["anomaly"], "length": r["length"],
                    "seed": r["seed"], "method": mname, **r["res"][mname],
                })
        for r in ex.map(_task_synth, synth_tasks, chunksize=8):
            for mname in METHOD_ORDER:
                if r["res"][mname] is None:
                    failed[mname] += 1
                    continue
                rows.append({
                    "dataset": "synth", "file": "", "window": -1,
                    "family": r["family"], "anomaly": r["anomaly"],
                    "length": r["length"], "seed": r["seed"],
                    "method": mname, **r["res"][mname],
                })

    fieldnames = ["dataset", "file", "window", "family", "anomaly", "length",
                  "seed", "method", *METRIC_COLS]
    out_real = os.path.join(OUT_DIR, "E5_real_instances.csv")
    out_synth = os.path.join(OUT_DIR, "E5_synth_instances.csv")
    with open(out_real, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if r["dataset"] == "real":
                w.writerow(r)
    with open(out_synth, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if r["dataset"] == "synth":
                w.writerow(r)
    print(f"[E5] 行数={len(rows)}  失败={failed}")
    print(f"[E5] 输出: {out_real} / {out_synth}")
    return out_real, out_synth


# ----------------------------------------------------------------------
# 汇总：dataset × anomaly × length × method
# ----------------------------------------------------------------------

def summarize(csv_path: str) -> dict:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    arr = {}
    for r in rows:
        key = (r["dataset"], r["anomaly"], int(r["length"]), r["method"])
        arr.setdefault(key, []).append(
            {k: float(r[k]) for k in METRIC_COLS})

    print(f"\n==== {os.path.basename(csv_path)} 汇总（length × method，mean） ====")
    print(f"{'anom':<9}{'L':>5}{'method':<9}{'n':>4}{'RMSE':>10}{'PSDcos':>8}"
          f"{'LogSpec':>8}{'bnd':>8}{'acf':>8}")
    for anom in INJECTORS:
        for length in LENGTHS:
            for m in METHOD_ORDER:
                vals = arr.get(("real" if "real" in csv_path else "synth",
                                anom, length, m), [])
                if not vals:
                    continue
                print(f"{anom:<9}{length:>5}{m:<9}{len(vals):>4}"
                      f"{np.mean([v['rmse'] for v in vals]):>10.4f}"
                      f"{np.nanmean([v['psd_cosine'] for v in vals]):>8.3f}"
                      f"{np.nanmean([v['log_spec_dist'] for v in vals]):>8.3f}"
                      f"{np.nanmean([v['boundary_jump'] for v in vals]):>8.3f}"
                      f"{np.nanmean([v['autocorr_decay_diff'] for v in vals]):>8.3f}")
    return arr


# ----------------------------------------------------------------------
# 图：长度趋势（RMSE / acf / boundary）
# ----------------------------------------------------------------------

def plot_trend(csv_paths: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    METHOD_LABELS = {"linear": "Linear", "spline": "Spline", "ssa": "SSA",
                     "unet": "UNet", "dssrr": "DSSRR"}
    COLORS = {"linear": "#9EACEA", "spline": "#8BC8EA", "ssa": "#2E8B57",
              "unet": "#FFA500", "dssrr": "#EA6668"}
    focus = ["linear", "spline", "ssa", "unet", "dssrr"]
    ANOM_LABEL = {"dropout": "Dropout (missing)", "step": "Step"}

    rows = []
    for p in csv_paths:
        rows += list(csv.DictReader(open(p, encoding="utf-8")))

    for anom in ["dropout", "step"]:
        fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
        metrics = [("rmse", "RMSE"), ("autocorr_decay_diff", "Autocorr decay diff"),
                   ("boundary_jump", "Boundary jump (norm.)")]
        for ci, (dataset, ds_label) in enumerate([("real", "Real Apollo (E5A)"),
                                                  ("synth", "Synthetic low-freq (E5B)")]):
            for mi, (metric, ylabel) in enumerate(metrics):
                ax = axes[mi, ci]
                for m in focus:
                    xs, ys = [], []
                    for length in LENGTHS:
                        vals = [float(r[metric]) for r in rows
                                if r["dataset"] == dataset
                                and r["anomaly"] == anom
                                and r["method"] == m
                                and int(r["length"]) == length
                                and np.isfinite(float(r[metric]))]
                        if vals:
                            xs.append(length)
                            ys.append(np.mean(vals))
                    ax.plot(xs, ys, marker="o", ms=4, lw=1.4,
                            color=COLORS[m], label=METHOD_LABELS[m])
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks(LENGTHS)
                ax.set_xticklabels([f"{L}\n({L/SR:.0f}s)" for L in LENGTHS],
                                   fontsize=8)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(f"{ds_label} — {ANOM_LABEL[anom]} — {ylabel}",
                             fontsize=9)
                ax.grid(alpha=0.3, lw=0.4, which="both")
                if mi == 0:
                    ax.legend(fontsize=8, frameon=False)
        axes[2, 0].set_xlabel("Anomaly length (samples)")
        axes[2, 1].set_xlabel("Anomaly length (samples)")
        fig.suptitle(f"E5 — {ANOM_LABEL[anom]} behavior vs anomaly length (lower is better)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(FIG_DIR, f"fig6_length_{anom}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    r1, r2 = run(workers=10)
    summarize(r1)
    summarize(r2)
    plot_trend([r1, r2])
