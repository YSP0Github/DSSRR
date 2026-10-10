"""
E6: 下游任务验证 —— 修复质量对 STA/LTA 事件检测与噪声底估计的影响

动机（用户方向）：DSSRR 的卖点是"统计/频谱保真 + 无缝边界"，必须证明
它对下游分析（事件检测、噪声底估计、频谱分析）的价值，而非仅指标好看。

设计：
- E6A 合成（定量主实验）：低频平稳背景 + 2 个已知月震型事件（固定 1/8、7/8
  位置，避开中间 1/2 注入区）。注入 dropout(500/5000 点)/spike → 8 方法修复
  → STA/LTA 检测。指标：
    keep_rate   : 真值中与已知事件匹配的触发，修复后被重新检测到的比例
    false_rate  : 修复后新增的、不与已知事件/真值触发匹配的误触事件数
    time_err    : 匹配触发的起始时刻偏差（中位数，s）
    noise_mad_err: 修复段内 MAD 相对真值段 MAD 的相对偏差
- E6B 真实 Apollo：9 段 12h 注入 dropout(500/5000)/spike → 修复 → STA/LTA。
  指标：触发总数相对真值的偏差、噪声底偏差。

异常区间约定（用户 2026-09-07）：真实异常修复时异常段 = 检测段 + 两侧
padding；本实验注入式异常直接采用注入区间（精确已知）。
"""

from __future__ import annotations

import csv
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, r"G:\SeisY\docs\dssrr_paper")

from docs.dssrr_paper.experiments.baselines import make_methods, METHOD_ORDER
from docs.dssrr_paper.experiments.E2_real import load_data_files
from docs.dssrr_paper.experiments.E5_length_scan import (
    SYN_FAMILIES, INJECTORS, SR,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

N_EVENTS = 2
EVENT_POS = [1 / 8.0, 7 / 8.0]     # 事件中心相对位置（避开注入区）
N_SYN = 20000                      # ~50 min
N_SEEDS = 10
STA_S, LTA_S, THR_ON, THR_OFF = 2.0, 30.0, 4.0, 1.2
MATCH_TOL_S = 5.0

METRIC_COLS = ["keep_rate", "false_rate", "time_err", "noise_mad_err"]


# ----------------------------------------------------------------------
# 合成事件 + 信号
# ----------------------------------------------------------------------

def add_events(x: np.ndarray, seed: int, sr: float) -> tuple[np.ndarray, list[float]]:
    """插入 2 个月震型事件（带通振荡 + 指数衰减），返回 (信号, 事件中心秒列表)。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = x.copy()
    centers = []
    for pos in EVENT_POS:
        c = int(pos * n)
        dur = int(rng.uniform(5, 12) * sr)
        t0, t1 = c - dur // 2, c + dur // 2
        t1 = min(t1, n - 1)
        tt = np.arange(t1 - t0 + 1) / sr
        sos_f = sp_butter(4, [0.3, 1.5], btype="bandpass", fs=sr, output="sos")
        sig = sp_sosfilt(sos_f, rng.standard_normal(len(tt)))
        env = np.exp(-tt / 3.0)
        amp = rng.uniform(15, 30) * (np.std(x) + 1e-9)
        out[t0:t1 + 1] += amp * sig * env
        centers.append(c / sr)
    return out, centers


from scipy.signal import butter as sp_butter, sosfilt as sp_sosfilt  # noqa: E402


# ----------------------------------------------------------------------
# STA/LTA 检测
# ----------------------------------------------------------------------

def sta_lta_ratio(x: np.ndarray, sr: float):
    """稳健 RMS/背景 比值（对低频平稳信号稳定）。

    短窗 RMS（1 s）除以 60 s 滑动平均背景；事件（5–12× 背景）触发
    显著，平稳背景比值稳定在 1 附近，修复段拉直线时比值趋于 0。
    """
    n_sta = max(2, int(1.0 * sr))
    n_bg = max(n_sta + 1, int(60.0 * sr))
    x = x - np.median(x)
    env = np.sqrt(np.convolve(x ** 2, np.ones(n_sta) / n_sta, mode="same"))
    bg = np.convolve(env, np.ones(n_bg) / n_bg, mode="same")
    r = env / (np.maximum(bg, 1e-12) + 1e-12)
    off = 0  # np.convolve(mode="same") 无延迟
    return r, off


def detect_triggers(r: np.ndarray, off: int, sr: float) -> list[tuple[float, float]]:
    """过阈值（>THR_ON）触发区间，合并间隙<2 s、最短 1 s，返回 [(t0,t1)] 秒。"""
    thr = THR_ON
    n = len(r)
    on = r > thr
    tr = []
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and (on[j + 1] or
                                 ((j + 1 - i) / sr < 2.0 and on[j + 1])):
                j += 1
            # 尾部允许 ≤2 s 间隙再接 on
            while j + 1 < n:
                gap = (j + 1 - i) / sr
                k = j + 1
                while k < n and not on[k]:
                    k += 1
                if k < n and (k - j - 1) / sr <= 2.0:
                    j = k
                    while j + 1 < n and on[j + 1]:
                        j += 1
                else:
                    break
            if (j - i + 1) / sr >= 1.0:
                tr.append(((i + off) / sr, (j + off) / sr))
            i = j + 1
        else:
            i += 1
    return tr


def match_events(triggers: list[tuple[float, float]], centers: list[float],
                 tol_s: float = MATCH_TOL_S) -> tuple[list[int], list[int]]:
    """匹配触发与已知事件中心。返回 (命中事件索引, 未命中触发索引)。"""
    hit, miss_t = [], []
    used = [False] * len(triggers)
    for ci, c in enumerate(centers):
        best = None
        for ti, (t0, t1) in enumerate(triggers):
            if used[ti]:
                continue
            if abs((t0 + t1) / 2 - c) <= tol_s:
                best = ti
                break
        if best is not None:
            used[best] = True
            hit.append(ci)
        else:
            miss_t.append(ci)
    false = [ti for ti in range(len(triggers)) if not used[ti]]
    return hit, false


# ----------------------------------------------------------------------
# 下游评估（对一个修复结果）
# ----------------------------------------------------------------------

def downstream_metrics(truth: np.ndarray, repaired: np.ndarray, s: int, e: int,
                       sr: float, centers: list[float]) -> dict:
    r_t, off_t = sta_lta_ratio(truth, sr)
    tr_t = detect_triggers(r_t, off_t, sr)
    r_r, off_r = sta_lta_ratio(repaired, sr)
    tr_r = detect_triggers(r_r, off_r, sr)

    hit_t, false_t = match_events(tr_t, centers)
    hit_r, false_r = match_events(tr_r, centers)

    keep = len(hit_r) / max(1, len(centers))
    # 新增误触发（修复后触发中未匹配事件数 − 真值本身未匹配事件数）
    false_rate = max(0.0, len(false_r) - len(false_t))
    # 匹配触发的起始时刻偏差（对每个命中事件取最近触发）
    errs = []
    for c in centers:
        cand = [abs(t0 - c) for t0, t1 in tr_r if abs((t0 + t1) / 2 - c) <= MATCH_TOL_S]
        if cand:
            errs.append(min(cand))
    time_err = float(np.median(errs)) if errs else np.nan

    # 修复段噪声底（MAD）相对真值
    seg_t = truth[s:e + 1]
    seg_r = repaired[s:e + 1]
    mad_t = np.median(np.abs(seg_t - np.median(seg_t))) + 1e-12
    mad_r = np.median(np.abs(seg_r - np.median(seg_r))) + 1e-12
    noise_mad_err = abs(mad_r - mad_t) / mad_t

    return {"keep_rate": keep, "false_rate": false_rate,
            "time_err": time_err, "noise_mad_err": noise_mad_err}


# ----------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------

def _task_synth(key: tuple) -> dict:
    fam, anom, length, seed_idx = key
    sr = SR
    seed = int(zlib.crc32(f"E6|{fam}|{anom}|{length}|{seed_idx}".encode()))
    base = SYN_FAMILIES[fam](seed, N_SYN, sr)
    truth, centers = add_events(base, seed, sr)
    corr, s, e = INJECTORS[anom](truth, seed, length)
    methods = make_methods(sr)
    out = {}
    for m in METHOD_ORDER:
        try:
            rep = methods[m].repair(corr, s, e)
            out[m] = downstream_metrics(truth, rep, s, e, sr, centers)
        except Exception:
            out[m] = None
    return {"dataset": "synth", "family": fam, "anomaly": anom,
            "length": length, "seed": seed_idx, "res": out}


def _task_real(key: tuple) -> dict:
    fname, data, sr, anom, length, seed_idx = key
    seed = int(zlib.crc32(f"E6|{fname}|{anom}|{length}|{seed_idx}".encode()))
    # 取整段中部子窗（约 40 min）作为处理段，保证上下文充足
    n = len(data)
    seg = data[n // 4:n // 4 + int(40 * 60 * sr)]
    truth = seg.copy()
    corr, s, e = INJECTORS[anom](truth, seed, length)
    centers = []  # 真实数据无已知事件；只用噪声底与触发稳定性
    methods = make_methods(sr)
    out = {}
    for m in METHOD_ORDER:
        try:
            rep = methods[m].repair(corr, s, e)
            r_t, off_t = sta_lta_ratio(truth, sr)
            r_r, off_r = sta_lta_ratio(rep, sr)
            n_tr_t = len(detect_triggers(r_t, off_t, sr))
            n_tr_r = len(detect_triggers(r_r, off_r, sr))
            seg_t = truth[s:e + 1]
            seg_r = rep[s:e + 1]
            mad_t = np.median(np.abs(seg_t - np.median(seg_t))) + 1e-12
            mad_r = np.median(np.abs(seg_r - np.median(seg_r))) + 1e-12
            out[m] = {"keep_rate": np.nan,
                      "false_rate": float(abs(n_tr_r - n_tr_t)),
                      "time_err": np.nan,
                      "noise_mad_err": float(abs(mad_r - mad_t) / mad_t)}
        except Exception:
            out[m] = None
    return {"dataset": "real", "file": fname, "anomaly": anom,
            "length": length, "seed": seed_idx, "res": out}


# ----------------------------------------------------------------------
# 运行
# ----------------------------------------------------------------------

def run(workers: int = 10) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    synth_tasks = [(fam, anom, length, seed_idx)
                   for fam in SYN_FAMILIES
                   for anom in INJECTORS
                   for length in [500, 5000]
                   for seed_idx in range(N_SEEDS)]
    print(f"[E6] 合成任务 = {len(synth_tasks)}")
    real_tasks = []
    for fname, data, sr in load_data_files():
        for anom in ["dropout", "step"]:
            for length in [500, 5000]:
                for seed_idx in range(3):
                    real_tasks.append((fname, data, sr, anom, length, seed_idx))
    print(f"[E6] 真实任务 = {len(real_tasks)}")

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_task_synth, synth_tasks, chunksize=4):
            for m in METHOD_ORDER:
                if r["res"][m] is None:
                    continue
                rows.append({"dataset": "synth", "file": "", "family": r["family"],
                             "anomaly": r["anomaly"], "length": r["length"],
                             "seed": r["seed"], "method": m, **r["res"][m]})
        for r in ex.map(_task_real, real_tasks, chunksize=4):
            for m in METHOD_ORDER:
                if r["res"][m] is None:
                    continue
                rows.append({"dataset": "real", "file": r["file"], "family": "",
                             "anomaly": r["anomaly"], "length": r["length"],
                             "seed": r["seed"], "method": m, **r["res"][m]})

    out_csv = os.path.join(OUT_DIR, "E6_downstream.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "file", "family", "anomaly",
                                          "length", "seed", "method", *METRIC_COLS])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[E6] 行数={len(rows)} 输出={out_csv}")
    return out_csv


# ----------------------------------------------------------------------
# 汇总 + 图
# ----------------------------------------------------------------------

def summarize(csv_path: str):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    print(f"\n==== E6 下游验证汇总（method × anomaly，mean） ====")
    for dataset, dlabel in [("synth", "E6A 合成(含已知事件)"), ("real", "E6B 真实Apollo")]:
        print(f"\n--- {dlabel} ---")
        print(f"{'anomaly':<9}{'L':>5}{'method':<9}{'n':>4}{'keep':>7}{'false':>7}"
              f"{'tErr':>7}{'madErr':>8}")
        for anom in ["dropout", "step"]:
            for length in [500, 5000]:
                for m in METHOD_ORDER:
                    vals = [r for r in rows if r["dataset"] == dataset
                            and r["anomaly"] == anom
                            and int(r["length"]) == length and r["method"] == m]
                    if not vals:
                        continue
                    keep = np.nanmean([float(v["keep_rate"]) for v in vals])
                    false = np.nanmean([float(v["false_rate"]) for v in vals])
                    terr = np.nanmedian([float(v["time_err"]) for v in vals])
                    mad = np.nanmean([float(v["noise_mad_err"]) for v in vals])
                    print(f"{anom:<9}{length:>5}{m:<9}{len(vals):>4}"
                          f"{keep:>7.2f}{false:>7.2f}{terr:>7.2f}{mad:>8.3f}")


def plot_downstream(csv_path: str, out_name: str = "fig7_downstream.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    METHOD_LABELS = {"linear": "Linear", "spline": "Spline", "wavelet": "Wavelet",
                     "ar": "AR(Burg)", "stft": "STFT", "ssa": "SSA",
                     "unet": "UNet", "dssrr": "DSSRR"}
    COLORS = {"linear": "#9EACEA", "spline": "#8BC8EA", "wavelet": "#C9A7E8",
              "ar": "#94D8C3", "stft": "#E1B98F", "ssa": "#2E8B57",
              "unet": "#FFA500", "dssrr": "#EA6668"}
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    synth = [r for r in rows if r["dataset"] == "synth"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [("keep_rate", "Event detection keep rate (higher better)"),
              ("false_rate", "False triggers per instance (lower better)"),
              ("noise_mad_err", "Noise-floor (MAD) error in repaired gap (lower better)"),
              ("time_err", "Matched trigger time error, s (lower better)")]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        groups = [("dropout", 500, "Dropout 500"),
                  ("dropout", 5000, "Dropout 5000"),
                  ("step", 500, "Step 500"),
                  ("step", 5000, "Step 5000")]
        xpos = np.arange(len(groups))
        w = 0.105
        for i, m in enumerate(METHOD_ORDER):
            ys = []
            for anom, length, _ in groups:
                vals = [float(r[metric]) for r in synth
                        if r["anomaly"] == anom and int(r["length"]) == length
                        and r["method"] == m and np.isfinite(float(r[metric]))]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.bar(xpos + (i - len(METHOD_ORDER) / 2 + 0.5) * w, ys, w,
                   label=METHOD_LABELS[m], color=COLORS[m],
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(xpos)
        ax.set_xticklabels([g[2] for g in groups], fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
        if metric == "keep_rate":
            ax.legend(fontsize=7, ncol=4, frameon=False, loc="lower right")
    fig.suptitle("E6 — downstream STA/LTA detection & noise-floor (synthetic, n=20/group)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(FIG_DIR, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    csv_path = run(workers=10)
    summarize(csv_path)
    plot_downstream(csv_path)
