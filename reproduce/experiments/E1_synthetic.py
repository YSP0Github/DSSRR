"""
E1: 统计稳健的合成数据 Monte Carlo 基准 (R1)

5 信号族 × 6 异常类型 × N 种子（每实例单个异常），统一公平评估。
输出：逐实例 CSV + 汇总表（mean±std、配对 Wilcoxon、改进倍数中位数）。
"""

from __future__ import annotations

import os
import zlib

import numpy as np
from scipy import signal as sp_signal
from scipy import stats

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docs.dssrr_paper.experiments.protocol import evaluate_fair
from docs.dssrr_paper.experiments.baselines import make_methods, METHOD_ORDER

SR = 6.625
N = 10000          # 信号样本数 (~1509 s)
N_SEEDS = 30
REF_MARGIN = int(60 * SR)   # 异常两侧至少 60 s 参考


# ----------------------------------------------------------------------
# 信号族生成器（返回真值波形）
# ----------------------------------------------------------------------

def gen_multitone(seed: int, n: int = N, sr: float = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    return (0.5 * np.sin(2 * np.pi * 0.1 * t)
            + 0.3 * np.sin(2 * np.pi * 0.5 * t)
            + 0.2 * np.sin(2 * np.pi * 1.0 * t)
            + 0.1 * rng.standard_normal(n))


def gen_chirp(seed: int, n: int = N, sr: float = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    f0, f1 = 0.05, 1.0
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-9) * t ** 2)
    return (0.6 * np.sin(phase) * (1 + 0.3 * np.sin(2 * np.pi * 0.02 * t))
            + 0.05 * rng.standard_normal(n))


def gen_coda(seed: int, n: int = N, sr: float = SR) -> np.ndarray:
    """带指数衰减包络的滤波噪声（模拟月震散射尾波）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    sos = sp_signal.butter(4, [0.02, 0.8], btype="bandpass", fs=sr, output="sos")
    sig = sp_signal.sosfilt(sos, rng.standard_normal(n))
    env = np.exp(-t / 300.0)
    return sig * env * 3.0


def gen_transient(seed: int, n: int = N, sr: float = SR) -> np.ndarray:
    """短时能量团 + 尾波（模拟月震事件）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    base = gen_coda(seed, n, sr) * 0.5
    burst_len = int(30 * sr)
    sos = sp_signal.butter(4, [0.05, 1.2], btype="bandpass", fs=sr, output="sos")
    burst = sp_signal.sosfilt(sos, rng.standard_normal(burst_len)) * 5.0
    b0 = int(n * 0.3)
    base[b0:b0 + burst_len] += burst
    return base


FAMILIES = {
    "multitone": gen_multitone,
    "chirp": gen_chirp,
    "coda": gen_coda,
    "transient": gen_transient,
}


# ----------------------------------------------------------------------
# 异常注入器（返回 (corrupted, (start, end))，单异常）
# ----------------------------------------------------------------------

def _rand_interval(rng, n: int, length: int) -> tuple[int, int]:
    lo = REF_MARGIN
    hi = n - REF_MARGIN - length
    if hi <= lo:
        hi = lo + 1
    start = int(rng.integers(lo, max(lo + 1, hi)))
    return start, start + length - 1


def inject_spike(x, seed):
    rng = np.random.default_rng(seed)
    corrupted = x.copy()
    i = int(rng.integers(REF_MARGIN, len(x) - REF_MARGIN))
    amp = rng.uniform(10, 100) * np.std(x)
    corrupted[i] += amp * (1.0 if rng.random() < 0.5 else -1.0)
    return corrupted, (i, i)


def inject_dropout_short(x, seed):
    rng = np.random.default_rng(seed)
    corrupted = x.copy()
    s, e = _rand_interval(rng, len(x), int(rng.integers(50, 200)))
    corrupted[s:e + 1] = 0.0
    return corrupted, (s, e)


def inject_dropout_long(x, seed):
    rng = np.random.default_rng(seed)
    corrupted = x.copy()
    s, e = _rand_interval(rng, len(x), int(rng.integers(500, 1000)))
    corrupted[s:e + 1] = 0.0
    return corrupted, (s, e)


def inject_noise_burst(x, seed):
    rng = np.random.default_rng(seed)
    corrupted = x.copy()
    s, e = _rand_interval(rng, len(x), int(rng.integers(30, 100)))
    corrupted[s:e + 1] += rng.standard_normal(e - s + 1) * np.std(x) * rng.uniform(5, 10)
    return corrupted, (s, e)


def inject_step(x, seed):
    rng = np.random.default_rng(seed)
    corrupted = x.copy()
    s, e = _rand_interval(rng, len(x), int(rng.integers(100, 400)))
    offset = rng.uniform(2, 5) * np.std(x) * (1.0 if rng.random() < 0.5 else -1.0)
    corrupted[s:e + 1] += offset
    return corrupted, (s, e)


def inject_saturation(x, seed):
    rng = np.random.default_rng(seed)
    corrupted = x.copy()
    s, e = _rand_interval(rng, len(x), int(rng.integers(100, 400)))
    corrupted[s:e + 1] = float(np.median(x[s:e + 1]))  # 平坦/饱和段
    return corrupted, (s, e)


ANOMALIES = {
    "spike": inject_spike,
    "dropout_short": inject_dropout_short,
    "dropout_long": inject_dropout_long,
    "noise_burst": inject_noise_burst,
    "step": inject_step,
    "saturation": inject_saturation,
}


# ----------------------------------------------------------------------
# 运行
# ----------------------------------------------------------------------

METRIC_COLS = ["rmse", "mae", "snr", "psd_cosine", "log_spec_dist",
               "boundary_jump", "autocorr_decay_diff", "energy_ratio"]


def run(n_seeds: int = N_SEEDS, out_dir: str | None = None) -> str:
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    failed = {m: 0 for m in METHOD_ORDER}

    for fam_name, gen in FAMILIES.items():
        for anom_name, inject in ANOMALIES.items():
            for seed_idx in range(n_seeds):
                # 确定性种子（hash() 跨进程随机，不可用；用 crc32）
                seed = int(zlib.crc32(f"{fam_name}|{anom_name}|{seed_idx}".encode()))
                truth = gen(seed)
                corrupted, (s, e) = inject(truth, seed)
                methods = make_methods(SR)
                for mname in METHOD_ORDER:
                    try:
                        repaired = methods[mname].repair(corrupted, s, e)
                        metrics = evaluate_fair(truth, repaired, s, e, SR)
                        if not metrics["valid"] or metrics["rmse"] > 1e6:
                            raise ValueError("unstable/failed")
                        rows.append({
                            "family": fam_name, "anomaly": anom_name,
                            "seed": seed_idx, "start": s, "end": e,
                            "length": e - s + 1, "method": mname,
                            **{k: metrics[k] for k in METRIC_COLS},
                        })
                    except Exception:
                        failed[mname] += 1

    out_csv = os.path.join(out_dir, "E1_instances.csv")
    import csv
    fieldnames = ["family", "anomaly", "seed", "start", "end", "length",
                  "method", *METRIC_COLS]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[E1] 实例数={len(rows)}  失败(排除)={failed}")
    print(f"[E1] 输出: {out_csv}")
    return out_csv


def summarize(csv_path: str) -> dict:
    import csv
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    arr = {}
    for r in rows:
        arr.setdefault((r["family"], r["anomaly"], r["method"]), []).append(
            {k: float(r[k]) for k in METRIC_COLS})

    # 汇总表：按 anomaly × method 的 mean±std
    print("\n==== E1 汇总（anomaly × method，RMSE 越小越好；SNR/PSD 越大越好） ====")
    print(f"{'anomaly':<14}{'method':<9}{'n':>4}{'RMSE':>10}{'SNR(dB)':>10}"
          f"{'PSDcos':>9}{'LogSpec':>9}{'bnd':>7}{'acf':>7}")
    for anom in ANOMALIES:
        for m in METHOD_ORDER:
            vals = [v for (fam, a, mm), lst in arr.items()
                    for v in lst if a == anom and mm == m]
            if not vals:
                continue
            rmse_m, rmse_s = np.mean([v["rmse"] for v in vals]), np.std([v["rmse"] for v in vals])
            snr_m = np.nanmean([v["snr"] for v in vals])
            psd_m = np.nanmean([v["psd_cosine"] for v in vals])
            log_m = np.nanmean([v["log_spec_dist"] for v in vals])
            bnd_m = np.nanmean([v["boundary_jump"] for v in vals])
            acf_m = np.nanmean([v["autocorr_decay_diff"] for v in vals])
            print(f"{anom:<14}{m:<9}{len(vals):>4}{rmse_m:>10.4f}{snr_m:>10.2f}"
                  f"{psd_m:>9.3f}{log_m:>9.3f}{bnd_m:>7.3f}{acf_m:>7.3f}")

    # 配对 Wilcoxon：每种异常类型，DSSRR vs 每个基线（跨信号族合并）
    print("\n==== E1 配对显著性（Wilcoxon，DSSRR 相对基线的 RMSE 降低） ====")
    print(f"{'anomaly':<14}{'baseline':<9}{'n':>4}{'med_impr':>10}{'p_value':>10}")
    for anom in ANOMALIES:
        d_rmse = [v["rmse"] for (fam, a, m), lst in arr.items()
                  for v in lst if a == anom and m == "dssrr"]
        for base in METHOD_ORDER[:-1]:
            b_rmse = [v["rmse"] for (fam, a, m), lst in arr.items()
                      for v in lst if a == anom and m == base]
            common = min(len(d_rmse), len(b_rmse))
            if common < 8:
                continue
            d = np.array(d_rmse[:common]); b = np.array(b_rmse[:common])
            ratio = b / (d + 1e-12)
            try:
                p = stats.wilcoxon(d, b, zero_method="wilcox").pvalue
            except ValueError:
                p = np.nan
            print(f"{anom:<14}{base:<9}{common:>4}{np.median(ratio):>10.2f}"
                  f"{p:>10.4g}")
    return arr


if __name__ == "__main__":
    csv_path = run()
    summarize(csv_path)
