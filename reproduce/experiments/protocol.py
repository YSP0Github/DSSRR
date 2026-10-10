"""
统一公平评估协议 (R0/R1)

所有对比方法使用同一污染输入、同一真值基准、同一指标函数。
禁止再出现"传统方法对污染数据评估、DSSRR 对原始真值评估"的不一致。

指标口径：
- RMSE/MAE/SNR/能量比：在异常核心区间 [start, end] 上计算（对真值）。
- PSD 相关指标（psd_cosine / log_spec_dist）：在"异常核心 + 两侧扩展窗"上，
  用多锥法平滑估计后计算（短异常段直接 FFT 的 PSD 方差过大，无意义）。
- boundary_jump：归一化边界跳变 = max(两端 |修复值 - 真值邻点|) / 邻域稳健尺度。
- autocorr_decay_diff：修复段与真值段归一化自相关首过零位置的相对差异。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import windows


def next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def multitaper_psd(x: np.ndarray, sr: float, nw: float = 4.0) -> np.ndarray:
    """多锥法 PSD（单边，返回 rfft 长度）。输入过短返回 None。"""
    x = np.asarray(x, dtype=float)
    if x.size < 8:
        return None
    x = x - np.mean(x)
    N = len(x)
    max_nw = max(0.75, min(8.0, N / 2.0 - 1e-6))
    NW = np.clip(nw, 0.75, max_nw)
    K = max(1, min(int(2 * NW - 1), 15))
    tapers = windows.dpss(N, NW, Kmax=K)
    n_fft = next_pow2(2 * N)
    n_rfft = n_fft // 2 + 1
    acc = np.zeros(n_rfft, dtype=np.float64)
    for k in range(K):
        acc += np.abs(np.fft.rfft(x * tapers[k], n=n_fft)) ** 2
    return acc / K


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _autocorr_first_zero(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return float(len(x))
    x = x - np.mean(x)
    s = np.std(x)
    if s < 1e-12:
        return float(len(x))
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-15)
    zc = np.where(ac < 0)[0]
    return float(zc[0]) if len(zc) else float(len(ac))


def evaluate_fair(
    truth: np.ndarray,
    repaired: np.ndarray,
    start: int,
    end: int,
    sr: float,
    spectral_pad: int | None = None,
) -> dict:
    """统一评估函数。

    Parameters
    ----------
    truth     : 注入异常前的真值波形（1-D）
    repaired  : 修复后的波形（1-D，与 truth 同长）
    start,end : 异常区间（闭区间样本索引）
    sr        : 采样率
    spectral_pad : 频谱指标用的扩展样本数；None 时自动取
                   max(128, 2*(end-start+1))
    """
    truth = np.asarray(truth, dtype=float)
    repaired = np.asarray(repaired, dtype=float)
    n = len(truth)
    start, end = int(start), int(end)

    seg_t = truth[start:end + 1]
    seg_r = repaired[start:end + 1]
    err = seg_r - seg_t

    metrics: dict = {}
    metrics["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    metrics["mae"] = float(np.mean(np.abs(err)))
    sig_pow = float(np.mean(seg_t ** 2))
    err_pow = float(np.mean(err ** 2)) + 1e-30
    metrics["snr"] = (
        float(10 * np.log10(sig_pow / err_pow)) if sig_pow > 0 else np.nan
    )
    metrics["energy_ratio"] = float(
        (np.var(seg_r) + 1e-30) / (np.var(seg_t) + 1e-30)
    )

    # 频谱指标：异常扩展窗（对短异常有意义的 PSD 比较）
    if spectral_pad is None:
        spectral_pad = max(128, 2 * (end - start + 1))
    lo = max(0, start - spectral_pad)
    hi = min(n - 1, end + spectral_pad)
    seg_t_wide = truth[lo:hi + 1]
    seg_r_wide = repaired[lo:hi + 1]
    psd_t = multitaper_psd(seg_t_wide, sr)
    psd_r = multitaper_psd(seg_r_wide, sr)
    if psd_t is not None and psd_r is not None:
        metrics["psd_cosine"] = _cosine_sim(psd_t, psd_r)
        metrics["log_spec_dist"] = float(np.sqrt(np.mean(
            (np.log10(psd_t + 1e-30) - np.log10(psd_r + 1e-30)) ** 2
        )))
    else:
        metrics["psd_cosine"] = np.nan
        metrics["log_spec_dist"] = np.nan

    # 边界跳变（归一化；尺度加下限防止恒值段/量化段退化）
    jumps = []
    if start > 0:
        jumps.append(abs(float(repaired[start] - truth[start - 1])))
    if end + 1 < n:
        jumps.append(abs(float(repaired[end] - truth[end + 1])))
    ctx = truth[max(0, start - 64):min(n, end + 65)]
    seg_std = float(np.std(truth[lo:hi + 1])) + 1e-12
    if len(ctx) > 0:
        med = np.median(ctx)
        local_scale = 1.4826 * np.median(np.abs(ctx - med))
    else:
        local_scale = 0.0
    scale = max(local_scale, 0.05 * seg_std)
    metrics["boundary_jump"] = float(max(jumps) / scale) if jumps else 0.0

    # 自相关衰减一致性（修复段 vs 真值段）
    zc_t = _autocorr_first_zero(seg_t)
    zc_r = _autocorr_first_zero(seg_r)
    metrics["autocorr_decay_diff"] = (
        float(abs(zc_r - zc_t) / max(zc_t, 1e-12)) if zc_t > 1e-12 else np.nan
    )

    metrics["valid"] = all(
        np.isfinite(metrics[k])
        for k in ("rmse", "mae", "snr", "psd_cosine", "log_spec_dist",
                  "boundary_jump", "autocorr_decay_diff", "energy_ratio")
    )
    return metrics


def mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))
