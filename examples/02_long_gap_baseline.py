# -*- coding: utf-8 -*-
"""DSSRR 示例 02 —— 长缺口修复：DSSRR vs 朴素基线（合成数据）。

长缺口（这里用 10 分钟）是 DSSRR 的主场：简单方法在长缺口上会迅速失效，
DSSRR 仍能保住功率谱形状。脚本把三种方法并排比较：

- **Linear**  —— 两端点直线插值（缺口一长就变成一条平线，谱全塌）。
- **FFT**     —— 从两侧健康段估谱、白相位合成（谱形对，但接缝与能量标定弱）。
- **DSSRR**   —— 双侧参考谱融合 + 基线对齐 + 边界平滑（本仓库的方法）。

输出：终端打印每种方法修复段的标准差与「对数 PSD 相对真值的 L2 距离」，
并把对比图保存为 ``02_long_gap_baseline.png``（matplotlib 缺失时仅打印统计）。

运行::

    python examples/02_long_gap_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import welch

from dssrr import DSSRR, auto_reference_length


def linear_repair(corrupted, s, e):
    """两端点直线插值：在缺口两端之间拉一条直线。"""
    n = len(corrupted)
    x = np.arange(n)
    known = np.ones(n, dtype=bool)
    known[s:e + 1] = False
    out = corrupted.copy()
    out[s:e + 1] = np.interp(x[s:e + 1], x[known], corrupted[known])
    return out


def fft_extrapolate(corrupted, s, e, sr, n_ref=1200, seed=42):
    """从健康参考段估计幅度谱，用随机相位做逆变换合成缺口段。"""
    rng = np.random.default_rng(seed)
    length = e - s + 1
    ref = np.concatenate([corrupted[max(0, s - n_ref):s],
                          corrupted[e + 1:e + 1 + n_ref]])
    ref_mean = float(np.mean(ref))
    ref = ref - ref_mean
    nfft = 1 << (length * 2 - 1).bit_length()
    amp = np.abs(np.fft.rfft(ref, n=nfft))
    phase = rng.uniform(-np.pi, np.pi, len(amp))
    synth = np.fft.irfft(amp * np.exp(1j * phase), n=nfft)[:length]
    scale = ref.std() / (synth.std() + 1e-12)
    out = corrupted.copy()
    out[s:e + 1] = synth * scale + ref_mean
    return out


def welch_psd(x, sr):
    """Welch 功率谱密度估计（统一参数，便于跨方法比较）。"""
    f, p = welch(x, fs=sr, nperseg=min(len(x), 4096),
                 noverlap=2048, scaling="density")
    return f, p


def log_psd_l2(a, b):
    """两个信号对数 PSD 的 RMS 距离（越小越接近）。"""
    _, pa = welch_psd(a, 6.625)
    _, pb = welch_psd(b, 6.625)
    return float(np.sqrt(np.mean(
        (np.log(pa + 1e-30) - np.log(pb + 1e-30)) ** 2)))


def main():
    sr = 6.625
    duration_sec = 3600
    t = np.arange(int(duration_sec * sr)) / sr

    # 月震式有色背景噪声（无强确定性单频，因为 DSSRR 保统计量而非相位）
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, len(t))
    colored = np.zeros_like(white)
    for i in range(1, len(white)):
        colored[i] = 0.98 * colored[i - 1] + 0.02 * white[i]
    data = 450.0 + 3.0 * colored

    # 挖一个 10 分钟的平缺口
    s = int(1700 * sr)
    e = s + int(600 * sr) - 1
    corrupted = data.copy()
    corrupted[s:e + 1] = 0.0

    ref_sec = auto_reference_length(e - s + 1, sr)

    lin = linear_repair(corrupted, s, e)
    fft = fft_extrapolate(corrupted, s, e, sr)
    dssrr = DSSRR(sr=sr, reference_sec=ref_sec, seed=42)
    rep = dssrr.repair(corrupted, s, e)

    # ---- 修复段统计 ----
    true_seg = data[s:e + 1]
    print(f"{'method':<10} {'std':>8}  log-PSD-L2 vs true")
    for name, seg in [("True", true_seg), ("Linear", lin[s:e + 1]),
                      ("FFT", fft[s:e + 1]), ("DSSRR", rep[s:e + 1])]:
        print(f"{name:<10} {np.std(seg):8.3f}  {log_psd_l2(seg, true_seg):10.3f}")

    # ---- 出图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure.")
        return

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1.3]})
    colors = {"Linear": "#95a5a6", "FFT": "#3498db", "DSSRR": "#16a085"}

    for ax, (name, sig) in zip(axes[:3],
                               [("Linear", lin), ("FFT", fft), ("DSSRR", rep)]):
        ax.axvspan(t[s], t[e], color="#e74c3c", alpha=0.12, zorder=0)
        ax.plot(t, data, color="#2c3e50", lw=0.6, alpha=0.45,
                label="True (hidden)")
        ax.plot(t, sig, color=colors[name], lw=0.8, label=f"{name} repaired")
        ax.set_ylabel(name, fontsize=11)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title(
        "10-minute dropout repair: linear vs FFT extrapolation vs DSSRR",
        fontsize=12)

    axp = axes[3]
    ft, pt = welch_psd(true_seg, sr)
    axp.plot(ft, pt, color="#2c3e50", lw=1.5, label="True")
    for name, sig in [("Linear", lin), ("FFT", fft), ("DSSRR", rep)]:
        _, p = welch_psd(sig[s:e + 1], sr)
        axp.plot(ft, p, color=colors[name], lw=1.2, label=name)
    axp.set_xlabel("Frequency (Hz)")
    axp.set_ylabel("PSD")
    axp.set_xscale("log")
    axp.set_yscale("log")
    axp.legend(frameon=False, fontsize=9)
    axp.spines["top"].set_visible(False)
    axp.spines["right"].set_visible(False)
    axp.set_xlim(0.005, 3)

    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "02_long_gap_baseline.png"
    fig.savefig(out, dpi=150)
    print(f"Figure saved to: {out}")


if __name__ == "__main__":
    main()
