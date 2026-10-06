# -*- coding: utf-8 -*-
"""DSSRR 示例 03 —— 在**真实 Apollo 月震数据**上修复一个长缺口。

与示例 02 的合成数据不同，这里用的是随仓库附带的真实存档数据
（``examples/data/`` 下的三个 Apollo 12/15/16 长周期垂向道，1976-01-13，
采样率 6.625 Hz）。做法是「先扣后验」：

1. 读入一段真实记录，作为**真值**。
2. 人为挖掉一段（置零），假装它丢失了。
3. 让 DSSRR 只看缺口两侧的健康数据来重建。
4. 把重建结果与真值对比（标准差 / 均值 / 对数 PSD 距离），并出图。

这样才能公平地量化「修复质量」，因为缺口内的真实波形是已知的。

运行::

    # 用仓库自带数据（默认）
    python examples/03_real_apollo_mseed.py

    # 用自己的数据目录
    python examples/03_real_apollo_mseed.py --data-dir /path/to/mseed --sr 6.625

参数：``--data-dir`` ``--station`` ``--sr`` ``--gap-sec`` ``--start-sec``
``--ref-sec`` ``--no-plot``。

依赖：numpy、scipy、obspy（读取 MiniSEED）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import welch

from dssrr import DSSRR, auto_reference_length

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"


def find_mseed(data_dir, station=None):
    """在 data_dir 中找第一个 MiniSEED 文件（可按台站名过滤）。"""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"[error] data directory not found: {data_dir}")
    files = sorted(data_dir.glob("*.mseed"))
    if station:
        files = [f for f in files if f".{station}." in f.name] or files
    if not files:
        raise SystemExit(
            f"[error] no .mseed files in {data_dir}. "
            "Download Apollo PSE data or point --data-dir elsewhere.")
    return files[0]


def welch_psd(x, sr):
    f, p = welch(x, fs=sr, nperseg=min(len(x), 4096),
                 noverlap=2048, scaling="density")
    return f, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="含 .mseed 的目录（缺省 examples/data）")
    ap.add_argument("--station", default="S12",
                    help="台站代码（S12 / S15 / S16），用于挑文件")
    ap.add_argument("--sr", type=float, default=6.625, help="采样率 Hz")
    ap.add_argument("--gap-sec", type=float, default=600.0, help="缺口长度（秒）")
    ap.add_argument("--start-sec", type=float, default=3600.0,
                    help="缺口起点（秒，相对文件开头）")
    ap.add_argument("--ref-sec", type=float, default=None,
                    help="参考窗长度（秒/侧）；缺省用 auto_reference_length")
    ap.add_argument("--no-plot", action="store_true", help="只打印统计，不出图")
    args = ap.parse_args(argv)

    sr = args.sr
    try:
        from obspy import read
    except ImportError:
        raise SystemExit(
            "[error] obspy is required for this example: pip install obspy")

    # ---------------------------------------------------------------
    # 1) 读取真实记录
    # ---------------------------------------------------------------
    path = find_mseed(args.data_dir, args.station)
    st = read(str(path))
    tr = st[0]
    data = tr.data.astype(float)
    print(f"file       : {path.name}")
    print(f"trace      : {tr.id}  {len(data)} samples @ {sr} Hz  "
          f"({len(data) / sr / 60:.1f} min)")
    print(f"counts     : min={data.min():.0f}  max={data.max():.0f}  "
          f"mean={data.mean():.1f}  std={data.std():.2f}")

    # ---------------------------------------------------------------
    # 2) 挖一个长缺口
    # ---------------------------------------------------------------
    start = int(args.start_sec * sr)
    end = start + int(args.gap_sec * sr) - 1
    if end >= len(data):
        raise SystemExit(
            f"[error] gap [{start}, {end}] exceeds trace length {len(data)}; "
            "lower --start-sec or --gap-sec")
    corrupted = data.copy()
    corrupted[start:end + 1] = 0.0
    print(f"injected   : samples [{start}, {end}]  "
          f"({args.gap_sec:.0f} s)")

    # ---------------------------------------------------------------
    # 3) 修复（把真值藏起来，只给 DSSRR 带缺口的数据）
    # ---------------------------------------------------------------
    ref_sec = args.ref_sec or auto_reference_length(end - start + 1, sr)
    model = DSSRR(sr=sr, reference_sec=ref_sec, seed=42)
    repaired, report = model.repair_with_report(
        corrupted, start, end,
        reference_before_sec=ref_sec,
        reference_after_sec=ref_sec,
        random_seed=42,
        # Apollo 存档是 10-bit 整数计数，勾选整数化以保持与原始数据同域
        quantize=True,
    )

    # ---------------------------------------------------------------
    # 4) 与真值对比
    # ---------------------------------------------------------------
    truth = data[start:end + 1]
    est = repaired[start:end + 1]
    _, pt = welch_psd(truth, sr)
    _, pe = welch_psd(est, sr)
    l2 = float(np.sqrt(np.mean((np.log(pe + 1e-30) - np.log(pt + 1e-30)) ** 2)))

    print("\n--- repair quality (vs held-out truth) ---")
    print(f"reference  : {ref_sec:.1f} s per side")
    print(f"mean       : true={truth.mean():8.2f}  repaired={est.mean():8.2f}")
    print(f"std        : true={truth.std():8.3f}  repaired={est.std():8.3f}")
    print(f"log-PSD L2 : {l2:.3f}   (smaller = closer spectrum)")
    print(f"changed    : {report['changed_samples']} / {end - start + 1} samples")
    v = report["verification"]
    print(f"energy_ratio       : {v['energy_ratio']:.4f}   (1.0 = matched)")
    print(f"envelope_continuity: {v['envelope_continuity']:.4f}   (lower better)")
    print(f"dc_continuity      : {v['dc_continuity']:.4f}   (lower better)")
    print(f"spectral_continuity: {v['spectral_continuity']:.4f}   "
          "(raw-periodogram cosine similarity; see note in 01_quickstart.py)")
    print(f"verifier verdict   : passed={v['passed']}  ({v['details']})")

    if args.no_plot:
        return

    # ---------------------------------------------------------------
    # 5) 出图
    # ---------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure.")
        return

    t = np.arange(len(data)) / sr / 60.0     # 分钟
    lo = max(0, start - int(600 * sr))
    hi = min(len(data), end + int(600 * sr))
    sl = slice(lo, hi)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(t[sl], data[sl], color="#2c3e50", lw=0.7, label="true (held out)")
    axes[0].plot(t[sl], repaired[sl], color="#e74c3c", lw=0.7, alpha=0.8,
                 label="DSSRR repaired")
    axes[0].axvspan(t[start], t[end], color="#ffd54f", alpha=0.25, zorder=0)
    axes[0].set_ylabel("counts")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].set_title(f"{tr.id}  gap {args.gap_sec:.0f} s @ "
                      f"{args.start_sec:.0f} s   (log-PSD L2 = {l2:.3f})")

    f1, p1 = welch_psd(truth, sr)
    f2, p2 = welch_psd(est, sr)
    axes[1].loglog(f1, p1, color="#2c3e50", lw=1.4, label="true")
    axes[1].loglog(f2, p2, color="#e74c3c", lw=1.2, alpha=0.85, label="repaired")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = HERE / "03_real_apollo_mseed.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved to: {out}")


if __name__ == "__main__":
    main()
