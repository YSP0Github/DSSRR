# -*- coding: utf-8 -*-
"""
E8 (Task 6): Transfer validation on Earth seismic data.

- One full day of IU.ANMO.00.BHZ (Albuquerque, NM; 40 Hz) downloaded from IRIS.
- Preprocess: merge, linear detrend, 4 Hz low-pass, decimate to 10 Hz
  (microseism band of interest is < 1 Hz).
- Automatically select quiet, stationary windows; inject flat dropouts of
  1, 5, 10, 30 min; repair with the same six methods used in E-RealInj.
- Report the full statistical-fidelity suite; demonstrate transferability.

Outputs (results_earth/):
  earth_instances.csv, earth_summary.csv, fig11_earth_transfer.png
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from obspy import read
from scipy import signal as sp_signal

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 统一图件样式模块（方法色号 / 字号 / 印刷尺寸），与本脚本同目录
SRL_STYLE_DIR = r"G:\SeisY\docs\dssrr_paper\submission_srl\image_code"
if SRL_STYLE_DIR not in sys.path:
    sys.path.insert(0, SRL_STYLE_DIR)
# 实验公共库（指标 / 结果读写）仍位于 experiments 目录，本脚本移入 image_code 后需补路径
EXPERIMENTS_DIR = r"G:\SeisY\docs\dssrr_paper\experiments"
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)
from srl_style import (  # noqa: E402
    apply_srl_style,
    DOUBLE_COL_WIDTH,
    METHOD_COLORS,
    PANEL_LABEL_KW,
    SAVE_KW,
    ANOMALY_FILL,
    ANOMALY_ALPHA,
    mlabel,
)

from seisy.core.anomaly_repair.comparison_methods import (  # noqa: E402
    LinearInterpolation,
    CubicSplineInterpolation,
    SSARepair,
    UNetRepair,
)
from seisy.core.anomaly_repair.reference_spectrum import (  # noqa: E402
    ReferenceSpectrumReplacer,
)
from _metrics_lib import all_metrics, psd_welch  # noqa: E402
import results_io as ri  # noqa: E402

RAW = r"G:\SeisY\docs\dssrr_paper\experiments\results_earth\raw\IU.ANMO.00.BHZ.day.mseed"
OUT_DIR = r"G:\SeisY\docs\dssrr_paper\experiments\results_earth"
TARGET_FS = 10.0
SEED = 7

# 结果目录里的案例名与台站代码（与 results_E2r_full_v5 的 <event> 层级对应）
CASE = "IU_ANMO"
STATION = "ANMO"

# gap length (s) -> number of independent trials
GAP_LENGTHS = [60, 300, 600, 1800]
GAP_REPS = [6, 5, 4, 3]

METHOD_ORDER = ["Linear", "CubicSpline", "FFT", "SSA", "UNet", "DSSRR"]


# ----------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------

class FFTInterpolation:
    def __init__(self, fs):
        self.fs = fs
        self.edge = int(15 * fs)

    def repair(self, data, start, end):
        data = np.asarray(data, dtype=float)
        repaired = data.copy()
        n = len(data)
        known = np.concatenate([data[:start], data[end + 1:]])
        fft_known = np.fft.rfft(known)
        n_low = max(1, len(fft_known) // 10)
        fft_smooth = np.zeros_like(fft_known)
        fft_smooth[:n_low] = fft_known[:n_low]
        smooth_full = np.fft.irfft(fft_smooth, n=n)
        lm = np.mean(data[max(0, start - self.edge):start])
        rm = np.mean(data[end + 1:min(n, end + 1 + self.edge)])
        target = (lm + rm) / 2
        dc = target - np.mean(smooth_full[start:end + 1])
        repaired[start:end + 1] = smooth_full[start:end + 1] + dc
        return repaired


def find_adaptive_boundary(data, start, end, sr):
    win_sec = 300
    win = int(win_sec * sr)
    left_end = start - int(1.0 * sr)
    left_start = max(0, left_end - win)
    left_seg = data[left_start:left_end]
    right_start = end + 1 + int(1.0 * sr)
    right_end = min(len(data), right_start + win)
    right_seg = data[right_start:right_end]
    c_l = np.polyfit(np.arange(len(left_seg)), left_seg, 1)
    c_r = np.polyfit(np.arange(len(right_seg)), right_seg, 1)
    dl = np.abs(left_seg - np.polyval(c_l, np.arange(len(left_seg))))
    dr = np.abs(right_seg - np.polyval(c_r, np.arange(len(right_seg))))
    s_adapt, e_adapt = start, end
    for i in range(start - 1, left_start, -1):
        idx = i - left_start
        if idx < len(dl) and dl[idx] < np.std(left_seg) * 0.5:
            s_adapt = i
            break
    for i in range(end + 1, right_end):
        idx = i - right_start
        if idx < len(dr) and dr[idx] < np.std(right_seg) * 0.5:
            e_adapt = i
            break
    return s_adapt, e_adapt


def repair_dssrr(data, start, end, fs, seed=SEED):
    replacer = ReferenceSpectrumReplacer(fs)
    s_adapt, e_adapt = find_adaptive_boundary(data, start, end, fs)
    anom_sec = (e_adapt - s_adapt + 1) / fs
    ref_len = float(np.clip(anom_sec / 2.0, 120.0, 600.0))
    seg, _ = replacer.replace(
        data.copy(), s_adapt, e_adapt,
        reference_before_sec=ref_len,
        reference_after_sec=ref_len,
        reference_gap_sec=1.0,
        random_seed=seed,
    )
    repaired = data.copy()
    repaired[start:end + 1] = seg[start:end + 1]
    return repaired


# ----------------------------------------------------------------------
# Preprocessing & clean-window selection
# ----------------------------------------------------------------------

def load_preprocessed():
    st = read(RAW)
    st.merge(fill_value="interpolate")
    tr = st[0]
    tr.detrend("linear")
    tr.taper(0.01)
    # 4 Hz low-pass then decimate 40 -> 10 (factor 4)
    tr.filter("lowpass", freq=4.0, corners=8, zerophase=True)
    tr.decimate(4, no_filter=True)
    x = tr.data.astype(float)
    x = x - np.mean(x)
    return x, tr.stats.sampling_rate, tr.stats.starttime


def rms_profile(x, fs, win_sec=100.0, step_sec=20.0):
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    centers = np.arange(win // 2, len(x) - win // 2, step)
    rms = np.array([np.sqrt(np.mean(x[c - win // 2:c + win // 2] ** 2))
                    for c in centers])
    return centers, rms


def select_trials(x, fs):
    centers, rms = rms_profile(x, fs)
    quiet_thresh = np.percentile(rms, 45)
    trials = []
    for gap_sec, n_reps in zip(GAP_LENGTHS, GAP_REPS):
        ref_sec = float(np.clip(gap_sec / 2.0, 120.0, 600.0))
        half_sec = gap_sec / 2.0 + ref_sec + 60.0
        half = int(half_sec * fs)
        gap_pts = int(gap_sec * fs)
        # local maximum rms within the required span for each candidate
        cand = []
        for ci, c in enumerate(centers):
            lo = np.searchsorted(centers, c - half)
            hi = np.searchsorted(centers, c + half)
            local_max = np.max(rms[lo:hi + 1]) if hi > lo else rms[ci]
            if c - half >= 0 and c + half < len(x) and local_max <= quiet_thresh:
                cand.append((local_max, c))
        cand.sort(key=lambda z: z[0])
        chosen = []
        for _, c in cand:
            s = c - gap_pts // 2
            e = s + gap_pts - 1
            if all(abs(c - c0) >= 2 * half + int(20 * fs) for c0 in chosen):
                chosen.append(c)
                trials.append((gap_sec, s, e))
            if len(chosen) >= n_reps:
                break
        if len(chosen) < n_reps:
            print(f"  WARNING: gap {gap_sec}s requested {n_reps}, found {len(chosen)}")
    return trials


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    import gc
    import time as _t

    os.makedirs(OUT_DIR, exist_ok=True)
    x, fs, starttime = load_preprocessed()
    print(f"preprocessed: n={len(x)} fs={fs} hours={len(x)/fs/3600:.2f}",
          flush=True)

    methods = {
        "Linear": LinearInterpolation(),
        "CubicSpline": CubicSplineInterpolation(),
        "FFT": FFTInterpolation(fs),
        "SSA": SSARepair(max_iter=30),
        "UNet": UNetRepair(
            patch_length=1536, n_epochs=10,
            model_path=r"G:\SeisY\docs\dssrr_paper\experiments\models\unet_synth.pt",
        ),
        "DSSRR": None,
    }

    trials = select_trials(x, fs)
    print(f"selected {len(trials)} trials", flush=True)

    writer = ri.ResultsWriter(OUT_DIR, experiment="earth")

    # ---- resume: 以案例产物是否齐全为准 ----
    inst_csv = os.path.join(OUT_DIR, "earth_instances.csv")
    rows = []
    metric_cols = ["rmse_anom", "wasserstein", "acf_l2", "envelope",
                   "spec_ent", "psd_cos", "runtime_s"]
    rep_cache = {}

    def _case_complete(leaf):
        need = ["before.png", "corrupted.mseed"]
        need += [f"after_{m}.png" for m in METHOD_ORDER]
        need += [f"repaired_{m}.mseed" for m in METHOD_ORDER]
        return all(os.path.isfile(os.path.join(leaf, f)) for f in need)

    # 同一 gap 长度有多次重复（GAP_REPS），必须用唯一层级名，
    # 否则后一次会覆盖前一次，stats_summary.csv 里的多行将指向同一套文件。
    _rep_idx = {}
    for gap_sec, s, e in sorted(trials):
        _rep_idx[gap_sec] = _rep_idx.get(gap_sec, 0) + 1
        level = f"{gap_sec}s_r{_rep_idx[gap_sec]:02d}"
        leaf = os.path.join(OUT_DIR, CASE, "dropout", level)
        # 明细表由 ResultsWriter._flush_case / rewrite_detail 写在**事件级**
        # （results_earth/IU_ANMO/stats_detail.csv），不是每个案例目录下，
        # 且靠 level 列区分试次。原先写成 leaf/stats_detail.csv 会导致
        # resume 分支必然 FileNotFoundError。
        detail_csv = os.path.join(OUT_DIR, CASE, "stats_detail.csv")

        # 已完整产出则直接复用，不重算
        if _case_complete(leaf):
            d = pd.read_csv(detail_csv)
            d = d[d["level"] == level]
            print(f"gap {gap_sec:5d}s [{s:7d},{e:7d}] 已存在，跳过", flush=True)
            for rec in d.to_dict("records"):
                writer.rows.append(rec)
                rows.append({
                    "gap_sec": gap_sec, "s": s, "e": e,
                    "method": rec["method"],
                    "rmse_anom": rec["rmse_anom"],
                    "wasserstein": rec["wasserstein"],
                    # 关键：非 resume 分支的 acf_l2 来自 all_metrics(..., acf_max_lag=30*fs)，
                    # 即"30 s 滞后"口径；而 stats_detail.csv 里的 acf_l2 是 results_io
                    # 自己那套（滞后更长）的口径，两者数值明显不同。30 s 口径存在
                    # earth_acf_l2_30s 列里，必须取它，否则同一份数据在"重跑"与
                    # "resume"两条路径下会得到不同的图与不同的统计表。
                    "acf_l2": rec.get("earth_acf_l2_30s", rec["acf_l2"]),
                    "envelope": rec.get("envelope_dist", np.nan),
                    "spec_ent": rec.get("spec_entropy_err", np.nan),
                    "psd_cos": rec.get("psd_cos", np.nan),
                    "runtime_s": rec.get("runtime_s", np.nan),
                })
            # 复用分支同样要回填代表案例，否则 make_figure 拿不到 rep_cache
            # （全部案例都命中 resume 时图会直接 KeyError 崩掉）。
            if gap_sec == 600:
                rep_cache["truth"] = x[s:e + 1].copy()
                rep_cache["s"], rep_cache["e"] = s, e
                for mname in METHOD_ORDER:
                    p = os.path.join(leaf, f"repaired_{mname}.mseed")
                    if os.path.isfile(p):
                        rep_cache[mname] = read(p)[0].data.astype(float)[s:e + 1].copy()
            continue

        truth = x[s:e + 1]
        corrupted = x.copy()
        corrupted[s:e + 1] = 0.0
        print(f"gap {gap_sec:5d}s [{s:7d},{e:7d}]", flush=True)

        repaired_by_method = {}
        runtimes = {}
        legacy_metrics = {}
        for mname in METHOD_ORDER:
            try:
                t0 = _t.perf_counter()
                if mname == "DSSRR":
                    repaired = repair_dssrr(corrupted, s, e, fs)
                else:
                    repaired = methods[mname].repair(corrupted, s, e)
                runtimes[mname] = _t.perf_counter() - t0
                m = all_metrics(truth, repaired[s:e + 1], fs,
                                acf_max_lag=int(30 * fs))
                legacy_metrics[mname] = m
                repaired_by_method[mname] = repaired
                print(f"    {mname:11s} W={m['wasserstein']:.3f} "
                      f"ACF={m['acf_l2']:.3f} RMSE={m['rmse_anom']:.3f} "
                      f"({runtimes[mname]:.1f}s)", flush=True)
            except Exception:
                import traceback
                traceback.print_exc()

        case_rows = writer.save_case(
            event=CASE, station=STATION, scenario="earth",
            anom_type="dropout", level=level,
            truth=x, corrupted=corrupted, intervals=[(int(s), int(e))],
            repaired_by_method=repaired_by_method, method_order=METHOD_ORDER,
            starttime=starttime, sr=fs,
            network="IU", location="00", channel="BHZ",
            meta_extra={"gap_sec": gap_sec, "s": int(s), "e": int(e)},
        )

        for row in case_rows:
            mname = row["method"]
            m = legacy_metrics.get(mname, {})
            row["runtime_s"] = runtimes.get(mname, np.nan)
            row["earth_wasserstein_30s"] = m.get("wasserstein", np.nan)
            row["earth_acf_l2_30s"] = m.get("acf_l2", np.nan)
            row["earth_envelope"] = m.get("envelope", np.nan)
            row["earth_spec_ent"] = m.get("spec_ent", np.nan)
            rows.append({
                "gap_sec": gap_sec, "s": s, "e": e, "method": mname,
                "rmse_anom": m.get("rmse_anom", np.nan),
                "wasserstein": m.get("wasserstein", np.nan),
                "acf_l2": m.get("acf_l2", np.nan),
                "envelope": m.get("envelope", np.nan),
                "spec_ent": m.get("spec_ent", np.nan),
                "psd_cos": m.get("psd_cos", np.nan),
                "runtime_s": runtimes.get(mname, np.nan),
            })

        # 代表案例（10 min）的修复段，供 Fig.11 时域/PSD 面板使用
        # 注意：s/e 与各方法数组必须取自**同一次**试次，否则面板 (a) 会错位。
        # 这里统一取最后一次 600 s 试次（与论文版 Fig.11 一致）。
        if gap_sec == 600:
            rep_cache["truth"] = truth.copy()
            rep_cache["s"], rep_cache["e"] = s, e
            for mname, rep in repaired_by_method.items():
                rep_cache[mname] = rep[s:e + 1].copy()

        # 增量落盘
        pd.DataFrame(rows).to_csv(inst_csv, index=False, encoding="utf-8-sig")
        writer.write_summary()
        writer.rewrite_detail()
        gc.collect()

    df = pd.DataFrame(rows)
    # 全部命中 resume 时循环内的增量落盘一次都不会执行，这里补写一次，
    # 保证 earth_instances.csv 与 earth_summary.csv 始终同源。
    df.to_csv(inst_csv, index=False, encoding="utf-8-sig")
    summary = df.groupby(["gap_sec", "method"])[metric_cols].mean().reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "earth_summary.csv"), index=False,
                   encoding="utf-8-sig")
    print("\n===== summary (Wasserstein / ACF L2) =====", flush=True)
    piv_w = summary.pivot(index="method", columns="gap_sec", values="wasserstein")
    piv_a = summary.pivot(index="method", columns="gap_sec", values="acf_l2")
    print("Wasserstein:\n", piv_w.round(3))
    print("ACF L2:\n", piv_a.round(3))

    writer.write_summary()
    writer.rewrite_detail()
    writer.write_provenance(script_path=os.path.abspath(__file__))

    make_figure(x, fs, summary, rep_cache, inst=df)
    return df, summary, rep_cache


def make_figure(x, fs, summary, rep, inst=None):
    """Figure 8：Earth transfer validation。

    与全篇其余 9 张图共用 srl_style：同一套方法色号、同一套子图标题样式
    （PANEL_LABEL_KW：9pt 粗体、左对齐）、无网格、AGU 双栏印刷宽度 7.09 in、
    1200 dpi 保存。此前本图使用 seaborn deep 调色板 + 网格 + 14x10 in / 200 dpi，
    与其余图件明显不一致。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    apply_srl_style()

    MARKERS = {"Linear": "o", "CubicSpline": "D", "FFT": "^",
               "SSA": "s", "UNet": "v", "DSSRR": "*"}
    MSIZE = {"Linear": 4.0, "CubicSpline": 3.6, "FFT": 4.4,
             "SSA": 4.0, "UNet": 4.0, "DSSRR": 6.0}
    LW = {"DSSRR": 1.6}
    PANELS = ["Linear", "CubicSpline", "FFT", "SSA", "UNet", "DSSRR"]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.0))

    # ---- (a) 代表案例（10 min 缺口）的波形放大 ----
    # 采用缺口内部的 2-min 短窗：10 min 全长窗下 6 条修复曲线会挤成一条色带，
    # 浅灰的 Linear 平坦桥完全看不出来；短窗内各方法的差别（平坦桥 vs 噪声纹理）
    # 才真正可辨。
    ax = axes[0, 0]
    s, e = int(rep["s"]), int(rep["e"])
    zc = (s + e) // 2
    half = int(60 * fs)                       # ±60 s -> 2 min 窗，完全落在缺口内
    z0, z1 = zc - half, zc + half
    t = (np.arange(z0, z1) - s) / fs          # 相对缺口起点的秒数
    ax.axvspan(t[0], t[-1], color=ANOMALY_FILL, alpha=ANOMALY_ALPHA, lw=0, zorder=0)
    ax.plot(t, x[z0:z1], color=METHOD_COLORS["GroundTruth"],
            lw=0.8, ls="--", label="Truth", zorder=5)
    for mname in PANELS:
        if mname in rep:
            seg = rep[mname]
            ax.plot(t, seg[z0 - s:z1 - s], color=METHOD_COLORS[mname],
                    lw=LW.get(mname, 0.9), alpha=0.95, label=mlabel(mname),
                    zorder=(4 if mname == "DSSRR" else 3))
    ax.set_xlabel("Time relative to gap start (s)")
    ax.set_ylabel("Amplitude (DU)")
    ax.set_title("(a) Waveform zoom inside the 10-min dropout", **PANEL_LABEL_KW)

    # ---- (b) 修复段的 PSD ----
    ax = axes[0, 1]
    f_t, p_t = psd_welch(rep["truth"], fs, nperseg=1024)
    ax.semilogy(f_t, p_t, color=METHOD_COLORS["GroundTruth"], lw=1.3, ls="--",
                label="Truth", zorder=5)
    for mname in PANELS:
        if mname in rep:
            f, p = psd_welch(rep[mname], fs, nperseg=1024)
            ax.semilogy(f, p, color=METHOD_COLORS[mname],
                        lw=LW.get(mname, 0.9), label=mlabel(mname),
                        zorder=(4 if mname == "DSSRR" else 2))
    ax.set_xlim(0.02, 4.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD (DU$^2$/Hz)")
    ax.set_title("(b) PSD of repaired segment", **PANEL_LABEL_KW)

    # ---- (c) Wasserstein / (d) ACF L2 vs 缺口长度 ----
    # 误差棒（2026-10-07，回应审稿意见 M2）：每个长度只有 3--6 个注入窗，
    # 取 **±1 标准差**（跨该长度的所有窗）。逐窗数值见 results_earth/earth_instances.csv。
    for ax, metric, title in [
        (axes[1, 0], "wasserstein", "(c) Amplitude fidelity vs gap length"),
        (axes[1, 1], "acf_l2", "(d) Temporal fidelity vs gap length"),
    ]:
        for mname in PANELS:
            d = summary[summary["method"] == mname].sort_values("gap_sec")
            sd = None
            if inst is not None:
                sdi = (inst[inst["method"] == mname]
                       .groupby("gap_sec")[metric].std().sort_index())
                sd = sdi.reindex(d["gap_sec"]).to_numpy()
            ax.errorbar(d["gap_sec"] / 60.0, d[metric], yerr=sd,
                        fmt="none", ecolor=METHOD_COLORS[mname], elinewidth=0.7,
                        capsize=2.0, capthick=0.7, alpha=0.6, zorder=1)
            ax.plot(d["gap_sec"] / 60.0, d[metric], marker=MARKERS[mname],
                    color=METHOD_COLORS[mname], lw=LW.get(mname, 1.1),
                    markersize=MSIZE[mname], label=mlabel(mname), zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 30])
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Gap length (min)")
        ax.set_ylabel("Wasserstein distance" if metric == "wasserstein"
                      else r"ACF $L_2$ distance")
        ax.set_title(title, **PANEL_LABEL_KW)

    # 四个面板共用同一组序列，图例统一提到整图顶部，避免各面板内部图例压住曲线。
    # 顶部预留带 0.065×5.0 in ≈ 0.33 in（原先 0.10×5.0 in = 0.50 in，图例离面板太远）；
    # 0.33 in 与 fig10（0.12×2.75 in）相当，两图的图例-面板间距因此一致。2026-10-09 改。
    fig.tight_layout(rect=(0, 0, 1, 0.935), w_pad=1.6, h_pad=1.4)
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", bbox_to_anchor=(0.5, 0.998),
               ncol=7, fontsize=6.5, frameon=False,
               handlelength=1.8, columnspacing=1.0)

    out_png = os.path.join(OUT_DIR, "fig11_earth_transfer.png")
    fig.savefig(out_png, **SAVE_KW)
    # 同步输出到 SRL 提交树（图号已与论文 Figure 8 对齐）
    srl_png = (r"G:\SeisY\docs\dssrr_paper\submission_srl\manuscript"
               r"\figures\fig08_earth_transfer.png")
    fig.savefig(srl_png, **SAVE_KW)
    plt.close(fig)
    print("saved", out_png)
    print("saved", srl_png)


if __name__ == "__main__":
    main()
