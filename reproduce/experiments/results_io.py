# -*- coding: utf-8 -*-
"""统一结果落盘工具 —— 复刻 ``results_E2r_full_v5`` 的目录与文件约定。

三个补充实验（消融 / 图7 长度扫描 / 地球数据迁移）共用本模块，保证与主实验
（``E2r_v5_batch.py``）产出**完全同构**的图像与数据，便于日后复核。

目录结构
--------
``<root>/``
    ``stats_summary.csv``               全局汇总（所有案例 × 方法）
    ``<case>/``                         每个案例一个目录（消融=事件，地球=台站）
        ``stats_detail.csv``            该案例的明细
        ``<anom_type>/<level>/``
            ``before.png``              干净参考 vs 注入异常后
            ``after_<Method>.png``      各方法修复结果
            ``corrupted.mseed``         注入异常后的数据
            ``repaired_<Method>.mseed`` 各方法修复后的数据

CSV 列与主实验 ``stats_summary.csv`` 完全兼容（仅在最前多一列 ``experiment``）::

    experiment,event,station,scenario,anom_type,level,n_segments,
    total_anom_sec,method,rmse,mae,rmse_anom,mae_anom,snr_db,psd_cos,
    amp_rmse,psd_log_rmse,wasserstein,acf_l2,envelope_dist,spec_entropy_err
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from obspy import Trace
from scipy import signal
from scipy.signal import hilbert
from scipy.stats import wasserstein_distance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 与主实验 plot_comparison 一致的配色
METHOD_COLORS = {
    "Linear": "#999999",
    "CubicSpline": "#64B5CD",
    "FFT": "#CCB974",
    "DSSRR": "#C44E52",
    "SSA": "#8172B3",
    "UNet": "#55A868",
    "Corrupted": "#DD8452",
}

CASE_COLS = ["experiment", "event", "station", "scenario", "anom_type",
             "level", "n_segments", "total_anom_sec", "anom_start",
             "anom_end", "method"]
METRIC_COLS = ["rmse", "mae", "rmse_anom", "mae_anom", "snr_db", "psd_cos",
               "amp_rmse", "psd_log_rmse", "wasserstein", "acf_l2",
               "envelope_dist", "spec_entropy_err"]
ALL_COLS = CASE_COLS + METRIC_COLS


# ----------------------------------------------------------------------
# 指标（与 E2r_v5_batch.compute_metrics / compute_stats_metrics 逐行一致）
# ----------------------------------------------------------------------

def _seg(data, intervals):
    idx = np.concatenate([np.arange(s, e + 1) for s, e in intervals])
    return data[idx]


def compute_stats_metrics(truth_seg, repaired_seg, sr):
    """统计特征保真度指标（Wasserstein / ACF L2 / 包络 / 谱熵）。"""
    metrics = {}
    metrics["wasserstein"] = wasserstein_distance(truth_seg, repaired_seg)

    def acf(x):
        x = x - np.mean(x)
        n = len(x)
        full = np.correlate(x, x, mode="full")[n - 1:]
        full = full / (full[0] + 1e-12)
        lags = np.arange(min(200, n))
        return full[lags]

    a_t = acf(truth_seg)
    a_r = acf(repaired_seg)
    L = min(len(a_t), len(a_r))
    metrics["acf_l2"] = float(np.sqrt(np.mean((a_t[:L] - a_r[:L]) ** 2)))

    e_t = np.abs(hilbert(truth_seg))
    e_r = np.abs(hilbert(repaired_seg))
    metrics["envelope_dist"] = wasserstein_distance(e_t, e_r)

    def spec_entropy(x):
        _, Pxx = signal.welch(x, fs=sr, nperseg=min(512, len(x) // 2),
                              scaling="density")
        P = Pxx / (Pxx.sum() + 1e-12)
        P = np.clip(P, 1e-12, None)
        return -np.sum(P * np.log(P)) / np.log(len(P))

    metrics["spec_entropy_err"] = abs(spec_entropy(truth_seg)
                                      - spec_entropy(repaired_seg))
    return metrics


def compute_metrics(truth, repaired, intervals, sr):
    """全套指标（与主实验 E2r_v5_batch.compute_metrics 同定义）。"""
    resid = repaired - truth
    rmse = np.sqrt(np.mean(resid ** 2))
    mae = np.mean(np.abs(resid))
    snr = 10 * np.log10(np.sum(truth ** 2) / np.sum(resid ** 2 + 1e-12))

    rmse_anom, mae_anom = np.nan, np.nan
    if intervals:
        mask = np.zeros(len(truth), dtype=bool)
        for s, e in intervals:
            mask[s:e + 1] = True
        if mask.sum() > 0:
            rmse_anom = np.sqrt(np.mean(resid[mask] ** 2))
            mae_anom = np.mean(np.abs(resid[mask]))

    _, psd_truth = signal.welch(truth, fs=sr, nperseg=min(4096, len(truth)))
    _, psd_rep = signal.welch(repaired, fs=sr, nperseg=min(4096, len(repaired)))
    cos_sim = (np.dot(psd_truth, psd_rep)
               / (np.linalg.norm(psd_truth) * np.linalg.norm(psd_rep) + 1e-12))

    f_amp, amp_truth = signal.periodogram(truth, fs=sr)
    _, amp_rep = signal.periodogram(repaired, fs=sr)
    freq_mask = (f_amp >= 0.01) & (f_amp < 1.0)
    amp_rmse = np.sqrt(np.mean((amp_truth[freq_mask] - amp_rep[freq_mask]) ** 2))

    psd_log_rmse = np.sqrt(np.mean((np.log10(psd_truth + 1e-12)
                                    - np.log10(psd_rep + 1e-12)) ** 2))

    stats_metrics = {}
    if intervals:
        stats_metrics = compute_stats_metrics(_seg(truth, intervals),
                                              _seg(repaired, intervals), sr)

    return {
        "rmse": rmse, "mae": mae, "snr_db": snr, "psd_cos": cos_sim,
        "rmse_anom": rmse_anom, "mae_anom": mae_anom,
        "amp_rmse": amp_rmse, "psd_log_rmse": psd_log_rmse,
        **stats_metrics,
    }


# ----------------------------------------------------------------------
# 落盘：mseed
# ----------------------------------------------------------------------

def save_mseed(data, out_path, starttime, station, sr,
               network="XA", location="01", channel="MHZ",
               encoding="FLOAT32"):
    """写出 mseed。

    重要
    ----
    默认使用 ``FLOAT32`` 编码。**不要**沿用主实验 ``E2r_v5_batch.save_mseed``
    里的 ``astype(np.int32)``：本项目的干净数据是 float64（例如 514.9996），
    直接转 int32 会**截断**，使存档与统计表所用的数组不一致，导致日后无法
    用存档复现 ``stats_*.csv``（实测插值类方法的 Wasserstein 会偏差 30–160%）。
    FLOAT32 往返误差约 3e-5，足以复现全部指标。
    """
    tr = Trace(data=np.asarray(data).astype(np.float32))
    tr.stats.network = network
    tr.stats.station = station
    tr.stats.location = location
    tr.stats.channel = channel
    tr.stats.sampling_rate = float(sr)
    tr.stats.starttime = starttime
    tr.write(out_path, format="MSEED", encoding=encoding)


# ----------------------------------------------------------------------
# 落盘：图像（与主实验 plot_comparison 同款 7 面板）
# ----------------------------------------------------------------------

def plot_comparison(truth, repaired, intervals, method_name, out_png, sr,
                    dpi=120):
    """7 面板对比图：时域全图 / 放大 / 残差 / 振幅谱 / 振幅谱残差 / PSD / PSD 残差。"""
    from obspy.signal.filter import highpass

    fig, axes = plt.subplots(7, 1, figsize=(14, 23))
    fig.subplots_adjust(hspace=0.45)
    fig.suptitle(f"{method_name} vs Truth", fontsize=12, y=0.995)
    t = np.arange(len(truth)) / sr
    n = len(truth)

    color = METHOD_COLORS.get(method_name, "#55A868")

    amin = min(s for s, e in intervals)
    amax = max(e for s, e in intervals)
    anom_len_sec = (amax - amin) / sr
    ref_sec = float(np.clip(anom_len_sec / 2.0, 120.0, 600.0))

    truth_proc = truth - np.mean(truth)
    repaired_proc = repaired - np.mean(repaired)
    truth_proc = highpass(truth_proc, freq=0.001, df=sr, corners=4,
                          zerophase=True)
    repaired_proc = highpass(repaired_proc, freq=0.001, df=sr, corners=4,
                             zerophase=True)

    ax = axes[0]
    ax.plot(t, truth, color="#4C72B0", lw=0.5, alpha=0.7, label="Truth")
    ax.plot(t, repaired, color=color, lw=0.5, alpha=0.7, label=method_name)
    ax.axvspan(amin / sr, amax / sr, color="red", alpha=0.2, label="Anomaly")
    ax.set_title("1. Time-Domain Full", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pad = int((ref_sec + 50.0) * sr)
    s_zoom = max(0, amin - pad)
    e_zoom = min(n, amax + pad)
    ax.plot(t[s_zoom:e_zoom], truth[s_zoom:e_zoom], color="#4C72B0", lw=0.8,
            alpha=0.8, label="Truth")
    ax.plot(t[s_zoom:e_zoom], repaired[s_zoom:e_zoom], color=color, lw=0.8,
            alpha=0.7, label=method_name)
    ax.axvspan(amin / sr, amax / sr, color="red", alpha=0.2,
               label=f"Anomaly ({anom_len_sec:.0f}s)")
    if method_name == "DSSRR":
        ax.axvspan(amin / sr - ref_sec, amin / sr, color="green", alpha=0.15,
                   label=f"Ref before ({ref_sec:.0f}s)")
        ax.axvspan(amax / sr, amax / sr + ref_sec, color="orange", alpha=0.15,
                   label=f"Ref after ({ref_sec:.0f}s)")
    ax.set_title(f"2. Time-Domain Zoom (Anomaly: {anom_len_sec:.0f}s)",
                 fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.02)

    ax = axes[2]
    ax.plot(t, repaired - truth, color="darkred", lw=0.5, alpha=0.7,
            label="Residual")
    ax.axvspan(amin / sr, amax / sr, color="red", alpha=0.2, label="Anomaly")
    ax.axhline(y=0, color="black", lw=0.8, linestyle="-")
    ax.axhline(y=5, color="gray", lw=0.8, linestyle="--", alpha=0.7,
               label="±5 threshold")
    ax.axhline(y=-5, color="gray", lw=0.8, linestyle="--", alpha=0.7)
    ax.set_title("3. Time-Domain Residual", fontsize=11)
    ax.set_ylabel("Residual (counts)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    f_amp, amp_truth = signal.periodogram(truth_proc, fs=sr, window="hann")
    _, amp_rep = signal.periodogram(repaired_proc, fs=sr, window="hann")
    psd_f, psd_truth = signal.welch(truth_proc, fs=sr,
                                    nperseg=min(4096, len(truth_proc)))
    _, psd_rep = signal.welch(repaired_proc, fs=sr,
                              nperseg=min(4096, len(repaired_proc)))

    ax = axes[3]
    ax.plot(f_amp, amp_truth, color="#4C72B0", lw=0.8, label="Truth",
            alpha=1.0)
    ax.plot(f_amp, amp_rep, color=color, lw=0.8, alpha=0.7,
            label=method_name)
    ax.margins(x=0.02)
    ax.set_title("4. Amplitude Spectrum (0-3Hz, 1mHz HP)", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Amplitude (counts^2·s)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    ax.plot(f_amp, amp_rep - amp_truth, color="darkred", lw=0.8, alpha=0.7,
            label=f"{method_name} - Truth")
    ax.axhline(y=0, color="black", lw=0.8, linestyle="-")
    ax.margins(x=0.02)
    ax.set_title("5. Amplitude Spectrum Residual (0-3Hz)", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Amplitude residual")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    ax.semilogy(psd_f, psd_truth, color="#4C72B0", lw=0.9, label="Truth",
                alpha=1.0)
    ax.semilogy(psd_f, psd_rep, color=color, lw=0.9, alpha=0.7,
                label=method_name)
    ax.margins(x=0.02)
    ax.set_title("6. PSD (Welch, 0-3Hz, 1mHz HP)", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[6]
    ax.semilogy(psd_f, np.abs(psd_rep - psd_truth) + 1e-12, color="darkred",
                lw=0.8, alpha=0.7, label=f"|{method_name} - Truth|")
    ax.margins(x=0.02)
    ax.set_title("7. PSD Residual (abs, 0-3Hz)", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD residual (abs)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 统一写出器
# ----------------------------------------------------------------------

class ResultsWriter:
    """收集逐案例指标，并按主实验结构写出目录、图像、数据与 CSV。"""

    def __init__(self, root, experiment, dpi=120):
        self.root = root
        self.experiment = experiment
        self.dpi = dpi
        self.rows = []
        os.makedirs(root, exist_ok=True)

    # -- 路径 ---------------------------------------------------------
    def case_dir(self, event, anom_type, level):
        d = os.path.join(self.root, event, anom_type, level)
        os.makedirs(d, exist_ok=True)
        return d

    # -- 单条案例落盘 --------------------------------------------------
    def save_case(self, *, event, station, scenario, anom_type, level,
                  truth, corrupted, intervals, repaired_by_method,
                  method_order, starttime, sr, sr_out=None,
                  meta_extra=None, plot=True, network="XA", location="01",
                  channel="MHZ", total_anom_sec=None):
        """写出一个案例的全部产物，并返回该案例的指标行列表。

        Parameters
        ----------
        repaired_by_method : dict[str, np.ndarray]
            已算好的各方法修复结果（全长度数组，便于直接落 mseed）。
        method_order : list[str]
            落盘与统计的方法顺序。
        starttime : obspy.UTCDateTime
            mseed 头段时间（与输入数据一致）。
        sr_out : float, optional
            mseed 写出的采样率，默认同 ``sr``。
        """
        leaf = self.case_dir(event, anom_type, level)
        sr_out = float(sr if sr_out is None else sr_out)

        if total_anom_sec is None:
            total_anom_sec = sum(e - s + 1 for s, e in intervals) / sr
        n_seg = len(intervals)

        # 显式记录全部异常区间：多段异常（如 spike）无法用 min/max 单区间还原，
        # 校验脚本据此才能逐段复算 rmse_anom / acf_l2 / spec_entropy_err。
        import json as _json
        with open(os.path.join(leaf, "intervals.json"), "w",
                  encoding="utf-8") as fh:
            _json.dump({"intervals": [[int(s), int(e)] for s, e in intervals]},
                       fh, indent=2)

        save_mseed(corrupted, os.path.join(leaf, "corrupted.mseed"),
                   starttime, station, sr_out, network=network,
                   location=location, channel=channel)
        if plot:
            plot_comparison(truth, corrupted, intervals, "Corrupted",
                            os.path.join(leaf, "before.png"), sr,
                            dpi=self.dpi)

        case_rows = []
        for mname in method_order:
            repaired = repaired_by_method.get(mname)
            if repaired is None:
                continue
            save_mseed(repaired, os.path.join(leaf, f"repaired_{mname}.mseed"),
                       starttime, station, sr_out, network=network,
                       location=location, channel=channel)
            if plot:
                plot_comparison(truth, repaired, intervals, mname,
                                os.path.join(leaf, f"after_{mname}.png"), sr,
                                dpi=self.dpi)

            metrics = compute_metrics(truth, repaired, intervals, sr)
            row = {
                "experiment": self.experiment,
                "event": event, "station": station, "scenario": scenario,
                "anom_type": anom_type, "level": level,
                "n_segments": n_seg, "total_anom_sec": total_anom_sec,
                "anom_start": int(min(s for s, _ in intervals)),
                "anom_end": int(max(e for _, e in intervals)),
                "method": mname,
            }
            row.update(metrics)
            if meta_extra:
                row.update(meta_extra)
            case_rows.append(row)

        self.rows.extend(case_rows)
        self._flush_case(event, case_rows)
        return case_rows

    # -- CSV ----------------------------------------------------------
    def _flush_case(self, event, case_rows):
        if not case_rows:
            return
        df = pd.DataFrame(case_rows)
        df = self._ordered(df)
        df.to_csv(os.path.join(self.root, event, "stats_detail.csv"),
                  index=False, encoding="utf-8-sig")

    def _ordered(self, df):
        cols = [c for c in ALL_COLS if c in df.columns]
        cols += [c for c in df.columns if c not in cols]
        return df[cols]

    def rewrite_detail(self):
        """按当前 ``self.rows`` 重写各案例的 stats_detail.csv。

        调用方若在 :meth:`save_case` 之后追加了自定义列，可用本方法把新列
        落进各案例的明细表。
        """
        for event in sorted({r["event"] for r in self.rows}):
            sub = pd.DataFrame([r for r in self.rows if r["event"] == event])
            sub = self._ordered(sub)
            sub.to_csv(os.path.join(self.root, event, "stats_detail.csv"),
                       index=False, encoding="utf-8-sig")

    def write_summary(self):
        """写出全局 stats_summary.csv（追加自定义列，如 runtime_s）。"""
        df = pd.DataFrame(self.rows)
        if df.empty:
            print("[ResultsWriter] 无结果可写")
            return df
        df = self._ordered(df)
        out = os.path.join(self.root, "stats_summary.csv")
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[ResultsWriter] summary -> {out}  ({len(df)} rows)")
        return df

    # -- 溯源 ----------------------------------------------------------
    def write_provenance(self, script_path=None, extra=None):
        """记录代码版本与关键文件哈希，供日后核对结果出处。

        写出 ``PROVENANCE.json``，包含：运行时间、git 提交与工作区是否脏、
        ``seisy/core/anomaly_repair`` 下各文件与本次实验脚本的 MD5、Python 版本。
        """
        import datetime
        import hashlib
        import json
        import subprocess
        import sys

        def _md5(path):
            try:
                with open(path, "rb") as fh:
                    return hashlib.md5(fh.read()).hexdigest()
            except OSError:
                return None

        repo = r"G:\SeisY"
        info = {
            "experiment": self.experiment,
            "root": self.root,
            "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "n_rows": len(self.rows),
        }

        def _git(*args):
            try:
                return subprocess.run(["git", "-C", repo, *args],
                                      capture_output=True, text=True,
                                      timeout=20).stdout.strip()
            except Exception:
                return None

        info["git_commit"] = _git("rev-parse", "HEAD")
        info["git_commit_time"] = _git("log", "-1", "--format=%ci")
        info["git_dirty"] = bool(_git("status", "--porcelain",
                                      "--", "seisy/core/anomaly_repair"))

        core_dir = os.path.join(repo, "seisy", "core", "anomaly_repair")
        if os.path.isdir(core_dir):
            info["core_module_md5"] = {
                fn: _md5(os.path.join(core_dir, fn))
                for fn in sorted(os.listdir(core_dir)) if fn.endswith(".py")
            }
        if script_path:
            info["script"] = os.path.basename(script_path)
            info["script_md5"] = _md5(script_path)
        if extra:
            info.update(extra)

        out = os.path.join(self.root, "PROVENANCE.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(info, fh, ensure_ascii=False, indent=2)
        print(f"[ResultsWriter] provenance -> {out}")
        return info
