"""
E-RealInj v5 全量批量实验
改进：
- 方案 B：分频带 Hilbert 瞬时相位锁定
- 方案 A 改进：真实波形交叉淡化（非线性外推）
- 增加统计特征保真度指标（wasserstein, acf_l2, envelope_dist, spec_entropy_err）
- v4 文件夹保留
"""
import numpy as np
import os
import sys
import pandas as pd
from obspy import read, Trace

sys.path.insert(0, r'G:\SeisY')
from scipy import signal
from scipy.stats import wasserstein_distance, skew, kurtosis

from seisy.core.anomaly_repair.comparison_methods import (
    LinearInterpolation,
    CubicSplineInterpolation,
    SSARepair,
    UNetRepair,
)
from seisy.core.anomaly_repair.reference_spectrum import ReferenceSpectrumReplacer


SR = 6.625
SEED = 7
AMP = 500.0
CLEAN_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h'
OUT_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v5'

SPIKE_WINDOW_SEC = 1500.0

ANOM_CONFIG = {
    'dropout': {'small': 300, 'medium': 900, 'large': 3600},
    'spike': {'small': (5, 1, 1), 'medium': (10, 3, 5), 'large': (20, 5, 10)},
}

ANOM_TYPES = ['dropout', 'spike']
LEVELS = ['small', 'medium', 'large']
SCENARIOS = ['stationary', 'transient']


class FFTInterpolation:
    def repair(self, data, start, end):
        repaired = data.copy()
        n = len(data)
        known = np.concatenate([data[:start], data[end+1:]])
        fft_known = np.fft.rfft(known)
        n_low = max(1, len(fft_known) // 10)
        fft_smooth = np.zeros_like(fft_known)
        fft_smooth[:n_low] = fft_known[:n_low]
        smooth_full = np.fft.irfft(fft_smooth, n=n)
        left_mean = np.mean(data[max(0, start-100):start])
        right_mean = np.mean(data[end+1:min(n, end+101)])
        target_mean = (left_mean + right_mean) / 2
        smooth_mean = np.mean(smooth_full[start:end+1])
        dc_offset = target_mean - smooth_mean
        repaired[start:end+1] = smooth_full[start:end+1] + dc_offset
        return repaired


METHODS = {
    'Linear': LinearInterpolation(),
    'CubicSpline': CubicSplineInterpolation(),
    'FFT': FFTInterpolation(),
    'SSA': SSARepair(max_iter=30),
    'UNet': UNetRepair(
        patch_length=1536,
        n_epochs=10,
        model_path=r'G:\SeisY\docs\dssrr_paper\experiments\models\unet_synth.pt',
    ),
    'DSSRR': None,
}


def find_adaptive_boundary(data, start, end, sr):
    win_sec = 300
    win = int(win_sec * sr)
    left_end = start - int(1.0 * sr)
    left_start = max(0, left_end - win)
    left_seg = data[left_start:left_end]
    right_start = end + 1 + int(1.0 * sr)
    right_end = min(len(data), right_start + win)
    right_seg = data[right_start:right_end]
    x_left = np.arange(len(left_seg))
    coeffs_left = np.polyfit(x_left, left_seg, 1)
    trend_left = np.polyval(coeffs_left, x_left)
    x_right = np.arange(len(right_seg))
    coeffs_right = np.polyfit(x_right, right_seg, 1)
    trend_right = np.polyval(coeffs_right, x_right)
    left_diff = np.abs(left_seg - trend_left)
    right_diff = np.abs(right_seg - trend_right)
    s_adapt = start
    for i in range(start-1, left_start, -1):
        idx = i - left_start
        if idx < len(left_diff) and left_diff[idx] < np.std(left_seg) * 0.5:
            s_adapt = i
            break
    e_adapt = end
    for i in range(end+1, right_end):
        idx = i - right_start
        if idx < len(right_diff) and right_diff[idx] < np.std(right_seg) * 0.5:
            e_adapt = i
            break
    return s_adapt, e_adapt


def find_seismic_center(data, sr, search_sec=3600):
    n = len(data)
    peak_idx = int(np.argmax(np.abs(data)))
    win = int(30 * sr)
    lo = max(0, peak_idx - win)
    hi = min(n, peak_idx + win)
    center = int((lo + hi) / 2)
    return center


def inject_anomaly(data, anom_type, level, seed, scenario='stationary'):
    rng = np.random.RandomState(seed)
    corrupted = data.copy()
    n = len(data)

    if scenario == 'stationary':
        center = int(n * 0.9)
    else:
        center = int(10000 * SR)

    if anom_type == 'dropout':
        len_sec = ANOM_CONFIG['dropout'][level]
        len_pts = int(len_sec * SR)
        start = center - len_pts // 2
        end = start + len_pts - 1
        start = max(1, start)
        end = min(n - 2, end)
        corrupted[start:end+1] = 0
        intervals = [(start, end)]
    elif anom_type == 'spike':
        n_spikes, w_min, w_max = ANOM_CONFIG['spike'][level]
        win_pts = int(SPIKE_WINDOW_SEC * SR)
        win_start = center - win_pts // 2
        win_end = win_start + win_pts - 1
        win_start = max(1, win_start)
        win_end = min(n - 2, win_end)
        win_pts_eff = win_end - win_start + 1
        cell = win_pts_eff // n_spikes
        jitter = int(cell * 0.15)
        intervals = []
        for k in range(n_spikes):
            width = rng.randint(w_min, w_max + 1)
            center_k = win_start + int((k + 0.5) * cell)
            s = center_k + rng.randint(-jitter, jitter + 1) - width // 2
            s = max(win_start, min(s, win_end - width))
            e = s + width - 1
            sign = rng.choice([-1, 1])
            corrupted[s:e+1] = data[s:e+1] + AMP * sign
            intervals.append((s, e))
        intervals.sort()
    return corrupted, intervals


def repair_intervals(method_name, corrupted, intervals, seed):
    repaired = corrupted.copy()
    n = len(repaired)

    if method_name == 'DSSRR':
        replacer = ReferenceSpectrumReplacer(SR)

        def ref_len_for(s, e):
            anom_sec = (e - s + 1) / SR
            return float(np.clip(anom_sec / 2.0, 120.0, 600.0))

        work = sorted(intervals)
        merged = []
        for s, e in work:
            if not merged:
                merged.append([s, e])
                continue
            ps, pe = merged[-1]
            gap_sec = (s - pe - 1) / SR
            need = ref_len_for(ps, pe) + ref_len_for(s, e) + 2.0
            if gap_sec < need:
                merged[-1][1] = max(pe, e)
            else:
                merged.append([s, e])

        for s0, e0 in merged:
            s_adapt, e_adapt = find_adaptive_boundary(repaired, s0, e0, SR)
            ref_len = ref_len_for(s_adapt, e_adapt)
            seg, _ = replacer.replace(
                repaired.copy(), s_adapt, e_adapt,
                reference_before_sec=ref_len,
                reference_after_sec=ref_len,
                reference_gap_sec=1.0,
                random_seed=seed,
            )
            # 合并区间（merged）只用于确定参考段长度与合成上下文；
            # 写回时严格限制在**原始检测到的异常段**，绝不覆盖尖峰之间的
            # 健康数据（论文：“...while only the originally detected interval
            # is written back”）。此前直接写回 [s0,e0] 会在 spike 场景把
            # 1500 s 窗内的大段正常数据一并替换掉（写回样本 65 -> 5256）。
            for ts, te in intervals:
                if s0 <= ts and te <= e0:
                    repaired[ts:te+1] = seg[ts:te+1]
                    repaired[ts:te+1] = np.round(repaired[ts:te+1])
    else:
        method = METHODS[method_name]
        for s0, e0 in intervals:
            seg = method.repair(repaired, s0, e0)
            repaired[s0:e0+1] = seg[s0:e0+1]
        repaired = np.clip(repaired, 0, 1e6)
        repaired = np.nan_to_num(repaired, nan=0.0, posinf=1e6, neginf=0.0)

    return repaired


def _seg(data, intervals):
    idx = np.concatenate([np.arange(s, e+1) for s, e in intervals])
    return data[idx]


def compute_stats_metrics(truth_seg, repaired_seg):
    """统计特征保真度指标"""
    metrics = {}
    metrics['wasserstein'] = wasserstein_distance(truth_seg, repaired_seg)

    # 自相关 L2
    def acf(x):
        x = x - np.mean(x)
        n = len(x)
        full = np.correlate(x, x, mode='full')[n-1:]
        full = full / (full[0] + 1e-12)
        lags = np.arange(min(200, n))
        return full[lags]
    a_t = acf(truth_seg)
    a_r = acf(repaired_seg)
    L = min(len(a_t), len(a_r))
    metrics['acf_l2'] = float(np.sqrt(np.mean((a_t[:L] - a_r[:L])**2)))

    # 包络分布
    from scipy.signal import hilbert
    e_t = np.abs(hilbert(truth_seg))
    e_r = np.abs(hilbert(repaired_seg))
    metrics['envelope_dist'] = wasserstein_distance(e_t, e_r)

    # 谱熵
    def spec_entropy(x):
        f, Pxx = signal.welch(x, fs=SR, nperseg=min(512, len(x)//2), scaling='density')
        P = Pxx / (Pxx.sum() + 1e-12)
        P = np.clip(P, 1e-12, None)
        return -np.sum(P * np.log(P)) / np.log(len(P))
    metrics['spec_entropy_err'] = abs(spec_entropy(truth_seg) - spec_entropy(repaired_seg))

    return metrics


def compute_metrics(truth, repaired, intervals=None):
    resid = repaired - truth
    rmse = np.sqrt(np.mean(resid**2))
    mae = np.mean(np.abs(resid))
    snr = 10 * np.log10(np.sum(truth**2) / np.sum(resid**2 + 1e-12))

    rmse_anom, mae_anom = np.nan, np.nan
    if intervals:
        mask = np.zeros(len(truth), dtype=bool)
        for s, e in intervals:
            mask[s:e+1] = True
        if mask.sum() > 0:
            rmse_anom = np.sqrt(np.mean(resid[mask]**2))
            mae_anom = np.mean(np.abs(resid[mask]))

    f_psd, psd_truth = signal.welch(truth, fs=SR, nperseg=min(4096, len(truth)))
    _, psd_rep = signal.welch(repaired, fs=SR, nperseg=min(4096, len(repaired)))
    cos_sim = np.dot(psd_truth, psd_rep) / (np.linalg.norm(psd_truth) * np.linalg.norm(psd_rep) + 1e-12)

    f_amp, amp_truth = signal.periodogram(truth, fs=SR)
    _, amp_rep = signal.periodogram(repaired, fs=SR)
    freq_mask = (f_amp >= 0.01) & (f_amp < 1.0)
    amp_rmse = np.sqrt(np.mean((amp_truth[freq_mask] - amp_rep[freq_mask])**2))

    log_psd_truth = np.log10(psd_truth + 1e-12)
    log_psd_rep = np.log10(psd_rep + 1e-12)
    psd_log_rmse = np.sqrt(np.mean((log_psd_truth - log_psd_rep)**2))

    # 新增：统计特征指标
    stats_metrics = {}
    if intervals:
        t_seg = _seg(truth, intervals)
        r_seg = _seg(repaired, intervals)
        stats_metrics = compute_stats_metrics(t_seg, r_seg)

    return {
        'rmse': rmse, 'mae': mae, 'snr_db': snr, 'psd_cos': cos_sim,
        'rmse_anom': rmse_anom, 'mae_anom': mae_anom,
        'amp_rmse': amp_rmse, 'psd_log_rmse': psd_log_rmse,
        **stats_metrics,
    }


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_comparison(truth, repaired, intervals, method_name, out_png):
    from obspy.signal.filter import highpass

    fig, axes = plt.subplots(7, 1, figsize=(14, 23))
    fig.subplots_adjust(hspace=0.45)
    fig.suptitle(f"{method_name} vs Truth", fontsize=12, y=0.995)
    t = np.arange(len(truth)) / SR
    n = len(truth)

    method_colors = {
        'Linear': '#999999',
        'CubicSpline': '#64B5CD',
        'FFT': '#CCB974',
        'DSSRR': '#C44E52',
        'SSA': '#8172B3',
        'UNet': '#55A868',
        'Corrupted': '#DD8452',
    }
    color = method_colors.get(method_name, '#55A868')

    amin = min(s for s, e in intervals)
    amax = max(e for s, e in intervals)
    anom_len_sec = (amax - amin) / SR

    # 计算参考段长度（和 DSSRR 一致：anom/2，限制120-600s）
    ref_sec = float(np.clip(anom_len_sec / 2.0, 120.0, 600.0))

    # 频谱预处理：全数据，去平均 + 1mHz 高通滤波
    truth_proc = truth - np.mean(truth)
    repaired_proc = repaired - np.mean(repaired)
    truth_proc = highpass(truth_proc, freq=0.001, df=SR, corners=4, zerophase=True)
    repaired_proc = highpass(repaired_proc, freq=0.001, df=SR, corners=4, zerophase=True)

    # 1. 时域全图
    ax = axes[0]
    ax.plot(t, truth, color="#4C72B0", lw=0.5, alpha=0.7, label="Truth")
    ax.plot(t, repaired, color=color, lw=0.5, alpha=0.7, label=method_name)
    ax.axvspan(amin/SR, amax/SR, color='red', alpha=0.2, label='Anomaly')
    ax.set_title("1. Time-Domain Full", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # 2. 时域放大（含参考段标记）
    ax = axes[1]
    pad_sec = ref_sec + 50.0
    pad = int(pad_sec * SR)
    s_zoom = max(0, amin - pad)
    e_zoom = min(n, amax + pad)
    ax.plot(t[s_zoom:e_zoom], truth[s_zoom:e_zoom], color="#4C72B0", lw=0.8, alpha=0.8, label="Truth")
    ax.plot(t[s_zoom:e_zoom], repaired[s_zoom:e_zoom], color=color, lw=0.8, alpha=0.7, label=method_name)
    ax.axvspan(amin/SR, amax/SR, color='red', alpha=0.2, label=f'Anomaly ({anom_len_sec:.0f}s)')
    # 只有 DSSRR 方法才显示参考段标记（其他方法不使用参考段）
    if method_name == 'DSSRR':
        ax.axvspan((amin/SR - ref_sec), amin/SR, color='green', alpha=0.15, label=f'Ref before ({ref_sec:.0f}s)')
        ax.axvspan(amax/SR, (amax/SR + ref_sec), color='orange', alpha=0.15, label=f'Ref after ({ref_sec:.0f}s)')
    ax.set_title(f"2. Time-Domain Zoom (Anomaly: {anom_len_sec:.0f}s)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    # 加x轴边距
    ax.margins(x=0.02)  # 左右各2%边距

    # 3. 时域残差
    ax = axes[2]
    resid = repaired - truth
    ax.plot(t, resid, color="darkred", lw=0.5, alpha=0.7, label="Residual")
    ax.axvspan(amin/SR, amax/SR, color='red', alpha=0.2, label='Anomaly')
    ax.axhline(y=0, color='black', lw=0.8, linestyle='-')
    ax.axhline(y=5, color='gray', lw=0.8, linestyle='--', alpha=0.7, label='±5 threshold')
    ax.axhline(y=-5, color='gray', lw=0.8, linestyle='--', alpha=0.7)
    ax.set_title("3. Time-Domain Residual", fontsize=11)
    ax.set_ylabel("Residual (counts)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # 计算频谱
    f_amp, amp_truth = signal.periodogram(truth_proc, fs=SR, window='hann')
    _, amp_rep = signal.periodogram(repaired_proc, fs=SR, window='hann')
    psd_f, psd_truth = signal.welch(truth_proc, fs=SR, nperseg=min(4096, len(truth_proc)))
    _, psd_rep = signal.welch(repaired_proc, fs=SR, nperseg=min(4096, len(repaired_proc)))

    # 4. 振幅谱（线性坐标，x轴0-3Hz）
    ax = axes[3]
    ax.plot(f_amp, amp_truth, color="#4C72B0", lw=0.8, label="Truth", alpha=1.0)
    ax.plot(f_amp, amp_rep, color=color, lw=0.8, alpha=0.7, label=method_name)
    ax.margins(x=0.02)
    ax.set_title("4. Amplitude Spectrum (0-3Hz, 1mHz HP)", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Amplitude (counts^2·s)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # 5. 振幅谱残差
    ax = axes[4]
    amp_resid = amp_rep - amp_truth
    ax.plot(f_amp, amp_resid, color="darkred", lw=0.8, alpha=0.7, label=f"{method_name} - Truth")
    ax.axhline(y=0, color='black', lw=0.8, linestyle='-')
    ax.margins(x=0.02)
    ax.set_title("5. Amplitude Spectrum Residual (0-3Hz)", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("Amplitude residual")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # 6. PSD
    ax = axes[5]
    ax.semilogy(psd_f, psd_truth, color="#4C72B0", lw=0.9, label="Truth", alpha=1.0)
    ax.semilogy(psd_f, psd_rep, color=color, lw=0.9, alpha=0.7, label=method_name)
    ax.margins(x=0.02)
    ax.set_title("6. PSD (Welch, 0-3Hz, 1mHz HP)", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # 7. PSD 残差
    ax = axes[6]
    psd_resid = psd_rep - psd_truth
    ax.semilogy(psd_f, np.abs(psd_resid) + 1e-12, color="darkred", lw=0.8, alpha=0.7, label=f"|{method_name} - Truth|")
    ax.margins(x=0.02)
    ax.set_title("7. PSD Residual (abs, 0-3Hz)", fontsize=11)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD residual (abs)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)



def save_mseed(data, out_path, orig_starttime, station):
    tr = Trace(data=data.astype(np.int32))
    tr.stats.network = 'XA'
    tr.stats.station = station
    tr.stats.location = '01'
    tr.stats.channel = 'MHZ'
    tr.stats.sampling_rate = SR
    tr.stats.starttime = orig_starttime
    tr.write(out_path, format='MSEED')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_stats = []

    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith('.mseed')]
    print(f"找到 {len(clean_files)} 个干净数据文件", flush=True)

    for fname in sorted(clean_files):
        parts = fname.replace('.mseed', '').split('.')
        station = parts[1]
        start_str = parts[4].split('-')[0]
        event_name = f"{station}_{start_str}"
        print(f"\n{'='*60}\n事件: {event_name}\n{'='*60}", flush=True)

        fpath = os.path.join(CLEAN_DIR, fname)
        st = read(fpath)
        tr = st[0]
        orig_starttime = tr.stats.starttime
        data = tr.data.astype(float)
        n = len(data)
        print(f"  数据长度: {n/SR/3600:.1f}h", flush=True)

        seismic_center = find_seismic_center(data, SR)
        print(f"  月震波形中心: {seismic_center/SR/60:.1f}min", flush=True)

        for scenario in SCENARIOS:
            event_dir = os.path.join(OUT_DIR, f"{event_name}_{scenario}")
            os.makedirs(event_dir, exist_ok=True)

            for anom_type in ANOM_TYPES:
                for level in LEVELS:
                    sub_dir = os.path.join(event_dir, anom_type, level)
                    os.makedirs(sub_dir, exist_ok=True)

                    corrupted, intervals = inject_anomaly(data, anom_type, level, SEED, scenario=scenario)
                    total_anom_sec = sum(e - s + 1 for s, e in intervals) / SR
                    n_seg = len(intervals)
                    print(f"  [{scenario}] {anom_type}/{level}: {n_seg} 段, 共 {total_anom_sec:.1f}s", flush=True)

                    save_mseed(corrupted, os.path.join(sub_dir, 'corrupted.mseed'), orig_starttime, station)
                    plot_comparison(data, corrupted, intervals, "Corrupted", os.path.join(sub_dir, 'before.png'))

                    for method_name in METHODS.keys():
                        try:
                            import time as _time
                            _t0 = _time.perf_counter()
                            print(f"      {method_name}...", end=" ", flush=True)
                            repaired = repair_intervals(method_name, corrupted, intervals, SEED)
                            stats = compute_metrics(data, repaired, intervals)

                            save_mseed(repaired, os.path.join(sub_dir, f'repaired_{method_name}.mseed'), orig_starttime, station)
                            plot_comparison(data, repaired, intervals, method_name, os.path.join(sub_dir, f'after_{method_name}.png'))
                            print(f"done ({_time.perf_counter() - _t0:.1f}s)", flush=True)

                            row = {
                                'event': event_name, 'station': station,
                                'scenario': scenario,
                                'anom_type': anom_type, 'level': level,
                                'n_segments': n_seg, 'total_anom_sec': total_anom_sec,
                                'method': method_name,
                                'rmse': stats['rmse'], 'mae': stats['mae'],
                                'rmse_anom': stats['rmse_anom'], 'mae_anom': stats['mae_anom'],
                                'snr_db': stats['snr_db'], 'psd_cos': stats['psd_cos'],
                                'amp_rmse': stats['amp_rmse'], 'psd_log_rmse': stats['psd_log_rmse'],
                                'wasserstein': stats.get('wasserstein', np.nan),
                                'acf_l2': stats.get('acf_l2', np.nan),
                                'envelope_dist': stats.get('envelope_dist', np.nan),
                                'spec_entropy_err': stats.get('spec_entropy_err', np.nan),
                            }
                            all_stats.append(row)
                        except Exception as e:
                            import traceback
                            print(f"FAILED ({e})", flush=True)
                            traceback.print_exc()
                            all_stats.append({
                                'event': event_name, 'station': station,
                                'scenario': scenario,
                                'anom_type': anom_type, 'level': level,
                                'n_segments': n_seg, 'total_anom_sec': total_anom_sec,
                                'method': method_name,
                                'rmse': np.nan, 'mae': np.nan,
                                'rmse_anom': np.nan, 'mae_anom': np.nan,
                                'snr_db': np.nan, 'psd_cos': np.nan,
                                'amp_rmse': np.nan, 'psd_log_rmse': np.nan,
                                'wasserstein': np.nan, 'acf_l2': np.nan,
                                'envelope_dist': np.nan, 'spec_entropy_err': np.nan,
                            })

            event_df = pd.DataFrame([r for r in all_stats if r['event'] == event_name])
            event_df.to_csv(os.path.join(event_dir, 'stats_detail.csv'), index=False, encoding='utf-8-sig')

    df = pd.DataFrame(all_stats)
    df.to_csv(os.path.join(OUT_DIR, 'stats_summary.csv'), index=False, encoding='utf-8-sig')
    print(f"\n{'='*60}\n总统计表已保存\n{'='*60}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()



