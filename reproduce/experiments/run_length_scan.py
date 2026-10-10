"""图7：长度效应精细扫描
20个不同长度的dropout，从2min到90min

除了原先的 length_scan_raw.csv 与 quick-look 图之外，本脚本还会按
``results_E2r_full_v5`` 的结构落盘完整产物（before/after PNG +
corrupted/repaired mseed + 明细/汇总 CSV），目录为 ``results_length_scan/``，
便于日后逐案例复核。
"""
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from obspy import read
from scipy.stats import wasserstein_distance
from scipy import signal as scipy_signal

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, r"G:\SeisY\docs\dssrr_paper")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import results_io as ri  # noqa: E402
from docs.dssrr_paper.experiments.baselines import make_methods, METHOD_ORDER

matplotlib.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['axes.grid'] = False

SR = 6.625

def compute_stats_metrics(truth_seg, repaired_seg):
    """和E2r_v5一致的统计指标计算"""
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
    return metrics

CLEAN_FILES = [
    r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h\XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed',
    r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h\XA.S12.01.MHZ.19761114_230000-19761115_115957.mseed',
    r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h\XA.S15.01.MHZ.19760125_160000-19760126_045957.mseed',
    r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h\XA.S16.01.MHZ.19761114_230000-19761115_115957.mseed',
]
# NOTE: the manuscript is now submitted to SRL; figures are read by LaTeX from
# submission_srl/manuscript/figures (this used to point at the old
# submission_jgr_planets tree, then briefly at submission_srl/figures, which the
# .tex does not read -- that mismatch silently left stale figures in the PDF).
FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'

# 20个长度点，对数间隔，从2min到90min
lengths_min = np.logspace(np.log10(2), np.log10(90), 20)
lengths_samples = [int(l * 60 * SR) for l in lengths_min]

print(f"长度点（min）: {[f'{l:.1f}' for l in lengths_min]}")

# 初始化方法
methods = make_methods(SR)
focus_methods = ['dssrr', 'linear', 'stft', 'ssa']
method_labels = {'dssrr': 'DSSRR', 'linear': 'Linear', 'stft': 'FFT', 'ssa': 'SSA'}
colors = {'dssrr': '#2c3e50', 'linear': '#95a5a6', 'stft': '#3498db', 'ssa': '#9b59b6'}
markers = {'dssrr': '*', 'linear': 'o', 'stft': '^', 'ssa': 's'}

# 跑每个长度（4个事件平均）
results = {m: {'lengths': [], 'wasserstein': [], 'acf_l2': []} for m in focus_methods}

# 统一结构的产物目录（与 results_E2r_full_v5 同构）
OUT_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\results_length_scan'
writer = ri.ResultsWriter(OUT_DIR, experiment='length_scan')

# 先收集所有事件的结果
all_results = []
for fidx, clean_file in enumerate(CLEAN_FILES):
    print(f"\n=== 处理事件 {fidx+1}/{len(CLEAN_FILES)}: {clean_file.split(chr(92))[-1][:20]}... ===")
    st = read(clean_file)
    data = st[0].data.astype(float)
    fname = os.path.basename(clean_file)
    parts = fname.replace('.mseed', '').split('.')
    station = parts[1]
    event = f"{station}_{parts[4].split('-')[0]}"
    start_sample = int(1 * 3600 * SR)
    end_sample = int(5 * 3600 * SR)
    clean_window = data[start_sample:end_sample]
    # 存档 mseed 的头段时间必须对应窗口起点（不是整文件起点）
    window_starttime = st[0].stats.starttime + start_sample / SR

    for i, length_samples in enumerate(lengths_samples):
        length_min = lengths_min[i]
        n = len(clean_window)
        s = n // 2 - length_samples // 2
        e = s + length_samples
        corrupted = clean_window.copy()
        corrupted[s:e] = 0.0

        intervals = [(int(s), int(e))]
        repaired_by_method = {}
        case_rows = []

        for mname in focus_methods:
            try:
                repaired = methods[mname].repair(corrupted, s, e)
                repaired_by_method[method_labels[mname]] = repaired
            except Exception as ex:
                print(f"    {mname} 失败: {ex}")

        if repaired_by_method:
            case_rows = writer.save_case(
                event=event, station=station,
                scenario='length_scan', anom_type='dropout',
                level=f'{length_min:05.2f}min',
                truth=clean_window, corrupted=corrupted, intervals=intervals,
                repaired_by_method=repaired_by_method,
                method_order=[method_labels[m] for m in focus_methods],
                starttime=window_starttime, sr=SR,
            )

        # 追加 length 与能量比（std_ratio），供 Fig.7(b) 的机制说明复核
        for row in case_rows:
            mname = row['method']
            key = [k for k, v in method_labels.items() if v == mname][0]
            seg_t = clean_window[s:e + 1]
            seg_r = repaired_by_method[mname][s:e + 1]
            row['length_min'] = float(length_min)
            row['std_ratio'] = float(seg_r.std() / (seg_t.std() + 1e-12))
            all_results.append({
                'length': length_min,
                'method': key,
                'wasserstein': row['wasserstein'],
                'acf_l2': row['acf_l2'],
            })

# 按长度和方法平均
import pandas as pd
df = pd.DataFrame(all_results)

# 落盘原始逐事件结果，供 submission_srl/image_code/draw_fig7_hd.py 复绘
RAW_CSV = r'G:\SeisY\docs\dssrr_paper\experiments\length_scan_raw.csv'
df.to_csv(RAW_CSV, index=False)
print(f"Raw results saved to {RAW_CSV}  ({len(df)} rows)")

# 统一结构的汇总表 + 明细（含 std_ratio）+ 溯源
writer.write_summary()
writer.rewrite_detail()
writer.write_provenance(script_path=os.path.abspath(__file__))

for mname in focus_methods:
    sub = df[df['method'] == mname].groupby('length')[['wasserstein', 'acf_l2']].mean().reset_index()
    results[mname]['lengths'] = sub['length'].values
    results[mname]['wasserstein'] = sub['wasserstein'].values
    results[mname]['acf_l2'] = sub['acf_l2'].values
# 画图
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) Wasserstein vs gap length
ax = axes[0]
for mname in focus_methods:
    ax.plot(results[mname]['lengths'], results[mname]['wasserstein'],
            marker=markers[mname], color=colors[mname], linewidth=2.0,
            markersize=6, markevery=2, label=method_labels[mname])

ax.set_xscale('log')
ax.set_xlabel('Gap length (min)', fontsize=10)
ax.set_ylabel('Wasserstein distance (lower better)', fontsize=10)
ax.set_title('(a) Amplitude distribution fidelity', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=9, frameon=False)

# (b) ACF L2 vs gap length
ax = axes[1]
for mname in focus_methods:
    ax.plot(results[mname]['lengths'], results[mname]['acf_l2'],
            marker=markers[mname], color=colors[mname], linewidth=2.0,
            markersize=6, markevery=2, label=method_labels[mname])

ax.set_xscale('log')
ax.set_xlabel('Gap length (min)', fontsize=10)
ax.set_ylabel('ACF $L_2$ distance (lower better)', fontsize=10)
ax.set_title('(b) Temporal correlation fidelity', fontsize=11, fontweight='bold', loc='left')
ax.legend(fontsize=9, frameon=False)

plt.tight_layout()
# 注意：论文 Fig.7 的正式渲染由 submission_srl/image_code/draw_fig7_hd.py 完成
# （截尾均值 + 期刊样式）。此处只输出一张同源 quick-look，避免覆盖高清版。
out_path = f'{FIG_DIR}/_backup/fig05_length_effect_quicklook.png'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\nFigure saved to {out_path}")
plt.close()
