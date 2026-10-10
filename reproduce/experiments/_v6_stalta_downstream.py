"""STA/LTA 下游可用性实验

检验论文 §Seam Boundary Fidelity 中的操作性主张：
"DSSRR 的输出可以直接进入 STA/LTA 检测流程，无需对修复区间做二次 taper/mask"。

对每个案例的修复结果与真值分别计算 STA/LTA 特征函数（CF），在异常区间内比较：
  1) n_trigger —— 落在异常区间（含两侧各 100 样本）内的触发次数（假触发）
  2) cf_wasserstein —— 异常区间内 CF 分布与真值 CF 分布的 Wasserstein 距离

只读取 v6 已落盘的 mseed 与真值，不重跑实验。
"""
import numpy as np
import os
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.stats import wasserstein_distance

ROOT = r'G:\SeisY\docs\dssrr_paper\experiments\results_E2r_full_v6'
CLEAN_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h'
OUT = r'G:\SeisY\docs\dssrr_paper\experiments\v6_stalta_downstream.csv'
SR = 6.625
NSTA, NLTA = int(1.0 * SR), int(30.0 * SR)   # 1 s / 30 s 标准短长窗
THR_ON, THR_OFF = 5.0, 3.0
PAD = 100                                     # 异常区间两侧各 100 样本计入假触发统计

METHODS = ['Linear', 'CubicSpline', 'SSA', 'FFT', 'UNet', 'DSSRR']

clean_map = {}
for f in sorted(os.listdir(CLEAN_DIR)):
    if not f.endswith('.mseed'):
        continue
    p = f[:-6].split('.')
    clean_map[f'{p[1]}_{p[4].split("-")[0]}'] = os.path.join(CLEAN_DIR, f)


def cf_of(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    return classic_sta_lta(x, NSTA, NLTA)


rows = []
for ev in sorted(os.listdir(ROOT)):
    p_ev = os.path.join(ROOT, ev)
    if not os.path.isdir(p_ev):
        continue
    head, scenario = ev.rsplit('_', 1)
    station, start = head.split('_', 1)
    clean = read(clean_map[f'{station}_{start}'])[0].data.astype(float)

    for anom in sorted(os.listdir(p_ev)):
        p_anom = os.path.join(p_ev, anom)
        if not os.path.isdir(p_anom):
            continue
        for lvl in sorted(os.listdir(p_anom)):
            cdir = os.path.join(p_anom, lvl)
            cf_path = os.path.join(cdir, 'corrupted.mseed')
            if not os.path.exists(cf_path):
                continue
            corr = read(cf_path)[0].data.astype(float)
            n = min(len(corr), len(clean))
            d = np.abs(corr[:n] - clean[:n])
            m = d > 1e-3 * np.std(clean[:n])
            s, e = int(np.where(m)[0][0]), int(np.where(m)[0][-1])

            # 只在异常区间附近 ±5 min 上计算，避免整条 12h 记录的开销
            lo, hi = max(0, s - int(300 * SR)), min(n, e + 1 + int(300 * SR))

            cf_truth = cf_of(clean[lo:hi])
            i0, i1 = s - lo, e + 1 - lo
            ref_cf = cf_truth[i0:i1]

            for mname in ['Truth'] + METHODS:
                if mname == 'Truth':
                    series = clean[:n]
                else:
                    fp = os.path.join(cdir, f'repaired_{mname}.mseed')
                    if not os.path.exists(fp):
                        continue
                    series = read(fp)[0].data.astype(float)[:n]
                cf = cf_of(series[lo:hi])
                trig = trigger_onset(cf, THR_ON, THR_OFF)
                j0, j1 = max(0, i0 - PAD), min(len(cf), i1 + PAD)
                n_trig = int(np.sum((trig[:, 0] >= j0) & (trig[:, 0] <= j1))) if len(trig) else 0
                seg_cf = cf[i0:i1]
                rows.append(dict(event=ev, scenario=scenario, anom_type=anom, level=lvl,
                                 method=mname, n_trigger=n_trig,
                                 cf_wasserstein=float(wasserstein_distance(ref_cf, seg_cf)),
                                 cf_std_ratio=float(seg_cf.std() / (ref_cf.std() + 1e-12))))

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'rows={len(df)} -> {OUT}\n')

order = ['Truth', 'DSSRR', 'FFT', 'UNet', 'SSA', 'CubicSpline', 'Linear']
sub = df[df.anom_type == 'dropout']
print('=== dropout cases: mean false triggers in anomaly window ===')
print(sub.groupby('method')['n_trigger'].agg(['mean', 'sum', 'max']).reindex(order).round(2).to_string())
print('\n=== all cases: CF distribution distance to truth (lower = better) ===')
print(df.groupby('method')['cf_wasserstein'].mean().reindex(order).round(4).to_string())
print('\n=== dropout cases: CF distribution distance to truth ===')
print(sub.groupby('method')['cf_wasserstein'].mean().reindex(order).round(4).to_string())
