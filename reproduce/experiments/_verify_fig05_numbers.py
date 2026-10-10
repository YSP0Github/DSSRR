"""核对正文中所有与长度扫描（图 5 / Table S2-S3）相关的数字是否与新数据一致。

口径与 fig05_length_effect.py 完全一致：逐长度对 4 个事件做截尾平均（去掉最好与最差）。
只读，不写任何文件。
"""
import numpy as np
import pandas as pd

CSV = r'G:\SeisY\docs\dssrr_paper\experiments\results_length_scan\stats_summary.csv'
EXCLUDE = [12.137540]
NAMES = ['DSSRR', 'Linear', 'FFT', 'SSA']


def trimmed(vals):
    v = np.sort(np.asarray(vals, dtype=float))
    return float(np.mean(v[1:-1])) if len(v) >= 4 else float(np.mean(v))


df = pd.read_csv(CSV)
df = df[df['method'].isin(NAMES)].copy()
mask = np.zeros(len(df), dtype=bool)
for L in EXCLUDE:
    mask |= np.isclose(df['length_min'].to_numpy(), L, atol=1e-6)
df = df[~mask]

# 逐长度截尾均值
res = {}
for n in NAMES:
    sub = df[df['method'] == n].groupby('length_min').agg(
        w=('wasserstein', trimmed), a=('acf_l2', trimmed), r=('std_ratio', trimmed)
    ).reset_index()
    res[n] = sub.set_index('length_min')

lens = res['DSSRR'].index.values
print(f'长度数（剔除 {EXCLUDE[0]} 后）: {len(lens)}')
print(f'长度列表: {[round(float(x),2) for x in lens]}')
print()

long6 = [L for L in lens if L >= 29]  # 30--90 min 六点
print(f'六个最长 gap: {[round(float(x),2) for x in long6]}')
print()

# --- ① L668-669: 六最长 gap 的均值 ---
print('=== ① 六最长 gap (30--90 min) 均值 ===')
for n in NAMES:
    w = res[n].loc[long6, 'w'].mean()
    a = res[n].loc[long6, 'a'].mean()
    print(f'  {n:8s} Wasserstein={w:.3f}   ACF L2={a:.3f}')
d_w = res['DSSRR'].loc[long6, 'w'].mean()
lin_w = res['Linear'].loc[long6, 'w'].mean()
ssa_a = res['SSA'].loc[long6, 'a'].mean()
d_a = res['DSSRR'].loc[long6, 'a'].mean()
print(f'  -> 稿件称: DSSRR 0.26 / 0.17; Linear 1.59 (W); SSA 0.99 (ACF)')
print(f'  -> 比值 W: Linear/DSSRR = {lin_w/d_w:.1f} (稿件 6.2)')
print(f'  -> 比值 ACF: SSA/DSSRR = {ssa_a/d_a:.1f} (稿件 6.0)')
print()

# --- ② L671: 插值类 ACF 斜率 per decade ---
print('=== ② 插值类 ACF L2 随长度的斜率 (per decade) ===')
x = np.log10(lens.astype(float))
for n in ['Linear', 'SSA']:
    y = res[n].loc[lens, 'a'].values
    slope = np.polyfit(x, y, 1)[0]
    print(f'  {n:8s} slope={slope:+.3f} per decade')
print('  -> 稿件称: +0.23 to +0.26 per decade')
print()

# --- ③ L673 / L678: ACF 区间 ---
print('=== ③ ACF L2 区间（全部 19 长度） ===')
for n in NAMES:
    y = res[n].loc[lens, 'a'].values
    print(f'  {n:8s} {y.min():.2f} -- {y.max():.2f}')
print('  -> 稿件称: DSSRR 0.07--0.33; FFT 0.07--0.24')
print()

# --- ④ L683-685: FFT std_ratio ---
print('=== ④ std_ratio（能量比）六最长 gap ===')
for n in ['FFT', 'DSSRR']:
    y = res[n].loc[long6, 'r'].values
    print(f'  {n:8s} {y.min():.3f} -- {y.max():.3f}')
# 逐 (event,length) 的 24 个案例（六长度）
fft_sub = df[(df['method'] == 'FFT') & (df['length_min'].isin(long6))]
print(f'  FFT 30--90min 的 (event,length) 案例数: {len(fft_sub)}')
print(f'  FFT 这些案例 std_ratio 是否全部 <1: {(fft_sub["std_ratio"] < 1).all()}')
print(f'  FFT std_ratio min/max (逐案例): {fft_sub["std_ratio"].min():.3f} / {fft_sub["std_ratio"].max():.3f}')
# 33 min 与 90 min 的截尾均值
for target in [33.0, 90.0]:
    near = lens[np.argmin(np.abs(lens - target))]
    print(f'  FFT @ {float(near):.2f} min: std_ratio={res["FFT"].loc[near, "r"]:.3f}')
print('  -> 稿件称: FFT 0.68--0.86 (均值), 0.87 @33min, 0.68 @90min; DSSRR 1.01--1.23')
print()

# --- ⑤ L690-691: Wasserstein 区间 + 4.3x ---
print('=== ⑤ Wasserstein 区间与分离度 ===')
for n in ['FFT', 'DSSRR']:
    y = res[n].loc[lens, 'w'].values
    print(f'  {n:8s} {y.min():.2f} -- {y.max():.2f}')
fft_long = res['FFT'].loc[long6, 'w'].mean()
d_long = res['DSSRR'].loc[long6, 'w'].mean()
print(f'  30--90min: FFT/DSSRR = {fft_long/d_long:.1f}x (稿件 4.3x)')
print()

# --- ⑥ L1205: DSSRR 是否在全部 19 长度领先 ---
print('=== ⑥ DSSRR 在全部长度 Wasserstein 最低？ ===')
n_win = int(np.sum(np.all([res['DSSRR'].loc[lens, 'w'].values <= res[n].loc[lens, 'w'].values + 1e-12
                           for n in NAMES[1:]], axis=0)))
print(f'  {n_win}/{len(lens)}')
