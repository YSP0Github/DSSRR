"""自动核对补充材料 Table S2/S3 是否与新长度扫描数据一致。

- Table S2 (tabS:lengthdisp): Wasserstein + ACF L2 两段
- Table S3 (tabS:lengthenergy): std_ratio
口径: 逐长度对 4 个事件做截尾均值 (best/worst 去掉) ± 全 4 事件标准差。
"""
import re
import numpy as np
import pandas as pd

CSV = r'G:\SeisY\docs\dssrr_paper\experiments\results_length_scan\stats_summary.csv'
TEX = r'G:\SeisY\docs\dssrr_paper\submission_srl\supplement\SRL-DSSRR-supplement.tex'
NAMES = ['DSSRR', 'Linear', 'FFT', 'SSA']
EXCLUDE = [12.137540]


def trimmed(v):
    v = np.sort(np.asarray(v, float))
    return float(np.mean(v[1:-1])) if len(v) >= 4 else float(np.mean(v))


df = pd.read_csv(CSV)
df = df[df['method'].isin(NAMES)].copy()
m = np.zeros(len(df), bool)
for L in EXCLUDE:
    m |= np.isclose(df['length_min'].to_numpy(), L, atol=1e-6)
df = df[~m]

agg = {}
for n in NAMES:
    s = df[df['method'] == n].groupby('length_min').agg(
        w=('wasserstein', trimmed), w_sd=('wasserstein', 'std'),
        a=('acf_l2', trimmed), a_sd=('acf_l2', 'std'),
        r=('std_ratio', trimmed), r_sd=('std_ratio', 'std'))
    agg[n] = s
lens = sorted(agg['DSSRR'].index)

# 解析 tex：找出所有 "X.XX & $a \pm b$ & ..." 行
txt = open(TEX, encoding='utf-8').read()
cell_re = re.compile(r'\$([0-9.]+)\s*\\pm\s*([0-9.]+)\$')
num_re = re.compile(r'^\s*([0-9]+\.[0-9]{2})\s*&')

parsed = []  # (length, [ (mean, sd) x4 ])
for l in txt.splitlines():
    mm = num_re.match(l)
    if not mm:
        continue
    cells = cell_re.findall(l)
    if len(cells) == 4:
        parsed.append((float(mm.group(1)), [(float(a), float(b)) for a, b in cells]))

print(f'解析到 {len(parsed)} 行数值（应为 19*3 = 57）')
# 分组：前 19 = S2 wasserstein, 中 19 = S2 acf, 后 19 = S3 energy
groups = ['S2-Wasserstein', 'S2-ACF', 'S3-std_ratio']
metrics = ['w', 'a', 'r']
bad = 0
for gi, gname in enumerate(groups):
    seg = parsed[gi * 19:(gi + 1) * 19]
    metric = metrics[gi]
    for (L, cells) in seg:
        Lx = lens[int(np.argmin(np.abs(np.array(lens) - L)))]
        for j, n in enumerate(NAMES):
            want_mean = agg[n].loc[Lx, metric]
            want_sd = agg[n].loc[Lx, metric + '_sd']
            got_mean, got_sd = cells[j]
            dm = abs(want_mean - got_mean)
            ds = abs(want_sd - got_sd)
            if dm > 0.0051 or ds > 0.0051:
                bad += 1
                print(f'  MISMATCH {gname} {L:.2f} {n:7s} '
                      f'tex={got_mean:.2f}±{got_sd:.2f}  data={want_mean:.3f}±{want_sd:.3f}')
print(f'\n不一致单元数: {bad}  (0 = 完全一致)')
