"""M4：真实档案案例的**定量一致性**评估（回应审稿意见 M4）

背景
----
审稿意见 M4：§3.7 的真实案例评估只有目视描述（"no visible seams, no artificial
smoothness…"），缺少定量指标，也没有"健康窗之间自然变异"的基准带。

做法（无需 ground truth）
------------------------
对每个案例取**主导异常窗** [s, e]（长度 L 个样本），然后：

1. **自然变异基准带**：在**输入记录**中，从异常窗之外**非重叠地**切出若干长度同为 L
   的"健康窗"；对**所有健康窗对** (i, j) 计算同一套指标 → 得到"两段互不相干的健康数据
   之间本来就会差多少"的分布（中位数 + 5–95 百分位）。
   （注：用 `修复前` 记录，因为异常窗之外 DSSRR 只改动了 <1.5% 的零散样本。）

2. **输入窗 / 修复窗**：分别用 `before[s:e]` 与 `after[s:e]` 去比每一个健康窗，
   得到两条分布。若修复窗落进自然带、而输入窗远在带外，即定量说明"修复后与档案其余
   部分不可区分"。

3. **接缝跳变**（boundary jump）：主导窗左右两端 |after - 相邻输入样本| / 局部背景 RMS，
   与健康数据中 |相邻样本差| / RMS 的自然分布比较。

指标与注入实验同口径（`E2r_v5_batch.compute_stats_metrics`）：
`wasserstein` / `acf_l2` / `envelope_dist` / `spec_entropy_err`，另加
`psd_log_rmse`（对数 PSD 距离）与 `std_ratio`（能量比，1 为最优）。

输出：`results_obs/M4_archival_metrics.csv`（逐窗对明细）+ 终端汇总表。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from obspy import read
from scipy.stats import wasserstein_distance
from scipy.signal import hilbert, welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SR = 6.625
OBS = os.path.join(HERE, "results_obs")
BEFORE_DIR = os.path.join(OBS, "修复前")
OUT = os.path.join(OBS, "M4_archival_metrics.csv")

# 与 fig09_archival_repair.py::CASES 保持一致（小时，相对记录起点）
CASES = [
    dict(key="C1", name="60-min dropout", station="S15 1972-09-17",
         file="XA.S15.00.MHZ.19720917_133809-19720917_233809.mseed",
         s_h=0.919, e_h=1.927),
    dict(key="C2", name="18-min spike cluster", station="S12 1976-01-13",
         file="XA.S12.01.MHZ.19760113_061654-19760113_161654.mseed",
         s_h=4.819, e_h=5.124),
    dict(key="C3", name="17-min step + recovery", station="S15 1976-01-13",
         file="XA.S15.01.MHZ.19760113_061655-19760113_161655.mseed",
         s_h=5.508, e_h=5.798),
]


# ----------------------------------------------------------------------------- 指标
def _acf_upto(x, maxlag=200):
    """与 E2r_v5_batch.compute_stats_metrics 里的 acf() **完全等价**，只算前 maxlag 个 lag。

    V5 用 np.correlate(x, x, 'full') 再取前 min(200, n) 个 lag，复杂度 O(n^2)；
    这里只算需要的 lag，复杂度 O(n*maxlag)，在 n≈24000 时快 ~120 倍，数值一致。
    """
    x = x - np.mean(x)
    n = len(x)
    L = min(maxlag, n)
    d0 = float(np.dot(x, x)) + 1e-12
    out = np.empty(L, dtype=float)
    for k in range(L):
        out[k] = np.dot(x[:n - k], x[k:]) / d0
    return out


def _spec_entropy(x):
    f, Pxx = welch(x, fs=SR, nperseg=min(512, len(x) // 2), scaling="density")
    P = Pxx / (Pxx.sum() + 1e-12)
    P = np.clip(P, 1e-12, None)
    return -np.sum(P * np.log(P)) / np.log(len(P))


def seg_metrics(x, y):
    """两段等长（或不等长）数据之间的统计保真度指标，全部**越小越好**，std_ratio 例外（1 最优）。"""
    m = {}
    m["wasserstein"] = float(wasserstein_distance(x, y))
    a, b = _acf_upto(x), _acf_upto(y)
    L = min(len(a), len(b))
    m["acf_l2"] = float(np.sqrt(np.mean((a[:L] - b[:L]) ** 2)))
    m["envelope_dist"] = float(wasserstein_distance(np.abs(hilbert(x)), np.abs(hilbert(y))))
    m["spec_entropy_err"] = float(abs(_spec_entropy(x) - _spec_entropy(y)))
    _, Px = welch(x, fs=SR, nperseg=min(4096, len(x)))
    _, Py = welch(y, fs=SR, nperseg=min(4096, len(y)))
    L2 = min(len(Px), len(Py))
    m["psd_log_rmse"] = float(np.sqrt(np.mean((np.log10(Px[:L2] + 1e-12)
                                               - np.log10(Py[:L2] + 1e-12)) ** 2)))
    m["std_ratio"] = float(np.std(x) / (np.std(y) + 1e-12))
    return m


METRICS = ["wasserstein", "acf_l2", "envelope_dist", "spec_entropy_err",
           "psd_log_rmse", "std_ratio"]
# 打印/表格里用的短名与小数位
LABEL = {"wasserstein": "Wasserstein (amplitude dist.)",
         "acf_l2": "ACF $L_2$ (temporal structure)",
         "envelope_dist": "Envelope distance",
         "spec_entropy_err": "Spectral entropy error",
         "psd_log_rmse": "log-PSD RMSE (spectral shape)",
         "std_ratio": "Energy ratio std(repaired)/std(healthy)"}
FMT = {"wasserstein": "%.3f", "acf_l2": "%.3f", "envelope_dist": "%.3f",
       "spec_entropy_err": "%.4f", "psd_log_rmse": "%.4f", "std_ratio": "%.3f"}


# ----------------------------------------------------------------------------- 主流程
def analyse(case):
    b = read(os.path.join(BEFORE_DIR, case["file"]))[0]
    a = read(os.path.join(OBS, case["file"]))[0]
    n = min(b.stats.npts, a.stats.npts)
    bef = b.data[:n].astype(float)
    aft = a.data[:n].astype(float)
    changed = np.abs(aft - bef) > 1e-9

    s = int(round(case["s_h"] * 3600 * SR))
    e = min(int(round(case["e_h"] * 3600 * SR)), n - 1)
    L = e - s + 1

    # ---- 健康窗：从输入记录里非重叠地切长度 L 的块，跳过与主导窗重叠者 ----
    starts = list(range(0, n - L + 1, L))
    healthy = [t for t in starts if t + L - 1 < s or t > e]
    H = [bef[t:t + L] for t in healthy]

    x_in, x_rep = bef[s:e + 1], aft[s:e + 1]

    rows = []
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            rows.append(dict(case=case["key"], kind="healthy-healthy", i=i, j=j,
                             **seg_metrics(H[i], H[j])))
    for j in range(len(H)):
        rows.append(dict(case=case["key"], kind="input-healthy", i=-1, j=j,
                         **seg_metrics(x_in, H[j])))
        rows.append(dict(case=case["key"], kind="repaired-healthy", i=-1, j=j,
                         **seg_metrics(x_rep, H[j])))

    # ---- 接缝跳变 ----
    # 定义（与 §3.4 一致）：|修复段端点的修复值 − 紧邻的原始样本|，用**接缝旁的局部背景 RMS**
    # 归一化（此处取 5 s 窗，不含异常本身）。同时给出**绝对计数**，因为档案流水线把
    # 输出写回整数计数，接缝跳变有 1 个 count 的量化下限。
    w = int(round(5 * SR))
    c1 = np.concatenate([[0.0], np.cumsum(bef)])
    c2 = np.concatenate([[0.0], np.cumsum(bef * bef)])
    m = (c1[w:] - c1[:-w]) / w
    v = (c2[w:] - c2[:-w]) / w - m ** 2
    roll_std = np.sqrt(np.maximum(v, 0.0))          # roll_std[i] = std(bef[i:i+w])

    def rms_at(i):
        """样本 i 附近的局部背景 RMS（取以 i 为起点的 5 s 窗；越界则回退）。"""
        j = min(max(i, 0), len(roll_std) - 1)
        s = float(roll_std[j])
        return s if s > 1e-9 else float(np.std(bef))

    idx = np.where(changed)[0]
    brk = np.where(np.diff(idx) > 1)[0]
    segs = np.split(idx, brk + 1)
    seam_rows = []
    for sg in segs:
        a0, b0 = int(sg[0]), int(sg[-1])
        if a0 == 0 or b0 >= n - 1:
            continue
        rl, rr = rms_at(a0 - w), rms_at(b0 + 1)
        dl, dr = abs(aft[a0] - bef[a0 - 1]), abs(aft[b0] - bef[b0 + 1])
        seam_rows.append(dict(case=case["key"], start=a0, stop=b0, length=len(sg),
                              d_left=dl, d_right=dr,
                              jump_left=dl / rl, jump_right=dr / rr))
    seam_df = pd.DataFrame(seam_rows)
    seam_df["jump_max"] = seam_df[["jump_left", "jump_right"]].max(axis=1)
    seam_df["d_max"] = seam_df[["d_left", "d_right"]].max(axis=1)

    # 主修复段（最长连续改动段）的两条接缝 —— 与 §3.4 的单异常注入最可比
    dom = seam_df.loc[seam_df["length"].idxmax()]

    # 自然参照：健康样本上 |相邻样本差| / 局部 RMS 的分布（同一把尺子）
    hp = np.where(changed, np.nan, bef)
    dstep = np.abs(np.diff(hp))
    rloc_full = np.empty(n, dtype=float)
    rloc_full[:len(roll_std)] = roll_std
    rloc_full[len(roll_std):] = roll_std[-1]
    rloc = rloc_full[:len(dstep)]
    step = dstep / np.where(rloc > 1e-9, rloc, np.nan)
    step = step[np.isfinite(step)]
    nat = dict(n_seg=len(seam_df), n_dom=int(dom["length"]),
               dom_left=float(dom["jump_left"]), dom_right=float(dom["jump_right"]),
               dom_left_cnt=float(dom["d_left"]), dom_right_cnt=float(dom["d_right"]),
               seam_median=float(seam_df["jump_max"].median()),
               seam_max=float(seam_df["jump_max"].max()),
               seam_max_cnt=float(seam_df["d_max"].max()),
               p50=float(np.percentile(step, 50)), p95=float(np.percentile(step, 95)),
               p999=float(np.percentile(step, 99.9)),
               p99_99=float(np.percentile(step, 99.99)), max_step=float(step.max()))
    nat["seam_pctile"] = float((step < nat["seam_max"]).mean() * 100)
    nat["dom_pctile"] = float((step < max(nat["dom_left"], nat["dom_right"])).mean() * 100)

    # 自然参照（以**数字计数**为单位，最直观；数据是整数计数）
    cnt = dstep[np.isfinite(dstep)]
    nat["cnt_p50"] = float(np.percentile(cnt, 50))
    nat["cnt_p95"] = float(np.percentile(cnt, 95))
    nat["cnt_p999"] = float(np.percentile(cnt, 99.9))
    nat["cnt_max"] = float(cnt.max())

    meta = dict(key=case["key"], name=case["name"], station=case["station"],
                L=L, L_min=L / SR / 60, n_healthy=len(H),
                n_changed=int(changed[s:e + 1].sum()),
                frac_changed=float(changed[s:e + 1].mean()),
                n_changed_all=int(changed.sum()))
    return rows, seam_df, nat, meta


def main():
    all_rows, seams, nats, metas = [], {}, {}, []
    for case in CASES:
        rows, seam_df, nat, meta = analyse(case)
        all_rows += rows
        seams[case["key"]] = seam_df
        nats[case["key"]] = nat
        metas.append(meta)
        print(f"[done] {case['key']} {case['name']}  L={meta['L_min']:.1f} min  "
              f"healthy windows={meta['n_healthy']}  changed={meta['frac_changed']*100:.1f}%",
              flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 116)
    print("M4 真实档案案例定量一致性（中位数 [5–95 百分位]；末列为修复窗在自然带中的百分位排名）")
    print("=" * 116)
    for meta in metas:
        c = meta["key"]
        sub = df[df["case"] == c]
        hh = sub[sub["kind"] == "healthy-healthy"]
        iv = sub[sub["kind"] == "input-healthy"]
        rv = sub[sub["kind"] == "repaired-healthy"]
        print(f"\n{meta['key']}  {meta['name']}  （{meta['station']}）")
        print(f"  主导窗 {meta['L_min']:.1f} min / {meta['L']} 样本；改动 "
              f"{meta['n_changed']} 样本（{meta['frac_changed']*100:.1f}%）；"
              f"健康窗 {meta['n_healthy']} 个 → {len(hh)} 个健康-健康窗对")
        print(f"  {'metric':<38}{'input':>22}{'repaired':>22}{'natural band':>24}{'pctile':>9}")
        for m in METRICS:
            hv = hh[m].to_numpy(dtype=float)
            ivv, rvv = iv[m].to_numpy(dtype=float), rv[m].to_numpy(dtype=float)

            def q(d):
                return f"{np.median(d):.3f} [{np.percentile(d,5):.3f}, {np.percentile(d,95):.3f}]"
            pct = float((hv < np.median(rvv)).mean() * 100)
            print(f"  {LABEL[m]:<38}{q(ivv):>22}{q(rvv):>22}{q(hv):>24}{pct:>8.1f}%")
        sd, nt = seams[c], nats[c]
        print(f"  主修复段（{nt['n_dom']} 样本）接缝跳变：左 {nt['dom_left']:.3f}（{nt['dom_left_cnt']:.0f} count）"
              f"  右 {nt['dom_right']:.3f}（{nt['dom_right_cnt']:.0f} count）"
              f"  → 自然相邻样本差第 {nt['dom_pctile']:.1f} 百分位")
        print(f"  全部 {nt['n_seg']} 个改动段：接缝跳变中位 {nt['seam_median']:.3f}  最大 "
              f"{nt['seam_max']:.3f}（{nt['seam_max_cnt']:.0f} count）"
              f"  → 自然第 {nt['seam_pctile']:.2f} 百分位")
        print(f"  健康数据 |相邻样本差|：p50={nt['cnt_p50']:.0f}  p95={nt['cnt_p95']:.0f}  "
              f"p99.9={nt['cnt_p999']:.0f}  max={nt['cnt_max']:.0f} count"
              f"   |  归一化后 p95={nt['p95']:.3f} p99.9={nt['p999']:.3f} max={nt['max_step']:.3f}")
        sd.to_csv(os.path.join(OBS, f"M4_seams_{c}.csv"), index=False, encoding="utf-8-sig")

    print(f"\n写出 {OUT}")
    print(f"      + M4_seams_C1/C2/C3.csv")


if __name__ == "__main__":
    main()
