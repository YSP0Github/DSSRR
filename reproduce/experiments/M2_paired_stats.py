"""M2：配对统计与离散度报告（回应审稿意见 M2）

审稿意见 M2 原文要点：
  1. Figures 3--5、8 均为点估计，无误差棒 / 置信区间 / 离散度；
  2. 无任何配对统计检验支撑"DSSRR 显著优于"的判断；
  3. 伪重复风险：大量 trial 复用同一批真实背景窗口（**6 个安静窗 + 瞬态窗**），
     独立背景数量远小于 trial 数，常规不配对检验会高估自由度。
  修改建议：① 补误差棒（bootstrap 95% CI 或 IQR）；② 按背景窗口配对比较
     （paired bootstrap / Wilcoxon），以背景窗口而非 trial 作为独立样本单位；
     ③ 报告效应量而非仅 p 值；明确随机种子与 trial 数的确定依据。

【事实核查：本脚本落实的样本结构】
  稿件中**没有 "trial" 概念**。真实结构（已逐条核对 CSV 与 E2r_v6_batch.py）：
    v6          : 72 个注入单元 = **6 条背景记录** × 2 场景 × 2 异常型 × 3 等级
                  注意 `event` 字段**已含场景后缀**（如 S12_19760113_070152_stationary），
                  剥离后才是真实记录名；PROVENANCE 里的 n_cases=12 实为"记录×场景"数。
    length_scan : 80 个单元 = 4 条背景记录 × 20 个长度
    earth       : 18 个注入窗（长度 60/300/600/1800 s，分别 6/5/4/3 个）；仅 1 条记录
  ⇒ 审稿人说的"6 个安静窗"是**对的**：v6 的独立背景记录数就是 **6**。
  ⇒ "伪重复"的真实结构 = 72 个单元嵌套在 6 条背景记录内。
  ⇒ 因此本脚本报告三个层级：
      unit           n = 72 / 80 / 18   （每单元是一次独立的异常注入实现）
      record         n = 6  / 4  / 1    （簇稳健主口径）
      record×scenario n = 12 / — / —    （敏感性：场景是否算独立）
  并在 n 很小、p 已触下限时明确标注。

输出目录：``experiments/results_stats/``
  dispersion_<ds>.csv   每 (方法[, 分组]) 的 n / 中位数 / IQR / bootstrap 95% CI
  paired_<ds>.csv       每 (指标, 基线) vs DSSRR 的配对检验 + 效应量 + Holm 校正
  perrecord_<ds>.csv    每背景记录的中位数差（用于"k/n 条记录方向一致"陈述）
  M2_SUMMARY.md         人类可读汇总（供写回复信与正文）
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# ---------------------------------------------------------------- 配置

ROOT = r"G:\SeisY\docs\dssrr_paper\experiments"
OUT_DIR = os.path.join(ROOT, "results_stats")

SEED = 7            # 与 E2r_v6_batch / length scan 落盘时一致
N_BOOT = 20000      # bootstrap 重采样次数
ALPHA = 0.05

METHOD_ORDER = ["Linear", "CubicSpline", "FFT", "SSA", "UNet", "DSSRR"]
REFERENCE = "DSSRR"     # 所有配对比较都以 DSSRR 为基准

# 指标的"最优值"。绝大多数指标越小越好（最优值 0），但 std_ratio 是
# 修复段与真值的标准差之比，**最优值是 1.0**（恰好相等），不是越小越好。
# 对这类指标，配对检验前必须先做 |x - optimal| 变换，否则方向会整体反转：
# DSSRR 的 std_ratio ≈ 1.05（贴近 1，更好）、插值类 ≈ 0.28（远离 1，更差），
# 不变换会把 DSSRR 判成"被所有基线碾压"。
# 注意：dispersion_table 仍报告**未变换**的原始值（描述性统计，读者要看
# std_ratio 本身的分布）；只有配对检验与 perrecord 的差值走变换。
OPTIMAL_VALUE = {"std_ratio": 1.0}


def orient(values, metric):
    """把指标变换成"越小越好"的形式：std_ratio -> |x - 1|，其余原样返回。"""
    opt = OPTIMAL_VALUE.get(metric)
    return values if opt is None else np.abs(values - opt)

DATASETS = {
    "v6": dict(
        path=os.path.join(ROOT, "results_E2r_full_v6", "stats_summary.csv"),
        unit_cols=["event", "anom_type", "level"],
        record_derive="event",          # 从 event 剥离 _stationary/_transient 得到真实记录
        cluster2_col="event",           # 记录×场景
        group_cols=[],
        metrics=["wasserstein", "envelope_dist", "spec_entropy_err", "acf_l2"],
        subset=None,
        label="E-RealInj 全量（72 单元 / 6 记录）",
    ),
    "v6_stat_dropout": dict(
        # 图 3 用的子集：stationary × dropout
        path=os.path.join(ROOT, "results_E2r_full_v6", "stats_summary.csv"),
        unit_cols=["event", "anom_type", "level"],
        record_derive="event",
        cluster2_col="event",
        group_cols=[],
        metrics=["wasserstein", "envelope_dist", "spec_entropy_err", "acf_l2"],
        subset=("scenario == 'stationary' and anom_type == 'dropout'"),
        label="E-RealInj 子集（stationary × dropout，18 单元 / 6 记录）——图 3 口径",
    ),
    "v6_dropout": dict(
        # 论文主场景：长时缺口。异常段 300/900/3600 s
        path=os.path.join(ROOT, "results_E2r_full_v6", "stats_summary.csv"),
        unit_cols=["event", "level"],
        record_derive="event",
        cluster2_col="event",
        group_cols=[],
        metrics=["wasserstein", "envelope_dist", "spec_entropy_err", "acf_l2"],
        subset="anom_type == 'dropout'",
        label="E-RealInj 仅 dropout（36 单元 / 6 记录；异常段 300--3600 s）",
    ),
    "v6_spike": dict(
        # 边界情形：孤立短尖峰。异常段仅 0.75/5.3/23.5 s（5/35/156 样本）
        path=os.path.join(ROOT, "results_E2r_full_v6", "stats_summary.csv"),
        unit_cols=["event", "level"],
        record_derive="event",
        cluster2_col="event",
        group_cols=[],
        metrics=["wasserstein", "envelope_dist", "spec_entropy_err", "acf_l2"],
        subset="anom_type == 'spike'",
        label="E-RealInj 仅 spike（36 单元 / 6 记录；异常段仅 0.75--23.5 s = 5--156 样本）",
    ),
    "length_scan": dict(
        path=os.path.join(ROOT, "results_length_scan", "stats_summary.csv"),
        unit_cols=["event", "length_min"],
        record_derive="event",          # 无场景后缀，正则不匹配即原值 → 4 条记录
        cluster2_col=None,
        group_cols=["length_min"],
        metrics=["wasserstein", "acf_l2", "std_ratio"],
        subset=None,
        label="长度扫描（80 单元 / 4 记录）",
    ),
    "earth": dict(
        path=os.path.join(ROOT, "results_earth", "earth_instances.csv"),
        unit_cols=["gap_sec", "s", "e"],
        record_derive=None,
        cluster2_col=None,
        group_cols=["gap_sec"],
        metrics=["wasserstein", "acf_l2"],
        subset=None,
        label="地球迁移（18 个注入窗 / 1 条记录）",
    ),
}


# ---------------------------------------------------------------- 统计工具

def bootstrap_ci_median(x, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """百分位法 bootstrap 95% CI（对中位数）。n<3 时返回 NaN。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_median_cluster(values, clusters, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """簇 bootstrap：以**记录**为重采样单位，先在簇内取中位数再抽簇。

    这是回应"伪重复"的核心手法——自由度按簇数算，而不是按单元数。
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    k = len(uniq)
    if k < 3:
        return np.nan, np.nan
    per_cluster = np.array([np.nanmedian(values[clusters == c]) for c in uniq])
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, k, size=(n_boot, k))
    meds = np.median(per_cluster[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def rank_biserial(d):
    """配对 Wilcoxon 的效应量：秩二列相关 r_rb = (W+ - W-) / (W+ + W-)，范围 [-1,1]。"""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return np.nan
    r = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_pos = r[d > 0].sum()
    w_neg = r[d < 0].sum()
    tot = w_pos + w_neg
    return float((w_pos - w_neg) / tot) if tot > 0 else np.nan


def holm(pvals):
    """Holm-Bonferroni 逐步校正。"""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if not ok.any():
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for i, j in enumerate(order):
        adj = (m - i) * p[j]
        running = max(running, adj)
        out[j] = min(running, 1.0)
    return out


def p_floor(n):
    """n 对配对、全同号时 Wilcoxon 双侧检验能达到的最小 p（即分辨率下限）。"""
    if n < 1:
        return np.nan
    return float(min(1.0, 2.0 / (2 ** n)))


def signflip_perm_test(d, stat="mean", n_perm=200000, seed=SEED, exact_max=16):
    """配对符号翻转置换检验（对均值的稳健检验）。

    论文的优势主张是**均值口径**（"reduces error by 32--37%" 由均值算得），
    而 Wilcoxon 检验的是中位数/秩。两者在重尾数据上会给出不同结论，
    因此这里额外给出针对**均值**与**中位数**的符号翻转置换检验：
    H0 下每个配对差值的符号可独立翻转，构造零分布。n<=16 用精确枚举，否则蒙特卡洛。
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n < 6:
        return np.nan
    f = np.mean if stat == "mean" else np.median
    obs = f(d)
    rng = np.random.RandomState(seed)
    if n <= exact_max:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
        null = f(signs * d, axis=1)
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
        null = f(signs * d, axis=1)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1))


def bootstrap_ci_mean(x, n_boot=N_BOOT, alpha=ALPHA, seed=SEED):
    """百分位法 bootstrap 95% CI（对均值）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    mus = x[idx].mean(axis=1)
    lo, hi = np.percentile(mus, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_test(a, b, cluster_ids=None, cluster2_ids=None,
                n_boot=N_BOOT, seed=SEED):
    """配对检验：DSSRR(a) vs 基线(b)，指标越小越好。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    d = a - b                      # 负值 = DSSRR 更优
    n = len(d)
    res = {"n_pairs": n, "median_diff": float(np.median(d)) if n else np.nan,
           "mean_diff": float(np.mean(d)) if n else np.nan,
           "frac_units_better": float(np.mean(d < 0)) if n else np.nan}

    if n >= 6 and np.any(d != 0):
        try:
            st = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
            res["W"] = float(st.statistic)
            res["p_unit"] = float(st.pvalue)
        except Exception:
            res["W"] = res["p_unit"] = np.nan
        try:
            res["p_unit_less"] = float(
                wilcoxon(a, b, alternative="less", zero_method="wilcox").pvalue)
        except Exception:
            res["p_unit_less"] = np.nan
        # 均值 / 中位数口径的符号翻转置换检验（论文优势主张是均值口径）
        res["p_perm_mean"] = signflip_perm_test(d, "mean", seed=seed)
        res["p_perm_median"] = signflip_perm_test(d, "median", seed=seed)
    else:
        res["W"] = res["p_unit"] = res["p_unit_less"] = np.nan
        res["p_perm_mean"] = res["p_perm_median"] = np.nan
    res["rank_biserial"] = rank_biserial(d)
    res["diff_ci_lo"], res["diff_ci_hi"] = bootstrap_ci_median(d, n_boot=n_boot, seed=seed)
    res["mean_diff_ci_lo"], res["mean_diff_ci_hi"] = bootstrap_ci_mean(
        d, n_boot=n_boot, seed=seed)

    def _cluster_block(ids, tag):
        cid = np.asarray(ids)[m]
        uniq = np.unique(cid)
        pa = np.array([np.nanmedian(a[cid == c]) for c in uniq])
        pb = np.array([np.nanmedian(b[cid == c]) for c in uniq])
        pd_ = pa - pb
        res[f"n_{tag}"] = len(uniq)
        res[f"frac_{tag}_better"] = float(np.mean(pd_ < 0)) if len(pd_) else np.nan
        if len(uniq) >= 5 and np.any(pd_ != 0):
            try:
                res[f"p_{tag}"] = float(
                    wilcoxon(pa, pb, alternative="two-sided",
                             zero_method="wilcox").pvalue)
            except Exception:
                res[f"p_{tag}"] = np.nan
        else:
            res[f"p_{tag}"] = np.nan
        res[f"p_{tag}_floor"] = p_floor(len(uniq))
        res[f"rank_biserial_{tag}"] = rank_biserial(pd_)
        lo, hi = bootstrap_ci_median_cluster(d, cid, n_boot=n_boot, seed=seed)
        res[f"diff_ci_lo_{tag}"], res[f"diff_ci_hi_{tag}"] = lo, hi

    if cluster_ids is not None:
        _cluster_block(cluster_ids, "record")
    else:
        res.update(n_record=np.nan, frac_record_better=np.nan, p_record=np.nan,
                   p_record_floor=np.nan, rank_biserial_record=np.nan,
                   diff_ci_lo_record=np.nan, diff_ci_hi_record=np.nan)
    if cluster2_ids is not None:
        _cluster_block(cluster2_ids, "recscen")
    else:
        res.update(n_recscen=np.nan, frac_recscen_better=np.nan, p_recscen=np.nan,
                   p_recscen_floor=np.nan, rank_biserial_recscen=np.nan,
                   diff_ci_lo_recscen=np.nan, diff_ci_hi_recscen=np.nan)
    return res


# ---------------------------------------------------------------- 主流程

def load(spec):
    df = pd.read_csv(spec["path"])
    if spec.get("subset"):
        df = df.query(spec["subset"]).copy()
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    if spec.get("record_derive"):
        df["_record"] = (df[spec["record_derive"]]
                         .astype(str)
                         .str.replace(r"_(stationary|transient)$", "", regex=True))
    else:
        df["_record"] = np.nan
    return df


def dispersion_table(df, spec):
    """每 (方法[, 分组]) 的 n / 中位数 / IQR / bootstrap 95% CI / 均值 / 标准差。"""
    rows = []
    group_cols = spec["group_cols"]
    metrics = spec["metrics"]
    unit_cols = spec["unit_cols"]
    has_record = spec.get("record_derive") is not None

    groups = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, sub in groups:
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(group_cols, key))
        for method in METHOD_ORDER:
            s = sub[sub["method"] == method]
            if s.empty:
                continue
            n_units = s.groupby(unit_cols, dropna=False).ngroups
            n_rec = s["_record"].nunique() if has_record else np.nan
            for metric in metrics:
                raw = s[metric].to_numpy(dtype=float)
                v = raw[np.isfinite(raw)]
                if len(v) == 0:
                    continue
                lo, hi = bootstrap_ci_median(v)
                mlo, mhi = bootstrap_ci_mean(v)
                if has_record:
                    lo_c, hi_c = bootstrap_ci_median_cluster(raw, s["_record"].to_numpy())
                else:
                    lo_c, hi_c = np.nan, np.nan
                rows.append(dict(
                    **base, method=method, metric=metric,
                    n=n_units, n_values=len(v), n_records=n_rec,
                    median=float(np.median(v)),
                    q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                    mean=float(np.mean(v)),
                    sd=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                    ci_lo=lo, ci_hi=hi,
                    ci_mean_lo=mlo, ci_mean_hi=mhi,
                    ci_lo_cluster=lo_c, ci_hi_cluster=hi_c,
                ))
    return pd.DataFrame(rows)


def paired_table(df, spec):
    """DSSRR vs 每个基线的配对检验（逐指标）。"""
    rows = []
    has_record = spec.get("record_derive") is not None
    for metric in spec["metrics"]:
        piv = df.pivot_table(index=spec["unit_cols"], columns="method",
                             values=metric, aggfunc="mean")
        if REFERENCE not in piv.columns:
            continue
        piv = orient(piv, metric)      # std_ratio 等"最优值非 0"的指标先统一方向
        rec = scen = None
        if has_record:
            # 单元索引里带 event(=记录×场景)，用 map 取回 _record
            ev = piv.index.get_level_values("event")
            lut = df.drop_duplicates("event").set_index("event")["_record"]
            rec = lut.reindex(ev).to_numpy()
            scen = np.asarray(ev)
        a = piv[REFERENCE].to_numpy(dtype=float)
        for base in METHOD_ORDER:
            if base == REFERENCE or base not in piv.columns:
                continue
            b = piv[base].to_numpy(dtype=float)
            r = paired_test(a, b, cluster_ids=rec, cluster2_ids=scen)
            r.update(metric=metric, baseline=base)
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for tag in ("unit", "record", "recscen", "perm_mean", "perm_median"):
        col = f"p_{tag}"
        if col in out.columns:
            out[f"p_{tag}_holm"] = np.nan
            for _, sub in out.groupby("metric"):
                out.loc[sub.index, f"p_{tag}_holm"] = holm(sub[col].to_numpy())
    return out


def perrecord_table(df, spec):
    """每背景记录的中位数（供"k/n 条记录方向一致"陈述）。"""
    if spec.get("record_derive") is None:
        return pd.DataFrame()
    rows = []
    for metric in spec["metrics"]:
        g = (df.groupby(["_record", "method"])[metric].median()
               .unstack("method"))
        g = orient(g, metric)      # 与 paired_table 同口径，保证 diff 方向一致
        for rec, r in g.iterrows():
            row = {"metric": metric, "record": rec}
            for m in METHOD_ORDER:
                if m in r.index:
                    row[m] = r[m]
            if REFERENCE in r.index:
                for base in METHOD_ORDER:
                    if base != REFERENCE and base in r.index:
                        row[f"diff_vs_{base}"] = r[REFERENCE] - r[base]
            rows.append(row)
    return pd.DataFrame(rows)


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 1e-4 else f"{p:.4f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    md = ["# M2 配对统计汇总", "",
          f"- 随机种子 seed = **{SEED}**（与 E2r_v6_batch / length scan 落盘一致）",
          f"- bootstrap 重采样 = {N_BOOT} 次，百分位法 95% CI",
          "- 效应量：配对 Wilcoxon 秩二列相关 $r_{rb}=(W_+-W_-)/(W_++W_-)\\in[-1,1]$",
          "- 指标均为 **越小越好** ⇒ 差值 (DSSRR − 基线) < 0 表示 DSSRR 更优",
          "- **例外**：`std_ratio`（能量比）的最优值是 **1.0**，故其配对统计量先做 $|x-1|$ "
          "变换，变换后同样满足「越小越好」；`dispersion_*` 表仍报告未变换的原始值",
          "- 层级：`unit` = 注入单元配对；`record` = 先按背景记录取中位数再配对（**主口径**）；"
          "`recscen` = 按记录×场景配对（敏感性）",
          "- `*_floor` = 该簇数下全同号时 Wilcoxon 双侧 p 的下限（分辨率极限）",
          ""]

    for name, spec in DATASETS.items():
        df = load(spec)
        disp = dispersion_table(df, spec)
        pair = paired_table(df, spec)
        perrec = perrecord_table(df, spec)
        disp.to_csv(os.path.join(OUT_DIR, f"dispersion_{name}.csv"),
                    index=False, encoding="utf-8-sig")
        pair.to_csv(os.path.join(OUT_DIR, f"paired_{name}.csv"),
                    index=False, encoding="utf-8-sig")
        if not perrec.empty:
            perrec.to_csv(os.path.join(OUT_DIR, f"perrecord_{name}.csv"),
                          index=False, encoding="utf-8-sig")

        print(f"\n{'=' * 78}\n{name}: {spec['label']}\n{'=' * 78}")
        n_rec = disp["n_records"].max() if "n_records" in disp else np.nan
        print(f"记录数 = {n_rec}")
        for metric in spec["metrics"]:
            sub = pair[pair["metric"] == metric]
            if sub.empty:
                continue
            tag = (f"（统计量 |x-{OPTIMAL_VALUE[metric]:g}|）"
                   if metric in OPTIMAL_VALUE else "")
            print(f"\n--- {metric} {tag} ---")
            for _, r in sub.iterrows():
                if np.isfinite(r.get("n_record", np.nan)):
                    rec_s = (f"rec {int(r['frac_record_better'] * r['n_record'])}"
                             f"/{int(r['n_record'])} "
                             f"p={fmt_p(r['p_record']):>8s}"
                             f"(floor {fmt_p(r['p_record_floor'])}) "
                             f"holm={fmt_p(r['p_record_holm']):>8s}")
                else:
                    rec_s = "rec n/a"
                print(f"  vs {r['baseline']:11s} "
                      f"Δmean={r['mean_diff']:+.4f} Δmed={r['median_diff']:+.4f} "
                      f"| Wilcoxon p={fmt_p(r['p_unit']):>9s} holm={fmt_p(r['p_unit_holm']):>9s} "
                      f"| 置换(均值) p={fmt_p(r['p_perm_mean']):>9s} holm={fmt_p(r['p_perm_mean_holm']):>9s} "
                      f"| r={r['rank_biserial']:+.3f} | {rec_s}")

        md += [f"## {name} —— {spec['label']}", "",
               f"背景记录数 **{int(n_rec) if np.isfinite(n_rec) else 'n/a'}**"
               f"（审稿意见所指的「6 个安静窗」即此）", ""]
        for metric in spec["metrics"]:
            sub = pair[pair["metric"] == metric]
            if sub.empty:
                continue
            tag = (f"（统计量 $|x - {OPTIMAL_VALUE[metric]:g}|$）"
                   if metric in OPTIMAL_VALUE else "")
            md += [f"### {metric}{tag}", "",
                   "| 基线 | n单元 | Δ均值 | 95% CI (Δ均值) | 置换p(均值) | p_holm | "
                   "Δ中位数 | Wilcoxon p | p_holm | r_rb "
                   "| n记录 | 记录方向一致 | p(record) | p_holm |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for _, r in sub.iterrows():
                nb = (int(r["frac_record_better"] * r["n_record"])
                      if np.isfinite(r["n_record"]) else np.nan)
                nrec = int(r["n_record"]) if np.isfinite(r["n_record"]) else "n/a"
                md.append(
                    f"| {r['baseline']} | {int(r['n_pairs'])} "
                    f"| {r['mean_diff']:+.4f} "
                    f"| [{r['mean_diff_ci_lo']:+.4f}, {r['mean_diff_ci_hi']:+.4f}] "
                    f"| {fmt_p(r['p_perm_mean'])} | {fmt_p(r['p_perm_mean_holm'])} "
                    f"| {r['median_diff']:+.4f} "
                    f"| {fmt_p(r['p_unit'])} | {fmt_p(r['p_unit_holm'])} "
                    f"| {r['rank_biserial']:+.3f} "
                    f"| {nrec} | {nb}/{nrec} "
                    f"| {fmt_p(r['p_record'])} | {fmt_p(r['p_record_holm'])} |")
            md.append("")
        md.append("")

    with open(os.path.join(OUT_DIR, "M2_SUMMARY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"\n\n输出目录：{OUT_DIR}")
    print("  " + "\n  ".join(sorted(os.listdir(OUT_DIR))))


if __name__ == "__main__":
    main()
