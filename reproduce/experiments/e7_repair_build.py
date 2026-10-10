"""E7 修复流水线 v4：对 1976-01 S12/S15/S16 MHZ 应用多级修复，写入新库

- 原始库只读：G:\\SeisY_Project\\database\\1976\\01
- 新库（去异常后）：G:\\SeisY_Project\\database_repaired\\1976\\01

v4 多级修复策略（用户 2026-09-08 反馈整合）：
- 连续缺失 ≤ 2 点（含单尖峰缺失）→ 线性插值（问题1：短缺失不必 DSSRR）
- 连续缺失 3-20 点 → Z 分数清理（|z|>4 离群点替换为插值）
- 孤立尖峰型段（无缺失）→ 局部 MAD 尖峰修复（滚动中值 + 插值）
- 月震保护窗内尖峰/短缺失 → MoonquakeProtectedSpikeReplacer（问题2：
  只修孤立尖峰，不动月震 coda 主体；coda 由目录保护窗整体保护）
- 连续缺失 > 20 点 / 能量 / 冻结 / 饱和 / 阶跃 → DSSRR 谱重建
  （DSSRR 主场；参考段 ≥ 异常长度一半、上限 600s —— 问题3）
- 参考段重叠合并：DSSRR 参考窗与相邻异常段重叠 → 两段及中间区域合并
  为一个修复单元（问题3，避免参考窗拾取相邻异常段）
- 事件保护：event 段修复后还原为原始值（防止修复单元外扩覆盖）
- 参考保护：DSSRR 修复前，参考窗内属于 event 的样本替换为背景中值
- gap 段（>max_repair_sec）：保留原始数据（如实反映缺口）
"""
import glob
import os
import sys
import time
import argparse

import numpy as np
from obspy import read, Stream

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"G:\SeisY")
from baselines import DSSRRRepair
from e7_detector import E7Detector, verify_catalog, protect_osc_clusters, _merge_simple

SR = 6.625
IN = r"G:\SeisY_Project\database\1976\01"
OUT = r"G:\SeisY_Project\database_repaired\1976\01"
CATALOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "E7_catalog_197601.csv")

# 修复层默认参数（= 论文 v5 批处理口径，与 seisy/core/repair_settings.py
# 的 repair 组默认一致；批处理不读用户设置，保证结果可复现）
_REPAIR_DEFAULTS = {
    "lin_max_run": 2,
    "z_max_run": 20,
    "miss_frac_lo": 0.30,
    "miss_frac_hi": 0.50,
    "z_thr": 4.0,
    "z_bg_sec": 30.0,
    "spike_k_sigma": 8.0,
    "ref_ratio": 0.5,
    "ref_min_sec": 30.0,
    "ref_max_sec": 600.0,
    "bkg_sec": 300.0,
    "osc_lo": 0.2,
    "osc_hi": 4.0,
    "std_raw_factor": 1.1,
    "std_bkg_factor": 2.0,
    "quantize_to_int": True,  # 写回前四舍五入为整数 counts（与原始 int32 库一致）
}


def _missing_mask(dseg):
    return (dseg <= -1) | (dseg == 0) | ~np.isfinite(dseg)


def max_missing_run(d, s, e):
    """段内最长连续缺失 run 长度（-1/0/NaN）。"""
    seg = d[s:e + 1]
    miss = _missing_mask(seg)
    if not miss.any():
        return 0
    best = cur = 0
    for v in miss:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def all_missing_runs(d, s, e):
    """段内全部缺失 run（-1/0/NaN）列表（不分长度），返回绝对样本索引。"""
    seg = d[s:e + 1]
    miss = _missing_mask(seg)
    if not miss.any():
        return []
    out, cur = [], []
    for i, v in enumerate(miss):
        if v:
            cur.append(s + i)
        else:
            if cur:
                out.append((cur[0], cur[-1]))
            cur = []
    if cur:
        out.append((cur[0], cur[-1]))
    return out


def ref_sec_for(L_samples, sr, min_ref=30.0, max_ref=600.0):
    """DSSRR 参考段长度：≥ 异常长度一半，且夹在 [30, 600] 秒（问题3）。"""
    L_sec = L_samples / sr
    return float(np.clip(max(min_ref, L_sec / 2.0), min_ref, max_ref))


def merge_ref_overlap(repaired, sr, max_len_sec=1800.0,
                      ref_ratio=0.5, ref_min_sec=30.0, ref_max_sec=600.0):
    """参考段重叠合并（问题3）：若某修复段的后参考窗（≥ L×ref_ratio，
    上限 ref_max_sec）与相邻修复段起点重叠，则合并两段及中间区域为一个
    修复单元。

    repaired: [(s, e, kind)] 样本索引，已排序；只合并修复段（event/gap 为墙）。
    迭代至稳定（合并后长度增大，参考窗增大，可能连锁合并）。
    合并后段长不得超过 max_len_sec（默认 1800s = detect 的 max_repair）：
    密集异常日（如 S12 01-05 全天 freeze 段遍布）若允许无限连锁，
    会把大半天空合并成一个超长修复单元，超出 DSSRR 重建能力；
    超限时放弃该对合并，交由 protect_ref_from_events 将参考窗内的
    相邻修复段替换为背景中值（防参考污染）。
    """
    max_len = int(max_len_sec * sr)
    out = [list(t) for t in repaired]
    changed = True
    guard = 0
    while changed and guard < 200:
        changed = False
        guard += 1
        for i in range(len(out) - 1):
            s, e, k = out[i]
            s2, e2, k2 = out[i + 1]
            L_sec = (e - s + 1) / sr
            ref = int(float(np.clip(max(ref_min_sec, L_sec * ref_ratio),
                                    ref_min_sec, ref_max_sec)) * sr)
            if s2 - e <= ref:
                new_len = max(e, e2) - s + 1
                if new_len <= max_len:
                    out[i] = [s, max(e, e2), k]
                    del out[i + 1]
                    changed = True
                    break
    return [(int(s), int(e), k) for s, e, k in out]


def linear_fill_seg(x, s, e, d):
    """连续缺失 ≤2 点：对缺失位置线性插值（保持其他点不变）。"""
    seg_d = d[s:e + 1]
    miss = _missing_mask(seg_d)
    if not miss.any():
        return x
    idx = np.arange(s, e + 1)
    known = ~miss
    if known.sum() < 2:
        # 极端（不应出现）：段内几乎全缺失 → 邻域中值
        # （v5.5：必须用 d 域非缺失点过滤，x 域被未修复缺失点
        # （d=0/-1 → x=-dc/-1-dc）污染时中值会落在缺失标记上，
        # 导致兜底修出 0/负值）
        ctx = x[(d > -1) & (d != 0) & np.isfinite(d) & np.isfinite(x)]
        med = float(np.median(ctx)) if len(ctx) else 0.0
        x[s:e + 1][miss] = med
        return x
    from scipy.interpolate import interp1d
    f = interp1d(idx[known], x[s:e + 1][known], kind="linear",
                 fill_value="extrapolate")
    x[s:e + 1][miss] = f(idx[miss])
    return x


def zscore_clean_seg(x, s, e, d, z_thr=4.0, bg_sec=30.0):
    """连续缺失 3-20 点：Z 分数清理 —— 以段外背景稳健统计为参照，
    替换 |z|>z_thr 的离群点与缺失标记为线性插值。"""
    nbg = int(bg_sec * SR)
    bg_idx = np.concatenate([np.arange(max(0, s - nbg), s),
                             np.arange(e + 1, min(len(x), e + 1 + nbg))])
    # v5.5：参考窗内的缺失标记点（x=-dc/-1-dc）剔除，避免污染背景中值
    keep = ((d[bg_idx] > -1) & (d[bg_idx] != 0) & np.isfinite(d[bg_idx]))
    bg = x[bg_idx[keep]]
    bg = bg[np.isfinite(bg)]
    if len(bg) < 30:
        bg = x[max(0, s - 3 * nbg):e + 1 + 3 * nbg]
        bg = bg[np.isfinite(bg)]
    if len(bg) < 8:
        bg = x[(d > -1) & (d != 0) & np.isfinite(d) & np.isfinite(x)]
    mu = float(np.median(bg)) if len(bg) else 0.0
    sd = 1.4826 * np.median(np.abs(bg - mu)) if len(bg) else 1.0
    sd = max(sd, 0.5)
    seg = x[s:e + 1]
    seg_d = d[s:e + 1]
    miss = _missing_mask(seg_d)
    z = np.abs(seg - mu) / sd
    bad = (z > z_thr) | miss
    if not bad.any():
        return x
    idx = np.arange(s, e + 1)
    known = ~bad
    if known.sum() >= 2:
        from scipy.interpolate import interp1d
        f = interp1d(idx[known], seg[known], kind="linear",
                     fill_value="extrapolate")
        x[s:e + 1][bad] = f(idx[bad])
    else:
        x[s:e + 1][bad] = mu
    return x


def spike_fill_seg(x, s, e, d, k_sigma=8.0):
    """孤立尖峰型段（无缺失标记）：局部滚动中值残差 + 稳健尺度，
    超阈值样本用线性插值替换（轻量尖峰修复，供保护窗外 spike 段使用）。"""
    from scipy.ndimage import median_filter
    seg = x[s:e + 1]
    med = median_filter(seg, size=7)
    res = np.abs(seg - med)
    scale = 1.4826 * median_filter(res, size=7) + 1e-9
    glob = 1.4826 * np.median(np.abs(x - np.median(x)))
    thr = k_sigma * np.maximum(scale, glob * 0.5)
    bad = res > thr
    if not bad.any():
        return x
    idx = np.arange(s, e + 1)
    known = ~bad
    if known.sum() >= 2:
        from scipy.interpolate import interp1d
        f = interp1d(idx[known], seg[known], kind="linear",
                     fill_value="extrapolate")
        x[s:e + 1][bad] = f(idx[bad])
    else:
        x[s:e + 1][bad] = float(np.median(seg[known])) if known.any() else 0.0
    return x


def dssrr_repair_with_guard(x, d, s, e, events, rest, sr, dc,
                            ref_ratio=0.5, ref_min_sec=30.0, ref_max_sec=600.0,
                            bkg_sec=300.0, osc_lo=0.2, osc_hi=4.0,
                            std_raw_factor=1.1, std_bkg_factor=2.0):
    """DSSRR 谱重建 + 修复后校验（防爆炸输出/伪振荡）。

    参考段 ≥ 异常长度 × ref_ratio（上限 ref_max_sec，问题3）；参考窗内
    event/相邻修复段替换为背景中值（protect_ref_from_events）；仅采纳修复区
    重建值（丢弃 DSSRR suffix_offset 对段后正常数据的整体平移）。

    修复后校验：若重建 std 超过 max(原始有效样本 std×std_raw_factor,
    段外背景 std×std_bkg_factor)，说明重建引入的能量明显超过原始异常或背景
    水平（DSSRR 退化输出），回退保留原值——避免"修复后反而更差/产生新异常"。
    对纯缺失长段（raw_std 大）阈值由 raw_std×factor 主导，几乎不回退；
    对正常段误检（raw_std≈背景）由背景×factor 兜底，防止重建出伪波动。

    能量保持 guard（v5）：段内有效 std 与 ±bkg_sec 背景 std 同量级
    （osc_lo ≤ ratio ≤ osc_hi，段是持续高动态背景的一部分）→ 不修。
    背景本身活跃时（振荡区/瞬态序列），其中的单阶跃段与低振幅衰减段也命中
    （都是信号活动的一部分）；孤立异常段 raw>>bkg（比值 > osc_hi）不命中、
    照常修复（S12 61711s 真阶跃 bkg 0.5、raw 高；53438s 多阶跃 ratio≈25 同样
    照修）。注：v4h 的 strong_disp 判据已移除（误伤 53438s 真异常，见 E7_RESULTS §0.11）。
    """
    L_sec = (e - s + 1) / sr
    ref = float(np.clip(max(ref_min_sec, L_sec * ref_ratio),
                        ref_min_sec, ref_max_sec))
    xx = protect_ref_from_events(x, s, e, events, rest, ref_sec=ref)
    out = DSSRRRepair(sr=sr, before_sec=ref, after_sec=ref, seed=42).repair(xx, s, e)

    # 原始有效样本 std（缺失标记不参与）
    seg_d = d[s:e + 1]
    vv = seg_d[(seg_d > -1) & (seg_d != 0) & np.isfinite(seg_d)]
    raw_std = float(vv.std()) if len(vv) > 3 else float("inf")
    # 段外背景稳健 std —— 用原始 d 而非逐步修复的 x（v4g：避免已修复值
    # 污染背景判定，防止零散修复改变后续 DSSRR 校验阈值导致误写）
    nb = int(bkg_sec * sr)
    ctx = np.concatenate([d[max(0, s - nb):s], d[e + 1:e + 1 + nb]])
    ctx = ctx[(ctx > -1) & (ctx != 0) & np.isfinite(ctx)]
    if len(ctx) >= 30:
        bkg_std = float(1.4826 * np.median(np.abs(ctx - np.median(ctx))))
    else:
        bkg_std = float(raw_std)
    bkg_std = max(bkg_std, 0.5)

    new_seg = out[s:e + 1]
    new_std = float(new_seg.std()) if len(new_seg) > 3 else 0.0
    limit = max(raw_std * std_raw_factor, bkg_std * std_bkg_factor)
    if new_std > limit:
        # 重建退化：保留原值（诚实反映，不编造能量）
        return x, False
    #   段内有效样本 std 与 ±bkg_sec 背景 std 同量级（osc_lo×–osc_hi×，
    #   段是持续高动态背景的一部分）→ 不修。
    #   2026-09-10 修正：缺失主导段（缺失占比 > 50%）的"有效样本"只是
    #   背景噪声残余（std≈背景，比值≈1）会被误判为振荡活动区而回退，
    #   导致大缺失未修复（S12 01-13 624.9-627.7min 64% 缺失保持 0）。
    #   振荡活动区要求段内大部分是真实信号 → 增加有效样本占比条件。
    d1_seg = d[s:min(e + 1, len(d) - 1)]
    if len(vv) >= 30 and len(d1_seg) >= 8:
        a1 = np.abs(np.diff(d1_seg))
        mx = float(np.max(a1))
        if mx > 0:
            spread = float(np.sum(a1)) / mx
            ratio = raw_std / bkg_std
            seg_dd = d[s:e + 1]
            miss_frac = float(np.mean(
                (seg_dd <= -1) | (seg_dd == 0) | ~np.isfinite(seg_dd)))
            osc_like = (miss_frac <= 0.5) and (osc_lo <= ratio <= osc_hi)
            if osc_like:
                return x, False
    # v5.3 修复区基线对齐 + 振幅匹配背景（用户反馈：修复后异常段有时整体
    # 偏高/偏低——S12 01-13 640-710min 三修复段均值偏 before +3~6 counts）。
    #   1) 基线：取修复段前后 20s 健康参考稳健中位 m_left/m_right；
    #      前后差显著（≥2×bkg_std，数据存在基线台阶）→ 对齐 m_left（修复段
    #      是前面数据的延续，与用户"手动把后面一段多选进异常段"的直觉等效，
    #      台阶保留在段末而非把重建段拉向 after）；前后一致 → 对称中位。
    #   2) 振幅：合成 std 超 bkg_std×std_bkg_factor → 压缩到该上限（只压缩
    #      不放大，防长参考窗 PSD 污染造成的合成伪振荡）。
    ctx_l = d[max(0, s - int(20 * sr)):s]
    ok_l = np.where((ctx_l > -1) & (ctx_l != 0) & np.isfinite(ctx_l))[0]
    ctx_lv = x[max(0, s - int(20 * sr)) + ok_l] if len(ok_l) >= 4 else np.array([])
    ctx_r = d[e + 1:e + 1 + int(20 * sr)]
    ok_r = np.where((ctx_r > -1) & (ctx_r != 0) & np.isfinite(ctx_r))[0]
    ctx_rv = x[e + 1 + ok_r] if len(ok_r) >= 4 else np.array([])
    m_left = float(np.median(ctx_lv)) if len(ctx_lv) >= 4 else float(np.median(new_seg))
    m_right = float(np.median(ctx_rv)) if len(ctx_rv) >= 4 else m_left
    if abs(m_right - m_left) >= 2.0 * max(bkg_std, 0.5):
        m_target = m_left
    else:
        m_target = 0.5 * (m_left + m_right)
    new_seg = new_seg - float(np.median(new_seg)) + m_target
    new_std = float(new_seg.std()) if len(new_seg) > 3 else 0.0
    target = max(bkg_std * std_bkg_factor, 0.5)
    if new_std > target and new_std > 1e-9:
        new_seg = m_target + (new_seg - m_target) * (target / new_std)
    x[s:e + 1] = new_seg
    return x, True


def moonquake_spike_repair(x, s, e):
    """月震保护窗内尖峰/短缺失：用 SeisY 的月震保护型局部尖峰修复
    （滚动中值 + 局部 MAD + 邻点回归判据，PCHIP/线性插值只修孤立尖峰，
    不重建月震 coda 主体）。"""
    from seisy.core.anomaly_repair import MoonquakeProtectedSpikeReplacer
    if getattr(moonquake_spike_repair, "_replacer", None) is None:
        moonquake_spike_repair._replacer = MoonquakeProtectedSpikeReplacer(
            sampling_rate=SR)
    repaired, _report = moonquake_spike_repair._replacer.replace(
        x[max(0, s - int(2 * SR)):e + 1 + int(2 * SR)])
    a0 = max(0, s - int(2 * SR))
    x[s:e + 1] = repaired[s - a0:e + 1 - a0]
    return x


def protect_ref_from_events(x, s, e, events, repaired_all, ref_sec=45.0):
    """DSSRR 修复前参考保护：修复单元两侧参考窗内属于 event 段、
    或属于其他 repair 段（尚未修复的相邻异常）的样本，
    替换为对应段前后正常区中值（防"参考窗拾取相邻异常段"导致的失败/伪振荡）。"""
    xx = x.copy()
    n = len(xx)
    nref = int(ref_sec * SR)
    r0 = max(0, s - nref)
    r1 = min(n - 1, e + nref)
    others = []
    for es, ee in repaired_all:
        if not (es == s and ee == e) and es <= r1 and ee >= r0:
            others.append((max(es, r0), min(ee, r1)))
    others += [(max(es, r0), min(ee, r1)) for es, ee in events if es <= r1 and ee >= r0]
    for a, b in others:
        if a > b:
            continue
        left = xx[max(0, a - nref):a]
        right = xx[b + 1:min(n - 1, b + nref) + 1]
        ctx = np.concatenate([left, right])
        ctx = ctx[np.isfinite(ctx)]
        if len(ctx) < 8:
            ctx = xx[np.isfinite(xx)]
        med = float(np.median(ctx)) if len(ctx) else 0.0
        xx[a:b + 1] = med
    return xx


def missing_runs_in(d, s, e, max_run=20):
    """段内孤立缺失 run（-1/0/NaN）列表，长度 ≤ max_run（问题2：
    月震 coda 内的短缺失也要修，用保护型尖峰修复而非整体重建）。"""
    seg = d[s:e + 1]
    miss = _missing_mask(seg)
    if not miss.any():
        return []
    out, cur = [], []
    for i, v in enumerate(miss):
        if v:
            cur.append(s + i)
        else:
            if cur:
                out.append((cur[0], cur[-1]))
            cur = []
    if cur:
        out.append((cur[0], cur[-1]))
    return [(a, b) for a, b in out if (b - a + 1) <= max_run]


def build_repaired_data(d, repaired, events, spikes, gaps, info, sr=SR,
                        params=None):
    """E7 多级修复流水线核心：对给定异常段列表分级修复。

    批处理(repair_one_file)与查看器手动重处理(ManualRepairDialog auto 模式)
    共用——保证手动重处理与全库批处理口径完全一致：
    检测 → 目录核验 → 参考重叠合并 → 逐段分级修复（freeze 只修缺失 run /
    缺失分级 linear→zscore→DSSRR / spike 尖峰修复 / 能量阶跃 DSSRR）→
    event 还原 → 月震保护窗尖峰修复 → gap 与未覆盖缺失保留原始值。
    返回 (原始计数域 out_data, 修复计数 dict)。

    params: 可选 dict，覆盖分级阈值与 DSSRR 参考/能量保护参数（键与
    seisy/core/repair_settings.py 的 repair 组一致）；None 时用默认值
    （= 论文 v5 批处理口径，保证可复现）。
    """
    p = _REPAIR_DEFAULTS if params is None else dict(_REPAIR_DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in p})
    lin_max_run = p["lin_max_run"]
    z_max_run = p["z_max_run"]
    miss_frac_lo = p["miss_frac_lo"]
    miss_frac_hi = p["miss_frac_hi"]
    z_thr = p["z_thr"]
    z_bg_sec = p["z_bg_sec"]
    spike_k = p["spike_k_sigma"]
    dssrr_kw = {k: p[k] for k in ("ref_ratio", "ref_min_sec", "ref_max_sec",
                                  "bkg_sec", "osc_lo", "osc_hi",
                                  "std_raw_factor", "std_bkg_factor")}

    dc = info["dc"]
    x = d.copy()
    x[~np.isfinite(x)] = 0.0
    x[x <= -1] = 0.0
    x = x - dc

    n_rep = n_lin = n_z = n_spike = n_dssrr = n_fb = 0
    seg_rows = []  # 每段真实执行路径（CSV 明细数据源）

    def _fix_missing_runs(s, e):
        """只修段内缺失 run（按 run 长度分级），正常/恒定部分保留。"""
        nonlocal x
        n_l = n_z_ = n_d = n_f = 0
        for rs, re in all_missing_runs(d, s, e):
            rl = re - rs + 1
            if rl <= lin_max_run:
                x = linear_fill_seg(x, rs, re, d)
                n_l += 1
            elif rl <= z_max_run:
                x = zscore_clean_seg(x, rs, re, d, z_thr=z_thr, bg_sec=z_bg_sec)
                n_z_ += 1
            else:
                rest = [(a, b) for a, b, _ in repaired[i + 1:]]
                x, _ok = dssrr_repair_with_guard(x, d, rs, re, events,
                                                 rest, sr, dc, **dssrr_kw)
                n_d += 1
                n_f += 0 if _ok else 1
        return n_l, n_z_, n_d, n_f

    # 防御：repaired 两两不重叠（detect/merge 回归防护；重叠段会
    # 导致明细 CSV 重复记录与同一区域被多次修复）
    repaired = _merge_simple(sorted(repaired))
    for i, (s, e, kind) in enumerate(repaired):
        max_run = max_missing_run(d, s, e)
        miss_frac = float(np.mean(_missing_mask(d[s:e + 1])))

        if kind == "freeze":
            # v4f：冻结（恒定保持）段不整体重建——DSSRR 会把遥测保持
            # 重建成伪波动（用户实测 60534-60747s 冻结修复后 std 3.9、
            # "修复后还不如不修复"）。只修段内缺失 run，恒定部分保留原值。
            nl, nz, nd, nf = _fix_missing_runs(s, e)
            n_lin += nl; n_z += nz; n_dssrr += nd; n_fb += nf
            n_rep += 1
            seg_rows.append((s, e, kind, "freeze_missing",
                             f"miss_frac={miss_frac:.3f} "
                             f"lin={nl} z={nz} dssrr={nd}"))
            continue

        # 分级按"最长连续缺失点数"（用户口径：≤lin_max_run 点线性/样条、
        # lin~z_max_run 点 Z 分数、>z_max_run 点 DSSRR），异常种类辅助判定：
        #   kind=zero（缺失主导）或段内缺失占比 > miss_frac_lo → 缺失分级
        #   kind=spike（孤立尖峰，无缺失）→ 尖峰修复
        #   其余（burst/step/sat，能量/饱和/阶跃）→ DSSRR
        if kind == "zero" or miss_frac > miss_frac_lo:
            if miss_frac <= miss_frac_hi:
                # v4f：缺失占比 ≤miss_frac_hi → 只修缺失 run，正常/恒定
                # 部分保留（此前整段 DSSRR 会把正常平稳数据重建成合成波）
                nl, nz, nd, nf = _fix_missing_runs(s, e)
                n_lin += nl; n_z += nz; n_dssrr += nd; n_fb += nf
                seg_rows.append((s, e, kind, "missing_runs",
                                 f"miss_frac={miss_frac:.3f} "
                                 f"lin={nl} z={nz} dssrr={nd}"))
            elif max_run <= lin_max_run:
                # 规则1（问题1）：单点/双点缺失 → 线性插值（不必 DSSRR）
                x = linear_fill_seg(x, s, e, d)
                n_lin += 1
                seg_rows.append((s, e, kind, "linear", f"max_run={max_run}"))
            elif max_run <= z_max_run:
                # 规则2：连续缺失 lin~z_max_run 点 → Z 分数清理
                x = zscore_clean_seg(x, s, e, d, z_thr=z_thr, bg_sec=z_bg_sec)
                n_z += 1
                seg_rows.append((s, e, kind, "zscore", f"max_run={max_run}"))
            else:
                # 规则4a：连续缺失 > z_max_run 点且缺失主导 → DSSRR（长缺失主场）
                rest = [(a, b) for a, b, _ in repaired[i + 1:]]
                x, _ok = dssrr_repair_with_guard(x, d, s, e, events, rest,
                                                 sr, dc, **dssrr_kw)
                n_dssrr += 1
                if not _ok:
                    n_fb += 1
                    # v5.2：DSSRR 回退时缺失点兜底（用户口径：≤2 点线性、
                    # 3-20 点 Z 分数），避免回退保留原值留下 0 值"深坑"
                    # （S12 01-13 195.4-195.7min step 段：1 个缺失点回退后
                    # 保持 0，查看器修复后图出现明显凹陷）。
                    nl, nz, nd, nf = _fix_missing_runs(s, e)
                    n_lin += nl; n_z += nz; n_dssrr += nd; n_fb += nf
                    seg_rows.append((s, e, kind, "dssrr_fb_missing",
                                     f"miss_frac={miss_frac:.3f} "
                                     f"lin={nl} z={nz} dssrr={nd}"))
                else:
                    seg_rows.append((s, e, kind, "dssrr",
                                     f"miss_frac={miss_frac:.3f}"))
        elif kind == "spike":
            # 规则3：孤立尖峰型段 → 先补缺失（如有）再局部 MAD 尖峰修复
            if miss_frac > 0:
                x = linear_fill_seg(x, s, e, d)
                n_lin += 1
            x = spike_fill_seg(x, s, e, d, k_sigma=spike_k)
            n_spike += 1
            seg_rows.append((s, e, kind, "spike", f"miss_frac={miss_frac:.3f}"))
        else:
            # 规则4b：能量/饱和/阶跃型（无缺失主导）→ DSSRR 谱重建
            # （DSSRR 主场；参考段 ≥ L×ref_ratio，上限 ref_max_sec —— 问题3）
            rest = [(a, b) for a, b, _ in repaired[i + 1:]]
            x, _ok = dssrr_repair_with_guard(x, d, s, e, events, rest,
                                             sr, dc, **dssrr_kw)
            n_dssrr += 1
            if not _ok:
                n_fb += 1
                # v5.2：DSSRR 回退时缺失点兜底（同规则4a），不留 0 值深坑
                nl, nz, nd, nf = _fix_missing_runs(s, e)
                n_lin += nl; n_z += nz; n_dssrr += nd; n_fb += nf
                seg_rows.append((s, e, kind, "dssrr_fb_missing",
                                 f"miss_frac={miss_frac:.3f} "
                                 f"lin={nl} z={nz} dssrr={nd}"))
            else:
                seg_rows.append((s, e, kind, "dssrr",
                                 f"miss_frac={miss_frac:.3f}"))
        n_rep += 1

    # event 区还原为原始值（修复单元外扩/padding 可能覆盖 event 边缘）
    for s, e in events:
        x[s:e + 1] = d[s:e + 1] - dc

    # 问题2：月震保护窗内尖峰/短缺失 → 月震保护型局部尖峰修复
    # （在 event 还原之后执行，避免被还原覆盖；只修孤立尖峰、不动 coda 主体）
    for s, e, kind in spikes:
        x = moonquake_spike_repair(x, s, e)
        n_spike += 1

    # 写回原始计数域
    out_data = x + dc
    # gap 段：保留原始值（包括 -1/0 标记）
    for s, e in gaps:
        out_data[s:e + 1] = d[s:e + 1]
    # v5.5：最终缺失兜底——非 event/gap 保护区修复后仍 ≤0 的点
    # （原始缺失修复段漏网、检测未覆盖孤立缺失、DSSRR/插值过冲产生的
    # 0/负值，含原始正常点被重建为 0 的极端情况）逐 run 线性插值强制
    # 收敛，保证"缺失残留=0（gap/event 豁免）"的验收口径成立。
    m_ev = np.zeros(len(d), dtype=bool)
    for s, e in events:
        m_ev[s:e + 1] = True
    m_gap = np.zeros(len(d), dtype=bool)
    for s, e in gaps:
        m_gap[s:e + 1] = True
    still = ~m_ev & ~m_gap & ((out_data <= 0) | ~np.isfinite(out_data))
    if np.any(still):
        runs = []
        cur = None
        for i in range(len(d)):
            if still[i]:
                if cur is None:
                    cur = [i, i]
                else:
                    cur[1] = i
            elif cur is not None:
                runs.append((cur[0], cur[1]))
                cur = None
        if cur is not None:
            runs.append((cur[0], cur[1]))
        for rs, re in runs:
            x = linear_fill_seg(x, rs, re, d)
        out_data = x + dc
        # 兜底在 x 域操作，gap/event 区 x=d-dc，x+dc 自动还原原始值；
        # 显式重做 gap 还原仅为防御
        for s, e in gaps:
            out_data[s:e + 1] = d[s:e + 1]
        # 极端情况下插值仍 ≤0/NaN 的非保护区点 → 计数下限 1（月震计数非负）
        bad = ~m_ev & ~m_gap & ((out_data <= 0) | ~np.isfinite(out_data))
        out_data[bad] = 1

    # 整数化：修复值（PSD 重建/Z 分数/插值）为实数值，原始月震计数为整数。
    # 默认四舍五入回整数 counts（量化误差 ±0.5 相对 DSSRR 重建不确定性可
    # 忽略；非修复区原始整数与 -1/0 缺失标记 round 后不变）。关闭时保留
    # 实数精度（下游高精度频谱/自由振荡分析用）。
    if p.get("quantize_to_int", True):
        out_data = np.round(out_data)
        # quantize 把 (0, 0.5) 的重建小值打回 0 → 又成缺失标记；
        # 非 event/gap 保护区 round 后仍 ≤0 的点 clip 到计数下限 1
        bad_q = ~m_ev & ~m_gap & ((out_data <= 0) | ~np.isfinite(out_data))
        if np.any(bad_q):
            out_data[bad_q] = 1

    return out_data, dict(rep=n_rep, lin=n_lin, z=n_z, spike=n_spike,
                          dssrr=n_dssrr, fb=n_fb,
                          segments=[{"s": s, "e": e, "kind": k, "method": m,
                                     "note": n}
                                    for s, e, k, m, n in seg_rows])


def write_catalog_csv(path, rows):
    """写出全月修复明细 CSV（UTF-8 with BOM，查看器 load_segments 直读）。

    rows: list[dict] —— station, day, start_s, end_s, start_time, end_time,
    len_s, kind, method, note。kind 含 repair/gap/event。
    """
    cols = ["station", "day", "start_s", "end_s", "start_time", "end_time",
            "len_s", "kind", "method", "note"]
    import csv as _csv
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _rows_from_cnt(cnt, events, gaps, t0, stn, day):
    """由 build 结果组装明细行（repair 真实 method + event/gap 条目）。"""
    rows = []
    for sg in cnt["segments"]:
        s, e = sg["s"], sg["e"]
        rows.append({
            "station": stn, "day": day,
            "start_s": s, "end_s": e,
            "start_time": str(t0 + s / SR)[:23],
            "end_time": str(t0 + e / SR)[:23],
            "len_s": round((e - s + 1) / SR, 1),
            "kind": "repair", "method": sg["method"], "note": sg["note"],
        })
    for s, e in events:
        rows.append({
            "station": stn, "day": day,
            "start_s": s, "end_s": e,
            "start_time": str(t0 + s / SR)[:23],
            "end_time": str(t0 + e / SR)[:23],
            "len_s": round((e - s + 1) / SR, 1),
            "kind": "event", "method": "catalog_protect", "note": "",
        })
    for s, e in gaps:
        rows.append({
            "station": stn, "day": day,
            "start_s": s, "end_s": e,
            "start_time": str(t0 + s / SR)[:23],
            "end_time": str(t0 + e / SR)[:23],
            "len_s": round((e - s + 1) / SR, 1),
            "kind": "gap", "method": "keep", "note": "no data",
        })
    return rows


def collect_day_rows(f, det):
    """重算某文件当日明细行（不写修复库）——B：手动重处理 apply 后刷新 CSV。

    返回 (station, day, rows)。
    """
    st = read(f)
    mhz = [tr for tr in st if tr.stats.channel == "MHZ"]
    if not mhz:
        return None
    tr = mhz[0]
    d = tr.data.astype(float)
    repaired, events, gaps, info = det.detect(d)
    day_str = os.path.basename(f).split(".")[2]
    repaired, events, spikes = verify_catalog(repaired, events, day_str)
    extra_spikes = []
    for s, e in events:
        for rs, re in missing_runs_in(d, s, e):
            extra_spikes.append((rs, re, "missing"))
    if extra_spikes:
        spikes = _merge_simple(sorted(spikes + extra_spikes))
    repaired = merge_ref_overlap(repaired, SR)
    _out, cnt = build_repaired_data(d, repaired, events, spikes, gaps,
                                    info, sr=SR)
    stn = os.path.basename(f).split(".")[1]
    day = os.path.basename(f).split(".")[2]
    return stn, day, _rows_from_cnt(cnt, events, gaps, tr.stats.starttime,
                                    stn, day)


def update_catalog_day(csv_path, station, day, rows, manual=None):
    """把某日明细行写回全月 CSV（替换该 station/day 全部旧行）。

    manual: (s, e) —— 与 repair 行重叠的段标记为 manual_apply
    （手动重处理覆盖过，方法与流水线判定不同）。
    """
    if manual:
        ms, me = manual
        for r in rows:
            if r["kind"] == "repair" and not (
                    r["end_s"] < ms or r["start_s"] > me):
                r["method"] = "manual_apply"
                r["note"] = (r["note"] + " | " if r["note"] else "") + \
                            "manual apply"
    import csv as _csv
    all_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8-sig") as f:
            all_rows = [dict(r) for r in _csv.DictReader(f)]
    all_rows = [r for r in all_rows
                if not (r.get("station") == station and r.get("day") == day)]
    all_rows.extend(rows)
    write_catalog_csv(csv_path, all_rows)


def repair_one_file(f, det, out_dir, verbose=True, catalog_rows=None,
                    params=None, use_catalog=True):
    st = read(f)
    mhz = [tr for tr in st if tr.stats.channel == "MHZ"]
    if not mhz:
        return 0, 0
    tr = mhz[0]
    d = tr.data.astype(float)
    repaired, events, gaps, info = det.detect(d)

    # 目录核验：真实月震保护（01-13 07:11 UTC 的 M 型月震 + 60min coda 保护窗；
    # 11:06-11:17 振荡簇无目录记录 → 解除 event 保护转 repair；
    # 保护窗内尖峰/短缺失 → spikes，单独走月震保护型尖峰修复）
    day_str = os.path.basename(f).split(".")[2]
    if use_catalog:
        repaired, events, spikes = verify_catalog(repaired, events, day_str)
    else:
        # 批量工具可关闭目录保护：与"无目录记录日"口径一致——
        # 检测器判定事件不构成保护依据，全部转回修复（burst 类）
        repaired, events, spikes = (
            _merge_simple(sorted(repaired + [(s, e, "burst")
                                             for s, e in events])),
            [], [])
    # v4h：严格只信目录——检测器判定事件/振荡簇不再保护（verify 无目录日
    # 已把 events 全部转 repair）；非目录"分散变化"段的防破坏由修复层
    # dssrr_repair_with_guard 的能量保持 guard 承担（重建抹平即回退）。
    # 问题2 增强：保护窗内 event 段（月震 coda）内部的孤立缺失 run（≤20 点）
    # 仍用月震保护型尖峰修复（只修孤立伪迹、不动 coda 主体）
    extra_spikes = []
    for s, e in events:
        for rs, re in missing_runs_in(d, s, e):
            extra_spikes.append((rs, re, "missing"))
    if extra_spikes:
        spikes = _merge_simple(sorted(spikes + extra_spikes))
    # 参考段重叠合并（问题3）：DSSRR 参考窗（≥L/2，上限 600s）与相邻
    # 修复段重叠 → 合并两段及中间区域为一个修复单元
    repaired = merge_ref_overlap(repaired, SR)

    out_data, cnt = build_repaired_data(
        d, repaired, events, spikes, gaps, info, sr=SR, params=params)
    n_rep, n_lin = cnt["rep"], cnt["lin"]
    n_z, n_spike = cnt["z"], cnt["spike"]
    n_dssrr, n_fb = cnt["dssrr"], cnt["fb"]

    # 写新库：恢复原始整数 dtype（int32 counts）。此前用 d.dtype（float64）
    # 误把修复库写成 FLOAT64 实数；build_repaired_data 已按
    # quantize_to_int 决定是否四舍五入，此处只负责 dtype 还原。
    tr2 = tr.copy()
    tr2.data = out_data.astype(tr.data.dtype)
    st_out = Stream()
    replaced = False
    for t in st:
        if t.stats.channel == "MHZ":
            st_out.append(tr2)
            replaced = True
        else:
            st_out.append(t.copy())
    assert replaced
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, os.path.basename(f))
    st_out.write(out_name, format="MSEED")
    # 明细收集（CSV 数据源）：修复段真实 method + event/gap 条目
    if catalog_rows is not None:
        stn = os.path.basename(f).split(".")[1]
        day = os.path.basename(f).split(".")[2]
        catalog_rows.extend(_rows_from_cnt(
            cnt, events, gaps, tr.stats.starttime, stn, day))
    if verbose:
        print(f"  wrote {os.path.basename(out_name)}  rep={n_rep} "
              f"(lin={n_lin} z={n_z} spike={n_spike} dssrr={n_dssrr}"
              f"[fb={n_fb}]) ev={len(events)} gap={len(gaps)}")
    return n_rep, len(gaps)


# 已知保护段基准（station, day, start_s, end_s, expected_changed）——
# 与 E7_RESULTS.md 各版本验收一致（v5.4 整数化后的期望值）
KNOWN_PROTECTED = [
    ("S12", "19760102", int(53438.8 * SR), int(53611.0 * SR), 1115),
    ("S12", "19760102", int(60534 * SR), int(60747 * SR), 0),
    ("S12", "19760102", int(61711.8 * SR), int(61726.8 * SR), 8),
    ("S16", "19760108", int(19219 * SR), int(27300 * SR), 13),
]


def verify_file(raw_path, fix_path, det, known=None):
    """C：单文件入库验收——缺失残留 / 0 坑 / 非修复区零改动 / 保护段。

    返回 dict 报告（不含异常时 OK）。
    """
    known = KNOWN_PROTECTED if known is None else known
    rep = {"file": os.path.basename(raw_path), "ok": True, "issues": []}

    def _issue(msg):
        rep["ok"] = False
        rep["issues"].append(msg)

    st = read(raw_path)
    mhz = [t for t in st if t.stats.channel == "MHZ"]
    if not mhz:
        return rep
    tr = mhz[0]
    d = tr.data.astype(float)
    stf = read(fix_path)
    tfl = [t for t in stf if t.stats.channel == "MHZ"]
    if not tfl:
        _issue("fix file has no MHZ")
        return rep
    fd = tfl[0].data.astype(float)
    if len(fd) != len(d):
        _issue(f"length mismatch raw={len(d)} fix={len(fd)}")
        return rep

    repaired, events, gaps, info = det.detect(d)
    day_str = os.path.basename(raw_path).split(".")[2]
    repaired, events, spikes = verify_catalog(repaired, events, day_str)
    extra = []
    for s, e in events:
        for rs, re in missing_runs_in(d, s, e):
            extra.append((rs, re, "missing"))
    if extra:
        spikes = _merge_simple(sorted(spikes + extra))
    repaired = merge_ref_overlap(repaired, SR)

    n = len(d)
    m_rep = np.zeros(n, bool)
    for s, e, _ in repaired:
        m_rep[s:e + 1] = True
    m_ev = np.zeros(n, bool)
    for s, e in events:
        m_ev[s:e + 1] = True
    m_gap = np.zeros(n, bool)
    for s, e in gaps:
        m_gap[s:e + 1] = True
    m_sp = np.zeros(n, bool)
    for s, e, _ in spikes:
        m_sp[s:e + 1] = True

    miss = (fd <= -1) | (fd == 0) | ~np.isfinite(fd)
    # 1) 缺失残留：gap/event 保护区内保留原值属设计，其余应为 0
    resid = miss & ~m_gap & ~m_ev
    n_resid = int(resid.sum())
    if n_resid:
        _issue(f"missing residual {n_resid} outside gap/event")
    # 2) 0 坑：修复区内 0 值
    n_zero = int((m_rep & (fd == 0)).sum())
    if n_zero:
        _issue(f"zero holes in repaired zones {n_zero}")
    # 3) 非修复区零改动（repair/event/gap/spike 之外必须逐样本一致）
    untouched = ~(m_rep | m_ev | m_gap | m_sp)
    n_leak = int((untouched & (d != fd)).sum())
    if n_leak:
        _issue(f"leak in untouched zones {n_leak}")
    # 4) 已知保护段基准
    stn = os.path.basename(raw_path).split(".")[1]
    for (kstn, kday, ks, ke, exp) in known:
        if kstn == stn and kday == day_str and ks < n and ke < n:
            got = int(np.sum(d[ks:ke] != fd[ks:ke]))
            if got != exp:
                _issue(f"protected {kstn} {kday} [{ks}-{ke}] "
                       f"changed={got} expected={exp}")
    rep["n_repair"] = len(repaired)
    rep["n_event"] = len(events)
    rep["n_gap"] = len(gaps)
    rep["missing_residual"] = n_resid
    rep["zero_holes"] = n_zero
    rep["untouched_leak"] = n_leak
    return rep


def verify_month(raw_root, fix_root, det, known=None,
                 stations=("S12", "S15", "S16")):
    """C：全月验收，返回 (汇总 dict, 逐文件列表)。

    只验证已建修复库的台站（默认 S12/S15/S16；原始库的 S14 等未处理台站
    不属于验收范围）。
    """
    out = {"ok": True, "files": []}
    files = []
    for stn in stations:
        files += sorted(glob.glob(
            os.path.join(raw_root, f"XA.{stn}.197601*.mseed")))
    for f in files:
        fix = os.path.join(fix_root, os.path.basename(f))
        if not os.path.exists(fix):
            out["ok"] = False
            out["files"].append({"file": os.path.basename(f), "ok": False,
                                 "issues": ["fix file missing"]})
            continue
        rep = verify_file(f, fix, det, known=known)
        out["files"].append(rep)
        if not rep["ok"]:
            out["ok"] = False
    out["n_files"] = len(out["files"])
    out["n_fail"] = sum(1 for r in out["files"] if not r["ok"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", default=["S12", "S15", "S16"])
    ap.add_argument("--days", nargs="+", default=None, help="如 19760113；默认全月")
    ap.add_argument("--catalog", default=CATALOG_PATH,
                    help="修复明细 CSV 输出路径（默认 E7_catalog_197601.csv）")
    ap.add_argument("--no-catalog", action="store_true", help="不写明细 CSV")
    ap.add_argument("--verify", action="store_true",
                    help="验收模式：只对修复库做每日验收（不重跑修复）")
    ap.add_argument("--settings", default=None,
                    help="读取 ~/.seisy_repair_settings.json 的 repair 组参数"
                         "（默认不读、用论文默认参数；显式传入 --settings 才生效）")
    args = ap.parse_args()

    det = E7Detector(sr=SR)

    if args.verify:
        ver = verify_month(IN, OUT, det)
        import json
        vpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "E7_verify_197601.json")
        with open(vpath, "w", encoding="utf-8") as f:
            json.dump(ver, f, ensure_ascii=False, indent=1)
        print(f"VERIFY: {ver['n_files']} files, {ver['n_fail']} FAIL "
              f"-> {vpath}")
        for r in ver["files"]:
            if not r["ok"]:
                print("  FAIL", r["file"], "; ".join(r["issues"]))
        return

    params = None
    if args.settings:
        import json as _json
        sp = os.path.join(os.path.expanduser("~"),
                          ".seisy_repair_settings.json")
        if not os.path.exists(sp):
            print("ERR --settings: file not found:", sp)
            sys.exit(2)
        with open(sp, encoding="utf-8") as f:
            params = _json.load(f).get("repair")
        print("--settings: using repair params from", sp)

    files = []
    for stn in args.stations:
        files += sorted(glob.glob(os.path.join(IN, f"XA.{stn}.197601*.mseed")))
    if args.days:
        files = [f for f in files if os.path.basename(f).split(".")[2] in args.days]

    catalog_rows = [] if not args.no_catalog else None
    t_start = time.time()
    tot_rep = tot_gap = 0
    for i, f in enumerate(files):
        t0 = time.time()
        try:
            nrep, ngap = repair_one_file(f, det, OUT,
                                         catalog_rows=catalog_rows,
                                         params=params)
        except Exception as ex:
            print("ERR", os.path.basename(f), ex)
            continue
        tot_rep += nrep; tot_gap += ngap
        print(f"[{i+1}/{len(files)}] {os.path.basename(f)} ({time.time()-t0:.1f}s)")
    if catalog_rows is not None:
        write_catalog_csv(args.catalog, catalog_rows)
        print("catalog:", args.catalog, f"({len(catalog_rows)} rows)")
    print(f"\nDONE {len(files)} files, {tot_rep} repairs, {tot_gap} gaps, {time.time()-t_start:.0f}s")
    print("output dir:", OUT)


if __name__ == "__main__":
    main()
