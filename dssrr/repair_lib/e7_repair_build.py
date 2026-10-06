"""E7 修复流水线：对 Apollo MHZ 日文件应用多级修复，写入修复库

输入 / 输出目录由命令行 ``--raw-root`` / ``--fix-root`` 指定，
**不硬编码任何机器上的路径**。目录结构约定::

    <raw_root>/1976/01/XA.S12.19760113.mseed     # 只读
    <fix_root>/1976/01/XA.S12.19760113.mseed     # 输出（自动创建）

多级修复策略：
- 连续缺失 ≤ 2 点（含单尖峰缺失）→ 线性插值（短缺失不必动用 DSSRR）
- 连续缺失 3-20 点 → Z 分数清理（|z|>4 离群点替换为插值）
- 孤立尖峰型段（无缺失）→ 局部 MAD 尖峰修复（滚动中值 + 插值）
- 月震保护窗内尖峰/短缺失 → MoonquakeProtectedSpikeReplacer
  （只修孤立尖峰，不动月震 coda 主体；coda 由目录保护窗整体保护）
- 连续缺失 > 20 点 / 能量 / 冻结 / 饱和 / 阶跃 → DSSRR 谱重建
  （参考段 ≥ 异常长度一半、上限 600 s）
- 参考段重叠合并：DSSRR 参考窗与相邻异常段重叠 → 两段及中间区域合并
  为一个修复单元（避免参考窗拾取相邻异常段）
- 事件保护：event 段修复后还原为原始值（防止修复单元外扩覆盖）
- 参考保护：DSSRR 修复前，参考窗内属于 event 的样本替换为背景中值
- gap 段（> ``max_repair_sec``）：保留原始数据（如实反映缺口）

本模块既被命令行入口 :mod:`dssrr.cli` 复用，也可直接运行::

    python -m dssrr.repair_lib.e7_repair_build \\
        --raw-root /data/apollo/raw --fix-root /data/apollo/repair
"""
import glob
import os
import sys
import time
import argparse

import numpy as np
from obspy import read, Stream

# DSSRR 后端统一走 dssrr.core，e7 流水线在本目录内相对导入（自包含，无外部依赖）
from ..core import ReferenceSpectrumReplacer
from ..discovery import find_counterpart
from .e7_detector import E7Detector, verify_catalog, protect_osc_clusters, _merge_simple

SR = 6.625
# 输入 / 输出根目录不再作为模块级常量硬编码：它们随每次调用而变，由
# ``main()`` 的 ``--raw-root`` / ``--fix-root`` 参数传入。修复库的默认
# 明细 CSV 也改为相对于 ``--fix-root`` 计算（见 main()）。

# 修复层默认参数（= 论文 v5 批处理口径，与 dssrr.config 的 repair 组默认一致；
# 批处理不读用户设置，保证结果可复现）
_REPAIR_DEFAULTS = {
    "lin_max_run": 2,
    "z_max_run": 20,
    "miss_frac_lo": 0.30,
    "miss_frac_hi": 0.50,
    "z_thr": 4.0,
    "z_bg_sec": 30.0,
    "spike_k_sigma": 8.0,
    "ref_ratio": 0.5,
    # 论文口径 120–600 s（旧默认 30 s 与 DSSRR 推荐不符）
    "ref_min_sec": 120.0,
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
    """段内全部缺失 run（-1/0/NaN）列表（不分长度），返回绝对样本索引。
    同时把 run 两侧紧邻的异常低值点纳入（缺失边缘过渡值，如 value=31
    紧邻 -1 缺失，不是标准缺失标记但同样是异常，不处理会留下凹陷）。"""
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
    # 扩展每个 run 的边缘，吃进紧邻异常低值点
    expanded = []
    for rs, re in out:
        rs, re = _expand_low_edges(d, rs, re)
        expanded.append((rs, re))
    return expanded


def _expand_low_edges(d, rs, re, max_grow=5, low_ratio=0.25):
    """把缺失 run 两侧紧邻的异常低值点纳入修复。

    正常背景约几百 counts，缺失边缘可能有几个远低于背景的过渡值
    （如 31 counts），不是 -1/0 标记但同样是异常。
    low_ratio: 低于局部健康中位 25% 判为异常低值。
    """
    n = len(d)

    def _local_median(pos, side):
        """从 pos 向外侧取 200 个有效点的中位（健康背景）。"""
        vals = []
        step = -1 if side == "left" else 1
        p = pos
        while 0 <= p < n and len(vals) < 200:
            v = d[p]
            if np.isfinite(v) and v > -1 and v != 0:
                vals.append(v)
            p += step
        if len(vals) < 10:
            return None
        return float(np.median(vals))

    # 向左扩展
    grow = 0
    while grow < max_grow and rs > 0:
        p = rs - 1
        v = d[p]
        if not np.isfinite(v) or v <= -1 or v == 0:
            rs = p
            grow += 1
            continue
        med = _local_median(p, "left")
        if med is not None and v < med * low_ratio:
            rs = p
            grow += 1
        else:
            break

    # 向右扩展
    grow = 0
    while grow < max_grow and re < n - 1:
        p = re + 1
        v = d[p]
        if not np.isfinite(v) or v <= -1 or v == 0:
            re = p
            grow += 1
            continue
        med = _local_median(p, "right")
        if med is not None and v < med * low_ratio:
            re = p
            grow += 1
        else:
            break

    return rs, re


def _find_segment_outliers(d, s, e, z_thr=4.0, min_run=1, max_run=20):
    """找段内非缺失但偏离局部背景的短 run（双向：高/低离群点）。
    用稳健 Z 分数（MAD），常见于缺失前的过渡异常（低值+高尖峰）。
    返回 [(rs,re),...] 绝对样本索引。"""
    seg = d[s:e + 1]
    valid = np.isfinite(seg) & (seg > -1) & (seg != 0)
    if valid.sum() < 10:
        return []
    med = float(np.median(seg[valid]))
    mad = float(np.median(np.abs(seg[valid] - med)))
    # LP 数据极安静时 MAD=0（整段几乎都是同一个直流值）。
    # 此时不能用全段 std（会被异常值本身污染），用固定偏差阈值：
    # 偏离中位数 > 15 counts 或 5%（取大）就算离群。
    if mad < 1e-6:
        dev_thr = max(15.0, med * 0.05)
        bad = valid & (np.abs(seg - med) > dev_thr)
        if not bad.any():
            return []
        out, cur = [], []
        for i, v in enumerate(bad):
            if v:
                cur.append(s + i)
            else:
                if cur and len(cur) <= 20:
                    out.append((cur[0], cur[-1]))
                cur = []
        if cur and len(cur) <= 20:
            out.append((cur[0], cur[-1]))
        return out
    # 正常情况：稳健 z
    scale = mad / 0.6745
    z = (seg - med) / scale
    bad = valid & (np.abs(z) > z_thr)
    if not bad.any():
        return []
    out, cur = [], []
    for i, v in enumerate(bad):
        if v:
            cur.append(s + i)
        else:
            if cur and min_run <= len(cur) <= max_run:
                out.append((cur[0], cur[-1]))
            cur = []
    if cur and min_run <= len(cur) <= max_run:
        out.append((cur[0], cur[-1]))
    return out


def _find_anomaly_runs(d, s, e, sr=None, dev_ratio=0.08,
                       span_sec=1.0, expand_thr=12.0, bridge_sec=8.0):
    """找段内非缺失但偏离背景的连续 run（不限长度，双向）。
    用于捕获 kind=zero/missing_runs 段内的长振荡/阶跃（有效但异常的值）。
    dev_ratio: 偏离中位数比例阈值（默认8%，LP背景约490→约39 counts）。
    sr: 采样率（用于外扩/桥接窗口）；None 时退化为像素级窗口。
    expand_thr: 拖尾外扩用的"局部瞬时跨度"阈值（counts）。纯偏差阈值
      只能圈住振荡的尖/谷，衰减拖尾会落在中位数 ±dev_ratio 内而漏检，
      导致局部 DSSRR 只重建尖谷、拖尾残留。本函数在偏差 run 基础上，沿
      两侧继续外扩，只要局部窗的 peak-to-peak 跨度仍显著高于平稳基线
      （> expand_thr）就继续吃进，直到真正回落基线。
    bridge_sec: 桥接间隔（秒）。大幅阻尼振荡在衰减过零处会出现短暂"安宁拍"
      （局部跨度接近基线、值靠拢中位数），被误切成多个 run。相邻 run 间隔
      ≤ bridge_sec 时视为同一次振荡（中间只隔一段过零/慢漂移），桥接成一个
      完整异常 run，交由局部 DSSRR 一次性重建，避免残留段间缝隙。
    返回 [(rs,re),...] 绝对样本索引。"""
    seg = d[s:e + 1]
    valid = np.isfinite(seg) & (seg > -1) & (seg != 0)
    if valid.sum() < 10:
        return []
    med = float(np.median(seg[valid]))
    thr = max(med * dev_ratio, 15.0)
    bad = valid & (np.abs(seg - med) > thr)
    if not bad.any():
        return []
    # 偏差点连成连续 run（段内局部索引）
    runs, i, n = [], 0, len(seg)
    while i < n:
        if bad[i]:
            j = i
            while j < n and bad[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1
    # 合并间隔 ≤30 样本的相邻 run（振荡中偶尔回到背景会被切断）
    merged = [runs[0][:]]
    for r in runs[1:]:
        if r[0] - merged[-1][1] <= 30:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])

    # 拖尾外扩：沿两侧延伸，把衰减到中位数附近的振荡尾巴完整包进来
    span_win = max(3, int(span_sec * sr)) if sr else 3
    # 慢衰减拖尾判据：每窗峰值跨度可能 < expand_thr（如阻尼振荡最后
    # 缓慢爬回基线的斜坡，1s 内只变 2-5 counts），但整体仍显著偏离健康
    # 基线 med。此时以"窗内中位数相对 med 的偏离"作为第二判据继续吃进。
    expand_dev = max(thr * 0.5, 8.0)

    def _local_span(lo, hi):
        vv = seg[lo:hi + 1][valid[lo:hi + 1]]
        if len(vv) < 2:
            return 0.0
        return float(vv.max() - vv.min())

    def _local_med_dev(lo, hi):
        vv = seg[lo:hi + 1][valid[lo:hi + 1]]
        if len(vv) < 2:
            return 0.0
        return float(abs(np.median(vv) - med))

    for r in merged:
        rs, re = r[0], r[1]
        # 左/右侧：瞬时跨度大（振荡主体）或相对基线偏离大（慢衰减拖尾）
        # 之一命中即继续吃进，直到真正回到基线附近
        left = int(rs)
        while left > 0:
            lo = max(0, left - span_win)
            if _local_span(lo, left - 1) > expand_thr or \
               _local_med_dev(lo, left - 1) > expand_dev:
                left = lo
            else:
                break
        r[0] = left
        right = int(re)
        while right < n - 1:
            hi = min(n - 1, right + span_win)
            if _local_span(right + 1, hi) > expand_thr or \
               _local_med_dev(right + 1, hi) > expand_dev:
                right = hi
            else:
                break
        r[1] = right
    # 外扩后重新合并相邻/重叠 run
    out = [merged[0][:]]
    for r in merged[1:]:
        if r[0] <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], r[1])
        else:
            out.append(r[:])
    # 桥接：相邻 run 间隔 ≤ bridge_sec 视为同一次振荡（隔一段过零/慢漂移），
    # 合并成完整异常 run，交由局部 DSSRR 一次性重建，消除段间缝隙残留。
    bridge = int(bridge_sec * sr) if sr else 30
    if len(out) > 1:
        out2 = [out[0][:]]
        for r in out[1:]:
            if r[0] - out2[-1][1] - 1 <= bridge:
                out2[-1][1] = max(out2[-1][1], r[1])
            else:
                out2.append(r[:])
        out = out2
    return [(s + rs, s + re) for rs, re in out]


def ref_sec_for(L_samples, sr, min_ref=120.0, max_ref=600.0):
    """DSSRR 参考段长度：≥ 异常长度一半，且夹在 [120, 600] 秒（论文口径）。"""
    L_sec = L_samples / sr
    return float(np.clip(max(min_ref, L_sec / 2.0), min_ref, max_ref))


def refine_segment_bounds(d, repaired, events, gaps, sr,
                          look_sec=8.0, far_sec=90.0, max_expand_sec=240.0,
                          k_sigma=2.5, max_len_sec=1800.0, skip_sec=35.0,
                          post_pad_sec=15.0):
    """异常边界外扩（边界完成）：把修复段两端延伸到“已回到健康基线”处。

    动机：尖峰/阶跃后常有残余抬升拖尾；若右边界切在拖尾内，DSSRR 的
    after 参考与 m_right 会被污染，修复段末端出现凸起。本步骤只调整
    检测给出的 [s,e]，不改 DSSRR 合成算法。

    规则：基线取两侧远窗（近端先跳过 skip_sec，避免把残余拖尾当成健康
    水平），从段端向外，若局部稳健均值仍显著偏离基线则继续吃进；达标后
    再垫 post_pad_sec（防止“刚碰到基线”后仍有小台阶）。碰到 event/gap
    墙、相邻修复段或 max_expand/max_len 则停。
    """
    if not repaired:
        return repaired
    n = len(d)
    look = max(1, int(look_sec * sr))
    far = max(look, int(far_sec * sr))
    skip = max(look, int(skip_sec * sr))
    max_expand = max(0, int(max_expand_sec * sr))
    max_len = int(max_len_sec * sr)
    post_pad = max(0, int(post_pad_sec * sr))

    # event / gap 为墙
    walls = []
    for a, b in list(events) + list(gaps):
        walls.append((int(a), int(b)))

    def _blocked(pos):
        for a, b in walls:
            if a <= pos <= b:
                return True
        return False

    def _collect(i0, i1):
        if i1 <= i0:
            return np.array([])
        seg = d[i0:i1]
        return seg[np.isfinite(seg) & (seg > -1) & (seg != 0)]

    def _healthy_baseline(s, e):
        """两侧远窗基线：近端 skip 之外取 far，按稳健尺度选更稳的一侧，
        两侧都有时取中位融合（避免残余拖尾污染 after 基线）。"""
        cands = []
        # left: [s-far-skip, s-skip)
        li0, li1 = max(0, s - far - skip), max(0, s - skip)
        lv = _collect(li0, li1)
        if len(lv) >= max(8, look // 2):
            cands.append((float(np.median(lv)),
                          float(1.4826 * np.median(np.abs(lv - np.median(lv))))))
        # right: (e+skip, e+skip+far]
        ri0, ri1 = min(n, e + 1 + skip), min(n, e + 1 + skip + far)
        rv = _collect(ri0, ri1)
        if len(rv) >= max(8, look // 2):
            cands.append((float(np.median(rv)),
                          float(1.4826 * np.median(np.abs(rv - np.median(rv))))))
        if not cands:
            return None
        if len(cands) == 2:
            # 两侧水平差大时优先更稳（MAD 小）的一侧，避免拖尾侧绑架
            cands.sort(key=lambda t: t[1])
            if abs(cands[0][0] - cands[1][0]) > 2.0 * max(cands[0][1], 0.5):
                return cands[0][0]
            return 0.5 * (cands[0][0] + cands[1][0])
        return cands[0][0]

    def _local_robust(pos0, pos1):
        seg = d[pos0:pos1]
        seg = seg[np.isfinite(seg) & (seg > -1) & (seg != 0)]
        if len(seg) < 4:
            return None
        return float(np.median(seg)), float(1.4826 * np.median(np.abs(seg - np.median(seg))))

    out = []
    for s, e, kind in repaired:
        s, e = int(s), int(e)
        if e - s + 1 >= max_len:
            out.append((s, e, kind))
            continue

        base = _healthy_baseline(s, e)
        if base is None:
            out.append((s, e, kind))
            continue

        # 右边界外扩
        expanded = 0
        while expanded < max_expand and e + 1 < n and e - s + 1 < max_len:
            nxt = min(n, e + 1 + look)
            stat = _local_robust(e + 1, nxt)
            if stat is None:
                break
            m, sc = stat
            thr = k_sigma * max(sc, 0.5)
            if abs(m - base) <= thr:
                break
            new_e = nxt - 1
            stop = False
            for a, b in walls:
                if a <= new_e and b >= e + 1:
                    stop = True
                    break
            if stop:
                break
            e = new_e
            expanded += look

        # 右端：达标后再垫 post_pad，吸收“刚碰到基线”的小台阶
        if post_pad > 0 and expanded > 0:
            pad = min(post_pad, max_len - (e - s + 1), n - 1 - e)
            new_e = e + pad
            stop = False
            for a, b in walls:
                if a <= new_e and b >= e + 1:
                    stop = True
                    break
            if not stop:
                e = new_e

        # 左边界外扩
        expanded = 0
        while expanded < max_expand and s - 1 >= 0 and e - s + 1 < max_len:
            nxt = max(0, s - look)
            stat = _local_robust(nxt, s)
            if stat is None:
                break
            m, sc = stat
            thr = k_sigma * max(sc, 0.5)
            if abs(m - base) <= thr:
                break
            new_s = nxt
            stop = False
            for a, b in walls:
                if b >= new_s and a <= s - 1:
                    stop = True
                    break
            if stop:
                break
            s = new_s
            expanded += look

        # 左端：同样垫 post_pad
        if post_pad > 0 and expanded > 0:
            pad = min(post_pad, max_len - (e - s + 1), s)
            new_s = s - pad
            stop = False
            for a, b in walls:
                if b >= new_s and a <= s - 1:
                    stop = True
                    break
            if not stop:
                s = new_s

        out.append((s, e, kind))

    # 外扩后可能重叠/相接，再合并一次
    out = _merge_simple(sorted(out))
    return out


def merge_ref_overlap(repaired, sr, max_len_sec=1800.0,
                      ref_ratio=0.5, ref_min_sec=120.0, ref_max_sec=600.0):
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
                            ref_ratio=0.5, ref_min_sec=120.0, ref_max_sec=600.0,
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
    out = ReferenceSpectrumReplacer(sr=sr).replace(
        xx, s, e, reference_before_sec=ref, reference_after_sec=ref,
        random_seed=42)[0]

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
    # ORIGINAL (constant median shift only; right 20s may still sit on residual):
    # ctx_l = d[max(0, s - int(20 * sr)):s]
    # ok_l = np.where((ctx_l > -1) & (ctx_l != 0) & np.isfinite(ctx_l))[0]
    # ctx_lv = x[max(0, s - int(20 * sr)) + ok_l] if len(ok_l) >= 4 else np.array([])
    # ctx_r = d[e + 1:e + 1 + int(20 * sr)]
    # ok_r = np.where((ctx_r > -1) & (ctx_r != 0) & np.isfinite(ctx_r))[0]
    # ctx_rv = x[e + 1 + ok_r] if len(ok_r) >= 4 else np.array([])
    # m_left = float(np.median(ctx_lv)) if len(ctx_lv) >= 4 else float(np.median(new_seg))
    # m_right = float(np.median(ctx_rv)) if len(ctx_rv) >= 4 else m_left
    # if abs(m_right - m_left) >= 2.0 * max(bkg_std, 0.5):
    #     m_target = m_left
    # else:
    #     m_target = 0.5 * (m_left + m_right)
    # new_seg = new_seg - float(np.median(new_seg)) + m_target
    # Fixed: longer quiet windows with residual skip + linear trend alignment
    # (replace synth linear trend with healthy left→right level; if right is
    # still dirty, anchor to left only).
    def _quiet_level(i0, i1):
        seg = d[i0:i1]
        ok = np.where((seg > -1) & (seg != 0) & np.isfinite(seg))[0]
        if len(ok) < 4:
            return None, None
        v = x[i0:i1][ok]
        if len(v) < 4:
            return None, None
        m = float(np.median(v))
        sc = float(1.4826 * np.median(np.abs(v - m)))
        return m, sc

    skip_r = int(35.0 * sr) if (e - s + 1) > int(60 * sr) else int(10.0 * sr)
    win_l0 = max(0, s - int(45 * sr))
    m_left, sc_left = _quiet_level(win_l0, s)
    r0 = min(len(d), e + 1 + skip_r)
    r1 = min(len(d), r0 + int(45 * sr))
    m_right, sc_right = _quiet_level(r0, r1)
    if m_left is None:
        m_left = float(np.median(new_seg))
        sc_left = 0.0
    if m_right is None:
        m_right = m_left
        sc_right = sc_left
    # 右端仍抬升/不稳（相对左端差 ≥2σ，或自身离散偏大）→ 不跟随
    right_clean = abs(m_right - m_left) < 2.0 * max(bkg_std, 0.5)
    if sc_right and sc_left and sc_right > 3.0 * max(sc_left, 0.5):
        right_clean = False
    nseg = len(new_seg)
    tt = np.arange(nseg, dtype=float) / max(1, nseg - 1)
    if right_clean:
        trend = m_left + (m_right - m_left) * tt
    else:
        trend = np.full(nseg, m_left, dtype=float)
    # 去掉合成段自身线性趋势，换上健康左→右趋势（真正 trend 对齐）
    try:
        a_fit, b_fit = np.polyfit(tt, new_seg, 1)
        new_seg = new_seg - (a_fit * tt + b_fit) + trend
    except Exception:
        new_seg = new_seg - float(np.median(new_seg)) + trend
    new_std = float(new_seg.std()) if len(new_seg) > 3 else 0.0
    target = max(bkg_std * std_bkg_factor, 0.5)
    if new_std > target and new_std > 1e-9:
        new_seg = trend + (new_seg - trend) * (target / new_std)

    # ── 边界交叉淡入淡出（2026-09-28 修复）──────────────────────────
    # DSSRR 内部已做过边界平滑，但上面的 polyfit 趋势对齐和 std 压缩
    # 对整个段（含边界平滑区）重新做了变换，破坏了交叉淡入淡出效果。
    # 这里用 x（与 new_seg 同域 = d-dc）的健康数据做两端融合，
    # 保证接缝连续且不重复加直流偏置。
    new_seg = _crossfade_edges(x, new_seg, s, e, sr)

    x[s:e + 1] = new_seg
    return x, True


def _crossfade_edges(d, new_seg, s, e, sr,
                     blend_sec=10.0, context_sec=30.0):
    """把合成段两端与原始健康数据交叉淡入淡出融合。

    blend_sec: 接缝处交叉淡入淡出长度（秒）
    context_sec: 从段外取健康趋势模板的长度（秒）
    只融合段边缘，段内部合成值不动。
    """
    n = len(new_seg)
    blend = min(int(blend_sec * sr), n // 4)
    if blend < 2:
        return new_seg
    ctx_n = int(context_sec * sr)
    result = new_seg.copy()

    def _healthy(i0, i1):
        """取有效（非缺失）样本。"""
        seg = d[i0:i1]
        mask = np.isfinite(seg) & (seg > -1) & (seg != 0)
        return seg[mask]

    def _edge_template(side):
        """从段外健康数据线性外推 blend 个点的模板。"""
        if side == "left":
            h = _healthy(max(0, s - ctx_n), s)
            if len(h) < 4:
                return None
            # 用最后几个点的线性趋势外推
            k = min(len(h), blend * 3)
            yy = h[-k:]
            xx = np.arange(k, dtype=float)
            try:
                a, b = np.polyfit(xx, yy, 1)
                t = np.arange(k, k + blend, dtype=float)
                return a * t + b
            except Exception:
                return np.full(blend, float(np.median(yy)))
        else:
            h = _healthy(e + 1, min(len(d), e + 1 + ctx_n))
            if len(h) < 4:
                return None
            k = min(len(h), blend * 3)
            yy = h[:k]
            xx = np.arange(k, dtype=float)
            try:
                a, b = np.polyfit(xx, yy, 1)
                t = np.arange(-blend, 0, dtype=float)
                return a * t + b
            except Exception:
                return np.full(blend, float(np.median(yy)))

    # 左端：真实模板 fade out，合成值 fade in
    tpl_l = _edge_template("left")
    if tpl_l is not None and len(tpl_l) == blend:
        fade = np.linspace(0, 1, blend)  # 0=全真实 → 1=全合成
        result[:blend] = tpl_l * (1 - fade) + result[:blend] * fade


    # 右端：合成值 fade out，真实模板 fade in
    tpl_r = _edge_template("right")
    if tpl_r is not None and len(tpl_r) == blend:
        fade = np.linspace(1, 0, blend)  # 1=全合成 → 0=全真实
        result[-blend:] = result[-blend:] * fade + tpl_r * (1 - fade)

    return result


def moonquake_spike_repair(x, s, e):
    """月震保护窗内尖峰/短缺失：稳健滚动中值 + 局部 MAD + 线性插值。

    本模块使用与本模块 ``spike_fill_seg`` 同口径的局部尖峰修复（超阈值孤立
    尖峰 + 短缺失标记替换，不重建月震 coda 主体），而不是
    :class:`~dssrr.repair_lib.anomaly_repair.MoonquakeProtectedSpikeReplacer`
    的完整版本；两者签名/就地修改语义保持一致。
    """
    from scipy.ndimage import median_filter
    seg = x[s:e + 1].copy()
    med = median_filter(seg, size=7)
    res = np.abs(seg - med)
    scale = 1.4826 * median_filter(res, size=7) + 1e-9
    glob = 1.4826 * np.median(np.abs(x - np.median(x)))
    thr = 8.0 * np.maximum(scale, glob * 0.5)
    bad = (res > thr)
    bad |= (seg <= -1) | (seg == 0) | ~np.isfinite(seg)
    if not bad.any():
        return x
    tt = np.arange(s, e + 1)
    known = ~bad
    if known.sum() >= 2:
        from scipy.interpolate import interp1d
        f = interp1d(tt[known], seg[known], kind="linear",
                     fill_value="extrapolate")
        x[s:e + 1][bad] = f(tt[bad])
    else:
        x[s:e + 1][bad] = float(np.median(seg[known])) if known.any() else 0.0
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
    dssrr.config 的 repair 组一致）；None 时用默认值
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
    # Fixed: expand edges until local level matches healthy baseline
    repaired = refine_segment_bounds(d, repaired, events, gaps, sr)

    # ── 三轮修复制（2026-09-28）─────────────────────────────────────────
    # Round 1: 先修所有 spike 和短缺失（≤lin_max_run 点线性插值），
    #          让 DSSRR 参考段里的尖峰/短缺失先被真实修复，不用中值替代。
    # Round 2: 再修中等缺失（lin_max_run+1 ~ z_max_run 点 Z 分数清理）。
    # Round 3: 最后 DSSRR 长缺失（>z_max_run 点）+ burst/step/sat 段。
    # 每段仍按原分级判定 method 标签，只是 run 级修复按长度分轮执行。

    # Step 1: 分类所有段，收集各轮任务
    # seg_plan: [(s, e, kind, miss_frac, max_run, primary_method)]
    # linear_runs / zscore_runs: 全局 run 列表（含所属段索引用于统计）
    spike_tasks = []       # (s, e)
    linear_runs = []       # (rs, re)
    zscore_runs = []       # (rs, re)
    dssrr_runs = []        # (rs, re) — run 级 DSSRR（来自 freeze/missing_runs 段内长洞）
    anomaly_runs = []      # (rs, re, idx) — 段内"有效但异常"run，需在缺失修复后再做局部DSSRR
    dssrr_segs = []        # (s, e) — 整段 DSSRR
    seg_plan = []          # 每段的决策结果，用于最后写 seg_rows
    run_stats = {}         # seg_idx -> [n_lin, n_z, n_dssrr, n_fb]

    for idx, (s, e, kind) in enumerate(repaired):
        max_run = max_missing_run(d, s, e)
        miss_frac = float(np.mean(_missing_mask(d[s:e + 1])))
        plan = {"s": s, "e": e, "kind": kind,
                "miss_frac": miss_frac, "max_run": max_run,
                "method": None, "note": ""}
        run_stats[idx] = [0, 0, 0, 0]

        if kind == "freeze":
            # 冻结段：只修缺失 run，恒定值保留
            plan["method"] = "freeze_missing"
            for rs, re in all_missing_runs(d, s, e):
                rl = re - rs + 1
                if rl <= lin_max_run:
                    linear_runs.append((rs, re, idx))
                elif rl <= z_max_run:
                    zscore_runs.append((rs, re, idx))
                else:
                    dssrr_runs.append((rs, re, idx))
        elif kind == "zero" or miss_frac > miss_frac_lo:
            # 段内异常值（非缺失但偏离背景的有效样本）占比
            _anruns = _find_anomaly_runs(d, s, e, sr=sr)
            if miss_frac <= miss_frac_hi:
                # 缺失占比 ≤50%：先只修缺失 run（缺失-1/0标记），正常样本保留。
                # 段内"有效但异常"的长振荡/阶跃（如阻尼振荡）不在此阶段处理，
                # 而是收集到 anomaly_runs，待所有缺失 run 修好后（参考段变为
                # 健康信号）再做局部 DSSRR——避免对正常样本整段重建。
                plan["method"] = "missing_runs"
                for rs, re in all_missing_runs(d, s, e):
                    rl = re - rs + 1
                    if rl <= lin_max_run:
                        linear_runs.append((rs, re, idx))
                    elif rl <= z_max_run:
                        zscore_runs.append((rs, re, idx))
                    else:
                        dssrr_runs.append((rs, re, idx))
                # 补充：段内非缺失但异常的有效样本（短尖峰/低值、长振荡/阶跃）。
                # missing_runs 只修 -1/0 标记，这些"有效异常值"会被漏掉。收集到
                # anomaly_runs，在缺失修复完成后再处理（参考段已健康）。
                for rs, re in _find_anomaly_runs(d, s, e, sr=sr):
                    anomaly_runs.append((rs, re, idx))
            elif max_run <= lin_max_run:
                plan["method"] = "linear"
                linear_runs.append((s, e, idx))
            elif max_run <= z_max_run:
                plan["method"] = "zscore"
                zscore_runs.append((s, e, idx))
            else:
                plan["method"] = "dssrr"
                dssrr_segs.append((s, e, idx))
        elif kind == "spike":
            plan["method"] = "spike"
            spike_tasks.append((s, e, idx, miss_frac))
        else:
            # burst/step/sat → DSSRR
            plan["method"] = "dssrr"
            dssrr_segs.append((s, e, idx))

        seg_plan.append(plan)

    # ── Round 0: 被合并到缺失/冻结段里的尖峰先修 ──
    # 检测器把 spike 和 zero/freeze 掩码 OR 在一起，紧邻缺失的尖峰会被
    # 合并到缺失段、不会单独成 spike 段。这里从全局 spike 掩码找回这些
    # 点，在缺失修复前先插值修掉，避免尖峰被 DSSRR 当异常重建。
    global_spike = info.get("masks", {}).get("spike")
    if global_spike is not None:
        # 标记独立 spike 段覆盖的位置（这些在 spike_tasks 里处理）
        handled = np.zeros(len(d), dtype=bool)
        for sp_s, sp_e, sp_idx, sp_mf in spike_tasks:
            handled[sp_s:sp_e + 1] = True
        # event 段内的尖峰后面由 moonquake_spike_repair 处理
        for ev_s, ev_e in events:
            handled[ev_s:ev_e + 1] = True
        # gap 段不动
        for g_s, g_e in gaps:
            handled[g_s:g_e + 1] = True
        # 找未处理的 spike 连续 run，线性插值
        orphan = global_spike & ~handled
        if orphan.any():
            cur0 = None
            for pi in range(len(d)):
                if orphan[pi]:
                    if cur0 is None:
                        cur0 = pi
                elif cur0 is not None:
                    x = linear_fill_seg(x, cur0, pi - 1, d)
                    n_spike += 1
                    cur0 = None
            if cur0 is not None:
                x = linear_fill_seg(x, cur0, len(d) - 1, d)
                n_spike += 1

    # ── Round 1: spike + linear ──
    for s, e, idx, miss_frac in spike_tasks:
        if miss_frac > 0:
            x = linear_fill_seg(x, s, e, d)
            run_stats[idx][0] += 1
            n_lin += 1
        x = spike_fill_seg(x, s, e, d, k_sigma=spike_k)
        n_spike += 1

    for rs, re, idx in linear_runs:
        x = linear_fill_seg(x, rs, re, d)
        run_stats[idx][0] += 1
        n_lin += 1

    # ── Round 2: zscore ──
    for rs, re, idx in zscore_runs:
        x = zscore_clean_seg(x, rs, re, d, z_thr=z_thr, bg_sec=z_bg_sec)
        run_stats[idx][1] += 1
        n_z += 1

    # ── Round 3: DSSRR ──
    # run 级 DSSRR（freeze/missing_runs 段内的长洞）
    for rs, re, idx in dssrr_runs:
        rest = [(a, b) for a, b, _ in repaired[idx + 1:]]
        x, _ok = dssrr_repair_with_guard(x, d, rs, re, events, rest,
                                         sr, dc, **dssrr_kw)
        run_stats[idx][2] += 1
        n_dssrr += 1
        if not _ok:
            run_stats[idx][3] += 1

    # 整段 DSSRR
    for s, e, idx in dssrr_segs:
        rest = [(a, b) for a, b, _ in repaired[idx + 1:]]
        x, _ok = dssrr_repair_with_guard(x, d, s, e, events, rest,
                                         sr, dc, **dssrr_kw)
        n_dssrr += 1
        if not _ok:
            # DSSRR 回退 → 缺失点兜底
            n_fb += 1
            nl = nz = nd = nf = 0
            for rs, re in all_missing_runs(d, s, e):
                rl = re - rs + 1
                if rl <= lin_max_run:
                    x = linear_fill_seg(x, rs, re, d)
                    nl += 1
                elif rl <= z_max_run:
                    x = zscore_clean_seg(x, rs, re, d, z_thr=z_thr, bg_sec=z_bg_sec)
                    nz += 1
                else:
                    rest2 = [(a, b) for a, b, _ in repaired[idx + 1:]]
                    x, _ok2 = dssrr_repair_with_guard(x, d, rs, re, events, rest2,
                                                      sr, dc, **dssrr_kw)
                    nd += 1
                    nf += 0 if _ok2 else 1
            run_stats[idx][0] += nl; run_stats[idx][1] += nz
            run_stats[idx][2] += nd; run_stats[idx][3] += nf
            n_lin += nl; n_z += nz; n_dssrr += nd; n_fb += nf
            seg_plan[idx]["method"] = "dssrr_fb_missing"
        # note 在下面统一生成

    # ── Round 4: 段内"有效但异常"run 的局部修复 ──
    # 必须在 Round 1-3（缺失/尖峰修复）之后执行：此时异常 run 前后的缺失
    # 已被填充为健康信号，DSSRR 能取到干净参考段，只重建振荡/阶跃本身而
    # 不触碰段内其余正常样本（代替原先"缺失+长异常→整段重建"的过度替换）。
    for rs, re, idx in anomaly_runs:
        rl = re - rs + 1
        if rl <= lin_max_run:
            ctx = x[(d > 100) & np.isfinite(x)]
            med = float(np.median(ctx)) if len(ctx) else float(np.median(x))
            x[rs:re + 1] = med
            run_stats[idx][0] += 1
            n_lin += 1
        elif rl <= z_max_run:
            x = zscore_clean_seg(x, rs, re, d, z_thr=z_thr, bg_sec=z_bg_sec)
            run_stats[idx][1] += 1
            n_z += 1
        else:
            rest = [(a, b) for a, b, _ in repaired[idx + 1:]]
            x, _ok = dssrr_repair_with_guard(x, d, rs, re, events, rest,
                                             sr, dc, **dssrr_kw)
            run_stats[idx][2] += 1
            n_dssrr += 1
            if not _ok:
                run_stats[idx][3] += 1
                n_fb += 1

    # Step 3: 生成 seg_rows（按段顺序）
    n_rep = len(seg_plan)
    for idx, plan in enumerate(seg_plan):
        s, e = plan["s"], plan["e"]
        nl, nz, nd, nf = run_stats[idx]
        method = plan["method"]
        mf = plan["miss_frac"]
        mr = plan["max_run"]
        if method in ("freeze_missing", "missing_runs"):
            note = f"miss_frac={mf:.3f} lin={nl} z={nz} dssrr={nd}"
        elif method == "dssrr_fb_missing":
            note = f"miss_frac={mf:.3f} lin={nl} z={nz} dssrr={nd}"
        elif method == "dssrr":
            note = f"miss_frac={mf:.3f}"
        elif method in ("linear", "zscore"):
            note = f"max_run={mr}"
        elif method == "spike":
            note = f"miss_frac={mf:.3f}"
        else:
            note = ""
        seg_rows.append((s, e, plan["kind"], method, note))

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

    合并模式：若 path 已存在，先读旧行，按 (station,day,start_s,end_s,kind,method)
    为 key 去重，新行覆盖旧行，其余旧行保留。这样分批跑（上半月/下半月）
    不会互相覆盖。

    rows: list[dict] —— station, day, start_s, end_s, start_time, end_time,
    len_s, kind, method, note。kind 含 repair/gap/event。
    """
    cols = ["station", "day", "start_s", "end_s", "start_time", "end_time",
            "len_s", "kind", "method", "note"]
    import csv as _csv, os
    old_rows = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                for r in _csv.DictReader(f):
                    k = (r.get("station",""), str(r.get("day","")),
                         r.get("start_s",""), r.get("end_s",""),
                         r.get("kind",""), r.get("method",""))
                    old_rows[k] = r
        except Exception:
            pass
    # 新行覆盖旧行（同 key）
    for r in rows:
        k = (str(r.get("station","")), str(r.get("day","")),
             str(r.get("start_s","")), str(r.get("end_s","")),
             str(r.get("kind","")), str(r.get("method","")))
        old_rows[k] = r
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(old_rows.values())


def _rows_from_cnt(cnt, events, gaps, t0, stn, day):
    """由 build 结果组装明细行（repair 真实 method + event/gap 条目）。
    完全落在 event（月震保护）内的 repair 段跳过——event 区最终会被还原为
    原始值，记录 repair 行会误导（看起来修了，实际没修）。"""
    rows = []
    for sg in cnt["segments"]:
        s, e = sg["s"], sg["e"]
        # 若 repair 段完全被某个 event 覆盖，跳过（只显示 catalog_protect 行）
        covered = any(es <= s and e <= ee for es, ee in events)
        if covered:
            continue
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
        # 关闭目录保护：检测器判定 events 全部转 repair，无 spikes
        repaired = _merge_simple(
            sorted(list(repaired) + [(a, b, "burst") for a, b in events]))
        events, spikes = [], []
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
    # 2026-09-28 重新标定：三轮修复制 + Round 0 尖峰预处理后的实际值
    ("S12", "19760102", int(53438.8 * SR), int(53611.0 * SR), 1117),
    ("S12", "19760102", int(60534 * SR), int(60747 * SR), 0),
    ("S12", "19760102", int(61711.8 * SR), int(61726.8 * SR), 1),
    ("S16", "19760108", int(19219 * SR), int(27300 * SR), 13),
]


def verify_file(raw_path, fix_path, det, known=None, use_catalog=True):
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
    if use_catalog:
        repaired, events, spikes = verify_catalog(repaired, events, day_str)
    else:
        repaired = _merge_simple(
            sorted(list(repaired) + [(a, b, "burst") for a, b in events]))
        events, spikes = [], []
    extra = []
    for s, e in events:
        for rs, re in missing_runs_in(d, s, e):
            extra.append((rs, re, "missing"))
    if extra:
        spikes = _merge_simple(sorted(spikes + extra))
    repaired = merge_ref_overlap(repaired, SR)
    # 与 build_repaired_data 保持一致：修复时 refine_segment_bounds 扩展了
    # 段边界，验收也必须做同样的扩展，否则扩展区域被改动会误报 leak。
    repaired = refine_segment_bounds(d, repaired, events, gaps, SR)

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
                 stations=("S12", "S15", "S16"), use_catalog=True):
    """C：全月验收，返回 (汇总 dict, 逐文件列表)。

    只验证已建修复库的台站（默认 S12/S15/S16；原始库的 S14 等未处理台站
    不属于验收范围）。
    """
    out = {"ok": True, "files": [], "skipped": []}
    # 从修复库递归扫描实际存在的修复文件（文件在 YYYY/MM/ 子目录中，
    # 未处理台站/日期不算失败）。
    fix_files = []
    for stn in stations:
        fix_files += sorted(glob.glob(
            os.path.join(fix_root, "**", f"XA.{stn}.*.mseed"),
            recursive=True))
    for fix in fix_files:
        # 映射回原始库的对应文件。**不能**用 os.path.relpath 直接镜像：
        # 修复库固定写在 <fix_root>/YYYY/MM/ 下，与原始库的目录布局无关，
        # 原始库是扁平目录时镜像出的路径不存在 → 每个文件都被静默 continue，
        # 最后 n_files=0 而 ok 仍是 True（假阳性：报告"验收通过"其实一个没验）。
        raw = find_counterpart(fix, fix_root, raw_root)
        if raw is None:
            # 原始库里确实没有对应文件（未处理的台站/日期）——照旧不算失败，
            # 但要**登记**下来，别再无声无息地跳过。
            out["skipped"].append(os.path.basename(fix))
            continue
        rep = verify_file(raw, fix, det, known=known,
                          use_catalog=use_catalog)
        out["files"].append(rep)
        if not rep["ok"]:
            out["ok"] = False
    out["n_files"] = len(out["files"])
    out["n_fail"] = sum(1 for r in out["files"] if not r["ok"])
    out["n_skipped"] = len(out["skipped"])
    # 一个文件都没能对上时，不能报"验收通过"——那是假阳性
    if out["n_files"] == 0 and fix_files:
        out["ok"] = False
        out["error"] = (
            "no repaired file could be matched back to the raw DB; check that "
            "the raw root really contains the source files (raw/fix layouts "
            "may differ)")
    return out


def main(argv=None):
    """命令行入口：批量修复一个月，并可选做验收。

    目录布局约定 ``<root>/<YYYY>/<MM>/XA.{站}.{YYYYMMDD}.mseed``。输入输出
    根目录必须显式给出（命令行参数或环境变量），模块内不保存任何默认路径。
    """
    ap = argparse.ArgumentParser(
        prog="python -m dssrr.repair_lib.e7_repair_build",
        description="E7 批量去异常：读取原始库，写入修复库，可选自动验收。")
    ap.add_argument("--raw-root", required=True,
                    help="原始库根目录（只读，YYYY/MM/ 结构）")
    ap.add_argument("--fix-root", required=True,
                    help="修复库输出根目录（自动创建，镜像 YYYY/MM 结构）")
    ap.add_argument("--year-month", default="197601",
                    help="处理的年月，形如 197601（默认 197601）")
    ap.add_argument("--stations", nargs="+", default=["S12", "S15", "S16"])
    ap.add_argument("--days", nargs="+", default=None, help="如 19760113；默认全月")
    ap.add_argument("--catalog", default=None,
                    help="修复明细 CSV 输出路径"
                         "（默认 <fix-root>/catalogs/E7_catalog_<YYYYMM>.csv）")
    ap.add_argument("--no-catalog", action="store_true", help="不写明细 CSV")
    ap.add_argument("--verify", action="store_true",
                    help="验收模式：只对修复库做每日验收（不重跑修复）")
    ap.add_argument("--settings", default=None,
                    help="读取 ~/.dssrr_settings.json 的 repair 组参数"
                         "（默认不读、用论文默认参数；显式传入 --settings 才生效）")
    args = ap.parse_args(argv)

    ym = str(args.year_month)
    if len(ym) != 6 or not ym.isdigit():
        print("ERR --year-month must be YYYYMM, e.g. 197601")
        sys.exit(2)
    yyyy, mm = ym[:4], ym[4:]

    in_dir = os.path.join(args.raw_root, yyyy, mm)
    out_dir = os.path.join(args.fix_root, yyyy, mm)
    catalog_path = args.catalog or os.path.join(
        args.fix_root, "catalogs", f"E7_catalog_{ym}.csv")

    if not os.path.isdir(in_dir):
        print("ERR --raw-root has no such month directory:", in_dir)
        sys.exit(2)

    det = E7Detector(sr=SR)

    if args.verify:
        ver = verify_month(in_dir, out_dir, det)
        import json
        os.makedirs(os.path.dirname(catalog_path), exist_ok=True)
        vpath = os.path.join(os.path.dirname(catalog_path),
                             f"E7_verify_{ym}.json")
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
        from ..paths import settings_read_path
        sp = settings_read_path()
        if not os.path.exists(sp):
            print("ERR --settings: file not found:", sp)
            sys.exit(2)
        with open(sp, encoding="utf-8") as f:
            params = _json.load(f).get("repair")
        print("--settings: using repair params from", sp)

    files = []
    for stn in args.stations:
        files += sorted(glob.glob(os.path.join(in_dir, f"XA.{stn}.{ym}*.mseed")))
    if args.days:
        files = [f for f in files if os.path.basename(f).split(".")[2] in args.days]

    catalog_rows = [] if not args.no_catalog else None
    t_start = time.time()
    tot_rep = tot_gap = 0
    for i, f in enumerate(files):
        t0 = time.time()
        try:
            nrep, ngap = repair_one_file(f, det, out_dir,
                                         catalog_rows=catalog_rows,
                                         params=params)
        except Exception as ex:
            print("ERR", os.path.basename(f), ex)
            continue
        tot_rep += nrep; tot_gap += ngap
        print(f"[{i+1}/{len(files)}] {os.path.basename(f)} ({time.time()-t0:.1f}s)")
    if catalog_rows is not None:
        write_catalog_csv(catalog_path, catalog_rows)
        print("catalog:", catalog_path, f"({len(catalog_rows)} rows)")
    print(f"\nDONE {len(files)} files, {tot_rep} repairs, {tot_gap} gaps, {time.time()-t_start:.0f}s")
    print("output dir:", out_dir)


if __name__ == "__main__":
    main()
