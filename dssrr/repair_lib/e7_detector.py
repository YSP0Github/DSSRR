"""
E7 检测器 v3 —— Apollo 月震（MHZ 通道）数据异常检测（正式版）
=============================================================
本模块是 DSSRR 批量修复流水线的「第一道闸」：在重建之前先判定哪些时间段
是**仪器伪迹**（应当修复），哪些是**真实月震事件**（必须原样保留）。

设计目标
--------
1. **不误伤月震** —— 事件形态判别采取保守策略（宁漏勿误），把真实事件
   从异常清单里剔除。
2. **异常段选取完整** —— 相邻异常段合并 + 双侧外扩 padding，避免 DSSRR
   取参考段时误拾到相邻异常导致质量下降或计算失败。

v3 相对 v2 的两项关键升级
-------------------------
1. 异常段合并阈值 5s → 120s：相距 <120s 的两个异常段合并为一个修复单元，
   避免 DSSRR 取前后参考段（默认 100s）时拾取相邻异常段。
2. 事件判定收紧（防「伪月震」误保护导致残留异常）：
   - 新增骤升-衰减形判据：段首 10% RMS 必须 ≥ 1.15 × 段中 RMS
     （月震「骤升 + 长 coda 衰减」形态；恒定矩形振荡簇 head≈mid 会被判非事件）
   - 新增缺失占比判据：段内缺失标记占比 < 5%
   因而「振荡 + 缺失交织、无骤升无衰减」的振荡簇现改判为 repair 整簇修复。

异常类别（均为仪器伪迹，月震不可能产生）
----------------------------------------
- MISSING: 零值 / -1 缺失标记（Apollo 原始计数中 -1 为缺数占位）
- FREEZE : 连续恒值（遥测冻结）
- SAT    : 饱和（|x| >= 1020，10-bit ADC 满量程）
- SPIKE  : 孤立尖峰（局部中值残差超阈值）
- STEP   : 阶跃（一阶差分稳健 z-score 超阈值）
- BURST  : 能量异常段（短窗 RMS vs 滑动背景 RMS 超阈值；经事件判别后保留）

不修复的两类
------------
- EVENT（事件保护）：骤升 + 长 coda 衰减的月震形态。
- GAP（长缺口）：段长超过 ``max_repair_sec``，超出 DSSRR 重建能力，
  保留原始数据并在目录标记，**绝不编造数据**。

边界约定
--------
- 形态段（段长 < 10s）：padding 5s（孤立伪迹，正常邻域）
- 长形态段与能量段   ：padding max(30s, 10%·段长)（覆盖完整受影响区域）

月震目录
--------
真实事件判定依赖一份合并月震目录（随包分发 ``moonquake_catalog.json``，
来源见下方常量注释）。用户可在 ``~/.dssrr_moonquake_catalog.json`` 放置
同结构的覆盖文件；旧 CSV 目录作为回退保留，兼容未更新环境。
"""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from scipy.ndimage import median_filter, binary_dilation, uniform_filter1d

SAT_HIGH = 1020.0
SAT_LOW = -1020.0

# ----------------------------------------------------------------------
# 月震事件目录核验：真实月震保护（避免将月震识别为异常）
# 来源：Gagnepain-Beyneix et al. (2006) / Lognonné et al. (2003) /
#       Apollo PSE Long-Period Event Catalog rev.1008c
# 统一合并为 moonquake_catalog.json（随包分发）；用户可在
# ~/.dssrr_moonquake_catalog.json 放置覆盖文件（同 JSON 结构）。
# 旧 CSV 保留为回退，兼容未更新环境。
# Date 格式 YYMMDDHHMM（UTC）
# ----------------------------------------------------------------------
_CATALOG_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "moonquake_catalog.json")
CATALOG_FILES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "apollo_catalog", "gagnepian_2006_catalog.csv"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "apollo_catalog", "lognonne_2003_catalog.csv"),
]


def _user_catalog_json():
    """用户覆盖目录文件路径（不存在则返回 ``None``）。

    路径解析集中在 :mod:`dssrr.paths`，同时兼容旧版
    ``~/.seisy_moonquake_catalog.json``。
    """
    from ..paths import catalog_override_read_path
    return catalog_override_read_path()


def _load_catalog_events_json():
    """从 JSON 文件加载事件 ISO 时间列表；优先用户覆盖文件。"""
    import json
    for p in (_user_catalog_json(), _CATALOG_JSON):
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            evs = data.get("events", [])
            if evs:
                return evs
        except Exception:
            continue
    return None


def load_lunar_event_seconds(day_str: str = "19760113"):
    """返回该日（YYYYMMDD）UTC 月震事件时间列表（当日 00:00 起秒）。

    优先读合并后的 moonquake_catalog.json；JSON 不可用时回退到
    旧 CSV（gagnepian_2006 + lognonne_2003）。
    """
    from datetime import datetime

    # 优先 JSON
    iso_events = _load_catalog_events_json()
    if iso_events is not None:
        day_events = []
        for iso in iso_events:
            try:
                dt = datetime.fromisoformat(iso)
            except ValueError:
                continue
            dstr = dt.strftime("%Y%m%d")
            if dstr == day_str:
                day_events.append((dt - datetime(dt.year, dt.month, dt.day)).total_seconds())
        return sorted(set(day_events))

    # 回退 CSV
    import pandas as pd
    day_events = []
    for p in CATALOG_FILES:
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        for v in df["Date"].astype(str):
            v = v.strip()
            if len(v) != 10 or not v.isdigit():
                continue
            yy, mm, dd, hh, mi = (int(v[0:2]), int(v[2:4]), int(v[4:6]),
                                  int(v[6:8]), int(v[8:10]))
            year = 2000 + yy if yy < 70 else 1900 + yy
            try:
                dt = datetime(year, mm, dd, hh, mi)
            except ValueError:
                continue  # 目录脏数据（如 7704712332）
            dstr = f"{year}{mm:02d}{dd:02d}"
            if dstr == day_str:
                day_events.append((dt - datetime(year, mm, dd)).total_seconds())
    return sorted(set(day_events))


def verify_catalog(repaired, events, day_str: str, tol_sec: float = 300.0,
                   sr: float = 6.625, protect_after_sec: float = 3600.0):
    """目录核验：真实月震保护（v4 扩展保护窗覆盖月震 coda）。

    - event 段：无目录事件命中 → 解除保护，转 repair（仪器伪迹，如振荡簇）
    - repair 段：命中目录保护窗 → 保护，转 event（真实月震，禁止整体修复）
    - 保护窗内的尖峰/短缺失段 → 返回为 spikes（保留修复资格，
      由"月震保护型局部尖峰修复"单独处理，只修孤立尖峰、不动 coda 主体）
    命中判据（秒）：事件时刻 t 的保护窗 = [t - tol_sec, t + protect_after_sec]，
    段与任一保护窗相交即命中。protect_after_sec 默认 3600s（1 小时）：
    1976-01-13 07:11 月震 coda 实测持续约 60 分钟（25860s → ~29460s 回落背景）。
    返回 (新 repaired 列表, 新 events 列表, 保护窗内 spike/missing 段列表)。
    """
    ev_times = load_lunar_event_seconds(day_str)
    if not ev_times:
        # v4g+：严格只信目录——无目录记录日，检测器判定的"事件"
        # （_is_real_event）不构成保护依据（可能把非月震信号误当事件），
        # 全部转回 repair（burst 类），由修复流程按校验门处理。
        return repaired + [(s, e, "burst") for s, e in events], [], []
    windows = [(t - tol_sec, t + protect_after_sec) for t in ev_times]

    def hit(s_sec, e_sec):
        return any(s_sec <= w1 and e_sec >= w0 for w0, w1 in windows)

    new_rep, new_ev, new_spike = [], [], []
    for s, e, k in repaired:
        if hit(s / sr, e / sr):
            if k in ("spike", "missing"):
                # 月震区内孤立尖峰/短缺失：保留修复资格（保护型尖峰修复）
                new_spike.append((s, e, k))
            else:
                # 月震主体与 coda 的持续振荡/长段：整体保护
                new_ev.append((s, e))
        else:
            new_rep.append((s, e, k))
    for s, e in events:
        if hit(s / sr, e / sr):
            new_ev.append((s, e))
        else:
            new_rep.append((s, e, "burst"))
    # 去重叠：转 repair 的小段可能落在原 repair 大段内部 → 合并为并集
    new_rep = _merge_simple(sorted(new_rep))
    new_ev = _merge_simple(sorted(new_ev))
    new_spike = _merge_simple(sorted(new_spike))
    return new_rep, new_ev, new_spike


def _merge_simple(segs):
    """重叠段合并（s <= 前一段 e 才算重叠；padding 相接但不重叠的段保持独立）。
    支持 (s,e) 或 (s,e,k) 元组，保留额外字段。"""
    if len(segs) <= 1:
        return segs
    out = [segs[0]]
    for seg in segs[1:]:
        s, e = seg[0], seg[1]
        if s <= out[-1][1]:
            prev = out[-1]
            if len(prev) > 2:
                out[-1] = (prev[0], max(prev[1], e), prev[2])
            else:
                out[-1] = (prev[0], max(prev[1], e))
        else:
            out.append(seg)
    return out


def protect_osc_clusters(repaired, d, sr,
                         cluster_gap_s=30.0, min_len_s=60.0,
                         spread=12.0, energy_ratio=1.3):
    """v4g：疑似真实信号的振荡/瞬态簇保护（不修复，转 event）。

    背景：step/burst 形态检测对"长持续强瞬态/振荡"过度敏感——如
    S16 19760108 05:20-08:08 UTC 的 2.8h 强振荡被切成 471 个 step 段、
    S15 19760122 11:00-11:46 UTC 的瞬态序列。这些段目录无月震记录
    （verify_catalog 不保护），若按常规修复（DSSRR 整段重建）会破坏
    真实信号（用户立场：宁可不过度、避免将月震/信号活动识别为异常）。
    因此对满足下述判据的 step/burst 簇**整体转 event（保护不修）**：

      1) 段 kind ∈ {step, burst}，相邻（间隔 <cluster_gap_s）成簇；
      2) 簇总长 > min_len_s（短促阶跃不构成振荡簇）；
      3) 变化分散度 sum|d1|/max|d1| > spread（持续瞬态的变化分散在大量
         样本上；单次大阶跃的变化集中（2-10），如 S12 53438s 真阶跃 ≈2）；
      4) 能量：簇内有效样本 MAD > energy_ratio × 簇外邻近（±300s）MAD。

    满足的簇从 repaired 移出并合并为 event（extra_events）——由调用方
    （catalog_build / repair_build）在 verify_catalog 之后调用，避免被
    "无目录记录解除保护"逻辑撤销。簇内 zero/freeze/sat/spike 段保留
    （缺失/冻结/饱和是数据问题，即使处于振荡区也要修）。
    返回 (new_repaired, extra_events)。
    """
    if not repaired:
        return repaired, []
    osc_kinds = {"step", "burst"}
    idxs = [i for i, (_, _, k) in enumerate(repaired) if k in osc_kinds]
    if not idxs:
        return repaired, []
    gap_smp = cluster_gap_s * sr
    clusters = []
    cur = [idxs[0]]
    for i in idxs[1:]:
        if repaired[i][0] - repaired[cur[-1]][1] < gap_smp:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    n = len(d)
    d1 = np.diff(d)
    remove = set()
    extra = []
    for cl in clusters:
        tot = sum(repaired[i][1] - repaired[i][0] + 1 for i in cl)
        if tot <= min_len_s * sr:
            continue
        s0 = min(repaired[i][0] for i in cl)
        e1 = max(repaired[i][1] for i in cl)
        d1_seg = d1[s0:min(e1, n - 1)]
        if len(d1_seg) < 8:
            continue
        a1 = np.abs(d1_seg)
        mx = float(np.max(a1))
        sm = float(np.sum(a1))
        if sm <= spread * mx:
            continue
        seg = d[s0:e1 + 1]
        v = seg[(seg > -1) & (seg != 0) & np.isfinite(seg)]
        if len(v) < 8:
            continue
        seg_mad = float(1.4826 * np.median(np.abs(v - np.median(v))))
        nb = int(300 * sr)
        ctx = np.concatenate([d[max(0, s0 - nb):s0],
                              d[e1 + 1:min(n, e1 + 1 + nb)]])
        ctx = ctx[(ctx > -1) & (ctx != 0) & np.isfinite(ctx)]
        if len(ctx) < 8:
            continue
        ctx_mad = float(1.4826 * np.median(np.abs(ctx - np.median(ctx))))
        ctx_mad = max(ctx_mad, 0.5)
        if seg_mad > energy_ratio * ctx_mad:
            remove.update(cl)
            extra.append((s0, e1))
    if remove:
        repaired = [r for i, r in enumerate(repaired) if i not in remove]
    if extra:
        extra = _merge_simple(sorted(extra))
    return repaired, extra


class E7Detector:
    def __init__(
        self,
        sr: float,
        # 尖峰
        spike_window_sec: float = 2.0,
        spike_k: float = 8.0,
        # 阶跃
        step_k: float = 12.0,
        # 冻结
        freeze_min_sec: float = 30.0,
        # 能量段
        rms_short_sec: float = 10.0,
        rms_bg_sec: float = 120.0,
        rms_ratio: float = 5.0,
        min_burst_sec: float = 5.0,
        # v4e：低动态量化日（全日中位短窗 RMS 很小，global floor 被平坦区拉低，
        # ratio_g 虚高 → 正常背景波动误判为能量异常）。此类日提高全局通道门槛
        # low_dynamic_gain 倍，只保留显著抬升的真能量段；step/freeze/zero 仍由
        # 形态通道负责，不受影响。
        low_dynamic_rms: float = 4.0,
        low_dynamic_gain: float = 3.0,
        # 合并/外扩
        # v3: 相距 <120s 的异常段合并为一个修复单元（DSSRR 前后参考默认 100s，
        #      防止参考窗拾取相邻异常段 → 计算失败/引入伪振荡）
        merge_gap_sec: float = 120.0,
        short_pad_sec: float = 5.0,
        long_pad_sec: float = 30.0,
        # 超长段处理上限：超过此长度且缺失主导才标记 gap 保留。
        # DSSRR 可重建较长缺失（参考段自动取 >=120s），2026-09-28
        # 从 1800s(30min) 提高到 7200s(2h)，53~100min 缺口尝试修复。
        max_repair_sec: float = 7200.0,
        # 事件判别
        event_morph_frac: float = 0.03,
        event_jitter_frac: float = 0.04,
        event_min_sec: float = 30.0,
        event_coda_ratio: float = 100.0,
        event_rms_smooth: float = 0.12,
        # v3 新增：骤升-衰减形判据（月震 head 能量必须 ≥ 1.15×中段）
        event_head_mid_ratio: float = 1.15,
        # v3 新增：段内缺失占比上限
        event_missing_frac: float = 0.05,
    ):
        self.sr = float(sr)
        self.spike_window = max(3, int(spike_window_sec * self.sr) | 1)
        self.spike_k = spike_k
        self.step_k = step_k
        self.freeze_min = max(1, int(freeze_min_sec * self.sr))
        self.rms_short = max(2, int(rms_short_sec * self.sr))
        self.rms_bg = max(10, int(rms_bg_sec * self.sr))
        self.rms_ratio = rms_ratio
        self.low_dynamic_rms = low_dynamic_rms
        self.low_dynamic_gain = low_dynamic_gain
        self.min_burst = max(1, int(min_burst_sec * self.sr))
        self.merge_gap = max(1, int(merge_gap_sec * self.sr))
        self.short_pad = max(1, int(short_pad_sec * self.sr))
        self.long_pad = max(1, int(long_pad_sec * self.sr))
        self.max_repair = max(1, int(max_repair_sec * self.sr))
        self.event_morph_frac = event_morph_frac
        self.event_jitter_frac = event_jitter_frac
        self.event_min = max(1, int(event_min_sec * self.sr))
        self.event_coda_ratio = event_coda_ratio
        self.event_rms_smooth = event_rms_smooth
        self.event_head_mid_ratio = event_head_mid_ratio
        self.event_missing_frac = event_missing_frac

    # ------------------------------------------------------------------
    def detect(self, data: np.ndarray):
        """返回 (修复段列表, 事件段列表, 缺口段列表, 掩码信息)。

        修复段: [(s, e, kind)] 左闭右闭，已外扩
        事件段: [(s, e)] 被保护的候选事件
        缺口段: [(s, e)] 超长断记/缺口，保留不修
        """
        if len(data) < 8:
            return [], [], [], {}
        d = np.asarray(data, dtype=float)

        # ---- 去直流（Apollo 原始计数带 ADC 基线）----
        good = d[(d > -1) & np.isfinite(d)]
        dc = float(np.median(good)) if len(good) else 0.0
        x = d.copy()
        x[~np.isfinite(x)] = 0.0
        x[x <= -1] = 0.0
        x = x - dc

        # ---- 形态掩码 ----
        # 缺失标记：原始计数为 0 / -1 / NaN（Apollo 缺数惯例）
        missing = (~np.isfinite(d)) | (d <= -1) | (d == 0)
        sat = (d >= SAT_HIGH) | (d <= SAT_LOW)
        zero = missing
        freeze = self._freeze_mask(d)   # 原始域检测恒值（遥测跳变-保持正常，阈值取 30s）
        spike = self._spike_mask(x)
        step = self._step_mask(x)
        morph = sat | zero | freeze | spike | step
        segs_morph = self._segments(morph)

        # ---- 能量掩码（排除形态区域）----
        energy = self._energy_mask(x)
        segs_energy = self._segments(energy & ~morph)

        # ---- 事件判别 ----
        kept_bursts: List[Tuple[int, int]] = []
        events: List[Tuple[int, int]] = []
        for s, e in segs_energy:
            if (e - s + 1) < self.min_burst:
                continue
            if self._is_real_event(x, s, e, morph, spike, missing):
                events.append((s, e))
            else:
                kept_bursts.append((s, e))

        # ---- 合并 ----
        all_segs = sorted(segs_morph + kept_bursts)
        merged = self._merge(all_segs)

        # ---- 分级：可修复 vs 长缺口 ----
        # 超长合并段（>max_repair）分流：
        #   - 段内缺失占比 > 50% → 真实长缺口（gap，保留原值）
        #   - 否则 → 低缺失长能量段（120s 合并的产物），拆回独立子段分别修复
        n = len(x)
        repaired: List[Tuple[int, int, str]] = []
        gaps: List[Tuple[int, int]] = []

        def _pad(s0, e0):
            L0 = e0 - s0 + 1
            if L0 <= 10 * self.sr:
                pad = self.short_pad
            else:
                # 长段 padding 固定上限（2×long_pad=60s），不随段长增长。
                # 旧逻辑 0.1*L0 对30分钟缺失每端加3分钟，把大量健康数据
                # 纳入 DSSRR 重建区域；DSSRR 参考段本身已取 >=120s 上下文。
                pad = min(self.long_pad * 2, max(self.long_pad, L0 // 20))
            return max(0, s0 - pad), min(n - 1, e0 + pad)

        masks = {"sat": sat, "zero": zero, "freeze": freeze,
                 "spike": spike, "step": step}

        def _primary_kind(s0, e0):
            """段的主异常类型（按原始段内各形态掩码占比）。"""
            seg = slice(s0, e0 + 1)
            fracs = {k: float(v[seg].mean()) for k, v in masks.items()}
            best = max(fracs, key=fracs.get)
            return best if fracs[best] > 0.01 else "burst"

        for s, e in merged:
            L = e - s + 1
            if L > self.max_repair:
                miss_frac = float(np.mean(missing[s:e + 1]))
                if miss_frac > 0.5:
                    gaps.append((s, e))
                else:
                    # 超长合并段（低缺失长能量段，120s 合并产物）拆为
                    # ≤max_repair 的连续块。块边界是"切"出来的，不是
                    # 真实异常边界，因此不 padding（padding 会把块外扩
                    # 进相邻段造成重叠；DSSRR 参考窗会自动取块外数据）。
                    # （2026-09-11 修复：此前拆回 all_segs 原始子段并各自
                    # padding，密集缺失日会产出数百个互相重叠的小段——
                    # S12 01-06 450 段中 410 段重叠、01-25 全天 1073 段；
                    # 动态 0.1·L0 padding 又把拆块扩进相邻短段。修复后
                    # 段两两不重叠，修复明细不再重复。）
                    for ss in range(s, e + 1, self.max_repair):
                        ee = min(ss + self.max_repair - 1, e)
                        repaired.append((ss, ee, _primary_kind(ss, ee)))
                continue
            a, b = _pad(s, e)
            repaired.append((a, b, _primary_kind(s, e)))

        info = {
            "dc": dc,
            "n_morph_segs": len(segs_morph),
            "n_burst_segs": len(segs_energy),
            "n_events": len(events),
            "n_kept_bursts": len(kept_bursts),
            "n_repaired": len(repaired),
            "n_gaps": len(gaps),
            "n_samples": n,
            "masks": masks,
        }
        return repaired, events, gaps, info

    # ------------------------------------------------------------------
    def _freeze_mask(self, x: np.ndarray) -> np.ndarray:
        """冻结（恒值保持）掩码：连续相同样本 ≥ 有效阈值。

        平台化常态抑制（2026-09-09 用户反馈 S12 19760102 误检 150 段）：
        低动态量化日"平坦保持"是数据常态——实测 19760102/19760105 相邻
        样本相同率 95%+、≥30s 恒定 run 达 256/341 个（p50≈40s），而正常
        活跃日 19760113 仅 3 个长 run。因此按全天恒定 run 数量自适应：
        - run 数量 > 50（量化平台常态）→ 有效阈值提到 120s，只保留显著
          更长的真冻结（19760105 的 357s/1685s 级冻结不受影响）
        - run 数量少（罕见冻结）→ 保持 freeze_min（30s）
        """
        d = np.zeros(len(x), dtype=bool)
        if len(x) < 2:
            return d
        same = np.diff(x) == 0
        idx = np.flatnonzero(same)
        if len(idx) == 0:
            return d
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        n_runs = sum(1 for g in groups if len(g) + 1 >= self.freeze_min)
        if n_runs > 50:
            thr = max(self.freeze_min, int(120 * self.sr))
        else:
            thr = self.freeze_min
        for g in groups:
            if len(g) + 1 >= thr:
                d[g[0]:g[-1] + 2] = True
        return d

    def _spike_mask(self, x: np.ndarray) -> np.ndarray:
        win = self.spike_window
        baseline = median_filter(x, size=win)
        residual = np.abs(x - baseline)
        scale = 1.4826 * median_filter(residual, size=win)
        # 全局尺度（稳健）
        glob = 1.4826 * np.median(np.abs(x - np.median(x)))
        floor = max(glob, 0.5)
        thr = self.spike_k * np.maximum(scale, floor * 0.5)
        return residual > thr

    def _step_mask(self, x: np.ndarray) -> np.ndarray:
        if len(x) < 4:
            return np.zeros(len(x), dtype=bool)
        d1 = np.diff(x)
        med = np.median(d1)
        mad = 1.4826 * np.median(np.abs(d1 - med))
        scale = max(mad, 0.5)
        big = np.abs(d1 - med) > self.step_k * scale
        m = np.zeros(len(x), dtype=bool)
        idx = np.flatnonzero(big)
        if len(idx):
            m[np.clip(idx, 0, len(x) - 1)] = True
            m[np.clip(idx + 1, 0, len(x) - 1)] = True
        return m

    def _energy_mask(self, x: np.ndarray) -> np.ndarray:
        rms_s = self._rms_sliding(x, self.rms_short)
        rms_b = self._rms_sliding(x, self.rms_bg)
        # 全局背景基准（长段能量不污染：用全日中位短窗 RMS）
        global_floor = np.median(rms_s) * 0.5 + 1e-9
        # 局部背景（短突发用）
        local_floor = np.maximum(rms_b, global_floor * 2.0)
        # 双通道：相对全局抬升（长段）或相对局部抬升（短突发）
        ratio_g = rms_s / global_floor
        ratio_l = rms_s / local_floor
        thr_g = self.rms_ratio * 2.0
        # v4e：低动态量化日（如 S12 19760102：全天平坦、med RMS≈3）的
        # global floor 被大量平坦区拉低，任何正常波动 ratio_g 都虚高触阈。
        # 提高全局通道门槛，仅保留显著抬升；正常日保持原阈值。
        if np.median(rms_s) < self.low_dynamic_rms:
            thr_g *= self.low_dynamic_gain
        m = (ratio_g > thr_g) | (ratio_l > self.rms_ratio)
        m = binary_dilation(m, structure=np.ones(3, dtype=bool))
        return m

    @staticmethod
    def _rms_sliding(x: np.ndarray, win: int) -> np.ndarray:
        m = uniform_filter1d(x ** 2, size=win, mode="reflect")
        return np.sqrt(np.maximum(m, 0.0))

    # ------------------------------------------------------------------
    def _is_real_event(self, x, s, e, morph, spike, missing=None) -> bool:
        L = e - s + 1
        if L < self.event_min:
            return False
        seg = x[s:e + 1]
        # 0) v3 新增：段内缺失标记占比（防"振荡+缺失交织"段）
        if missing is not None and len(missing) == len(x):
            if np.mean(missing[s:e + 1]) > self.event_missing_frac:
                return False
        # 1) 形态干净度
        if np.mean(morph[s:e + 1]) > self.event_morph_frac:
            return False
        if np.mean(spike[s:e + 1]) > self.event_jitter_frac:
            return False
        # 2) v3 新增：骤升-衰减形判据 —— 月震"骤升+长 coda 衰减"，
        #    段首 10% RMS 必须 ≥ 1.15 × 段中 30% RMS
        #    （恒定矩形振荡簇 head≈mid 会被判非事件，转 repair 整簇修复）
        head = max(3, int(0.1 * L))
        mid = max(3, int(0.3 * L))
        rms_head = np.sqrt(np.mean(seg[:head] ** 2) + 1e-12)
        rms_mid = np.sqrt(np.mean(seg[head:head + mid] ** 2) + 1e-12)
        if rms_head < self.event_head_mid_ratio * rms_mid:
            return False
        # 3) coda 衰减：尾部仍保有能量（异常段尾部快速归零）
        tail = max(2, int(0.1 * L))
        rms_tail = np.sqrt(np.mean(seg[-tail:] ** 2) + 1e-12)
        if rms_head > 1e-6 and rms_tail < rms_head / self.event_coda_ratio:
            return False
        # 4) 包络平滑：短窗 RMS 相邻变化率中位小
        r = self._rms_sliding(seg, self.rms_short)
        r = r / (np.max(r) + 1e-12)
        if len(r) > 3:
            dr = np.abs(np.diff(r))
            if np.median(dr) > self.event_rms_smooth:
                return False
        return True

    # ------------------------------------------------------------------
    @staticmethod
    def _segments(mask: np.ndarray) -> List[Tuple[int, int]]:
        if not np.any(mask):
            return []
        diff = np.diff(mask.astype(np.int8), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))

    def _merge(self, segs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(segs) <= 1:
            return segs
        merged = [segs[0]]
        for s, e in segs[1:]:
            if s - merged[-1][1] <= self.merge_gap:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged
