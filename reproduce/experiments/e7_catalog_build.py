"""E7 主流水线 v3：检测 S12/S15/S16 1976-01 MHZ，输出异常目录 CSV（含 repair/event/gap）
v3：目录核验（月震保护）+ 120s 合并 + 事件判定收紧"""
import glob
import os
import numpy as np
import pandas as pd
from obspy import read
from e7_detector import (E7Detector, verify_catalog, protect_osc_clusters,
                         load_lunar_event_seconds, _merge_simple)
from e7_repair_build import merge_ref_overlap, missing_runs_in

SR = 6.625
IN = r"G:\SeisY_Project\database\1976\01"
OUT_CSV = r"G:\SeisY\docs\dssrr_paper\experiments\E7_catalog_197601.csv"
STATIONS = ["S12", "S15", "S16"]


def process_file(f, det):
    st = read(f)
    mhz = [tr for tr in st if tr.stats.channel == "MHZ"]
    if not mhz:
        return None
    tr = mhz[0]
    d = tr.data.astype(float)
    repaired, events, gaps, info = det.detect(d)
    # 目录核验（真实月震保护 + coda 保护窗 + 保护窗内尖峰分流）。
    # v4h：严格只信目录——检测器判定的事件（_is_real_event）与形态
    # 振荡簇均不再作为保护依据（无目录日 verify 会把 events 全部转
    # repair）；非目录"分散变化"段的防破坏由修复层能量保持 guard 承担。
    day_str = os.path.basename(f).split(".")[2]
    repaired, events, spikes = verify_catalog(repaired, events, day_str)
    all_events = events
    # 与修复流水线一致：保护窗内 event 段（月震 coda）内部的孤立缺失
    # run（≤20 点）也走保护型尖峰修复 → 目录须标注（否则查看器漏标）
    extra_spikes = []
    for s, e in all_events:
        for rs, re in missing_runs_in(d, s, e):
            extra_spikes.append((rs, re, "missing"))
    if extra_spikes:
        spikes = _merge_simple(sorted(spikes + extra_spikes))
    # 与修复流水线保持一致：参考段重叠合并（问题3）——DSSRR 参考窗
    # （≥L/2，上限600s）与相邻修复段重叠时，两段及中间区域合并为一个
    # 修复单元。目录必须反映实际修复单元，否则查看器标注/明细与真实
    # 修复不一致（长异常后紧跟异常、中间看似正常段会被分别显示）。
    repaired = merge_ref_overlap(repaired, SR)
    t0 = tr.stats.starttime
    rows = []
    for s, e, kind in repaired:
        rows.append({
            "station": tr.stats.station, "day": t0.strftime("%Y-%m-%d"),
            "start_s": float(s) / SR, "end_s": float(e) / SR,
            "start_time": str(t0 + s / SR), "end_time": str(t0 + (e + 1) / SR),
            "len_s": (e - s + 1) / SR, "kind": "repair", "subtype": kind,
            "source": "",
        })
    for s, e, kind in spikes:
        rows.append({
            "station": tr.stats.station, "day": t0.strftime("%Y-%m-%d"),
            "start_s": float(s) / SR, "end_s": float(e) / SR,
            "start_time": str(t0 + s / SR), "end_time": str(t0 + (e + 1) / SR),
            "len_s": (e - s + 1) / SR, "kind": "repair", "subtype": f"{kind}+moon",
            "source": "moon",
        })
    for s, e in all_events:
        rows.append({
            "station": tr.stats.station, "day": t0.strftime("%Y-%m-%d"),
            "start_s": float(s) / SR, "end_s": float(e) / SR,
            "start_time": str(t0 + s / SR), "end_time": str(t0 + (e + 1) / SR),
            "len_s": (e - s + 1) / SR, "kind": "event", "subtype": "",
            "source": "moon",
        })
    for s, e in gaps:
        rows.append({
            "station": tr.stats.station, "day": t0.strftime("%Y-%m-%d"),
            "start_s": float(s) / SR, "end_s": float(e) / SR,
            "start_time": str(t0 + s / SR), "end_time": str(t0 + (e + 1) / SR),
            "len_s": (e - s + 1) / SR, "kind": "gap", "subtype": "",
            "source": "",
        })
    return rows


def main():
    det = E7Detector(sr=SR)
    files = []
    for stn in STATIONS:
        files += sorted(glob.glob(os.path.join(IN, f"XA.{stn}.197601*.mseed")))
    all_rows = []
    for i, f in enumerate(files):
        try:
            rows = process_file(f, det)
        except Exception as ex:
            print("ERR", os.path.basename(f), ex)
            continue
        if rows is None:
            print("NO MHZ", os.path.basename(f))
            continue
        all_rows += rows
        cnt = {}
        for r in rows:
            cnt[r["kind"]] = cnt.get(r["kind"], 0) + 1
        print(f"[{i+1}/{len(files)}] {os.path.basename(f)}  {cnt}")
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\nTOTAL rows:", len(df))
    for k in ["repair", "event", "gap"]:
        sub = df[df.kind == k]
        print(f"  {k}: {len(sub)} segs, {round(sub['len_s'].sum()/60, 1)} min")
    print("by station:", df.groupby("station").size().to_dict())
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
