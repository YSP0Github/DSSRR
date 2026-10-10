# -*- coding: utf-8 -*-
"""E-RealInj v6 全量批量实验。

与 v5 的差别
------------
1. **代码版本**：DSSRR 走当前 ``seisy/core/anomaly_repair``（含论文实验后加入的
   6 处数值修正，以及趋势时间轴安全间隙几何修正）。
2. **参考段协议**：``ref_len = clip(anomaly/2, 120 s, 600 s)``（论文正文所述协议）。
   ``results_E2r_full_v5`` 归档实际由 **下限 15 s** 的旧脚本产出
   （见 ``__pycache__/E2r_v5_batch.cpython-311.pyc``，时间戳 2026-09-22 15:46），
   与论文自述协议不符；v6 按论文协议重跑。
3. **写回协议**：合并区间只用于确定参考段长度与合成上下文，**只写回原始检测到的
   异常段**（论文：“...while only the originally detected interval is written
   back”）。旧脚本把整个合并区间写回，在 spike 场景会把 1500 s 窗内的健康数据
   一并替换（写回样本 65 -> 5256，整体 RMSE 0.285 -> 0.451）。
4. **落盘**：走 ``results_io.ResultsWriter`` —— FLOAT32 mseed（无损可复算）、
   ``intervals.json``（多段异常区间）、``PROVENANCE.json`` 溯源、
   每案例 ``stats_detail.csv`` + 全局 ``stats_summary.csv``；
   目录结构与图像样式与 v5 同构。
5. **输出目录**：``results_E2r_full_v6``，**绝不覆盖 v5**。

协议（异常注入 / 自适应边界 / 修复 / 指标定义）全部复用 ``E2r_v5_batch``，
保证 v5↔v6 的差异**只来自代码版本**。

用法
----
    python E2r_v6_batch.py                 # 默认输出 results_E2r_full_v6
    python E2r_v6_batch.py <输出目录>       # 自定义输出目录

注意：含 UNet，需在装有 torch 的环境运行（G:/miniconda3/envs/seisy/python.exe）。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from obspy import read

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, HERE)

import E2r_v5_batch as V5          # noqa: E402  复用协议
import results_io as RIO           # noqa: E402  统一归档

SR = V5.SR
SEED = V5.SEED
CLEAN_DIR = V5.CLEAN_DIR
V5_DIR = os.path.join(HERE, "results_E2r_full_v5")
DEFAULT_OUT = os.path.join(HERE, "results_E2r_full_v6")
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT


def main():
    if os.path.abspath(OUT_DIR) == os.path.abspath(V5_DIR):
        raise SystemExit("拒绝写入 v5 目录：v6 重跑不得覆盖 results_E2r_full_v5")

    print(f"{'='*64}\nE2r v6 全量批量实验\n  输出目录: {OUT_DIR}\n{'='*64}",
          flush=True)
    writer = RIO.ResultsWriter(OUT_DIR, experiment="E2r_full_v6")

    clean_files = sorted(f for f in os.listdir(CLEAN_DIR)
                         if f.endswith(".mseed"))
    _only = os.environ.get("E2R_ONLY")
    if _only:
        clean_files = [f for f in clean_files if _only in f]
        print(f"[E2R_ONLY={_only}] 仅处理 {len(clean_files)} 个文件")
    print(f"找到 {len(clean_files)} 个干净数据文件", flush=True)

    t_all = time.perf_counter()
    for fname in clean_files:
        parts = fname.replace(".mseed", "").split(".")
        station = parts[1]
        start_str = parts[4].split("-")[0]
        event_name = f"{station}_{start_str}"

        st = read(os.path.join(CLEAN_DIR, fname))
        tr = st[0]
        orig_starttime = tr.stats.starttime
        data = tr.data.astype(float)
        n = len(data)
        print(f"\n{'='*64}\n事件: {event_name}  ({n/SR/3600:.1f}h)\n{'='*64}",
              flush=True)

        for scenario in V5.SCENARIOS:
            case = f"{event_name}_{scenario}"
            for anom_type in V5.ANOM_TYPES:
                for level in V5.LEVELS:
                    corrupted, intervals = V5.inject_anomaly(
                        data, anom_type, level, SEED, scenario=scenario)
                    total_anom_sec = sum(e - s + 1 for s, e in intervals) / SR
                    print(f"  [{scenario}] {anom_type}/{level}: "
                          f"{len(intervals)} 段, 共 {total_anom_sec:.1f}s",
                          flush=True)

                    repaired_by_method = {}
                    for mname in V5.METHODS.keys():
                        t0 = time.perf_counter()
                        print(f"      {mname}...", end=" ", flush=True)
                        try:
                            repaired_by_method[mname] = V5.repair_intervals(
                                mname, corrupted, intervals, SEED)
                            print(f"done ({time.perf_counter()-t0:.1f}s)",
                                  flush=True)
                        except Exception as exc:      # noqa: BLE001
                            import traceback
                            print(f"FAILED ({exc})", flush=True)
                            traceback.print_exc()

                    writer.save_case(
                        event=case, station=station, scenario=scenario,
                        anom_type=anom_type, level=level,
                        truth=data, corrupted=corrupted, intervals=intervals,
                        repaired_by_method=repaired_by_method,
                        method_order=list(V5.METHODS.keys()),
                        starttime=orig_starttime, sr=SR,
                        total_anom_sec=total_anom_sec,
                    )

    df = writer.write_summary()
    writer.write_provenance(script_path=__file__,
                            extra={"protocol": "E2r_v5_batch (reused)",
                                   "sr": SR, "seed": SEED,
                                   "n_cases": int(df["event"].nunique())
                                   if not df.empty else 0})
    print(f"\n{'='*64}\nv6 完成，用时 {time.perf_counter()-t_all:.1f}s\n{'='*64}")


if __name__ == "__main__":
    main()
