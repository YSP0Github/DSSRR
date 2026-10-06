# -*- coding: utf-8 -*-
"""DSSRR command-line batch repair (``dssrr-batch``).

Cross-platform (Linux / macOS / Windows PowerShell) entry point for repairing a
whole archive month by month.  It runs **the same pipeline as the desktop GUI's
Batch Repair page**, but without importing PyQt, so it works on headless
servers.

Pipeline
--------
::

    scan scope  ->  E7 anomaly detection  ->  per-file multi-level repair
                ->  moonquake-catalogue protection  ->  write catalog CSV
                ->  automatic verification  ->  summary JSON

Directory convention::

    <raw_root>/1976/01/XA.S12.19760113.mseed     # input, read-only
    <fix_root>/1976/01/XA.S12.19760113.mseed     # output, mirror structure
    <catalog_dir>/E7_catalog_197601.csv          # per-month repair detail
    <catalog_dir>/verify_197601.json             # per-month verification
    <catalog_dir>/batch_summary.json             # overall summary

Usage
-----
::

    dssrr-batch --raw-root /data/apollo/raw --fix-root /data/apollo/repair \\
        --stations S12,S15 --d0 19760101 --d1 19760229

Exit status is ``0`` when verification passes for every repaired file, and
``1`` when any file failed or the arguments were invalid.  See ``docs/CLI.md``
for the full option list and worked examples.

Programmatic use: :func:`quick_scan` and :func:`batch_repair` are importable
directly and take plain Python arguments.

Dependencies: numpy, obspy, pandas (no PyQt5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import discovery

DEFAULT_SR = 6.625
SEC_PER_FILE = 0.9  # 1976-01 measured baseline (same estimate as the GUI)


# ----------------------------------------------------------------------
# 文件扫描 —— 实现见 dssrr.discovery（与 GUI batch_repair 共用同一份规则）
# ----------------------------------------------------------------------
_iter_scope_files = discovery.iter_scope_files


def quick_scan(raw_root, stations, d0, d1):
    """快速统计：文件数、按年月、总字节、预计耗时（不读文件内容）。"""
    n, nbytes = 0, 0
    by_ym = {}
    for f, y, m in _iter_scope_files(raw_root, stations, d0, d1):
        n += 1
        by_ym[f"{y:04d}-{m:02d}"] = by_ym.get(f"{y:04d}-{m:02d}", 0) + 1
        try:
            nbytes += os.path.getsize(f)
        except OSError:
            pass
    return {
        "files": n, "bytes": nbytes, "by_ym": by_ym,
        "est_sec": n * SEC_PER_FILE,
    }


# ----------------------------------------------------------------------
# 批处理主流程（等价 GUI BatchWorker.run，无 Qt 信号）
# ----------------------------------------------------------------------
def batch_repair(raw_root, fix_root, stations, d0, d1,
                 catalog_dir=None, use_catalog=True,
                 det_kwargs=None, params=None, sr=DEFAULT_SR,
                 log=None, stop_flag=None):
    """执行批量去异常并返回汇总 dict。stop_flag 为可选的 callable() -> bool。"""
    if log is None:
        log = lambda msg: print(msg)  # noqa: E731
    from dssrr.repair_lib.e7_repair_build import (repair_one_file,
                                                  write_catalog_csv,
                                                  verify_file)
    from dssrr.repair_lib.e7_detector import E7Detector

    det = E7Detector(sr=sr, **(det_kwargs or {}))
    if catalog_dir is None:
        catalog_dir = os.path.join(fix_root, "catalogs")

    files = list(_iter_scope_files(raw_root, stations, d0, d1))
    total = len(files)
    log(f"[scope] {total} file(s), stations={','.join(stations) or 'ALL'}, "
        f"{d0}~{d1}")
    if total == 0:
        return {"files": 0, "repaired": 0, "gaps": 0, "errors": [],
                "months": [], "verify": None, "elapsed": 0.0}

    t0 = time.time()
    summary = {"files": total, "repaired": 0, "gaps": 0, "errors": [],
               "months": [], "verify": None, "elapsed": 0.0}
    groups = {}
    for f, y, m in files:
        groups.setdefault(f"{y:04d}{m:02d}", []).append(f)

    for i, ym in enumerate(sorted(groups), 1):
        if stop_flag and stop_flag():
            log("[stop] user requested stop")
            break
        out_ym = os.path.join(fix_root, ym[:4], ym[4:6])
        rows = []
        log(f"[{i}/{len(groups)}] month {ym}: {len(groups[ym])} file(s)")
        for f in sorted(groups[ym]):
            if stop_flag and stop_flag():
                break
            try:
                nrep, ngap = repair_one_file(
                    f, det, out_ym, verbose=False, catalog_rows=rows,
                    params=params, use_catalog=use_catalog)
                summary["repaired"] += nrep
                summary["gaps"] += ngap
                log(f"  repaired {os.path.basename(f)}  "
                    f"(rep={nrep} gap={ngap})")
            except Exception as ex:
                summary["errors"].append(
                    (os.path.basename(f), str(ex)[:120]))
                log(f"  ERR {os.path.basename(f)}: {ex}")
        if rows and catalog_dir:
            try:
                os.makedirs(catalog_dir, exist_ok=True)
                cpath = os.path.join(catalog_dir, f"E7_catalog_{ym}.csv")
                write_catalog_csv(cpath, rows)
                log(f"[catalog] {ym}: {len(rows)} row(s) -> {cpath}")
            except Exception as ex:
                log(f"[catalog] {ym} ERR: {ex}")
        summary["months"].append(ym)

    # 自动验收：只验收本次处理过的文件
    log("[verify] verifying repaired files...")
    try:
        ver = {"ok": True, "n_files": 0, "n_fail": 0, "n_skip": 0,
               "by_month": {}}
        for ym in sorted(groups):
            r_raw = os.path.join(raw_root, ym[:4], ym[4:6])
            r_fix = os.path.join(fix_root, ym[:4], ym[4:6])
            ym_files, ym_fail = 0, 0
            for f in sorted(groups[ym]):
                fb = os.path.basename(f)
                fix_path = os.path.join(r_fix, fb)
                if not os.path.exists(fix_path):
                    ver["n_skip"] += 1
                    log(f"  SKIP {fb}: no repaired file")
                    continue
                try:
                    rep = verify_file(f, fix_path, det,
                                      use_catalog=use_catalog)
                    ver["n_files"] += 1
                    ym_files += 1
                    if not rep.get("ok"):
                        ver["n_fail"] += 1
                        ym_fail += 1
                        ver["ok"] = False
                        log(f"  FAIL {fb}: "
                            f"{'; '.join(rep.get('issues', []))}")
                except Exception as ex:
                    ver["n_fail"] += 1
                    ym_fail += 1
                    ver["ok"] = False
                    log(f"  ERR {fb}: {ex}")
            ver["by_month"][ym] = {"ok": ym_fail == 0, "n_files": ym_files,
                                   "n_fail": ym_fail}
        if catalog_dir:
            try:
                os.makedirs(catalog_dir, exist_ok=True)
                vpath = os.path.join(
                    catalog_dir,
                    "verify_" + "_".join(sorted(summary["months"]))[:12]
                    + ".json")
                with open(vpath, "w", encoding="utf-8") as f:
                    json.dump(ver, f, ensure_ascii=False, indent=1)
                log(f"[verify] json -> {vpath}")
            except Exception as ex:
                log(f"[verify] json ERR: {ex}")
        summary["verify"] = ver
    except Exception as ex:
        summary["verify"] = {"ok": False, "error": str(ex)[:200]}

    summary["elapsed"] = time.time() - t0
    log(f"[done] repaired={summary['repaired']} gaps={summary['gaps']} "
        f"errors={len(summary['errors'])} elapsed={summary['elapsed']:.1f}s "
        f"verify_ok={summary['verify'].get('ok') if summary['verify'] else None}")
    return summary


# ----------------------------------------------------------------------
# 参数解析 / 入口
# ----------------------------------------------------------------------
def _positive_date(v):
    v = v.strip()
    if not (len(v) == 8 and v.isdigit()):
        raise argparse.ArgumentTypeError(f"bad YYYYMMDD: {v!r}")
    return int(v)


def _run(args):
    if not args.raw_root or not os.path.isdir(args.raw_root):
        sys.exit(f"[error] raw root not found: {args.raw_root!r}")
    stations = [s.strip() for s in (args.stations or "").split(",")
                if s.strip()]
    d0, d1 = sorted((args.d0, args.d1))
    r = quick_scan(args.raw_root, stations, d0, d1)
    print(f"[scan] {r['files']} file(s), {r['bytes']/1024**3:.2f} GB, "
          f"{len(r['by_ym'])} month(s), est. {r['est_sec']/60:.1f} min")

    det_kwargs, params = {}, None
    if args.settings:
        try:
            from dssrr.paths import settings_read_path
            from dssrr.repair_lib import repair_settings as RS
            s = RS.load_settings()
            det_kwargs = dict(s.get("detect", {}) or {})
            params = dict(s.get("repair", {}) or {})
            print(f"[settings] loaded user settings "
                  f"({settings_read_path()})")
        except Exception as ex:
            print(f"[warning] settings load failed, using defaults: {ex}")

    summary = batch_repair(
        args.raw_root, args.fix_root, stations, d0, d1,
        catalog_dir=args.catalog_dir, use_catalog=not args.no_catalog,
        det_kwargs=det_kwargs, params=params, sr=args.sr)

    # 汇总落盘
    os.makedirs(args.catalog_dir, exist_ok=True) if args.catalog_dir else None
    out_json = args.summary_json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1,
                  default=lambda o: o if not hasattr(o, "isoformat") else
                  str(o))
    print(f"[summary] -> {out_json}")
    ok = summary["verify"] and summary["verify"].get("ok")
    if ok:
        print("[RESULT] VERIFY OK")
    else:
        print("[RESULT] VERIFY FAIL "
              f"(fail={summary['verify'].get('n_fail', '?')} "
              f"skip={summary['verify'].get('n_skip', '?')})" if ok is False
              else "[RESULT] VERIFY FAIL (see json)")
    return 0 if ok else 1


def build_parser():
    p = argparse.ArgumentParser(
        prog="dssrr batch-repair",
        description="DSSRR 批量去异常命令行工具（Linux/PowerShell 通用）")
    p.add_argument("--raw-root", required=True,
                   help="raw 库根目录；目录结构不限，文件名需为 "
                        "NET.STA.YYYYMMDD.mseed 或 "
                        "NET.STA.LOC.CHA.START-END.mseed")
    p.add_argument("--fix-root", required=True,
                   help="修复库输出根目录（自动创建）")
    p.add_argument("--stations", default="",
                   help="逗号分隔台站列表，如 S12,S15；缺省=全部")
    p.add_argument("--d0", type=_positive_date, default=19760101,
                   help="开始日期 YYYYMMDD")
    p.add_argument("--d1", type=_positive_date, default=19760201,
                   help="结束日期 YYYYMMDD")
    p.add_argument("--catalog-dir", default=None,
                   help="catalog/verify 输出目录；缺省 = <fix-root>/catalogs")
    p.add_argument("--no-catalog", action="store_true",
                   help="关闭月震目录保护（检测器判定事件全部转 repair）")
    p.add_argument("--settings", action="store_true",
                   help="读取用户设置 ~/.dssrr_settings.json "
                        "（检测/修复参数）；缺省=内置默认")
    p.add_argument("--sr", type=float, default=DEFAULT_SR,
                   help=f"采样率 Hz（缺省 {DEFAULT_SR}）")
    p.add_argument("--summary-json", default=None,
                   help="汇总 JSON 输出路径；缺省 = <catalog-dir>/"
                        "batch_summary.json")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.summary_json is None:
        cat = args.catalog_dir or os.path.join(args.fix_root, "catalogs")
        args.summary_json = os.path.join(cat, "batch_summary.json")
    if args.catalog_dir is None:
        args.catalog_dir = os.path.join(args.fix_root, "catalogs")
    try:
        rc = _run(args)
    except KeyboardInterrupt:
        print("\n[interrupted by user]", file=sys.stderr)
        rc = 130
    sys.exit(rc)


if __name__ == "__main__":
    main()
