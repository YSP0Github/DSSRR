# -*- coding: utf-8 -*-
"""DSSRR 示例 05 —— 用 Python API 跑批量修复（等价于 ``dssrr-batch`` 命令）。

命令行工具 ``dssrr-batch`` 适合在服务器上批量处理整个归档；如果你想把批量
修复嵌进自己的脚本/流水线，直接用它的两个函数即可：

- :func:`dssrr.cli.quick_scan` —— 只统计文件数、体量、预计耗时，不读内容。
- :func:`dssrr.cli.batch_repair` —— 真正跑：E7 检测 → 多级修复 →
  月震目录保护 → 写 catalog CSV → 自动验收 → 汇总 dict。

为了完全自包含，本脚本会先在一个临时目录里**造一个迷你归档**（Apollo 风格
的 ``<raw>/YYYY/MM/XA.STN.YYYYMMDD.mseed``），再对它跑批量修复，最后打印汇总。

运行::

    python examples/05_batch_cli.py

依赖：numpy、obspy、pandas（批量流水线不需要 PyQt5）。
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def build_synthetic_archive(root, stations=("S12", "S15"), days=("19760113",),
                            sr=6.625, hours=1.0, seed=0):
    """造一个 Apollo 风格的迷你归档，并人为注入几处异常。

    目录结构：``<root>/YYYY/MM/XA.<STN>.<YYYYMMDD>.mseed``
    每文件注入一段全零（缺失）与一处孤立尖峰，确保修复流水线有事可做。
    """
    from obspy import Trace, UTCDateTime, Stream

    rng = np.random.default_rng(seed)
    n = int(hours * 3600 * sr)
    written = []
    for stn in stations:
        for day in days:
            # 有色的背景噪声 + 常数基线，模拟原始计数
            white = rng.normal(0.0, 1.0, n)
            colored = np.zeros(n)
            for i in range(1, n):
                colored[i] = 0.98 * colored[i - 1] + 0.02 * white[i]
            counts = np.round(500.0 + 3.0 * colored).astype(np.int32)

            # 注入：一段 90 s 全零（缺失）+ 一处孤立尖峰
            g0 = int(0.25 * n)
            counts[g0:g0 + int(90 * sr)] = 0
            counts[int(0.75 * n)] = 1023

            y, m, d = int(day[:4]), int(day[4:6]), int(day[6:8])
            start = UTCDateTime(y, m, d, 0, 0, 0)
            tr = Trace(data=counts)
            tr.stats.network = "XA"
            tr.stats.station = stn
            tr.stats.location = "01"
            tr.stats.channel = "MHZ"
            tr.stats.sampling_rate = sr
            tr.stats.starttime = start

            out_dir = Path(root) / f"{y:04d}" / f"{m:02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            fpath = out_dir / f"XA.{stn}.{day}.mseed"
            Stream([tr]).write(str(fpath), format="MSEED")
            written.append(fpath)
    return written


def main():
    try:
        import obspy  # noqa: F401
    except ImportError:
        raise SystemExit("[error] obspy is required: pip install obspy")

    from dssrr.cli import quick_scan, batch_repair

    tmp = Path(tempfile.mkdtemp(prefix="dssrr_example_"))
    raw_root = tmp / "raw"
    fix_root = tmp / "fix"
    try:
        files = build_synthetic_archive(str(raw_root))
        print(f"[setup] synthetic archive: {raw_root}")
        print(f"[setup] wrote {len(files)} file(s):")
        for f in files:
            print(f"          {f.relative_to(raw_root)}")

        stations = ["S12", "S15"]
        d0, d1 = 19760113, 19760113

        # ---- 1) 先扫一遍：文件数 / 体量 / 预计耗时 ----
        scan = quick_scan(str(raw_root), stations, d0, d1)
        print(f"\n[scan] files={scan['files']}  "
              f"bytes={scan['bytes']}  months={list(scan['by_ym'])}  "
              f"est={scan['est_sec']:.1f}s")

        # ---- 2) 批量修复 ----
        print("\n[batch] running...")
        summary = batch_repair(
            raw_root=str(raw_root),
            fix_root=str(fix_root),
            stations=stations,
            d0=d0, d1=d1,
            # catalog_dir 缺省 = <fix_root>/catalogs
        )

        # ---- 3) 汇总 ----
        print("\n--- summary ---")
        print(f"files            : {summary['files']}")
        print(f"repaired units   : {summary['repaired']}")
        print(f"long gaps kept   : {summary['gaps']}")
        print(f"errors           : {len(summary['errors'])}")
        print(f"elapsed          : {summary['elapsed']:.1f} s")
        v = summary.get("verify") or {}
        print(f"verify ok        : {v.get('ok')}  "
              f"(checked={v.get('n_files')} fail={v.get('n_fail')} "
              f"skip={v.get('n_skip')})")

        print("\n[outputs]")
        for p in sorted(fix_root.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(fix_root)}")
        print(f"\n(temp dir {tmp} will be removed on exit; comment out the "
              f"shutil.rmtree call below to keep the results)")
    finally:
        # 想保留结果以便查看就把下面这行注释掉
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
