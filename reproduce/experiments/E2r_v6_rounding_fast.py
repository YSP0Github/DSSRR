"""快速取整核对：只算 anomaly 段上的 Wasserstein（与 V5.compute_stats_metrics 同一实现）。

E2r_v6_rounding_check.py 会对每个单元调用完整 compute_metrics（含两次 welch + 两次
periodogram，286k 样本），864 次调用要跑十几分钟。而 wasserstein 指标只取 [s,e] 段
（dropout 2k--24k、spike 5--156 样本），因此可以秒级算完。

口径与 E2r_v6_rounding_check.py 完全一致：
  - 真值 = CLEAN_DIR 里对应干净记录
  - 对**所有方法**输出施加 np.round 后重算
  - 按 dropout / spike 分层求均值
"""
from __future__ import annotations

import os
import sys
import glob
import json

import numpy as np
import pandas as pd
from obspy import read
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import E2r_v6_rounding_check as RC  # noqa: E402  复用 _clean / ROOT / METHODS

OUT = os.path.join(HERE, "results_stats", "rounding_check_fast.csv")


def main():
    rows = []
    for case_dir in sorted(glob.glob(os.path.join(RC.ROOT, "*_stationary"))
                           + glob.glob(os.path.join(RC.ROOT, "*_transient"))):
        case = os.path.basename(case_dir)
        record = case.rsplit("_", 1)[0]
        if record not in RC._clean:
            continue
        truth = read(RC._clean[record])[0].data.astype(float)

        for anom in ("dropout", "spike"):
            for level_dir in sorted(glob.glob(os.path.join(case_dir, anom, "*"))):
                level = os.path.basename(level_dir)
                ij = os.path.join(level_dir, "intervals.json")
                if not os.path.isfile(ij):
                    continue
                with open(ij) as fh:
                    obj = json.load(fh)
                intervals = obj["intervals"] if isinstance(obj, dict) else obj
                intervals = [(int(a), int(b)) for a, b in intervals]

                idx = np.concatenate([np.arange(s, e + 1) for s, e in intervals])
                t_seg = truth[idx]
                for m in RC.METHODS:
                    p = os.path.join(level_dir, f"repaired_{m}.mseed")
                    if not os.path.isfile(p):
                        continue
                    rep = read(p)[0].data.astype(float)
                    r_seg = rep[idx]
                    rows.append(dict(
                        case=case, record=record, anom_type=anom, level=level, method=m,
                        wasserstein_raw=wasserstein_distance(t_seg, r_seg),
                        wasserstein_round=wasserstein_distance(t_seg, np.round(r_seg)),
                    ))
        print(f"{case} done", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n单元数 = {df.groupby(['case', 'anom_type', 'level']).ngroups}, 行数 = {len(df)}")
    for anom in ("dropout", "spike"):
        sub = df[df["anom_type"] == anom]
        t = sub.groupby("method")[["wasserstein_raw", "wasserstein_round"]].mean()
        t = t.loc[[m for m in RC.METHODS if m in t.index]].round(3)
        print(f"\n=== {anom}（{sub.groupby(['case', 'level']).ngroups} 个单元）===")
        print(t.to_string())
        print("  raw 最优:", t["wasserstein_raw"].idxmin(),
              " round 最优:", t["wasserstein_round"].idxmin())
    print(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
