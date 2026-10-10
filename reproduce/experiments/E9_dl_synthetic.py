# -*- coding: utf-8 -*-
"""
E9 (Task 7): Deep-learning baseline on in-domain synthetic data.

The self-supervised 1-D U-Net (models/unet_synth.pt) is trained on the E1
synthetic signal families, so synthetic test instances are its *in-domain*
regime. We compare the same six methods as E-RealInj on fresh synthetic
instances (4 signal families x 6 anomaly types x 6 seeds), reporting both
pointwise RMSE and the statistical-fidelity suite.

This isolates the deep-learning deployment boundary: the U-Net is strong
in-domain but its training distribution does not exist for unlabeled real
planetary archives (cross-domain result in Table 1), whereas DSSRR is
training-free and domain-independent.

Outputs (results_dl_synth/):
  dl_synth_instances.csv, dl_synth_summary.csv
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, r"G:\SeisY")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seisy.core.anomaly_repair.comparison_methods import (  # noqa: E402
    LinearInterpolation,
    CubicSplineInterpolation,
    SSARepair,
    UNetRepair,
)
from E8_earth_transfer import FFTInterpolation, repair_dssrr  # noqa: E402
from E1_synthetic import FAMILIES, ANOMALIES  # noqa: E402
from _metrics_lib import all_metrics  # noqa: E402

OUT_DIR = r"G:\SeisY\docs\dssrr_paper\experiments\results_dl_synth"
FS = 6.625
N_REPS = 6
SEED_OFFSET = 100
METHOD_ORDER = ["Linear", "CubicSpline", "FFT", "SSA", "UNet", "DSSRR"]
MIN_STAT_LEN = 16


def main():
    import argparse, gc
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-family", default=None)
    ap.add_argument("--only-anomaly", default=None)
    ap.add_argument("--combine", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    inst_csv = os.path.join(OUT_DIR, "dl_synth_instances.csv")

    # ---- combine-only mode: build summary tables from instances ----
    if args.combine:
        df = pd.read_csv(inst_csv)
        metric_cols = ["rmse_anom", "wasserstein", "acf_l2", "envelope",
                       "spec_ent", "psd_cos"]
        summary = df.groupby("method")[metric_cols].mean().reindex(
            METHOD_ORDER).reset_index()
        summary.to_csv(os.path.join(OUT_DIR, "dl_synth_summary.csv"),
                       index=False, encoding="utf-8-sig")
        by_anom = df.groupby(["anomaly", "method"])[metric_cols].mean().reset_index()
        by_anom.to_csv(os.path.join(OUT_DIR, "dl_synth_by_anomaly.csv"),
                       index=False, encoding="utf-8-sig")
        print("===== in-domain synthetic overall means =====")
        print(summary.round(4).to_string(index=False))
        print("\n===== Wasserstein by anomaly x method =====")
        print(by_anom.pivot(index="anomaly", columns="method",
                            values="wasserstein").round(3))
        return df, summary

    methods = {
        "Linear": LinearInterpolation(),
        "CubicSpline": CubicSplineInterpolation(),
        "FFT": FFTInterpolation(FS),
        "SSA": SSARepair(max_iter=30),
        "UNet": UNetRepair(
            patch_length=1536, n_epochs=10,
            model_path=r"G:\SeisY\docs\dssrr_paper\experiments\models\unet_synth.pt",
        ),
        "DSSRR": None,
    }

    rows = []
    done = set()
    if os.path.isfile(inst_csv):
        old = pd.read_csv(inst_csv)
        rows = old.to_dict("records")
        for r in rows:
            done.add((r["family"], r["anomaly"], int(r["rep"]), r["method"]))

    fam_names = list(FAMILIES)
    anom_names = list(ANOMALIES)
    if args.only_family is not None:
        fam_names = [f for f in fam_names if f == args.only_family]
    if args.only_anomaly is not None:
        anom_names = [a for a in anom_names if a == args.only_anomaly]
    for fam in fam_names:
        for anom in anom_names:
            for rep in range(N_REPS):
                seed = int(zlib.crc32(
                    f"{fam}|{anom}|{SEED_OFFSET + rep}".encode()))
                truth = FAMILIES[fam](seed)
                corrupted, (s, e) = ANOMALIES[anom](truth, seed)
                truth_seg = truth[s:e + 1]
                gap_len = e - s + 1
                for mname in METHOD_ORDER:
                    if (fam, anom, rep, mname) in done:
                        continue
                    row = {"family": fam, "anomaly": anom, "rep": rep,
                           "gap_len": gap_len, "method": mname}
                    try:
                        if mname == "DSSRR":
                            repaired = repair_dssrr(corrupted, s, e, FS)
                        else:
                            repaired = methods[mname].repair(corrupted, s, e)
                        rep_seg = repaired[s:e + 1]
                        if gap_len < MIN_STAT_LEN:
                            from _metrics_lib import rmse
                            from scipy.stats import wasserstein_distance
                            row.update(rmse_anom=rmse(truth_seg, rep_seg),
                                       wasserstein=float(
                                           wasserstein_distance(truth_seg, rep_seg)),
                                       acf_l2=np.nan, envelope=np.nan,
                                       spec_ent=np.nan, psd_cos=np.nan)
                        else:
                            row.update(all_metrics(truth_seg, rep_seg, FS,
                                                   acf_max_lag=200))
                        del repaired
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        row.update(rmse_anom=np.nan, wasserstein=np.nan,
                                   acf_l2=np.nan, envelope=np.nan,
                                   spec_ent=np.nan, psd_cos=np.nan)
                    rows.append(row)
                    done.add((fam, anom, rep, mname))
                    gc.collect()
            # incremental save after each anomaly group
            pd.DataFrame(rows).to_csv(inst_csv, index=False, encoding="utf-8-sig")
            print(f"done {fam}/{anom}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(inst_csv, index=False, encoding="utf-8-sig")

    # Only build overall summary when running the full set (no group filter).
    if args.only_family is not None or args.only_anomaly is not None:
        print("single-group run complete", flush=True)
        return df, None

    metric_cols = ["rmse_anom", "wasserstein", "acf_l2", "envelope",
                   "spec_ent", "psd_cos"]
    summary = df.groupby("method")[metric_cols].mean().reindex(
        METHOD_ORDER).reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "dl_synth_summary.csv"), index=False,
                   encoding="utf-8-sig")

    by_anom = df.groupby(["anomaly", "method"])[metric_cols].mean().reset_index()
    by_anom.to_csv(os.path.join(OUT_DIR, "dl_synth_by_anomaly.csv"),
                    index=False, encoding="utf-8-sig")

    print("\n===== in-domain synthetic overall means =====")
    print(summary.round(4).to_string(index=False))
    print("\n===== Wasserstein by anomaly x method =====")
    print(by_anom.pivot(index="anomaly", columns="method",
                        values="wasserstein").round(3))
    return df, summary


if __name__ == "__main__":
    main()
