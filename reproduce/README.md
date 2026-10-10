# Reproduction package — DSSRR

Analysis scripts and instance-level result tables behind every figure and table in

> **Dual-Sided Spectral Reference Reconstruction (DSSRR) for Long-Duration Anomaly
> Repair in Apollo Lunar Seismic Records**
> S. Yu and X. Li — submitted to *Seismological Research Letters*.

This directory is the archive referred to by the paper's **Data and Resources**
statement. It lives in the same public, versioned repository as the released
`dssrr` package.

---

## 1. What is here

```
reproduce/
├── README.md                 <- this file
├── experiments/              <- experiment drivers + statistics + verification
├── image_code/               <- figure-generation scripts (all Figs 1-10, S1)
└── data/                     <- instance-level tables and the archived waveforms
    ├── e_realinj/            <- Table 1 / Fig 3 / Fig 6  (E-RealInj)
    │   ├── stats_summary.csv        432 rows = 72 units x 6 methods
    │   ├── PROVENANCE.json          seeds, git commit, module checksums
    │   ├── example_case/            waveforms for Figs 4 and 10
    │   │   ├── medium/              corrupted + 6 repaired + intervals.json
    │   │   └── large/
    │   └── ground_truth/            clean 12-h reference record (S12, 1976-01-13)
    ├── length_scan/          <- Fig 5 / Tables S2-S3 (gap-length sweep)
    ├── earth_transfer/       <- Fig 8 (IU.ANMO, Earth broadband)
    ├── in_domain/            <- Table S1 (labelled synthetic benchmark)
    ├── ablation/             <- Fig 7 (component ablation)
    ├── archival/             <- Fig 9 / Table 2 (Apollo archival cases)
    │   ├── M4_*.csv                 window-level consistency metrics
    │   └── pairs/{before,repaired}/ the 5 repaired observational pairs
    └── stats/                <- paired tests / dispersion tables (M2)
```

---

## 2. Figure / table → script → data

| Paper item | Script | Input | Notes |
|---|---|---|---|
| Fig. 1 | `image_code/fig01_anomaly_types.py` | Apollo raw miniSEED | anomaly morphology panel |
| Fig. 2 | `image_code/fig02a_workflow_panel.py`, `fig02bc_repair_panel.py`, `fig02_method_overview.py` | — | schematic |
| **Table 1**, Fig. 3 | `experiments/E2r_v6_batch.py` → `E2r_v5_batch.py` | Apollo raw miniSEED | writes `data/e_realinj/stats_summary.csv` |
| Fig. 4 | `image_code/fig04_psd_comparison.py` | `data/e_realinj/example_case/medium/`, `ground_truth/` | |
| **Fig. 5** | `experiments/run_length_scan.py` → `image_code/fig05_length_effect.py` | `data/length_scan/stats_summary.csv` | 19 lengths, 4 methods |
| Fig. 6 | `experiments/_v6_stalta_downstream.py` → `image_code/fig06_stalta_downstream.py` | **full** E-RealInj case tree (36 cases) | see §5 |
| Fig. 7 | `experiments/ablation_study_v3.py` → `image_code/fig07ab_ablation_panel.py`, `fig07cd_fusion_panel.py` | `data/ablation/ablation_v3_results.csv` | |
| Fig. 8 | `image_code/fig08_earth_transfer.py` | `data/earth_transfer/earth_instances.csv`; IU.ANMO BHZ raw | |
| Fig. 9 | `image_code/fig09_archival_repair.py` | `data/archival/pairs/` | 3 cases x before/after |
| Fig. 10 | `image_code/fig10_failure_mode.py` | `data/e_realinj/example_case/large/`, `ground_truth/` | |
| **Table 2** | `experiments/M4_archival_metrics.py` | `data/archival/pairs/` | writes `M4_archival_metrics.csv` |
| **Table S1** | `experiments/E9_dl_synthetic.py` | synthetic (seeded) | writes `data/in_domain/*.csv` |
| **Tables S2-S3** | `experiments/_regen_supp_tables.py` | `data/stats/dispersion_length_scan.csv` | regenerates the LaTeX tables |
| Paired tests (Fig. 3/5/8 error bars) | `experiments/M2_paired_stats.py` | `data/*/stats_summary.csv` | writes `data/stats/paired_*.csv` |
| Fig. S1 | `image_code/figS1_background.py` | Apollo raw miniSEED | |
| Fig. S2 | `dssrr` package GUI | — | screenshot of the released tools |

Supporting modules imported by the drivers: `experiments/protocol.py`,
`experiments/results_io.py`, `experiments/_metrics_lib.py`,
`experiments/baselines.py` (the seven comparison methods), `experiments/unet_model.py`
(1-D U-Net definition).

Verification / audit scripts are included for provenance:
`_audit_baselines.py`, `_audit_weakness.py`, `_audit_ssa.py`,
`_audit_ssa_unify_fig05b.py`, `_audit_dssrr_quantize.py`,
`_verify_fig05_numbers.py`, `_verify_supp_tables.py`.

---

## 3. How the scripts were run

The scripts are archived **exactly as executed**, i.e. unmodified. They were run
from inside the authors' working tree with the repository root on `sys.path`:

```bash
cd <repo-root>/docs/dssrr_paper/experiments
MPLBACKEND=Agg python run_length_scan.py       # e.g. Fig. 5 data
python M2_paired_stats.py                      # paired statistics
cd ../submission_srl/image_code
MPLBACKEND=Agg python fig05_length_effect.py   # Fig. 5
```

Environment used: Python 3.11, `numpy` / `scipy` / `pandas` / `obspy` /
`matplotlib`, plus PyTorch (CPU) for the 1-D U-Net baseline.

---

## 4. Dependencies — please read

The scripts were written for the authors' own tree, and this is reflected in the
archive. **Two things must be adapted before they will run elsewhere:**

1. **Absolute paths.** Most scripts contain hard-coded paths such as
   `G:\SeisY\docs\dssrr_paper\experiments\...`. Edit these (or override the
   environment variables `E2R_CASE`, `FAILCASE`, `LENGTH_SUMMARY` where offered)
   to point at your checkout.
2. **The `seisy` package.** The drivers import the comparison methods and the
   reference-spectrum replacer from `seisy.core.anomaly_repair.*`, which belongs
   to the **SeisY** application (the software named in Data and Resources),
   not to this repository:

   ```python
   from seisy.core.anomaly_repair.comparison_methods import (
       LinearInterpolation, CubicSplineInterpolation, SSARepair, UNetRepair)
   from seisy.core.anomaly_repair.reference_spectrum import ReferenceSpectrumReplacer
   ```

   The **canonical, packaged implementation of the DSSRR algorithm itself** is
   `dssrr/core.py` in this repository (and on PyPI as `dssrr`); the module paths
   differ, so the drivers are not drop-in runnable against the PyPI package
   without a shim. The full baseline suite (`comparison_methods.py`) is not part
   of the released package.

We deliberately archived the scripts verbatim rather than rewriting their paths
and imports, so that what is published is exactly what produced the numbers in
the paper.

---

## 5. What is *not* bundled

* **Full E-RealInj case tree** (72 units x corrupted + 6 repaired miniSEED,
  ~820 MB). Only one example case (S12, 1976-01-13, medium and large dropout) is
  included, which is what Figs 4 and 10 need. Fig. 6 walks all 36 dropout cases
  and therefore needs the full tree, regenerable with `E2r_v6_batch.py`.
* **Raw Apollo records.** Available from the IRIS Data Services and the NASA
  Planetary Data System Geosciences Node (network `XA`, station `MHZ`).
* **The Earth broadband record** (IU.ANMO BHZ) — available from the IRIS FDSN
  web service.
* The 1-D U-Net **trained weights** (regenerable with `unet_model.py` / the
  training entry point).

---

## 6. Provenance and integrity

Each results directory carries a `PROVENANCE.json` recording the run time, the
Python version, the git commit of the source tree, and an MD5 of every
`seisy/core/anomaly_repair/*.py` module involved. Those checksums are how the
paper's numbers were traced back to the exact code that produced them — for
example, the `reference_spectrum.py` checksum distinguishes runs made before and
after the `quantize` default changed.

Metric definitions (Wasserstein distance, ACF $L_2$, energy ratio, PSD cosine,
envelope distance, spectral-entropy error) are implemented in
`experiments/_metrics_lib.py` and described in the paper's *Data and Methods*.

`MANIFEST.md5` lists an MD5 for every file in this directory (except itself).
It was computed on the **repository** content, where the repository's
`.gitattributes` normalises text files to LF; `.mseed` files are stored
byte-exact. Verify a fresh clone with:

```bash
cd reproduce && md5sum -c MANIFEST.md5
```
