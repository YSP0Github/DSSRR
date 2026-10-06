# DSSRR examples

Runnable, self-contained examples that walk from the smallest possible call to
the full batch pipeline. Every script is heavily commented and can be run
straight from a source checkout — no `pip install` required (each one adds the
repository root to `sys.path`).

| Script | What it shows | Needs |
|---|---|---|
| [`01_quickstart.py`](01_quickstart.py) | Minimal repair on synthetic data; the two hard invariants (gap filled, outside untouched); the full report dict | numpy |
| [`02_long_gap_baseline.py`](02_long_gap_baseline.py) | Why long gaps need DSSRR: side-by-side with linear interpolation and FFT extrapolation | numpy, scipy, matplotlib |
| [`03_real_apollo_mseed.py`](03_real_apollo_mseed.py) | Repair a real Apollo PSE record (bundled data), then score against the held-out truth | numpy, scipy, obspy, matplotlib |
| [`04_report_plot.py`](04_report_plot.py) | Produce a publication-style repair report figure | numpy, scipy, matplotlib |
| [`05_batch_cli.py`](05_batch_cli.py) | Drive the batch pipeline programmatically (`quick_scan` + `batch_repair`) | numpy, obspy, pandas |

## Running

```bash
# from the repository root
python examples/01_quickstart.py
python examples/02_long_gap_baseline.py
python examples/03_real_apollo_mseed.py                 # uses examples/data/
python examples/04_report_plot.py
python examples/05_batch_cli.py
```

`02`, `03` and `04` write PNG files next to the script (they are git-ignored).

### Using your own data

`03_real_apollo_mseed.py` accepts any directory of MiniSEED files:

```bash
python examples/03_real_apollo_mseed.py --data-dir /path/to/mseed --sr 6.625
```

See [`data/README.md`](data/README.md) for the provenance and licence of the
bundled Apollo records.

## Notes

- Examples `02`–`04` set the matplotlib backend to `Agg` so they also run on
  headless machines. The GUI and `report_plot` still use `Qt5Agg` when a Qt
  binding is available.
- `05_batch_cli.py` builds a throw-away synthetic archive in a temp directory,
  so it never touches your real data.
