# Changelog

All notable changes to the `dssrr` package are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `reproduce/experiments/_verify_unet_gap_regime.py`, the check behind the
  supplement's *Note on the deployment regime* (training / in-domain /
  E-RealInj gap-length ranges and the flatness of the 1-D U-Net Wasserstein
  degradation across 300--3600 s), plus the corresponding README entry and a
  refreshed `MANIFEST.md5`. `reproduce/` is documentation-only and is not
  installed with the package, so this does not warrant a version bump.

## [0.1.1] — 2026-10-10

### Added
- **Reproduction package** (`reproduce/`): the analysis drivers, the figure
  scripts and the instance-level result tables behind every figure and table in
  the accompanying paper, together with the repaired observational pairs of the
  archival case studies and an `md5` manifest. This is the archive referenced by
  the paper's *Data and Resources* statement. It is **not** installed with the
  Python package.

### Changed
- **Safety gap is now configurable** in the GUI: a new `reference_gap_sec`
  setting (default 0.5 s, the archival-pipeline value) exposes the gap kept
  between an anomaly boundary and the reference windows. The controlled
  experiments in the paper use 1.0 s.
- GUI: layout and wording fixes in the repair-settings panel and the manual
  repair dialog; Chinese translations extended to cover the new setting.

### Fixed
- GUI: several controls no longer overflow on small screens, and panel resizing
  now propagates correctly to child widgets.

## [0.1.0] — 2026-10-06

Initial release.

### Added
- `dssrr.core.ReferenceSpectrumReplacer` — the training-free, two-sided spectral
  reference reconstruction pipeline (power-spectral-density and amplitude-statistics
  fusion, reference-block overlap-add synthesis, optional `quantize` back to
  integer counts).
- `dssrr.spectral`, `dssrr.synthesizer`, `dssrr.verifier` — the algorithm layers.
- `dssrr.config` — every threshold in a single `DEFAULT_CONFIG` dict.
- `dssrr.cli` / `dssrr-batch` — command-line batch repair.
- `dssrr.gui` / `dssrr-gui` — the PyQt5 desktop application (viewer, interactive
  manual repair, batch page), with English/Chinese translations.
- `dssrr.repair_lib` — the Apollo-specific detection and repair pipeline,
  including the merged moonquake catalogue used for event protection.
- `examples/` — runnable examples with three 12-hour Apollo 12/15/16 fixtures.
