"""Default DSSRR configuration — the single source of truth for all tunables.

Every time parameter is in **seconds**, every frequency parameter is in **Hz**.

How it is used
--------------
:func:`get_config` returns a deep copy of :data:`DEFAULT_CONFIG` with optional
per-section overrides applied.  Nothing in the package mutates the module-level
dictionary, so callers can freely customise their own copy.

    from dssrr.config import get_config
    cfg = get_config({"synthesis": {"reference_block_sec": 30.0}})

Sections
--------
``detect``
    Anomaly-detection thresholds.  Only the batch/GUI pipelines read this
    section (``dssrr.repair_lib``); the core algorithm never looks at it.
``reference``
    Reference-window geometry used by the pipelines.
``spectral``
    Power-spectral-density estimation (:mod:`dssrr.spectral`).
``synthesis``
    Replacement-waveform synthesis (:mod:`dssrr.synthesizer`).
``layered_synthesis``
    Band-prioritised synthesis used by
    :class:`dssrr.repair_lib.anomaly_repair.layered_reference_spectrum.LayeredReferenceSpectrumReplacer`.
``joining``
    Boundary treatment and baseline alignment (:mod:`dssrr.core`).
``verification``
    Acceptance thresholds for :class:`dssrr.verifier.QualityVerifier`.

The extra sections (``detect`` / ``reference`` / ``layered_synthesis``) are
deliberately kept in this one dictionary even though the lightweight core does
not read them: it means the GUI, the batch pipeline and the library all share a
single configuration object instead of maintaining parallel copies.
"""
from __future__ import annotations

import copy

DEFAULT_CONFIG = {
    # ---------------------------------------------------------------- detection
    # Read by dssrr.repair_lib (E7 batch pipeline / GUI).  Not used by core.
    "detect": {
        "sta_len_sec": 0.5,
        "lta_len_sec": 10.0,
        "sta_lta_threshold": 5.0,
        "z_score_threshold": 4.0,
        "diff_threshold_factor": 5.0,
        "expand_sec": 0.5,
        "min_gap_sec": 2.0,
    },

    # ---------------------------------------------------------------- reference
    "reference": {
        "length_factor": 1.5,
        "min_sec": 120.0,
        "max_sec": 600.0,
        # Legacy default safety margin.  Kept for backward compatibility with
        # saved configurations; the modern call path always passes an explicit
        # ``reference_gap_sec`` to ``replace()``.
        "safety_margin_sec": 0.3,
        "stationarity_cv_thresh": 0.3,
    },

    # ------------------------------------------------------- spectral estimation
    "spectral": {
        # The DSSRR estimator is Welch's averaged periodogram (Hann window,
        # 50% overlap; see dssrr/spectral.py).  "method" is kept for backward
        # compatibility with saved configs; no other value is implemented.
        # There is deliberately NO taper / time-bandwidth key: DSSRR uses no
        # DPSS (Slepian) tapers, so no such parameter exists or is read.
        "method": "welch",
        "smooth_bandwidth_hz": 0.005,
        "n_fft": None,
        # 0.0 = use the whole selected reference interval.  Set a positive
        # value only to crop an explicit near-edge context window, which helps
        # when the reference is non-stationary.
        "local_context_sec": 0.0,
    },

    # ------------------------------------------------------------ synthesis
    "synthesis": {
        # Long replacements are assembled from healthy reference blocks so
        # their spectrum and time-domain texture stay physically plausible.
        "method": "reference_overlap_add",
        "reference_block_sec": 60.0,
        "reference_overlap_sec": 2.0,
        "max_iter": 200,
        "tolerance": 1e-6,
        "envelope_perturbation": 0.05,
        # ``random_seed`` is injected per call by the replacer; ``None`` means
        # "draw fresh entropy" and therefore gives a non-reproducible result.
    },

    # -------------------------------------------------- layered (band-priority)
    # Read by LayeredReferenceSpectrumReplacer only.
    "layered_synthesis": {
        # Ultra-low-frequency lunar free-oscillation work usually cares more
        # about millihertz-to-long-period continuity than broadband texture.
        "low_band_max_hz": 0.08,
        "high_band_min_hz": 0.25,
        "custom_band_min_hz": 0.001,
        "custom_band_max_hz": 0.08,
        "transition_hz": 0.01,
        "low_band_target_weight": 0.92,
        "low_band_background_weight": 0.18,
        "high_band_target_weight": 0.88,
        "high_band_background_weight": 0.22,
        "custom_band_target_weight": 0.90,
        "custom_band_background_weight": 0.20,
        "low_residual_scale": 0.45,
        "high_background_scale": 0.75,
        "custom_residual_scale": 0.60,
        "low_blend_strength": 0.45,
        "high_blend_strength": 0.35,
        "custom_blend_strength": 0.40,
        "context_sec": 120.0,
    },

    # -------------------------------------------------------------- joining
    "joining": {
        "overlap_factor": 2.5,
        "overlap_min_samples": 64,
        "tukey_alpha": 0.3,
        "dc_removal": True,
        "phase_correction": True,
        # Automatically shift a stable post-event baseline when it is clearly
        # offset; otherwise bridge the replacement core only.  One of
        # "auto" | "bridge" | "post_shift" | "off".
        "baseline_alignment": "auto",
        "baseline_context_sec": 20.0,
        # Smooth only the replacement edges *inside* the selected core.  The
        # samples outside the user selection remain untouched, and original
        # anomaly samples are never mixed back into the repair.
        "boundary_blend_fraction": 0.08,
        "boundary_blend_min_samples": 8,
        "boundary_blend_max_sec": 20.0,
        "boundary_context_sec": 20.0,
    },

    # ----------------------------------------------------------- verification
    "verification": {
        "spectral_similarity_thresh": 0.95,
        "envelope_jump_thresh": 0.15,
        "dc_jump_thresh": 0.05,
        "autocorr_diff_thresh": 0.20,
        "energy_ratio_range": [0.7, 1.3],
    },
}


def get_config(overrides: dict | None = None) -> dict:
    """Return a deep copy of :data:`DEFAULT_CONFIG` with ``overrides`` applied.

    A value that is itself a ``dict`` and matches an existing top-level section
    is merged *into* that section (shallow update), so callers can override a
    single key without restating the whole section:

    >>> get_config({"synthesis": {"reference_block_sec": 30.0}})
    ...["synthesis"]["method"]
    'reference_overlap_add'

    A key that does not exist yet is simply added, and a non-dict override
    replaces the existing entry outright.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in cfg:
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg
