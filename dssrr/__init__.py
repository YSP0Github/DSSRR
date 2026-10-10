"""DSSRR — Dual-Sided Spectral Reference Reconstruction.

Training-free, statistically faithful long-gap repair for planetary and
seismic time series.  DSSRR fills a corrupted or missing interval by fusing
power-spectral-density estimates taken from the healthy data on **both** sides
of the gap, synthesising a replacement with realistic waveform texture, and
blending it into the surrounding trace.  No training data and no model fitting
are required, and the method stays stable for gaps from seconds to over
90 minutes.

Quick start
-----------
>>> import numpy as np
>>> from dssrr import DSSRR
>>> sr = 6.625                       # Apollo long-period sampling rate
>>> model = DSSRR(sr=sr, reference_sec=300, seed=42)
>>> repaired = model.repair(data, start=anomaly_start, end=anomaly_end)

Package layout
--------------
The package is layered so that the scientific core stays lightweight:

``dssrr.core`` / ``spectral`` / ``synthesizer`` / ``verifier`` / ``config``
    The algorithm itself.  Depends on **numpy + scipy only** — importing
    :mod:`dssrr` never pulls in a GUI toolkit.
``dssrr.cli``
    Cross-platform command-line batch repair (``dssrr-batch``).
``dssrr.gui``
    PyQt5 desktop application (``dssrr-gui``): trace viewer, manual repair
    panel and batch runner.
``dssrr.repair_lib``
    Apollo-specific detection/repair pipeline (E7 detector, multi-level repair
    routing, report plotting) used by the GUI and the batch command.
``dssrr.paths``
    Canonical locations of the optional user settings / catalogue files.
``dssrr.i18n``
    Single source of truth for the language of user-facing report text
    (default **English**; override with the ``DSSRR_LANG`` environment
    variable).  The system locale is deliberately not consulted.
``dssrr.gui.i18n`` + ``dssrr.gui.translations_zh``
    Bilingual UI support.  A ``QTranslator`` subclass backed by a plain
    ``{english: chinese}`` dict makes every existing ``self.tr()`` call site
    translatable without a ``.qm`` toolchain.  Still Qt-only — the core
    library never imports it.

Citation
--------
    Yu, S., Li, X. (2026). Dual-Sided Spectral Reference Reconstruction for
    long-duration artifact repair in Apollo lunar seismic records.
    *Seismological Research Letters* (submitted).
    Software (Zenodo DOI): https://doi.org/10.5281/zenodo.23192456

License: MIT (see ``LICENSE``).
"""
from __future__ import annotations
import numpy as np
from .core import ReferenceSpectrumReplacer
from .config import get_config

__version__ = "0.1.1"

def dssrr_gui():
    """启动 DSSRR 桌面 GUI（查看器 / 手动重处理 / 批量去异常）。

    惰性导入 dssrr.gui，避免核心库 import 时强制依赖 PyQt5。
    """
    from .gui import launch_gui
    launch_gui()


def dssrr_batch(argv=None):
    """命令行批量去异常（Linux / macOS / Windows PowerShell 通用）。

    惰性导入 dssrr.cli，命令行模式无需安装 PyQt5。
    """
    from .cli import main
    return main(argv)


__all__ = ["DSSRR", "ReferenceSpectrumReplacer", "auto_reference_length",
           "dssrr_gui", "dssrr_batch", "__version__"]


class DSSRR:
    """High-level convenience wrapper around :class:`ReferenceSpectrumReplacer`.

    Use this when you just want to repair a trace.  For full control over the
    configuration dictionary (spectral estimation, synthesis, boundary joining
    and verification thresholds), use :class:`ReferenceSpectrumReplacer`
    directly.

    Parameters
    ----------
    sr : float
        Sampling rate in Hz.  Must be positive.  Apollo long-period data is
        6.625 Hz.
    reference_sec : float, optional
        Default reference window length **per side**, in seconds (default
        300.0).  For long gaps prefer :func:`auto_reference_length`.
    seed : int, optional
        Random seed used for deterministic synthesis (default 42).  Pass
        ``None`` at call time for fresh randomness — note that the result is
        then not reproducible.
    quantize : bool, optional
        If ``True``, round the repaired samples to integers using
        round-half-to-even.  Enable this only for traces stored as raw integer
        counts (e.g. the Apollo archive); leave it ``False`` for physical-unit
        data such as velocity or displacement.  Can be overridden per call.
    config : dict, optional
        Configuration overrides merged onto :data:`dssrr.config.DEFAULT_CONFIG`.

    Examples
    --------
    >>> import numpy as np
    >>> from dssrr import DSSRR, auto_reference_length
    >>> sr = 6.625
    >>> data = 500.0 + np.random.default_rng(0).normal(0, 3, 20000)
    >>> data[6000:9001] = 0.0                     # a 7.5-minute dropout
    >>> ref = auto_reference_length(3001, sr)     # 226.5 s per side
    >>> model = DSSRR(sr=sr, reference_sec=ref, seed=42)
    >>> repaired = model.repair(data, 6000, 9000)
    >>> repaired.shape == data.shape
    True

    Notes
    -----
    DSSRR reproduces the **second-order statistics** of the surrounding trace
    (power spectrum, autocorrelation, energy).  The original phase inside the
    gap is not recoverable, so a one-off transient that happened during the
    dropout cannot be reconstructed.
    """

    def __init__(self, sr: float, reference_sec: float = 300.0,
                 seed: int = 42, quantize: bool = False,
                 config: dict | None = None):
        if sr <= 0:
            raise ValueError("Sampling rate must be positive")
        self.sr = float(sr)
        self.reference_sec = float(reference_sec)
        self.seed = seed
        self.quantize = bool(quantize)
        self._replacer = ReferenceSpectrumReplacer(sr=sr, config=config)

    def repair(self, data: np.ndarray, start: int, end: int,
               *, reference_before_sec: float | None = None,
               reference_after_sec: float | None = None,
               safety_gap_sec: float | None = None,
               seed: int | None = None,
               quantize: bool | None = None) -> np.ndarray:
        """Repair ``data[start:end + 1]`` in place-free fashion.

        Parameters
        ----------
        data : numpy.ndarray
            1-D waveform.  Never modified in place — a repaired copy is
            returned, and every sample outside ``[start, end]`` is copied
            through bit-identically.
        start, end : int
            Inclusive sample indices of the anomaly core.
        reference_before_sec, reference_after_sec : float, optional
            Reference window length per side.  Default to ``reference_sec``.
        safety_gap_sec : float, optional
            Guard interval between the anomaly core and each reference window,
            in seconds.  ``None`` (the default) defers to the replacer's own
            default so there is a single source of truth for this value.
        seed : int, optional
            Override the instance seed.  ``None`` means "use the instance
            seed"; to request fresh randomness pass ``seed=None`` to the
            constructor instead.
        quantize : bool, optional
            Override the instance ``quantize`` flag for this call only.

        Returns
        -------
        numpy.ndarray
            A repaired copy of ``data`` with the same shape and dtype layout
            (float; integer input becomes integer again when ``quantize``).
        """
        rb = self.reference_sec if reference_before_sec is None else reference_before_sec
        ra = self.reference_sec if reference_after_sec is None else reference_after_sec
        s = self.seed if seed is None else seed
        q = self.quantize if quantize is None else bool(quantize)
        kwargs = dict(
            reference_before_sec=rb,
            reference_after_sec=ra,
            random_seed=s,
            quantize=q,
        )
        if safety_gap_sec is not None:
            kwargs["reference_gap_sec"] = float(safety_gap_sec)
        repaired, _ = self._replacer.replace(
            np.asarray(data, dtype=float), int(start), int(end), **kwargs)
        return repaired

    def repair_with_report(self, data: np.ndarray, start: int, end: int,
                           **kwargs) -> tuple[np.ndarray, dict]:
        """Repair and also return the full provenance / quality report.

        Every :meth:`ReferenceSpectrumReplacer.replace` keyword is forwarded,
        so this is the way to reach ``reference_gap_sec``, ``quantize``,
        ``random_seed`` and friends while still using the convenience wrapper.

        Returns
        -------
        (numpy.ndarray, dict)
            The repaired trace and a report describing the reference windows
            used, baseline alignment, verification metrics and known
            limitations.
        """
        return self._replacer.replace(np.asarray(data, dtype=float),
                                      int(start), int(end), **kwargs)


def auto_reference_length(anomaly_samples: int, sr: float,
                          min_sec: float = 120.0, max_sec: float = 600.0) -> float:
    """Pick a reference-window length from the anomaly duration.

    Returns ``clip(anomaly_duration / 2, min_sec, max_sec)`` seconds — the
    protocol used for the DSSRR paper (120–600 s for Apollo data).  Half the
    anomaly length is enough for a stable spectral estimate, while the bounds
    keep the window short enough to stay locally stationary.

    Examples
    --------
    >>> auto_reference_length(int(60 * 6.625), 6.625)     # 60 s anomaly
    120.0
    >>> auto_reference_length(int(2000 * 6.625), 6.625)   # very long anomaly
    600.0
    """
    anom_sec = float(anomaly_samples) / float(sr)
    return float(np.clip(anom_sec / 2.0, min_sec, max_sec))

