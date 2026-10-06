# -*- coding: utf-8 -*-
"""Backward-compatibility alias for :mod:`dssrr.spectral`.

The DSSRR code base used to carry **two** copies of the spectral estimator: the
public one in ``dssrr/spectral.py`` and a private one here.  They were verified
to be byte-identical, so the private copy was deleted and this module now
re-exports the single canonical implementation.

There is now exactly one place to change the PSD estimator:
:mod:`dssrr.spectral`.

See also
--------
:class:`dssrr.spectral.SpectralEstimator`
    The real implementation (Welch averaged periodogram with two-sided
    log-domain fusion).
"""
from __future__ import annotations

from ...spectral import SpectralEstimator

__all__ = ["SpectralEstimator"]
