# -*- coding: utf-8 -*-
"""Backward-compatibility alias for :mod:`dssrr.verifier`.

The DSSRR code base used to carry **two** copies of the quality verifier: the
public one in ``dssrr/verifier.py`` and a private one here.  They were verified
to be byte-identical, so the private copy was deleted and this module now
re-exports the single canonical implementation.

There is now exactly one place to change the acceptance metrics:
:mod:`dssrr.verifier`.

See also
--------
:class:`dssrr.verifier.QualityVerifier`
    The real implementation (spectral / envelope / DC / autocorrelation checks).
"""
from __future__ import annotations

from ...verifier import QualityVerifier

__all__ = ["QualityVerifier"]
