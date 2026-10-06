# -*- coding: utf-8 -*-
"""Backward-compatibility alias for :mod:`dssrr.core`.

The DSSRR code base used to carry **two** copies of
:class:`~dssrr.core.ReferenceSpectrumReplacer`: the public one in
``dssrr/core.py`` and a private one here.  The private copy was a stale fork
that had fallen behind the public one, and — more dangerously — used a different
default for the safety gap (``reference_gap_sec=0.5`` instead of ``1.0``).

Both copies were compared over a battery of synthetic traces and produced
bit-identical output once the safety gap was pinned, so the private copy was
deleted.  This module now re-exports the single canonical implementation.

.. warning::
   The default ``reference_gap_sec`` is now **1.0 s** (the published library
   value), where this legacy import path previously defaulted to 0.5 s.  Every
   call site inside DSSRR passes the gap explicitly, so nothing in the package
   changes behaviour — but if you imported the replacer from this module and
   relied on the 0.5 s default, pass ``reference_gap_sec=0.5`` explicitly.

See also
--------
:class:`dssrr.core.ReferenceSpectrumReplacer`
    The real implementation.
:func:`dssrr.core.replace_by_reference_spectrum`
    One-shot convenience wrapper.
:func:`dssrr.core.auto_reference_length`
    Reference-window length heuristic.
"""
from __future__ import annotations

from ...core import (
    ReferenceSpectrumReplacer,
    auto_reference_length,
    replace_by_reference_spectrum,
)

__all__ = [
    "ReferenceSpectrumReplacer",
    "replace_by_reference_spectrum",
    "auto_reference_length",
]
