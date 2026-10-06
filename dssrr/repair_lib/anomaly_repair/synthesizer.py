# -*- coding: utf-8 -*-
"""Backward-compatibility alias for :mod:`dssrr.synthesizer`.

The DSSRR code base used to carry **two** copies of the waveform synthesizer:
the public one in ``dssrr/synthesizer.py`` and a private one here.  The private
copy had drifted: it kept a legacy ``gap_samples=None`` fallback that read the
obsolete ``reference.safety_margin_sec`` key, while the public copy had already
dropped it.

The two were compared token-by-token and, apart from that fallback and the
default value of ``gap_samples`` (``None`` vs ``0``), the code was identical.
The public implementation therefore won, and the legacy ``None`` spelling is
still accepted (it now simply means "no safety gap") so old call sites keep
working.

There is now exactly one place to change the synthesis code:
:mod:`dssrr.synthesizer`.

See also
--------
:class:`dssrr.synthesizer.SignalSynthesizer`
    The real implementation (spectral shaping, overlap-add block assembly,
    phase-constrained boundaries, trend correction).
"""
from __future__ import annotations

from ...synthesizer import SignalSynthesizer

__all__ = ["SignalSynthesizer"]
