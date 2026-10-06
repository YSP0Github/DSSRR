# -*- coding: utf-8 -*-
"""Guard the "single source of truth" invariant for the core algorithm.

DSSRR historically carried two copies of the reference-spectrum repair stack:
the public one (``dssrr/core.py``, ``spectral.py``, ``synthesizer.py``,
``verifier.py``, ``config.py``) and a private fork under
``dssrr/repair_lib/anomaly_repair/``.

The fork had drifted — most visibly, it still defaulted the safety gap to
0.5 s while the public code used 1.0 s — which meant the GUI and the library
could silently disagree.  The fork has been removed and replaced with thin
re-export modules.

These tests fail loudly if anyone ever re-introduces a second implementation:
they assert that the two import paths hand back *the very same object*, not
merely two classes that happen to look alike.
"""
import numpy as np

from dssrr import config as public_config
from dssrr import core as public_core
from dssrr import spectral as public_spectral
from dssrr import synthesizer as public_synthesizer
from dssrr import verifier as public_verifier

from dssrr.repair_lib.anomaly_repair import config as forked_config
from dssrr.repair_lib.anomaly_repair import reference_spectrum as forked_core
from dssrr.repair_lib.anomaly_repair import spectral as forked_spectral
from dssrr.repair_lib.anomaly_repair import synthesizer as forked_synthesizer
from dssrr.repair_lib.anomaly_repair import verifier as forked_verifier


def test_replacer_is_the_same_object():
    assert forked_core.ReferenceSpectrumReplacer is public_core.ReferenceSpectrumReplacer


def test_spectral_estimator_is_the_same_object():
    assert forked_spectral.SpectralEstimator is public_spectral.SpectralEstimator


def test_synthesizer_is_the_same_object():
    assert forked_synthesizer.SignalSynthesizer is public_synthesizer.SignalSynthesizer


def test_verifier_is_the_same_object():
    assert forked_verifier.QualityVerifier is public_verifier.QualityVerifier


def test_get_config_is_the_same_object():
    assert forked_config.get_config is public_config.get_config
    assert forked_config.DEFAULT_CONFIG is public_config.DEFAULT_CONFIG


def test_helper_functions_are_reexported():
    assert forked_core.auto_reference_length is public_core.auto_reference_length
    assert (
        forked_core.replace_by_reference_spectrum
        is public_core.replace_by_reference_spectrum
    )


def test_safety_gap_defaults_agree_across_import_paths():
    """The legacy 0.5 s fork must not come back through a stale default."""
    import inspect

    public_default = inspect.signature(
        public_core.ReferenceSpectrumReplacer.replace
    ).parameters["reference_gap_sec"].default
    assert public_default == 1.0

    from dssrr.repair_lib.anomaly_repair import (
        LayeredReferenceSpectrumReplacer,
    )

    layered_default = inspect.signature(
        LayeredReferenceSpectrumReplacer.replace
    ).parameters["reference_gap_sec"].default
    assert layered_default == public_default


def test_synthesizer_accepts_legacy_none_gap():
    """``gap_samples=None`` was the old private signature; it must still work."""
    from dssrr.config import get_config
    from dssrr.synthesizer import SignalSynthesizer

    rng = np.random.default_rng(0)
    ref = 500.0 + rng.normal(0, 1, 4000)
    synth = SignalSynthesizer(6.625, get_config({"synthesis": {"random_seed": 0}}))
    psd = np.ones(2049)
    feats = {"target_mean": float(ref.mean()), "target_std": float(ref.std())}

    a = synth.synthesize(psd, feats, ref, ref, 500, gap_samples=None)
    b = synth.synthesize(psd, feats, ref, ref, 500, gap_samples=0)
    assert a.shape == (500,)
    assert np.array_equal(a, b)
