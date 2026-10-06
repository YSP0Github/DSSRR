# -*- coding: utf-8 -*-
"""End-to-end tests for :class:`dssrr.repair_lib.repair_engine.StreamRepair`.

These exercise the exact code path the desktop GUI and the batch pipeline use,
so they catch regressions that the pure-library tests cannot see — in
particular the routing between the two-sided and the layered reference-spectrum
replacers, and the ``quantize_to_int`` toggle.

The tests need obspy to build a ``Stream``; they are skipped when it is absent.
"""
import numpy as np
import pytest

obspy = pytest.importorskip("obspy")

from obspy import Stream, Trace  # noqa: E402

from dssrr.repair_lib.repair_engine import StreamRepair  # noqa: E402

SR = 6.625


def make_stream(n=20000, seed=0, dtype=np.float64):
    """Coloured-noise trace with a flat dropout in the middle."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.97 * x[i - 1] + 0.03 * w[i]
    x = 500.0 + 4.0 * x

    start, end = 6000, 9000
    x[start:end + 1] = 0.0
    if np.issubdtype(dtype, np.integer):
        x = np.round(x)
    tr = Trace(data=x.astype(dtype))
    tr.stats.sampling_rate = SR
    tr.stats.network, tr.stats.station = "XA", "S12"
    tr.stats.channel = "MHZ"
    return Stream([tr]), start, end


def run_method(method_id, params=None, dtype=np.float64):
    st, start, end = make_stream(dtype=dtype)
    method_configs = [{"id": method_id, "params": params or {}}]
    segment_info = {
        "trace_index": 0,
        "time_range": (start / SR, end / SR),
    }
    out = StreamRepair._apply_depulse_segment(st, method_configs, segment_info)
    return st, out, start, end


@pytest.mark.parametrize(
    "method_id",
    ["reference_spectrum", "layered_reference_spectrum"],
)
def test_reference_methods_repair_and_preserve_outside(method_id):
    st, out, start, end = run_method(method_id, {"random_seed": 0})
    assert out is not None
    original = st[0].data
    repaired = out[0].data

    # The dropout is gone ...
    assert not np.allclose(repaired[start:end + 1], 0.0)
    # ... and nothing outside the selected core was touched.
    assert np.array_equal(repaired[:start], original[:start])
    assert np.array_equal(repaired[end + 1:], original[end + 1:])


@pytest.mark.parametrize(
    "method_id",
    ["reference_spectrum", "layered_reference_spectrum"],
)
def test_quantize_toggle_controls_output_dtype(method_id):
    """``quantize_to_int`` must actually decide integer vs float output.

    Regression guard: the rounding used to be applied unconditionally inside
    the replacer, which made the GUI checkbox a no-op.
    """
    # quantize ON -> integer-valued samples inside the core
    _, on, start, end = run_method(
        method_id, {"random_seed": 0, "quantize_to_int": True})
    core_on = on[0].data[start:end + 1]
    assert np.all(core_on == np.round(core_on))

    # quantize OFF -> real-valued samples inside the core
    _, off, start, end = run_method(
        method_id, {"random_seed": 0, "quantize_to_int": False})
    core_off = off[0].data[start:end + 1]
    frac_non_integer = float(np.mean(core_off != np.round(core_off)))
    assert frac_non_integer > 0.5, (
        "quantize_to_int=False must leave the repaired samples real-valued, "
        f"but {frac_non_integer:.1%} were integers"
    )


def test_integer_input_round_trips_to_integer():
    """An int32 archive trace must come back as integers when quantize is on."""
    _, out, start, end = run_method(
        "reference_spectrum", {"random_seed": 0, "quantize_to_int": True},
        dtype=np.int32)
    core = out[0].data[start:end + 1]
    assert np.all(core == np.round(core))


def test_quantize_defaults_to_on_for_archival_counts():
    """The GUI default is quantize=True, matching the paper protocol."""
    _, out, start, end = run_method("reference_spectrum", {"random_seed": 0})
    core = out[0].data[start:end + 1]
    assert np.all(core == np.round(core))
