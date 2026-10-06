"""DSSRR core: two-sided reference-spectrum replacement of one anomaly interval.

This is the heart of the method.  Given a 1-D waveform and an inclusive sample
range ``[start, end]`` that is known to be corrupted (flat dropout, glitch,
saturated section, ...), it builds a replacement for **that range only** and
returns a repaired copy of the whole trace.

Pipeline
--------
:meth:`ReferenceSpectrumReplacer.replace` runs these steps in order:

1. **Reference extraction** — :meth:`_extract_reference` cuts a healthy window
   from each side of the anomaly, separated from it by a safety gap
   (``reference_gap_sec``).  Windows are clipped at the trace edges and short
   windows are tolerated (one-sided repair is allowed).
2. **Local context selection** — :meth:`_select_local_reference_segments`
   optionally trims the references to the samples nearest the gap
   (``spectral.local_context_sec``), which keeps a distant trend or event from
   setting the amplitude of a short gap.
3. **Spectral estimation** — :class:`dssrr.spectral.SpectralEstimator` fuses the
   two reference power spectra into one target PSD plus a set of phase
   features.
4. **Synthesis** — :class:`dssrr.synthesizer.SignalSynthesizer` produces a
   replacement of exactly ``end - start + 1`` samples with that spectrum and
   physically plausible time-domain texture.
5. **Baseline alignment** — :meth:`_align_replacement_baseline` removes the
   synthesised DC location and bridges the (possibly different) levels of the
   two references, so no step appears at either join.
6. **Boundary blending** — :meth:`_smooth_replacement_boundaries` fades the
   first and last few samples of the replacement into the *real* neighbouring
   waveform, guaranteeing continuity without linear extrapolation.
7. **Write-back** — only ``repaired[start:end + 1]`` is touched.  Every sample
   outside the interval is copied through unchanged, and when ``quantize`` is
   enabled the repaired range alone is rounded back to integers.
8. **Verification** — :class:`dssrr.verifier.QualityVerifier` scores spectral
   similarity, envelope continuity, DC continuity and energy ratio.

Two invariants worth remembering
--------------------------------
* **Geometry must be shared.**  The safety gap in samples
  (``round(reference_gap_sec * sr)``) is threaded through both synthesis and
  baseline alignment.  Any step that builds a joint time axis across the hole
  must use the same value, otherwise the after-reference is treated as if it
  started at the anomaly edge and the right-hand target level is biased.
* **The gap is not recoverable.**  PSD matching preserves statistical frequency
  energy, not the original phase; a transient inside the interval is treated as
  contamination and removed.  The report returned alongside the data states
  this under ``"limitations"``.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import get_config
from .spectral import SpectralEstimator
from .synthesizer import SignalSynthesizer
from .verifier import QualityVerifier


class ReferenceSpectrumReplacer:
    """Replace one selected interval using two-sided reference spectra.

    Parameters
    ----------
    sr : float
        Sampling rate in Hz.  Must be positive.
    config : dict, optional
        Overrides merged onto :data:`dssrr.config.DEFAULT_CONFIG`; see
        :func:`dssrr.config.get_config`.  Leave it as ``None`` to use the
        defaults.

    Examples
    --------
    >>> import numpy as np
    >>> from dssrr.core import ReferenceSpectrumReplacer
    >>> rng = np.random.default_rng(0)
    >>> data = 500.0 + rng.normal(0, 2.0, 12000)
    >>> data[4000:6001] = 0.0
    >>> replacer = ReferenceSpectrumReplacer(sr=6.625)
    >>> repaired, report = replacer.replace(data, 4000, 6000, random_seed=0)
    >>> report["method"], report["length"]
    ('reference_spectrum', 2001)

    Notes
    -----
    The instance is stateless with respect to the data: the same replacer can
    be reused across many traces and intervals.  Only the optional ``config``
    is retained.
    """

    def __init__(self, sr: float, config: dict | None = None):
        if sr <= 0:
            raise ValueError("sr must be positive")
        self.sr = float(sr)
        self.config = get_config(config)
        self.estimator = SpectralEstimator(self.sr, self.config)
        self.synthesizer = SignalSynthesizer(self.sr, self.config)
        self.verifier = QualityVerifier(self.sr, self.config)

    def replace(
        self,
        data: np.ndarray,
        start: int,
        end: int,
        *,
        # Default reference windows follow the DSSRR paper protocol
        # (L = clip(anomaly/2, 120 s, 600 s)); callers may override.
        reference_before_sec: float = 120.0,
        reference_after_sec: float = 120.0,
        reference_gap_sec: float = 1.0,
        random_seed: int | None = 0,
        quantize: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Replace ``data[start:end + 1]`` from healthy two-sided references.

        Parameters
        ----------
        data : 1-D waveform. The input is never modified in place.
        start, end : anomaly core bounds, inclusive sample indices.
        reference_before_sec, reference_after_sec : maximum reference lengths.
        reference_gap_sec : safety gap between the anomaly and each reference.
        random_seed : deterministic phase seed. ``None`` uses fresh randomness.
        quantize : if True, round the repaired segment to integers
            (round-half-to-even). Use this only for raw integer counts
            (e.g. Apollo archival counts); leave it False for physical
            float traces (velocity / displacement units).
        """
        values = np.asarray(data, dtype=float)
        if values.ndim != 1:
            raise ValueError("data must be a 1-D array")
        n = len(values)
        start, end = int(start), int(end)
        if n == 0 or start < 0 or end >= n or start > end:
            raise ValueError("invalid anomaly interval")
        if reference_before_sec < 0 or reference_after_sec < 0:
            raise ValueError("reference lengths cannot be negative")
        if reference_gap_sec < 0:
            raise ValueError("reference_gap_sec cannot be negative")

        ref_before, before_bounds = self._extract_reference(
            values, start, end, reference_before_sec, reference_gap_sec, side="before"
        )
        ref_after, after_bounds = self._extract_reference(
            values, start, end, reference_after_sec, reference_gap_sec, side="after"
        )
        if ref_before is None and ref_after is None:
            raise ValueError(
                "at least one healthy reference segment is required for "
                "reference-spectrum replacement"
            )

        work_before, work_after = self._select_local_reference_segments(
            ref_before, ref_after
        )
        # Safety gap in samples. Every step that builds a *joint* time axis
        # across the hole (trend extrapolation, baseline alignment) must use
        # the same geometry, otherwise the after-reference is treated as if it
        # started right at the anomaly edge and the target right-hand level is
        # biased.
        gap_samples = int(round(reference_gap_sec * self.sr))

        synthesis_config = dict(self.config.get("synthesis", {}))
        synthesis_config["random_seed"] = random_seed
        estimator = self.estimator
        synthesizer = SignalSynthesizer(
            self.sr,
            {**self.config, "synthesis": synthesis_config},
        )
        anomaly_length = end - start + 1
        target_psd, phase_features = estimator.estimate(
            work_before, work_after, anomaly_length
        )
        replacement = synthesizer.synthesize(
            target_psd,
            phase_features,
            work_before,
            work_after,
            anomaly_length,
            gap_samples=gap_samples,
        )
        replacement, baseline_report = self._align_replacement_baseline(
            replacement, work_before, work_after, gap_samples=gap_samples
        )
        replacement = self._smooth_replacement_boundaries(
            values, replacement, start, end
        )

        repaired = values.copy()
        # 只修改异常段，不修改异常段外的数据（修复 suffix_offset 影响外部数据的 bug）
        repaired[start:end + 1] = replacement
        # 整数化：原始数据是整数计数，修复后的数据也应量化为整数
        # 用 round-half-even，与 raw int32 归档一致
        if quantize:
            repaired[start:end + 1] = np.round(repaired[start:end + 1])
        verification_after = work_after
        verification = self.verifier.verify(
            values, repaired, start, end, work_before, verification_after
        )

        report = {
            "method": "reference_spectrum",
            "start": start,
            "end": end,
            "length": anomaly_length,
            "sampling_rate": self.sr,
            "reference_before": {
                "start": before_bounds[0],
                "end": before_bounds[1],
                "length": len(ref_before) if ref_before is not None else 0,
            },
            "reference_after": {
                "start": after_bounds[0],
                "end": after_bounds[1],
                "length": len(ref_after) if ref_after is not None else 0,
            },
            "working_reference_before_length": len(work_before) if work_before is not None else 0,
            "working_reference_after_length": len(work_after) if work_after is not None else 0,
            "working_reference_context_sec": float(
                max(
                    len(work_before) if work_before is not None else 0,
                    len(work_after) if work_after is not None else 0,
                ) / self.sr
            ),
            "random_seed": random_seed,
            "target_mean": phase_features.get("target_mean"),
            "target_std": phase_features.get("target_std"),
            "replacement_mean": float(np.mean(replacement)),
            "replacement_std": float(np.std(replacement)),
            "baseline_alignment": baseline_report,
            "shifted_suffix_samples": 0,
            "boundary_blend_samples": self._boundary_blend_length(anomaly_length),
            "changed_samples": int(np.count_nonzero(
                np.abs(repaired[start:end + 1] - values[start:end + 1]) > 1e-12
            )),
            "verification": verification,
            "limitations": [
                "PSD matching preserves statistical frequency energy, not the original phase.",
                "A transient event inside the selected interval is treated as contamination and removed.",
                "Reference non-stationarity can bias the fused target spectrum.",
            ],
        }
        return repaired, report

    def _align_replacement_baseline(
        self,
        replacement: np.ndarray,
        ref_before: np.ndarray,
        ref_after: np.ndarray,
        gap_samples: int = 0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Bridge different healthy-reference baselines across the repair.

        A spectral synthesis estimates one target distribution from both
        references. If the references have different DC levels, that target
        sits between the two levels and creates a step at one or both joins.
        Remove only the synthesized DC location and add a smooth baseline
        bridge. The healthy samples outside the selected interval are never
        changed.
        """
        result = np.asarray(replacement, dtype=float).copy()
        joining = self.config.get("joining", {})
        mode = str(joining.get("baseline_alignment", "auto")).strip().lower()
        if mode in {"off", "none", "disabled", "false", "0"} or len(result) == 0:
            return result, {
                "mode": "off",
                "before": None,
                "after": None,
                "offset": None,
                "suffix_offset": 0.0,
            }

        context_sec = float(joining.get("baseline_context_sec", 60.0))
        context_samples = max(4, int(round(context_sec * self.sr)))
        before = (
            np.asarray(ref_before, dtype=float)[-context_samples:]
            if ref_before is not None else np.array([])
        )
        after = (
            np.asarray(ref_after, dtype=float)[:context_samples]
            if ref_after is not None else np.array([])
        )
        before = before[np.isfinite(before)]
        after = after[np.isfinite(after)]
        if len(before) < 4 or len(after) < 4:
            return result, {
                "mode": "bridge",
                "before": None,
                "after": None,
                "offset": None,
                "suffix_offset": 0.0,
            }

        # 用线性回归从整个参考段外推趋势，而不是只用紧邻窗口的中位数
        # 这样能更好地估计长时程漂移
        n_before_ref = len(ref_before) if ref_before is not None else 0
        n_after_ref = len(ref_after) if ref_after is not None else 0
        n_result = len(result)

        gap = int(gap_samples)
        if n_before_ref >= 10 and n_after_ref >= 10:
            # 合并两侧参考段，时间轴：左侧为负，右侧为正。
            # 异常段占 t=0..n_result-1，中间隔着安全间隙 gap，
            # 因此后参考段第一个样本位于 t=n_result+gap。
            # 时间轴必须计入该间隙，否则两侧参考段会被人为拉近 gap 个样本，
            # 拟合出的整体趋势斜率偏大、右端目标电平系统性偏高。
            t_before = np.arange(n_before_ref, dtype=float) - n_before_ref + 1
            t_after = np.arange(n_after_ref, dtype=float) + n_result + gap
            t_all = np.concatenate([t_before, t_after])
            v_all = np.concatenate([ref_before.astype(float), ref_after.astype(float)])
            # 线性回归
            slope_all, intercept_all = np.polyfit(t_all, v_all, 1)
            # 外推到异常段两端
            before_level = float(intercept_all + slope_all * 0)  # t=0
            after_level = float(intercept_all + slope_all * (n_result - 1))  # t=n-1
        else:
            before_level = float(np.median(before))
            after_level = float(np.median(after))
        replacement_center = float(np.median(result))
        offset = after_level - before_level
        before_scale = self._robust_scale(before)
        after_scale = self._robust_scale(after)
        if mode == "auto":
            stable_scale = max(before_scale, after_scale, 1e-12)
            scale_ratio = max(before_scale, after_scale) / max(
                min(before_scale, after_scale), 1e-12
            )
            # A short reference window can sit on different phases of a slow
            # oscillation. Its medians may differ by several robust scales even
            # though there is no DC step. Only shift the suffix when the level
            # difference is clearly larger than that normal local variation.
            mode = (
                "post_shift"
                if abs(offset) > max(8.0 * stable_scale, 0.5)
                and scale_ratio <= 2.5
                else "bridge"
            )

        suffix_offset = before_level - after_level if mode == "post_shift" else 0.0
        bridge_after = before_level if mode == "post_shift" else after_level
        bridge = np.linspace(
            before_level,
            bridge_after,
            len(result),
            endpoint=True,
        )
        result = result - replacement_center + bridge
        return result, {
            "mode": mode,
            "before": before_level,
            "after": after_level,
            "offset": offset,
            "suffix_offset": suffix_offset,
            "before_scale": before_scale,
            "after_scale": after_scale,
            "context_samples": int(min(len(before), len(after))),
        }

    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        if mad > 0:
            return 1.4826 * mad
        # 整数计数数据可能有很多连续相同值，导致 MAD=0
        # 此时用 std 作为 fallback
        std = float(np.std(values))
        return max(std, 1e-12)

    def _select_local_reference_segments(
        self,
        ref_before: np.ndarray,
        ref_after: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Use the nearest healthy context for local spectrum estimation.

        The full extracted references remain available for provenance, while
        only the edge-nearest samples drive reconstruction. This prevents a
        distant trend or event from setting the amplitude of a short gap.
        Set ``spectral.local_context_sec`` to ``0`` to disable trimming.
        """
        local_sec = float(self.config.get("spectral", {}).get(
            "local_context_sec", 0.0
        ))
        if local_sec <= 0:
            return ref_before, ref_after
        local_samples = max(4, int(round(local_sec * self.sr)))
        before = ref_before[-local_samples:].copy() if ref_before is not None else None
        after = ref_after[:local_samples].copy() if ref_after is not None else None
        return before, after

    def _extract_reference(
        self,
        data: np.ndarray,
        start: int,
        end: int,
        length_sec: float,
        gap_sec: float,
        *,
        side: str,
    ) -> tuple[np.ndarray | None, tuple[int, int]]:
        gap = int(round(gap_sec * self.sr))
        length = int(round(length_sec * self.sr))
        if length <= 0:
            return None, (0, -1)
        if side == "before":
            ref_end = start - gap
            ref_start = max(0, ref_end - length)
        else:
            ref_start = end + 1 + gap
            ref_end = min(len(data), ref_start + length)

        if ref_end <= ref_start:
            return None, (ref_start, ref_end - 1)
        reference = np.asarray(data[ref_start:ref_end], dtype=float)
        reference = reference[np.isfinite(reference)]
        if len(reference) < 4:
            return None, (ref_start, ref_end - 1)
        return reference.copy(), (ref_start, ref_end - 1)

    def _smooth_replacement_boundaries(
        self,
        data: np.ndarray,
        replacement: np.ndarray,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Blend replacement edges with real healthy waveform on both sides.

        Phase-constrained boundary: instead of linear extrapolation, use the
        actual healthy waveform immediately outside the anomaly as the blend
        template.  This guarantees waveform continuity at both joins because
        the edge of the replacement is literally fading into real data.
        """
        result = np.asarray(replacement, dtype=float).copy()
        n = len(result)
        edge = self._boundary_blend_length(n)
        if edge <= 1:
            return result

        # Left: extrapolate from real data ending at start-1, fade into synth.
        if start > 0:
            context = self._boundary_context_length(edge)
            left_ctx = data[max(0, start - context):start]
            template = self._edge_trend(left_ctx, edge, side="left")
            if template is not None and len(template) == edge:
                fade = self._smoothstep(edge)  # 0 at join → 1 inside
                result[:edge] = template * (1.0 - fade) + result[:edge] * fade
            else:
                anchor = float(data[start - 1])
                if np.isfinite(anchor):
                    result[0] = anchor

        # Right: extrapolate from real data starting at end+1, fade into synth.
        if end + 1 < len(data):
            context = self._boundary_context_length(edge)
            right_ctx = data[end + 1:end + 1 + context]
            template = self._edge_trend(right_ctx, edge, side="right")
            if template is not None and len(template) == edge:
                fade = self._smoothstep(edge)[::-1]  # 1 inside → 0 at join
                result[-edge:] = result[-edge:] * fade + template * (1.0 - fade)
            else:
                anchor = float(data[end + 1])
                if np.isfinite(anchor):
                    result[-1] = anchor

        return result

    def _boundary_blend_length(self, anomaly_length: int) -> int:
        joining = self.config.get("joining", {})
        if anomaly_length < 4:
            return 0
        fraction = float(joining.get("boundary_blend_fraction", 0.08))
        min_samples = int(joining.get("boundary_blend_min_samples", 8))
        max_sec = float(joining.get("boundary_blend_max_sec", 30.0))
        max_samples = max(1, int(round(max_sec * self.sr)))
        edge = max(min_samples, int(round(anomaly_length * fraction)))
        return max(0, min(edge, max_samples, anomaly_length // 2))

    def _boundary_context_length(self, edge: int) -> int:
        joining = self.config.get("joining", {})
        context_sec = float(joining.get("boundary_context_sec", 20.0))
        context_samples = int(round(context_sec * self.sr))
        return max(edge, context_samples, 4)

    @staticmethod
    def _smoothstep(length: int) -> np.ndarray:
        x = np.linspace(0.0, 1.0, length)
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _edge_trend(
        context: np.ndarray,
        count: int,
        *,
        side: str,
    ) -> np.ndarray | None:
        values = np.asarray(context, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            return None

        fit_len = min(len(values), max(4, min(count * 2, 64)))
        if side == "left":
            local = values[-fit_len:]
            anchor = float(local[-1])
            slope = ReferenceSpectrumReplacer._robust_slope(local)
            steps = np.arange(1, count + 1, dtype=float)
            return anchor + slope * steps

        local = values[:fit_len]
        anchor = float(local[0])
        slope = ReferenceSpectrumReplacer._robust_slope(local)
        steps = np.arange(-(count - 1), 1, dtype=float)
        return anchor + slope * (steps - 1.0)

    @staticmethod
    def _robust_slope(values: np.ndarray) -> float:
        diffs = np.diff(np.asarray(values, dtype=float))
        diffs = diffs[np.isfinite(diffs)]
        if len(diffs) == 0:
            return 0.0
        slope = float(np.median(diffs))
        scale = float(np.median(np.abs(diffs)))
        if scale > 1e-12:
            slope = float(np.clip(slope, -6.0 * scale, 6.0 * scale))
        return slope


def replace_by_reference_spectrum(
    data: np.ndarray,
    sr: float,
    start: int,
    end: int,
    **kwargs,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convenience function for one-shot reference-spectrum replacement."""
    return ReferenceSpectrumReplacer(sr).replace(data, start, end, **kwargs)

def auto_reference_length(
    anomaly_samples: int,
    sr: float,
    min_sec: float = 120.0,
    max_sec: float = 600.0,
) -> float:
    """clip(anomaly_sec/2, min_sec, max_sec)"""
    anom_sec = max(0.0, float(anomaly_samples) / float(sr))
    return float(np.clip(anom_sec / 2.0, float(min_sec), float(max_sec)))
