"""Layered reference-spectrum replacement with band-prioritized synthesis.

DSSRR's core repair (``ReferenceSpectrumReplacer``) rebuilds the missing
samples from a single target power spectrum estimated from the two healthy
reference windows that flank the anomaly.  That is the right default when the
signal of interest is broadband, but it can under-represent a narrow scientific
band that carries most of the information -- for example the ultra-low-frequency
(ULF) content of lunar free oscillations, which sits far below the noise floor
of the rest of the spectrum.

This module adds one conservative extra layer on top of the two-sided result:

1. Run the ordinary two-sided reference repair (geometry, baseline bridge,
   boundary smoothing and optional integer quantization are inherited
   unchanged from :class:`~dssrr.core.ReferenceSpectrumReplacer`).
2. In a user-selected band ``[band_min_hz, band_max_hz]`` with a smooth
   ``transition_hz`` roll-off, synthesize an additional reference-guided
   component so the chosen band dominates the repaired interval.  Outside the
   band the base two-sided result is left untouched.

Because step 1 is shared, the two methods differ **only** by the band-priority
layer; this is deliberate and is covered by ``tests/test_single_source_of_truth.py``.

Typical use::

    from dssrr.repair_lib.anomaly_repair import LayeredReferenceSpectrumReplacer

    replacer = LayeredReferenceSpectrumReplacer(sr=6.625)
    repaired, info = replacer.replace(
        data, start, end,
        focus_mode="ultra_low",     # or "band"
        band_min_hz=0.02,
        band_max_hz=0.10,
    )

See Also
--------
dssrr.core.ReferenceSpectrumReplacer : the two-sided base method.
dssrr.config : the ``layered_synthesis`` section tunes this layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .reference_spectrum import ReferenceSpectrumReplacer
from .synthesizer import SignalSynthesizer


class LayeredReferenceSpectrumReplacer(ReferenceSpectrumReplacer):
    """Reference-spectrum replacement with frequency-band prioritization."""

    def __init__(self, sr: float, config: dict | None = None):
        super().__init__(sr, config)
        self.layered_cfg = self.config.get("layered_synthesis", {})

    def replace(
        self,
        data: np.ndarray,
        start: int,
        end: int,
        *,
        focus_mode: str = "ultra_low",
        band_min_hz: float | None = None,
        band_max_hz: float | None = None,
        transition_hz: float | None = None,
        random_seed: int | None = 0,
        # Default reference windows follow the DSSRR paper protocol
        # (L = clip(anomaly/2, 120 s, 600 s)); callers may override.
        reference_before_sec: float = 120.0,
        reference_after_sec: float = 120.0,
        reference_gap_sec: float = 1.0,
        quantize: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Replace a selected interval while prioritizing one frequency band.

        The workflow is the current two-sided reference-spectrum repair plus
        one extra reference-guided layer in the requested band. Geometry
        (safety gap, baseline bridge, boundary smoothing) and the optional
        integer quantization follow :meth:`ReferenceSpectrumReplacer.replace`
        exactly, so the two methods differ only by the band-priority layer.
        """
        values = np.asarray(data, dtype=float)
        if values.ndim != 1:
            raise ValueError("data must be a 1-D array")
        n = len(values)
        start, end = int(start), int(end)
        if n == 0 or start < 0 or end >= n or start > end:
            raise ValueError("invalid anomaly interval")
        if reference_before_sec <= 0 or reference_after_sec <= 0:
            raise ValueError("reference lengths must be positive")
        if reference_gap_sec < 0:
            raise ValueError("reference_gap_sec cannot be negative")

        ref_before, before_bounds = self._extract_reference(
            values,
            start,
            end,
            reference_before_sec,
            reference_gap_sec,
            side="before",
        )
        ref_after, after_bounds = self._extract_reference(
            values,
            start,
            end,
            reference_after_sec,
            reference_gap_sec,
            side="after",
        )
        if ref_before is None or ref_after is None:
            raise ValueError(
                "both before and after reference data are required for "
                "layered reference-spectrum replacement"
            )

        work_before, work_after = self._select_local_reference_segments(
            ref_before, ref_after
        )

        synthesis_config = dict(self.config.get("synthesis", {}))
        synthesis_config["random_seed"] = random_seed
        estimator = self.estimator
        synthesizer = SignalSynthesizer(
            self.sr,
            {**self.config, "synthesis": synthesis_config},
        )

        anomaly_length = end - start + 1
        # 安全间隙（样本数）：后参考段实际起点为 end+1+gap_samples。
        # 趋势外推与基线对齐必须使用同一几何，否则右侧目标电平被系统性偏移
        # （与双侧参考实现保持一致，见 reference_spectrum.replace）。
        gap_samples = int(round(reference_gap_sec * self.sr))
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
        replacement, band_report = self._apply_frequency_priority(
            replacement,
            target_psd,
            focus_mode=focus_mode,
            band_min_hz=band_min_hz,
            band_max_hz=band_max_hz,
            transition_hz=transition_hz,
            ref_before=ref_before,
            ref_after=ref_after,
        )
        replacement, baseline_report = self._align_replacement_baseline(
            replacement, work_before, work_after, gap_samples=gap_samples
        )
        replacement = self._smooth_replacement_boundaries(
            values, replacement, start, end
        )

        repaired = values.copy()
        # 只修改异常段，不修改异常段外的数据（与双侧参考实现一致：
        # 旧版分层实现对 repaired[end+1:] 施加 suffix_offset，会污染异常段
        # 之外的原始数据并引入人造台阶，此行为已废弃）。
        repaired[start:end + 1] = replacement
        # 整数化（可选）：与双侧参考同口径，仅当调用方显式开启时取整。
        if quantize:
            repaired[start:end + 1] = np.round(repaired[start:end + 1])
        verification_after = work_after
        verification = self.verifier.verify(
            values, repaired, start, end, work_before, verification_after
        )

        report = {
            "method": "layered_reference_spectrum",
            "focus_mode": band_report["focus_mode"],
            "start": start,
            "end": end,
            "length": anomaly_length,
            "sampling_rate": self.sr,
            "frequency_band_hz": band_report["frequency_band_hz"],
            "transition_hz": band_report["transition_hz"],
            "band_target_weight": band_report["band_target_weight"],
            "band_background_weight": band_report["band_background_weight"],
            "layer_blend_strength": band_report["layer_blend_strength"],
            "residual_scale": band_report["residual_scale"],
            "reference_before": {
                "start": before_bounds[0],
                "end": before_bounds[1],
                "length": len(ref_before),
            },
            "reference_after": {
                "start": after_bounds[0],
                "end": after_bounds[1],
                "length": len(ref_after),
            },
            "working_reference_before_length": len(work_before),
            "working_reference_after_length": len(work_after),
            "working_reference_context_sec": float(
                max(len(work_before), len(work_after)) / self.sr
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
                "The selected focus band is emphasized while other bands keep supporting texture.",
                "Reference non-stationarity can bias the fused target spectrum.",
            ],
        }
        return repaired, report

    def _apply_frequency_priority(
        self,
        replacement: np.ndarray,
        target_psd: np.ndarray,
        *,
        focus_mode: str,
        band_min_hz: float | None,
        band_max_hz: float | None,
        transition_hz: float | None,
        ref_before: np.ndarray,
        ref_after: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Conservatively add a reference-guided layer in the focus band."""
        del target_psd
        mode, band_min, band_max, transition, weights = self._resolve_focus_band(
            focus_mode, band_min_hz, band_max_hz, transition_hz
        )
        base = np.asarray(replacement, dtype=float)
        focus_from_context = self._focus_component_from_context(
            base,
            ref_before,
            ref_after,
            band_min,
            band_max,
            transition,
        )
        focus_from_base = self._fft_focus_component(
            base, band_min, band_max, transition
        )
        background = base - focus_from_base

        if mode == "high":
            residual_scale = float(weights["background_scale"])
            # Keep the synthesized waveform as the phase/texture carrier and
            # only replace the selected band's difference. Reconstructing
            # ``focus + background`` from two different FFT windows can bring
            # the concatenation boundary back into the repair as a step.
            layered = base + residual_scale * (
                focus_from_context - focus_from_base
            )
            blend_strength = float(weights["blend_strength"])
            target_weight = float(self.layered_cfg.get("high_band_target_weight", 0.88))
            background_weight = float(self.layered_cfg.get("high_band_background_weight", 0.22))
        elif mode == "custom":
            residual_scale = float(weights["residual_scale"])
            layered = base + residual_scale * (
                focus_from_context - focus_from_base
            )
            blend_strength = float(weights["blend_strength"])
            target_weight = float(self.layered_cfg.get("custom_band_target_weight", 0.90))
            background_weight = float(self.layered_cfg.get("custom_band_background_weight", 0.20))
        else:
            # Keep the measured broadband texture intact. Multiplying the
            # out-of-band residual by 0.45 makes a low-band repair look like
            # staircase quantization when the source contains fine texture.
            residual_scale = 1.0
            layered = base + focus_from_context - focus_from_base
            blend_strength = float(weights["blend_strength"])
            target_weight = float(self.layered_cfg.get("low_band_target_weight", 0.92))
            background_weight = float(self.layered_cfg.get("low_band_background_weight", 0.18))

        blend_strength = float(np.clip(blend_strength, 0.0, 1.0))
        # The context FFT is calculated across two joins (reference -> repair
        # and repair -> reference). Those joins are not physical data and can
        # produce a low-frequency transient at the first/last samples of the
        # selected interval. Fade the correction in over a few periods while
        # retaining the stable two-sided synthesis at the actual joins.
        correction = layered - base
        correction *= self._frequency_correction_taper(
            len(base), band_max, transition
        )
        result = base + blend_strength * correction

        references = [ref for ref in (ref_before, ref_after) if ref is not None and len(ref)]
        if references:
            reference = np.concatenate(references).astype(float)
            target_mean = float(np.mean(reference))
            target_std = float(np.std(reference))
            current_std = float(np.std(result))
            if target_std > 1e-12 and current_std > 1e-12:
                scale = float(np.clip(target_std / current_std, 0.65, 1.35))
                result = (result - np.mean(result)) * scale
            else:
                result = np.zeros_like(result)
            result += target_mean

        return result, {
            "focus_mode": mode,
            "frequency_band_hz": (float(band_min), float(band_max)),
            "transition_hz": float(transition),
            "band_target_weight": float(target_weight),
            "band_background_weight": float(background_weight),
            "layer_blend_strength": float(blend_strength),
            "residual_scale": float(residual_scale),
        }

    def _frequency_correction_taper(
        self,
        length: int,
        band_max: float,
        transition: float,
    ) -> np.ndarray:
        """Return a smooth edge taper for the layered-only correction."""
        if length < 4:
            return np.ones(max(0, length), dtype=float)

        # Two periods are enough to suppress the context-FFT transient while
        # keeping most of a long manually selected interval available for the
        # requested frequency emphasis. The transition band also contributes
        # to the transient width, so include it in the lower bound.
        safe_band = max(float(band_max), 1.0 / max(length / self.sr, 1.0))
        edge = int(round(2.0 * self.sr / safe_band))
        edge = max(edge, int(round(2.0 * float(transition) * self.sr)))
        edge = min(max(2, edge), length // 2)
        if edge <= 1:
            return np.ones(length, dtype=float)

        taper = np.ones(length, dtype=float)
        x = np.linspace(0.0, 1.0, edge)
        fade = x * x * (3.0 - 2.0 * x)
        taper[:edge] = fade
        taper[-edge:] = fade[::-1]
        return taper

    def _resolve_focus_band(
        self,
        focus_mode: str,
        band_min_hz: float | None,
        band_max_hz: float | None,
        transition_hz: float | None,
    ) -> tuple[str, float, float, float, dict[str, float]]:
        mode = str(focus_mode or "ultra_low").strip().lower()
        nyquist = self.sr / 2.0
        transition = float(
            transition_hz
            if transition_hz is not None
            else self.layered_cfg.get("transition_hz", 0.01)
        )
        transition = max(0.0, min(transition, nyquist))

        if mode in {"ultra_low", "low", "ultra-low"}:
            mode = "ultra_low"
            band_min = 0.0
            default_max = float(self.layered_cfg.get("low_band_max_hz", 0.08))
            band_max = default_max if band_max_hz is None else float(band_max_hz)
            band_max = min(max(band_max, 1e-9), nyquist)
            weights = {
                "residual_scale": float(self.layered_cfg.get("low_residual_scale", 0.45)),
                "blend_strength": float(self.layered_cfg.get("low_blend_strength", 0.45)),
            }
        elif mode in {"high", "high_frequency", "high-frequency"}:
            mode = "high"
            default_min = float(self.layered_cfg.get("high_band_min_hz", 0.25))
            band_min = default_min if band_min_hz is None else float(band_min_hz)
            band_min = max(0.0, min(band_min, nyquist - 1e-9))
            if band_max_hz is None or float(band_max_hz) <= band_min:
                band_max = nyquist
            else:
                band_max = min(float(band_max_hz), nyquist)
            weights = {
                "background_scale": float(self.layered_cfg.get("high_background_scale", 0.75)),
                "blend_strength": float(self.layered_cfg.get("high_blend_strength", 0.35)),
            }
        elif mode in {"custom", "band", "frequency_band"}:
            mode = "custom"
            default_min = float(self.layered_cfg.get("custom_band_min_hz", 0.001))
            default_max = float(self.layered_cfg.get("custom_band_max_hz", 0.08))
            band_min = default_min if band_min_hz is None else float(band_min_hz)
            band_max = default_max if band_max_hz is None else float(band_max_hz)
            band_min = max(0.0, min(band_min, nyquist))
            band_max = max(band_min + 1e-9, min(band_max, nyquist))
            weights = {
                "residual_scale": float(self.layered_cfg.get("custom_residual_scale", 0.60)),
                "blend_strength": float(self.layered_cfg.get("custom_blend_strength", 0.40)),
            }
        else:
            raise ValueError(f"unknown focus_mode: {focus_mode}")

        return mode, float(band_min), float(band_max), float(transition), weights

    def _focus_component_from_context(
        self,
        replacement: np.ndarray,
        ref_before: np.ndarray,
        ref_after: np.ndarray,
        band_min: float,
        band_max: float,
        transition: float,
    ) -> np.ndarray:
        context_samples = int(round(
            float(self.layered_cfg.get("context_sec", 120.0)) * self.sr
        ))
        context_samples = max(len(replacement), context_samples, 4)
        before = np.asarray(ref_before, dtype=float)[-context_samples:]
        after = np.asarray(ref_after, dtype=float)[:context_samples]
        combo = np.concatenate([before, replacement, after])
        component = self._fft_focus_component(combo, band_min, band_max, transition)
        start = len(before)
        return component[start:start + len(replacement)]

    def _fft_focus_component(
        self,
        signal: np.ndarray,
        band_min: float,
        band_max: float,
        transition: float,
    ) -> np.ndarray:
        values = np.asarray(signal, dtype=float)
        n = len(values)
        if n == 0:
            return values.copy()
        n_fft = self._next_pow2(max(2, n * 2))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)
        mask = self._band_profile(
            freqs,
            band_min,
            band_max,
            transition,
            target_weight=1.0,
            background_weight=0.0,
        )
        spectrum = np.fft.rfft(values, n=n_fft)
        return np.fft.irfft(spectrum * mask, n=n_fft)[:n]

    @staticmethod
    def _band_profile(
        freqs: np.ndarray,
        band_min: float,
        band_max: float,
        transition: float,
        *,
        target_weight: float,
        background_weight: float,
    ) -> np.ndarray:
        freqs = np.asarray(freqs, dtype=float)
        profile = np.full_like(freqs, float(background_weight), dtype=float)
        target_weight = float(target_weight)
        background_weight = float(background_weight)
        if transition <= 1e-12 or band_max <= band_min:
            profile[(freqs >= band_min) & (freqs <= band_max)] = target_weight
            return profile

        left_start = max(0.0, band_min - transition)
        left_inside = (freqs >= left_start) & (freqs < band_min)
        inside = (freqs >= band_min) & (freqs <= band_max)
        right_inside = (freqs > band_max) & (freqs <= band_max + transition)

        profile[inside] = target_weight

        if np.any(left_inside):
            x = (freqs[left_inside] - left_start) / max(transition, 1e-12)
            fade = x * x * (3.0 - 2.0 * x)
            profile[left_inside] = background_weight + (
                target_weight - background_weight
            ) * fade

        if np.any(right_inside):
            x = (freqs[right_inside] - band_max) / max(transition, 1e-12)
            fade = x * x * (3.0 - 2.0 * x)
            profile[right_inside] = target_weight - (
                target_weight - background_weight
            ) * fade

        return profile

    @staticmethod
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p


def replace_by_layered_reference_spectrum(
    data: np.ndarray,
    sr: float,
    start: int,
    end: int,
    **kwargs,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convenience function for layered reference-spectrum replacement."""
    return LayeredReferenceSpectrumReplacer(sr).replace(data, start, end, **kwargs)
