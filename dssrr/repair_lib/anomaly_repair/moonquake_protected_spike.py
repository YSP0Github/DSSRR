"""Local spike repair that protects non-stationary moonquake waveforms.

Standard de-spiking (MAD / z-score / moving-average) assumes the trace is
stationary background noise, so it happily "cleans" the sharp onset of a real
moonquake and destroys the very event the analyst cares about.  This module is
built for the opposite priority: remove instrument artefacts, but never touch a
genuine event.

Strategy (deliberately conservative -- prefer missing an artefact over
corrupting an event):

- **Detection.** Flag samples that deviate from a rolling median by more than
  ``threshold_sigma`` local standard deviations, while also requiring a minimum
  *global* sigma so a quiet window is not over-triggered.  Only short runs
  (``max_width_samples``) are candidates.
- **Rescue path.** A separate branch handles extreme isolated peaks and short
  instrument-clipping plateaus that the rolling-median test under-detects,
  using a higher ``strong_peak_sigma`` plus a neighbour-return-ratio test.
- **Never reconstruct long events.** Anything wider than the configured limits
  is left as-is; the class only interpolates a handful of samples across a
  spike (PCHIP, shape-preserving) and returns a per-sample mask of what changed.

The class is dependency-light (numpy + scipy only) and stateless apart from the
sampling rate, so it can be reused across traces::

    from dssrr.repair_lib.anomaly_repair import MoonquakeProtectedSpikeReplacer

    replacer = MoonquakeProtectedSpikeReplacer(sampling_rate=6.625)
    repaired, report = replacer.replace(data)

See Also
--------
dssrr.repair_lib.repair_engine.StreamRepair.moonquake_protected_spike_cleaning
    The ObsPy Stream wrapper used by the manual/batch GUI.
dssrr.repair_lib.e7_detector
    Catalogue-aware event protection used by the batch pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator


class MoonquakeProtectedSpikeReplacer:
    """Repair isolated sample spikes without reconstructing the event core."""

    def __init__(self, sampling_rate: float):
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        self.sampling_rate = float(sampling_rate)

    @staticmethod
    def _odd_window(samples: int, length: int) -> int:
        window = max(3, int(samples))
        if window % 2 == 0:
            window += 1
        if length < 3:
            return length if length % 2 else max(1, length - 1)
        return min(window, length if length % 2 else length - 1)

    @classmethod
    def _rolling_median(cls, values: np.ndarray, window: int) -> np.ndarray:
        if len(values) < 3:
            return np.full_like(values, np.median(values) if len(values) else 0.0)
        effective = cls._odd_window(window, len(values))
        pad = effective // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, effective)
        return np.median(windows, axis=-1)

    @staticmethod
    def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            return []
        breaks = np.flatnonzero(np.diff(indices) > 1) + 1
        groups = np.split(indices, breaks)
        return [(int(group[0]), int(group[-1])) for group in groups]

    def detect(
        self,
        data: np.ndarray,
        *,
        threshold_sigma: float = 8.0,
        minimum_global_sigma: float = 4.0,
        detection_window_sec: float = 0.75,
        scale_window_sec: float = 4.0,
        max_width_samples: int = 2,
        neighbour_return_ratio: float = 0.35,
        strong_peak_rescue: bool = True,
        strong_peak_sigma: float = 16.0,
        strong_peak_global_sigma: float = 6.0,
        strong_peak_ratio: float = 1.25,
        rescue_max_width_samples: int = 4,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return a conservative mask and detection diagnostics.

        A point is eligible only when it is a robust local outlier, belongs to
        a short run, and both immediate neighbours are much closer to the
        local waveform. This last condition protects real oscillatory event
        structure, which usually has energy on both sides of a peak. The
        optional rescue path is restricted to very large local peaks and
        repeated samples at an observed clipping boundary.
        """
        values = np.asarray(data, dtype=float)
        if values.ndim != 1:
            raise ValueError("data must be a 1-D array")
        if threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be positive")
        if minimum_global_sigma <= 0:
            raise ValueError("minimum_global_sigma must be positive")
        if detection_window_sec <= 0 or scale_window_sec <= 0:
            raise ValueError("window lengths must be positive")
        if max_width_samples < 1:
            raise ValueError("max_width_samples must be at least 1")
        if not 0 < neighbour_return_ratio <= 1:
            raise ValueError("neighbour_return_ratio must be in (0, 1]")
        if strong_peak_sigma <= 0:
            raise ValueError("strong_peak_sigma must be positive")
        if strong_peak_global_sigma <= 0:
            raise ValueError("strong_peak_global_sigma must be positive")
        if strong_peak_ratio < 1:
            raise ValueError("strong_peak_ratio must be at least 1")
        if rescue_max_width_samples < 1:
            raise ValueError("rescue_max_width_samples must be at least 1")

        mask = np.zeros(len(values), dtype=bool)
        valid = np.isfinite(values) & (values != -1)
        if len(values) < 3 or valid.sum() < 3:
            return mask, {
                "candidate_runs": [],
                "repaired_runs": [],
                "skipped_runs": [],
                "rescue_runs": [],
                "saturation_runs": [],
                "threshold_sigma": float(threshold_sigma),
                "minimum_global_sigma": float(minimum_global_sigma),
                "invalid_samples": int(np.count_nonzero(~valid)),
            }

        safe_values = values.copy()
        valid_indices = np.flatnonzero(valid)
        safe_values[~valid] = np.interp(
            np.flatnonzero(~valid), valid_indices, safe_values[valid_indices])
        detection_window = max(3, int(round(detection_window_sec * self.sampling_rate)))
        scale_window = max(3, int(round(scale_window_sec * self.sampling_rate)))
        local_median = self._rolling_median(safe_values, detection_window)
        residual = safe_values - local_median
        local_scale = 1.4826 * self._rolling_median(np.abs(residual), scale_window)

        global_median = float(np.median(safe_values))
        global_scale = 1.4826 * float(np.median(np.abs(safe_values - global_median)))
        global_scale = max(global_scale, np.finfo(float).eps)
        scale_floor = max(global_scale * 0.05, np.finfo(float).eps)
        local_scale = np.maximum(local_scale, scale_floor)
        local_threshold = threshold_sigma * local_scale
        global_threshold = minimum_global_sigma * global_scale
        candidate_mask = valid & (
            np.abs(residual) > np.maximum(local_threshold, global_threshold)
        )

        # Apollo files use repeated ADC-boundary samples for clipping and -1
        # for missing data. A short repeated maximum can be invisible to a
        # rolling median, so detect it independently of the residual mask.
        saturation_mask = np.zeros(len(values), dtype=bool)
        valid_values = safe_values[valid]
        observed_high = float(np.max(valid_values))
        high_count = int(np.count_nonzero(valid & (safe_values == observed_high)))
        below_high = valid_values[valid_values < observed_high]
        high_reference = (
            float(np.percentile(below_high, 99.5))
            if below_high.size else observed_high
        )
        if (
            high_count >= 2
            and observed_high - high_reference >= max(4.0, 8.0 * global_scale)
        ):
            saturation_mask = valid & (safe_values == observed_high)
            candidate_mask |= saturation_mask

        candidate_runs = self._runs(candidate_mask)
        accepted_runs: list[tuple[int, int]] = []
        skipped_runs: list[dict[str, Any]] = []
        rescue_runs: list[dict[str, Any]] = []
        saturation_runs = self._runs(saturation_mask)
        for start, end in candidate_runs:
            width = end - start + 1
            reason = None
            if width > int(max_width_samples):
                reason = "run_too_long"
            elif start == 0 or end == len(values) - 1:
                reason = "edge_run"
            else:
                peak = float(np.max(np.abs(residual[start:end + 1])))
                left = float(abs(residual[start - 1]))
                right = float(abs(residual[end + 1]))
                neighbour_limit = max(
                    peak * float(neighbour_return_ratio),
                    2.0 * max(float(local_scale[start - 1]), float(local_scale[end + 1])),
                )
                if max(left, right) > neighbour_limit:
                    reason = "neighbours_not_returned"

            # A genuinely extreme peak can sit inside a high-energy event,
            # where the normal neighbour-return test is intentionally strict.
            # Rescue only the peak itself, never the surrounding active run.
            if reason is not None and strong_peak_rescue:
                peak = float(np.max(np.abs(residual[start:end + 1])))
                is_short_saturation = (
                    width <= int(rescue_max_width_samples)
                    and np.all(saturation_mask[start:end + 1])
                )
                strong_indices = []
                for index in range(start, end + 1):
                    if saturation_mask[index]:
                        strong_indices.append(index)
                        continue
                    local_peak = abs(float(residual[index]))
                    neighbour_peak = max(
                        abs(float(residual[index - 1])),
                        abs(float(residual[index + 1])),
                    ) if 0 < index < len(values) - 1 else np.inf
                    if (
                        local_peak >= float(strong_peak_sigma) * max(
                            float(local_scale[index]), np.finfo(float).eps)
                        and local_peak >= float(strong_peak_global_sigma) * global_scale
                        and local_peak >= float(strong_peak_ratio) * max(neighbour_peak, np.finfo(float).eps)
                    ):
                        strong_indices.append(index)
                strong_runs = self._runs(
                    np.isin(np.arange(len(values)), strong_indices))
                if is_short_saturation:
                    accepted_runs.append((start, end))
                    rescue_runs.append({
                        "start": start,
                        "end": end,
                        "reason": "short_saturation_plateau",
                    })
                    continue
                if strong_runs:
                    for rescue_start, rescue_end in strong_runs:
                        if rescue_end - rescue_start + 1 <= int(rescue_max_width_samples):
                            accepted_runs.append((rescue_start, rescue_end))
                            rescue_runs.append({
                                "start": rescue_start,
                                "end": rescue_end,
                                "reason": "extreme_isolated_peak",
                            })
                    if any(
                        rescue_start >= start and rescue_end <= end
                        for rescue_start, rescue_end in strong_runs
                    ):
                        continue
            if reason is None:
                accepted_runs.append((start, end))
            else:
                skipped_runs.append({"start": start, "end": end, "reason": reason})

        mask[:] = False
        for start, end in accepted_runs:
            mask[start:end + 1] = True
        return mask, {
            "candidate_runs": [
                {"start": start, "end": end, "width": end - start + 1}
                for start, end in candidate_runs
            ],
            "repaired_runs": [
                {"start": start, "end": end, "width": end - start + 1}
                for start, end in accepted_runs
            ],
            "skipped_runs": skipped_runs,
            "rescue_runs": rescue_runs,
            "saturation_runs": [
                {"start": start, "end": end, "width": end - start + 1}
                for start, end in saturation_runs
            ],
            "detection_window_samples": detection_window,
            "scale_window_samples": scale_window,
            "threshold_sigma": float(threshold_sigma),
            "minimum_global_sigma": float(minimum_global_sigma),
            "strong_peak_rescue": bool(strong_peak_rescue),
            "strong_peak_sigma": float(strong_peak_sigma),
            "strong_peak_global_sigma": float(strong_peak_global_sigma),
            "strong_peak_ratio": float(strong_peak_ratio),
            "rescue_max_width_samples": int(rescue_max_width_samples),
            "invalid_samples": int(np.count_nonzero(~valid)),
            "observed_high": observed_high,
        }

    @staticmethod
    def _repair_run(values: np.ndarray, start: int, end: int, radius: int = 2) -> np.ndarray:
        repaired = values.copy()
        left_start = max(0, start - max(1, radius))
        right_end = min(len(values), end + 1 + max(1, radius))
        support = np.concatenate((np.arange(left_start, start), np.arange(end + 1, right_end)))
        support = support[np.isfinite(values[support]) & (values[support] != -1)]
        target = np.arange(start, end + 1)
        if len(support) < 2:
            return repaired
        if len(support) >= 4:
            try:
                repaired[target] = PchipInterpolator(support, values[support])(target)
                return repaired
            except (ValueError, TypeError):
                pass
        repaired[target] = np.interp(target, support, values[support])
        return repaired

    def replace(
        self,
        data: np.ndarray,
        *,
        threshold_sigma: float = 8.0,
        minimum_global_sigma: float = 4.0,
        detection_window_sec: float = 0.75,
        scale_window_sec: float = 4.0,
        max_width_samples: int = 2,
        neighbour_return_ratio: float = 0.35,
        interpolation_radius_samples: int = 2,
        strong_peak_rescue: bool = True,
        strong_peak_sigma: float = 16.0,
        strong_peak_global_sigma: float = 6.0,
        strong_peak_ratio: float = 1.25,
        rescue_max_width_samples: int = 4,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Repair only accepted isolated spikes and preserve all other samples."""
        values = np.asarray(data, dtype=float)
        mask, report = self.detect(
            values,
            threshold_sigma=threshold_sigma,
            minimum_global_sigma=minimum_global_sigma,
            detection_window_sec=detection_window_sec,
            scale_window_sec=scale_window_sec,
            max_width_samples=max_width_samples,
            neighbour_return_ratio=neighbour_return_ratio,
            strong_peak_rescue=strong_peak_rescue,
            strong_peak_sigma=strong_peak_sigma,
            strong_peak_global_sigma=strong_peak_global_sigma,
            strong_peak_ratio=strong_peak_ratio,
            rescue_max_width_samples=rescue_max_width_samples,
        )
        repaired = values.copy()
        for run in report["repaired_runs"]:
            repaired = self._repair_run(
                repaired,
                run["start"],
                run["end"],
                radius=interpolation_radius_samples,
            )
        report.update({
            "method": "moonquake_protected_spike",
            "sampling_rate": self.sampling_rate,
            "changed_samples": int(np.count_nonzero(repaired != values)),
            "candidate_count": len(report["candidate_runs"]),
            "repaired_count": len(report["repaired_runs"]),
            "skipped_count": len(report["skipped_runs"]),
            "interpolation_radius_samples": int(interpolation_radius_samples),
        })
        return repaired, report


def replace_by_moonquake_protected_spikes(
    data: np.ndarray,
    sampling_rate: float,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convenience wrapper for the moonquake-protected local repair method."""
    return MoonquakeProtectedSpikeReplacer(sampling_rate).replace(data, **kwargs)
