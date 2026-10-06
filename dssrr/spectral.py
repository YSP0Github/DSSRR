"""Spectral feature estimator — Welch periodogram with two-sided log-domain fusion.

This module implements step 3 of the DSSRR pipeline: turning the two healthy
reference windows into **one** target power spectral density plus a small set of
phase/amplitude features used later by the synthesizer.

How the target spectrum is built
--------------------------------
1. Estimate the one-sided PSD of each reference window separately with
   :func:`scipy.signal.welch` (Hann window, 50 % overlap, per-segment mean
   removed).
2. Interpolate both onto a common frequency axis and fuse them in the **log**
   domain (geometric mean).  Averaging in log space keeps a strong spectral line
   present on one side only from being washed out by the other side, and it is
   far more robust to a single noisy reference than arithmetic averaging.
3. Optionally smooth the fused PSD with the configured bandwidth
   (``spectral.smooth_bandwidth_hz``).

The frequency axis is produced with ``rfft``/``rfftfreq`` so it is guaranteed to
be strictly increasing.

.. note::
   **Historical naming.**  This estimator used to be described as a
   "multitaper" (Thomson) estimator.  That was never true: the implementation
   has always been Welch's averaged periodogram and has never used DPSS/Slepian
   tapers or a time-bandwidth product (``spectral.time_bandwidth`` was a dead
   config key that nothing read).  The misleading private method name
   ``_multitaper_psd`` was renamed to ``_psd`` for the same reason.  Please do
   not reintroduce the multitaper wording in papers or docs.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SpectralEstimator:
    """基于 Welch 平均周期图的频谱特征估计。"""

    def __init__(self, sr: float, config: dict):
        self.sr = sr
        self.cfg = config.get("spectral", {})

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def estimate(
        self,
        ref_before: np.ndarray,
        ref_after: np.ndarray,
        anomaly_length: int,
    ) -> Tuple[np.ndarray, dict]:
        """
        估计目标频谱特征。

        Returns
        -------
        (target_psd, phase_features)
            target_psd: 目标功率谱密度（长度 = n_fft // 2 + 1）
            phase_features: dict
        """
        n_fft = self.cfg.get("n_fft") or self._next_pow2(anomaly_length * 2)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        # Estimate the spectrum of fluctuations, not the large ADC/DC
        # baseline. The baseline is carried separately in phase_features.
        psd_before = self._psd_or_none(ref_before, n_fft)
        psd_after = self._psd_or_none(ref_after, n_fft)

        # 频谱平滑
        psd_before = self._smooth_spectrum(psd_before, freqs)
        psd_after = self._smooth_spectrum(psd_after, freqs)

        # 融合
        target_psd = self._fuse_spectra(psd_before, psd_after, 0.5)

        # 相位特征
        phase_features = self._extract_phase_features(ref_before, ref_after)
        refs = [r for r in (ref_before, ref_after) if r is not None and len(r)]
        references = np.concatenate(refs).astype(float)
        phase_features["target_mean"] = float(np.mean(references))
        phase_features["target_std"] = float(np.std(references))

        return target_psd, phase_features

    def _psd_or_none(
        self, segment: np.ndarray | None, n_fft: int
    ) -> np.ndarray | None:
        """Welch PSD；缺失/过短的参考段返回 None（单侧模式）。"""
        if segment is None or len(segment) < 8:
            return None
        _, psd = self._psd(segment, n_fft)
        return psd

    def _psd(
        self, segment: np.ndarray, n_fft: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Welch 平均周期图 —— DSSRR 的谱估计器（不提供多窗/DPSS 选项）。

        实现细节：Hann 窗，段长 ``min(N, max(64, N//4))`` 样本，50\% 重叠，
        每段去均值（DC 基线由 ``phase_features`` 另行承载），``nfft=n_fft``
        保证两侧共用同一频率轴。
        """
        from scipy.signal import welch

        segment = np.asarray(segment, dtype=float)
        segment = segment - np.mean(segment)
        N = len(segment)
        if N < 8:
            freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)
            return freqs, np.zeros_like(freqs)
        nperseg = int(self.cfg.get("welch_nperseg", 0)) or 0
        if nperseg <= 0:
            nperseg = min(N, max(64, N // 4))
        nperseg = int(np.clip(nperseg, 16, min(N, max(16, n_fft))))
        freqs, psd = welch(
            segment, fs=self.sr, window="hann", nperseg=nperseg,
            noverlap=nperseg // 2, nfft=n_fft, scaling="density",
            detrend="constant",
        )
        return freqs, np.asarray(psd, dtype=float)

    # ------------------------------------------------------------------
    # 频谱融合
    # ------------------------------------------------------------------

    def _fuse_spectra(
        self,
        psd_before: np.ndarray | None,
        psd_after: np.ndarray | None,
        ratio: float,
    ) -> np.ndarray:
        """对数域加权融合；单侧缺失时直接返回另一侧。"""
        if psd_before is None and psd_after is None:
            raise ValueError("at least one PSD is required for fusion")
        if psd_before is None:
            return np.asarray(psd_after, dtype=float)
        if psd_after is None:
            return np.asarray(psd_before, dtype=float)
        log_b = np.log(psd_before + 1e-30)
        log_a = np.log(psd_after + 1e-30)
        return np.exp((1.0 - ratio) * log_b + ratio * log_a)

    # ------------------------------------------------------------------
    # 频谱平滑
    # ------------------------------------------------------------------

    def _smooth_spectrum(self, psd: np.ndarray | None, freqs: np.ndarray) -> np.ndarray | None:
        if psd is None:
            return None
        smooth_bw = self.cfg.get("smooth_bandwidth_hz", 2.0)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        kernel_size = max(1, int(smooth_bw / df))
        if kernel_size <= 1:
            return psd.copy()
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(psd, kernel, mode="same")

    # ------------------------------------------------------------------
    # 相位特征
    # ------------------------------------------------------------------

    def _extract_phase_features(
        self, ref_before: np.ndarray | None, ref_after: np.ndarray | None
    ) -> dict:
        gd_before = self._group_delay(ref_before) if ref_before is not None else None
        gd_after = self._group_delay(ref_after) if ref_after is not None else None
        t_b = np.arange(len(gd_before)) / self.sr if gd_before is not None else np.array([0.0])
        t_a = np.arange(len(gd_after)) / self.sr if gd_after is not None else np.array([0.0])
        slope_b = np.polyfit(t_b, gd_before, 1) if len(t_b) > 1 else (0.0, 0.0)
        slope_a = np.polyfit(t_a, gd_after, 1) if len(t_a) > 1 else (0.0, 0.0)

        refs = [r for r in (ref_before, ref_after) if r is not None and len(r)]
        if refs:
            freqs, psd = self._psd(
                np.concatenate(refs),
                self._next_pow2(sum(len(r) for r in refs)),
            )
        else:
            freqs, psd = np.array([0.0]), np.array([1.0])
        dom_freq = float(freqs[np.argmax(psd[1:]) + 1]) if len(psd) > 1 else self.sr / 4

        return {
            "group_delay_before": gd_before,
            "group_delay_after": gd_after,
            "phase_slopes": (slope_b, slope_a),
            "dominant_freq": dom_freq,
        }

    @staticmethod
    def _group_delay(segment: np.ndarray) -> np.ndarray:
        from scipy.signal import hilbert
        analytic = hilbert(segment)
        return np.unwrap(np.angle(analytic))

    @staticmethod
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p
