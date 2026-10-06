"""Quality verifier — acceptance metrics for a completed repair.

This module implements step 8 of the DSSRR pipeline.  It never changes the data;
it only *scores* the result so that an automatic pipeline can decide whether a
repair is trustworthy, and so that a human reviewer has numbers to look at.

Metrics produced by :meth:`QualityVerifier.verify`
--------------------------------------------------
``spectral_continuity``
    Similarity between the PSD of the repaired segment and the PSD of the
    surrounding healthy data.  1.0 means the spectral shape is preserved.
``envelope_continuity``
    Relative jump in the local amplitude envelope across each join.
``dc_continuity``
    Relative step in the mean level across each join.
``autocorr_consistency``
    Difference between the autocorrelation of the repaired segment and that of
    the references.
``energy_ratio``
    Repaired-segment energy divided by the reference energy (ideally ~1).

The thresholds these are compared against live under the ``verification``
section of :data:`dssrr.config.DEFAULT_CONFIG`.

Dependencies: numpy, scipy.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.signal import hilbert

logger = logging.getLogger(__name__)


class QualityVerifier:
    """验证异常段修复结果的质量。"""

    def __init__(self, sr: float, config: dict):
        self.sr = sr
        self.cfg = config.get("verification", {})

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def verify(
        self,
        original: np.ndarray,
        repaired: np.ndarray,
        anomaly_start: int,
        anomaly_end: int,
        ref_before: np.ndarray | None = None,
        ref_after: np.ndarray | None = None,
    ) -> dict:
        """
        综合质量验证。

        Returns
        -------
        report : dict {
            "passed"              : bool — 整体是否通过,
            "spectral_continuity" : float — 频谱余弦相似度,
            "envelope_continuity" : float — 包络跳变量 (越小越好),
            "dc_continuity"       : float — 直流跳变量,
            "autocorr_consistency": float — 自相关衰减率差异,
            "energy_ratio"        : float — 修复段能量 / 健康参考能量,
            "details"             : str  — 描述信息,
        }
        """
        report = {
            "passed": True,
            "spectral_continuity": 0.0,
            "envelope_continuity": 0.0,
            "dc_continuity": 0.0,
            "autocorr_consistency": 0.0,
            "energy_ratio": 1.0,
            "details": "",
        }

        repair_segment = repaired[anomaly_start : anomaly_end + 1]
        original_segment = original[anomaly_start : anomaly_end + 1]

        # ── 频谱连续性 ──
        if ref_before is not None and ref_after is not None:
            report["spectral_continuity"] = self._spectral_continuity(
                ref_before, repair_segment, ref_after
            )

        # ── 包络连续性 ──
        report["envelope_continuity"] = self._envelope_continuity(
            ref_before, ref_after, repair_segment
        )

        # ── 直流连续性 ──
        report["dc_continuity"] = self._dc_continuity(
            ref_before, ref_after, repair_segment
        )

        # ── 自相关一致性 ──
        if ref_before is not None:
            report["autocorr_consistency"] = self._autocorr_consistency(
                ref_before, repair_segment
            )

        # ── 能量比 ──
        report["energy_ratio"] = self._energy_ratio(
            original_segment, repair_segment, ref_before, ref_after
        )

        # ── 综合判定 ──
        passed, details = self._judge(report)
        report["passed"] = passed
        report["details"] = details

        return report

    # ------------------------------------------------------------------
    # 各项检查
    # ------------------------------------------------------------------

    def _spectral_continuity(
        self,
        ref_before: np.ndarray,
        repair: np.ndarray,
        ref_after: np.ndarray,
    ) -> float:
        """
        频谱连续性：修复段与前后参考段的频谱余弦相似度。
        取两侧中的最小值作为报告值。
        """
        n_fft = self._next_pow2(max(len(ref_before), len(repair), len(ref_after)) * 2)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        psd_before = self._psd(ref_before, n_fft)
        psd_repair = self._psd(repair, n_fft)
        psd_after = self._psd(ref_after, n_fft)

        # 只比较有效频率范围（忽略 DC 和极高频）
        lo = max(1, int(0.5 / self.sr * n_fft))
        hi = min(n_fft // 2, int(0.45 * n_fft))

        sim_before = self._cosine_sim(psd_before[lo:hi], psd_repair[lo:hi])
        sim_after = self._cosine_sim(psd_repair[lo:hi], psd_after[lo:hi])

        return min(sim_before, sim_after)

    def _envelope_continuity(
        self,
        ref_before: np.ndarray | None,
        ref_after: np.ndarray | None,
        repair: np.ndarray,
    ) -> float:
        """包络连续性：接缝处 Hilbert 包络跳变量（归一化）。"""
        env_repair = np.abs(hilbert(repair))
        jumps = []

        if ref_before is not None and len(ref_before) > 0:
            env_before = np.abs(hilbert(ref_before))
            jump = abs(float(env_before[-1] - env_repair[0]))
            norm = float(np.mean(env_before[-max(1, len(env_before) // 4) :]) + 1e-10)
            jumps.append(jump / norm)

        if ref_after is not None and len(ref_after) > 0:
            env_after = np.abs(hilbert(ref_after))
            jump = abs(float(env_repair[-1] - env_after[0]))
            norm = float(np.mean(env_after[: max(1, len(env_after) // 4)]) + 1e-10)
            jumps.append(jump / norm)

        return max(jumps) if jumps else 0.0

    def _dc_continuity(
        self,
        ref_before: np.ndarray | None,
        ref_after: np.ndarray | None,
        repair: np.ndarray,
    ) -> float:
        """直流连续性：重叠区两端均值差（归一化）。"""
        jumps = []
        if ref_before is not None and len(ref_before) > 0:
            mu_ref = float(np.mean(ref_before[-max(1, len(ref_before) // 4) :]))
            mu_rep = float(np.mean(repair[: max(1, len(repair) // 4)]))
            norm = float(np.std(ref_before) + 1e-10)
            jumps.append(abs(mu_ref - mu_rep) / norm)

        if ref_after is not None and len(ref_after) > 0:
            mu_ref = float(np.mean(ref_after[: max(1, len(ref_after) // 4)]))
            mu_rep = float(np.mean(repair[-max(1, len(repair) // 4) :]))
            norm = float(np.std(ref_after) + 1e-10)
            jumps.append(abs(mu_ref - mu_rep) / norm)

        return max(jumps) if jumps else 0.0

    def _autocorr_consistency(
        self, ref: np.ndarray, repair: np.ndarray
    ) -> float:
        """自相关一致性：衰减率差异。"""
        def _autocorr_decay(segment):
            ac = np.correlate(segment, segment, mode="full")
            ac = ac[len(ac) // 2 :]
            ac = ac / (ac[0] + 1e-15)
            # 首次过零点位置作为衰减速度的代理
            zero_crossings = np.where(ac < 0)[0]
            if len(zero_crossings) == 0:
                return len(ac)
            return float(zero_crossings[0])

        decay_ref = _autocorr_decay(ref)
        decay_rep = _autocorr_decay(repair)

        if decay_ref < 1e-10:
            return 0.0
        diff = abs(decay_ref - decay_rep) / decay_ref
        return float(diff)

    def _energy_ratio(
        self,
        original: np.ndarray,
        repair: np.ndarray,
        ref_before: np.ndarray | None = None,
        ref_after: np.ndarray | None = None,
    ) -> float:
        """修复段能量与健康参考能量的比值。"""
        references = [
            ref for ref in (ref_before, ref_after)
            if ref is not None and len(ref)
        ]
        if references:
            ref_mean_energy = float(np.mean([
                np.mean(np.asarray(ref, dtype=float) ** 2)
                for ref in references
            ]))
            e_orig = ref_mean_energy * len(repair) + 1e-30
        else:
            # Backward-compatible fallback when no healthy reference exists.
            e_orig = float(np.sum(original ** 2) + 1e-30)
        e_repair = float(np.sum(repair ** 2) + 1e-30)
        return e_repair / e_orig

    # ------------------------------------------------------------------
    # 综合判定
    # ------------------------------------------------------------------

    def _judge(self, report: dict) -> Tuple[bool, str]:
        """根据各项指标综合判定是否通过。"""
        issues = []

        if report["spectral_continuity"] < self.cfg["spectral_similarity_thresh"]:
            issues.append(
                f"频谱相似度 {report['spectral_continuity']:.3f} "
                f"< {self.cfg['spectral_similarity_thresh']}"
            )

        if report["envelope_continuity"] > self.cfg["envelope_jump_thresh"]:
            issues.append(
                f"包络跳变 {report['envelope_continuity']:.3f} "
                f"> {self.cfg['envelope_jump_thresh']}"
            )

        if report["dc_continuity"] > self.cfg["dc_jump_thresh"]:
            issues.append(
                f"直流跳变 {report['dc_continuity']:.3f} "
                f"> {self.cfg['dc_jump_thresh']}"
            )

        if report["autocorr_consistency"] > self.cfg["autocorr_diff_thresh"]:
            issues.append(
                f"自相关差异 {report['autocorr_consistency']:.3f} "
                f"> {self.cfg['autocorr_diff_thresh']}"
            )

        er = report["energy_ratio"]
        lo, hi = self.cfg["energy_ratio_range"]
        if er < lo or er > hi:
            issues.append(f"能量比 {er:.3f} 超出 [{lo}, {hi}]")

        passed = len(issues) == 0
        details = "; ".join(issues) if issues else "所有检查通过"
        return passed, details

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _psd(segment: np.ndarray, n_fft: int) -> np.ndarray:
        segment = np.asarray(segment, dtype=float)
        segment = segment - np.mean(segment)
        fft_vals = np.fft.rfft(segment, n=n_fft)
        return np.abs(fft_vals) ** 2 / n_fft

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-15 or nb < 1e-15:
            return 0.0
        return float(dot / (na * nb))

    @staticmethod
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p
