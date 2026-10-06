"""
替换信号合成器 — 频谱整形 + 局部边界约束
========================================

本模块负责 DSSRR 流水线的第三步（见 :mod:`dssrr.core` 的流程说明）：
给定**目标功率谱密度**和两侧健康参考段，合成一段长度与缺失区间完全相同的
替代波形，并保证它与两侧真实数据在拼接处连续。

核心思想
--------
单纯按目标谱做逆变换（"白相位"合成）能匹配统计能量，但会丢掉波形质感，且
两端电平对不上。因此本模块在频谱整形之外叠加了四道时域后处理：

1. :meth:`SignalSynthesizer._apply_boundary_constraints`
   —— 边界电平约束；
2. :meth:`SignalSynthesizer._apply_phase_constrained_boundary`
   —— 相位约束边界（用真实数据的相位延长边界，而非线性外推）；
3. :meth:`SignalSynthesizer._correct_edge_dc_drift`
   —— 端点直流漂移校正；
4. :meth:`SignalSynthesizer._apply_trend_correction`
   —— 从两侧参考段线性外推长时程趋势。

长缺失段（可达 90 分钟）不适合一次性合成，因此默认方法
``reference_overlap_add`` 会把健康参考段切成块、重叠相加地铺满整个缺口，
使合成结果在时域纹理上仍然像真实地震记录。

关键参数：``gap_samples``
------------------------
异常段与**后**参考段之间存在安全间隙（安全间隙，见 ``reference_gap_sec``）。
趋势外推要跨越整个缺口建立统一时间轴，必须把该间隙计入，否则后参考段会被
人为拉近若干样本，拟合出的趋势斜率偏大、右端目标电平系统性偏移。
调用方应传入 ``round(reference_gap_sec * sr)``。

依赖：numpy、scipy。本模块不依赖 PyQt5/obspy。
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import hilbert

logger = logging.getLogger(__name__)


def _as_gap_samples(value: int | None) -> int:
    """Normalise a safety-gap argument to a non-negative sample count.

    ``None`` is accepted for backward compatibility with callers that used the
    older internal signature (where ``None`` meant "fall back to the configured
    safety margin").  The modern call path always passes an explicit integer, so
    ``None`` is simply treated as "no gap".
    """
    if value is None:
        return 0
    return max(0, int(value))


class SignalSynthesizer:
    """基于频谱约束的替换信号合成器。"""

    def __init__(self, sr: float, config: dict):
        self.sr = sr
        self.cfg = config.get("synthesis", {})

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def synthesize(
        self,
        target_psd: np.ndarray,
        phase_features: dict,
        ref_before: np.ndarray,
        ref_after: np.ndarray,
        length: int,
        gap_samples: int | None = 0,
    ) -> np.ndarray:
        """
        合成替代信号。

        Parameters
        ----------
        target_psd     : 目标功率谱密度（单边，长度 = n_fft//2+1）
        phase_features : 相位结构特征
        ref_before     : 前参考段（已清洗）
        ref_after      : 后参考段（已清洗）
        length         : 输出信号长度
        gap_samples    : 异常段与后参考段之间的安全间隙长度（样本数）。
                         趋势外推的时间轴必须计入该间隙，否则后侧参考的
                         时间轴会整体偏移 gap，导致外推出的右端目标电平
                         系统性偏低/偏高。``None`` 视为 0（向后兼容）。

        Returns
        -------
        替代时域信号
        """
        gap = _as_gap_samples(gap_samples)
        method = self.cfg.get("method", "iterative_fft")
        if method == "reference_overlap_add":
            shaped = self._reference_overlap_add(
                target_psd, ref_before, ref_after, length
            )
            # 同步完整后处理，避免 overlap-add 跳过边界/相位/趋势步骤
            shaped = self._apply_boundary_constraints(shaped, ref_before, ref_after)
            shaped = self._apply_phase_constrained_boundary(shaped, ref_before, ref_after)
            shaped = self._correct_edge_dc_drift(shaped, ref_before, ref_after)
            shaped = self._apply_trend_correction(
                shaped, ref_before, ref_after, gap_samples=gap
            )
            return shaped
        if method == "arma":
            shaped = self._arma_synthesis(ref_before, length)
            shaped = self._apply_boundary_constraints(shaped, ref_before, ref_after)
            shaped = self._apply_phase_constrained_boundary(shaped, ref_before, ref_after)
            shaped = self._correct_edge_dc_drift(shaped, ref_before, ref_after)
            shaped = self._apply_trend_correction(
                shaped, ref_before, ref_after, gap_samples=gap
            )
            return shaped
        return self._spectral_shaping(target_psd, ref_before, ref_after, length)

    def _reference_overlap_add(
        self,
        target_psd: np.ndarray,
        ref_before: np.ndarray,
        ref_after: np.ndarray,
        length: int,
    ) -> np.ndarray:
        """Build a natural long replacement from healthy reference blocks.

        A single random-phase IFFT has the right second-order spectrum but
        looks like broadband noise over a long interval.  Circularly sampled
        reference blocks preserve measured waveform texture, while each
        block's amplitude spectrum is corrected to the fused target PSD.
        Short crossfades remove block boundaries without restoring anomaly
        samples.
        """
        references = [
            np.asarray(ref, dtype=float)
            for ref in (ref_before, ref_after)
            if ref is not None and len(ref) >= 4
        ]
        if not references:
            return np.zeros(length, dtype=float)

        seed = self.cfg.get("random_seed")
        rng = np.random.default_rng(seed)
        target = np.concatenate(references)
        target_mean = float(np.mean(target))
        target_std = float(np.std(target))

        block_sec = float(self.cfg.get("reference_block_sec", 60.0))
        block_len = max(16, int(round(block_sec * self.sr)))
        block_len = min(block_len, max(len(ref) for ref in references))
        overlap_sec = float(self.cfg.get("reference_overlap_sec", 2.0))
        overlap = max(1, int(round(overlap_sec * self.sr)))
        overlap = min(overlap, max(1, block_len // 4))

        result = np.empty(max(0, length), dtype=float)
        written = 0
        first = True
        while written < length:
            reference = references[int(rng.integers(0, len(references)))]
            start = int(rng.integers(0, len(reference)))
            indices = (start + np.arange(block_len)) % len(reference)
            block = reference[indices]
            take = min(block_len, length - written)

            if first:
                result[:take] = block[:take]
                written = take
                first = False
                continue

            # Keep at least one newly written sample for a short final block.
            crossfade = min(overlap, written, max(0, take - 1))
            join_start = written - crossfade
            fade = np.linspace(0.0, 1.0, crossfade, endpoint=False)
            result[join_start:written] = (
                result[join_start:written] * (1.0 - fade)
                + block[:crossfade] * fade
            )
            remaining = take - crossfade
            if remaining > 0:
                result[written:written + remaining] = block[crossfade:take]
            written += remaining

        # Use the natural reference-block sequence only as a phase template,
        # then impose the fused target PSD over the complete replacement.
        # This keeps the spectral contract global instead of approximate per
        # block, while avoiding independent random phases at every block.
        n_fft = max(self._next_pow2(length * 2), (len(target_psd) - 1) * 2)
        template_fft = np.fft.rfft(result - np.mean(result), n=n_fft)
        template_freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)
        target_nfft = max(2, (len(target_psd) - 1) * 2)
        target_freqs = np.fft.rfftfreq(target_nfft, d=1.0 / self.sr)
        target_amp = np.sqrt(np.maximum(
            np.interp(
                template_freqs,
                target_freqs,
                np.asarray(target_psd, dtype=float),
                left=float(target_psd[0]),
                right=float(target_psd[-1]),
            ),
            0.0,
        ))
        if np.max(target_amp) > 1e-15:
            shaped_fft = target_amp * np.exp(1j * np.angle(template_fft))
            shaped_fft[0] = 0.0
            if n_fft % 2 == 0:
                shaped_fft[-1] = complex(float(np.real(shaped_fft[-1])), 0.0)
            result = np.fft.irfft(shaped_fft, n=n_fft)[:length]

        if target_std > 1e-12 and np.std(result) > 1e-12:
            result = (result - np.mean(result)) * (target_std / np.std(result))
        result += target_mean
        return result

    # ------------------------------------------------------------------
    # 频谱整形法
    # ------------------------------------------------------------------

    def _spectral_shaping(
        self,
        target_psd: np.ndarray,
        ref_before: np.ndarray,
        ref_after: np.ndarray,
        length: int,
    ) -> np.ndarray:
        """白噪声 → 频域整形 → 局部边界约束。"""
        n_fft = self._next_pow2(length * 2)
        target_amp = np.sqrt(np.maximum(target_psd[: n_fft // 2 + 1], 0.0))

        # Step 1: 频谱整形
        shaped = self._shape_noise(target_amp, n_fft, length)

        # Restore the physical baseline and fluctuation scale from healthy
        # references after spectral shaping of the demeaned signal.
        references = [ref for ref in (ref_before, ref_after) if ref is not None and len(ref)]
        if references:
            reference = np.concatenate(references).astype(float)
            target_mean = float(np.mean(reference))
            target_std = float(np.std(reference))
            shaped_std = float(np.std(shaped))
            if target_std > 1e-12 and shaped_std > 1e-12:
                shaped = (shaped - np.mean(shaped)) * (target_std / shaped_std)
            else:
                shaped = np.zeros_like(shaped)
            shaped += target_mean

        # Step 2: 局部边界约束（均值、RMS、一阶差分）
        shaped = self._apply_boundary_constraints(shaped, ref_before, ref_after)
        # 逐侧端部统计匹配（均值/RMS/差分尺度），与流水线文档 Step 2 对齐
        if ref_before is not None and len(ref_before):
            shaped = self._constrain_edge(
                shaped, ref_before,
                edge=max(1, min(int(0.5 * self.sr), len(shaped) // 4)),
                side="head",
            )
        if ref_after is not None and len(ref_after):
            shaped = self._constrain_edge(
                shaped, ref_after,
                edge=max(1, min(int(0.5 * self.sr), len(shaped) // 4)),
                side="tail",
            )

        # Step 3: 分频带 Hilbert 瞬时相位锁定（方案 B）
        shaped = self._apply_phase_constrained_boundary(shaped, ref_before, ref_after)

        # Step 4: 边缘直流漂移校正
        shaped = self._correct_edge_dc_drift(shaped, ref_before, ref_after)

        # Step 5: 时域趋势校正
        shaped = self._apply_trend_correction(shaped, ref_before, ref_after)

        return shaped

    def _shape_noise(
        self, target_amp: np.ndarray, n_fft: int, output_length: int
    ) -> np.ndarray:
        """白噪声 → 频域整形 → IFFT。"""
        n_half = len(target_amp)
        seed = self.cfg.get("random_seed")
        rng = np.random.default_rng(seed)
        random_phases = rng.uniform(-np.pi, np.pi, n_half)
        spectrum = target_amp * np.exp(1j * random_phases)
        # DC carries the baseline handled above; randomizing it creates a
        # large artificial offset. Nyquist must remain real for irfft.
        spectrum[0] = 0.0
        if n_fft % 2 == 0:
            spectrum[-1] = complex(float(np.real(spectrum[-1])), 0.0)
        signal = np.fft.irfft(spectrum, n=n_fft)
        return signal[:output_length]

    # ------------------------------------------------------------------
    # 局部边界约束
    # ------------------------------------------------------------------

    def _apply_boundary_constraints(
        self,
        signal: np.ndarray,
        ref_before: np.ndarray | None,
        ref_after: np.ndarray | None,
    ) -> np.ndarray:
        """
        对合成信号施加局部边界约束：
        - 匹配端部局部均值
        - 匹配端部局部 RMS
        - 匹配端部一阶差分的稳健尺度
        """
        result = signal.copy()
        n = len(result)
        references = [ref for ref in (ref_before, ref_after) if ref is not None and len(ref)]
        if not references:
            return result

        edge = max(1, min(int(0.5 * self.sr), n // 4))
        target_mean = float(np.mean(np.concatenate(references)))
        current_mean = float(np.mean(result))
        result += target_mean - current_mean

        # Match the healthy fluctuation scale without allowing an extreme
        # correction when a short reference is unstable.
        target_std = float(np.mean([np.std(ref) for ref in references]))
        current_std = float(np.std(result))
        if target_std > 1e-12 and current_std > 1e-12:
            scale = np.clip(target_std / current_std, 0.25, 4.0)
            result = (result - target_mean) * scale + target_mean

        # Correct only the low-order boundary trend; preserve the generated
        # spectral fluctuations instead of rebuilding them from raw diffs.
        left_error = 0.0
        right_error = 0.0
        if ref_before is not None and len(ref_before):
            left_error = float(np.mean(ref_before[-edge:]) - np.mean(result[:edge]))
        if ref_after is not None and len(ref_after):
            right_error = float(np.mean(ref_after[:edge]) - np.mean(result[-edge:]))
        if ref_before is not None and ref_after is not None:
            result += np.linspace(left_error, right_error, n)
        elif ref_before is not None:
            result += left_error
        else:
            result += right_error
        return result

    def _constrain_edge(
        self,
        signal: np.ndarray,
        reference: np.ndarray,
        edge: int,
        side: str,
    ) -> np.ndarray:
        """约束信号端部的均值、RMS 和差分尺度。"""
        result = signal.copy()
        n = len(result)

        # 参考段端部统计
        ref_tail = reference[-edge:] if side == "head" else reference[:edge]
        ref_mean = float(np.mean(ref_tail))
        ref_rms = float(np.sqrt(np.mean(ref_tail ** 2) + 1e-30))
        ref_diff = np.diff(ref_tail)
        ref_diff_scale = float(np.median(np.abs(ref_diff)) + 1e-30)

        # 信号端部
        if side == "head":
            sig_slice = result[:edge]
        else:
            sig_slice = result[-edge:]

        sig_mean = float(np.mean(sig_slice))
        sig_rms = float(np.sqrt(np.mean(sig_slice ** 2) + 1e-30))
        sig_diff = np.diff(sig_slice)
        sig_diff_scale = float(np.median(np.abs(sig_diff)) + 1e-30)

        # Step 1: 去均值
        corrected = sig_slice - sig_mean + ref_mean

        # Step 2: RMS 缩放（限制范围 0.25~4.0）
        if sig_rms > 1e-15:
            rms_scale = ref_rms / sig_rms
            rms_scale = np.clip(rms_scale, 0.25, 4.0)
            corrected = (corrected - ref_mean) * rms_scale + ref_mean

        # Step 3: 差分尺度约束（限制范围 0.25~4.0）
        if sig_diff_scale > 1e-15:
            diff_scale = ref_diff_scale / sig_diff_scale
            diff_scale = np.clip(diff_scale, 0.25, 4.0)
            # 通过累积差分重建，保持均值不变
            diff_corrected = np.diff(corrected) * diff_scale
            corrected = np.concatenate([[corrected[0]], np.cumsum(diff_corrected) + corrected[0]])
            # 重新对齐均值
            corrected = corrected - np.mean(corrected) + ref_mean

        if side == "head":
            result[:edge] = corrected
        else:
            result[-edge:] = corrected

        return result

    # ------------------------------------------------------------------
    # ARMA 备选
    # ------------------------------------------------------------------

    def _arma_synthesis(self, reference: np.ndarray, length: int) -> np.ndarray:
        try:
            from scipy.linalg import solve_toeplitz
            order = min(20, len(reference) // 4)
            if order < 1:
                return np.random.normal(0, np.std(reference) + 1e-15, length)
            acf = np.correlate(reference, reference, mode="full")
            acf = acf[len(acf) // 2:]
            acf = acf / (acf[0] + 1e-15)
            r = acf[:order + 1]
            ar_coeffs = solve_toeplitz(r[:order], r[1:order + 1])
            noise = np.random.normal(0, np.std(reference), length + order)
            output = np.zeros(length + order)
            for i in range(order, len(output)):
                output[i] = noise[i] - np.dot(ar_coeffs, output[i - order:i][::-1])
            return output[order:].copy()
        except Exception as exc:
            logger.warning("ARMA 合成失败: %s", exc)
            return np.random.normal(0, np.std(reference) + 1e-15, length)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hermitian(spectrum_half: np.ndarray, length: int) -> np.ndarray:
        n_half = len(spectrum_half)
        full = np.zeros(length, dtype=complex)
        full[:n_half] = spectrum_half
        n_neg = min(n_half - 2, length // 2 - 1)
        if n_neg > 0:
            pos_part = spectrum_half[1:1 + n_neg]
            full[length // 2 + 1:length // 2 + 1 + n_neg] = np.conj(pos_part[::-1])
        return full

    @staticmethod
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p <<= 1
        return p

    # ------------------------------------------------------------------
    # 分频带 Hilbert 瞬时相位锁定（方案 B）
    # ------------------------------------------------------------------

    def _apply_phase_constrained_boundary(
        self,
        signal: np.ndarray,
        ref_before: np.ndarray | None,
        ref_after: np.ndarray | None,
    ) -> np.ndarray:
        """
        时域 Hilbert 相位先验约束：
        1. 对合成信号做 Hilbert 变换，得到解析信号（振幅+相位）
        2. 提取两侧参考段的瞬时相位趋势
        3. 将两侧相位趋势外推到异常段内部，形成贯穿整个异常段的相位先验
        4. 保持合成信号的振幅包络，将相位替换为先验相位
        """
        from scipy.signal import hilbert

        n = len(signal)
        if n < 10:
            return signal

        # 去均值后再做 Hilbert 变换（直流分量会污染解析信号计算）
        signal_mean = float(np.mean(signal))
        signal_centered = signal - signal_mean

        # 合成信号的解析信号
        analytic_synth = hilbert(signal_centered)
        env_synth = np.abs(analytic_synth)  # 保持振幅包络
        phase_synth = np.unwrap(np.angle(analytic_synth))  # unwrap 后的相位

        # 提取两侧参考段的相位
        phase_before = None
        phase_after = None

        if ref_before is not None and len(ref_before) >= 10:
            ref_before_centered = ref_before - float(np.mean(ref_before))
            analytic_ref_before = hilbert(ref_before_centered)
            phase_before = np.unwrap(np.angle(analytic_ref_before))

        if ref_after is not None and len(ref_after) >= 10:
            ref_after_centered = ref_after - float(np.mean(ref_after))
            analytic_ref_after = hilbert(ref_after_centered)
            phase_after = np.unwrap(np.angle(analytic_ref_after))

        if phase_before is None and phase_after is None:
            return signal

        # 构造贯穿整个异常段的目标相位
        target_phase = phase_synth.copy()

        # 左侧：用参考段的末端相位斜率外推到合成段左端
        if phase_before is not None:
            # 参考段末端的相位斜率（每样本）
            n_ref = len(phase_before)
            # 用最后 10% 或至少 5 个点估计斜率
            slope_window = max(5, n_ref // 10)
            # The last W samples span W-1 intervals, so the per-sample slope
            # must divide by W-1.  (Single source of truth: dssrr/core.py.)
            slope_before = (phase_before[-1] - phase_before[-slope_window]) / max(1, slope_window - 1)

            # 合成段左端的当前相位
            current_left = phase_synth[0]
            # 参考段末端的相位
            ref_end_phase = phase_before[-1]

            # 计算需要的相位偏移：让合成段左端的相位与参考段末端连续
            # 但考虑斜率：如果参考段在上升，合成段左端也应该继续上升
            offset_left = ref_end_phase - current_left
            # 将偏移 wrap 到 [-pi, pi]
            offset_left = (offset_left + np.pi) % (2 * np.pi) - np.pi

            # 应用偏移到整个合成段
            target_phase = target_phase + offset_left

            # 然后调整斜率：让合成段左端的相位斜率接近参考段的斜率
            # 当前合成段的平均斜率
            # n samples span n-1 intervals.
            current_slope = (target_phase[-1] - target_phase[0]) / max(1, n - 1)
            # 混合斜率：从左端的 slope_before 渐变到右端的 current_slope
            # 这里简单处理：只调整左端 20% 的斜率
            n_left = n // 5
            t_left = np.linspace(0, 1, n_left)
            # 左端的目标斜率是 slope_before，右端保持 current_slope
            target_slope_left = slope_before
            # 计算需要额外添加的斜率差
            slope_diff_left = target_slope_left - current_slope
            # 在左端 20% 区域添加渐变的斜率差
            extra_slope_left = slope_diff_left * (1 - t_left)
            # 积分得到相位偏移
            extra_phase_left = np.concatenate([[0], np.cumsum(extra_slope_left)])
            target_phase[:n_left] += extra_phase_left[:n_left]

        # 右侧：用参考段的起始相位斜率外推到合成段右端
        if phase_after is not None:
            n_ref = len(phase_after)
            slope_window = max(5, n_ref // 10)
            # Use the same convention as the left window: first W samples span
            # W-1 intervals, and clamp the window to the available length.
            w_after = min(slope_window, max(1, n_ref - 1))
            slope_after = (phase_after[w_after] - phase_after[0]) / max(1, w_after)

            # 合成段右端的当前相位
            current_right = target_phase[-1]
            # 参考段起始的相位
            ref_start_phase = phase_after[0]

            # 计算需要的相位偏移：让合成段右端的相位与参考段起始连续
            offset_right = ref_start_phase - current_right
            offset_right = (offset_right + np.pi) % (2 * np.pi) - np.pi

            # 应用偏移到右端 20% 区域
            n_right = n // 5
            t_right = np.linspace(0, 1, n_right)
            # 从 0 渐变到 offset_right
            right_offset = offset_right * t_right
            target_phase[-n_right:] += right_offset

            # 调整右端的斜率
            current_slope = (target_phase[-1] - target_phase[-n_right-1]) / n_right
            slope_diff_right = slope_after - current_slope
            extra_slope_right = slope_diff_right * t_right
            extra_phase_right = np.concatenate([[0], np.cumsum(extra_slope_right)])
            target_phase[-n_right:] += extra_phase_right[:n_right]

        # 用调整后的相位和原始振幅重构信号，加回均值
        result = env_synth * np.exp(1j * target_phase)
        result = np.real(result) + signal_mean

        return result


    def _correct_edge_dc_drift(
        self,
        signal: np.ndarray,
        ref_before: np.ndarray | None,
        ref_after: np.ndarray | None,
    ) -> np.ndarray:
        """
        相位锁定后的边缘直流漂移校正。

        Hilbert 相位偏移会改变小波系数的实部均值，导致信号两端
        整体水平偏移。本方法将 replacement 两端的局部均值校正到
        与参考段边缘均值一致，用平滑过渡避免引入新跳变。
        """
        result = signal.copy()
        n = len(result)

        # 边缘校正长度：取 10s 数据
        edge_sec = min(10.0, n / self.sr / 6)
        edge = max(1, int(edge_sec * self.sr))
        edge = min(edge, n // 4)

        # ---- 左端直流校正 ----
        if ref_before is not None and len(ref_before) >= edge and edge > 0:
            ref_left_mean = float(np.mean(ref_before[-edge:]))
            rep_left_mean = float(np.mean(result[:edge]))
            drift = ref_left_mean - rep_left_mean

            if abs(drift) > 1e-6:
                # 从左端 drift 渐变到 0（edge 处），用 smoothstep 避免二次跳变
                t = np.linspace(0.0, 1.0, edge)
                smooth = t * t * (3.0 - 2.0 * t)  # smoothstep
                correction = drift * (1.0 - smooth)
                result[:edge] += correction

        # ---- 右端直流校正 ----
        if ref_after is not None and len(ref_after) >= edge and edge > 0:
            ref_right_mean = float(np.mean(ref_after[:edge]))
            rep_right_mean = float(np.mean(result[-edge:]))
            drift = ref_right_mean - rep_right_mean

            if abs(drift) > 1e-6:
                t = np.linspace(0.0, 1.0, edge)
                smooth = t * t * (3.0 - 2.0 * t)
                # Weight ramps UP toward the right join: full drift at the last
                # sample (where it must match R_a), fading to 0 inside.
                # (The mirrored weight drift*(1-smooth) would put the whole
                # offset inside the segment and leave a residual step at the
                # join; fixed to match the canonical dssrr/core.py pipeline.)
                correction = drift * smooth
                result[-edge:] += correction

        return result


    def _apply_trend_correction(
        self,
        signal: np.ndarray,
        ref_before: np.ndarray | None,
        ref_after: np.ndarray | None,
        gap_samples: int | None = 0,
    ) -> np.ndarray:
        """
        时域趋势校正：从两侧参考段外推趋势线到异常段内。

        月震数据在长时程内有缓慢变化，紧邻参考段均值不能代表
        整段缺失的真值均值。本方法用整个参考段拟合线性趋势，
        外推到异常段两端，叠加平滑的水平校正。
        """
        result = signal.copy()
        n = len(result)
        if n < 10:
            return result

        n_before = len(ref_before) if ref_before is not None else 0
        n_after = len(ref_after) if ref_after is not None else 0

        # 拟合两侧参考段的线性趋势
        # ref_before: 时间轴从 -n_before 到 -1（异常段前）
        # ref_after: 时间轴从 n 到 n+n_after-1（异常段后）
        slope_left = 0.0
        intercept_left = 0.0
        slope_right = 0.0
        intercept_right = 0.0

        if ref_before is not None and n_before >= 10:
            t = np.arange(n_before, dtype=float)
            t_rel = t - n_before + 1  # 从 -n_before+1 到 0
            slope_left, intercept_left = np.polyfit(t_rel, ref_before, 1)
            slope_left = float(slope_left)
            intercept_left = float(intercept_left)

        if ref_after is not None and n_after >= 10:
            t = np.arange(n_after, dtype=float)
            # 异常段占 t=0..n-1，后参考段第一个样本位于 t=n+gap。
            # 时间轴必须计入安全间隙，否则整条右侧趋势线会水平偏移 gap，
            # 外推到 t=n-1 的目标电平出现系统性偏差。
            t_rel = t + n + _as_gap_samples(gap_samples)
            slope_right, intercept_right = np.polyfit(t_rel, ref_after, 1)
            slope_right = float(slope_right)
            intercept_right = float(intercept_right)

        # 计算异常段两端的目标均值
        target_left = None
        target_right = None

        if ref_before is not None and n_before >= 10:
            target_left = intercept_left + slope_left * 0  # t=0（异常段开始）

        if ref_after is not None and n_after >= 10:
            target_right = intercept_right + slope_right * (n - 1)  # t=n-1（异常段结束）

        # 如果只有一侧，用那一侧
        if target_left is None and target_right is not None:
            target_left = target_right
        if target_right is None and target_left is not None:
            target_right = target_left

        if target_left is None or target_right is None:
            return result

        # 当前信号两端均值
        edge_len = min(30, n // 4)
        current_left = float(np.mean(result[:edge_len]))
        current_right = float(np.mean(result[-edge_len:]))

        # 需要校正的量
        delta_left = target_left - current_left
        delta_right = target_right - current_right

        # 构造从 delta_left 到 delta_right 的线性趋势校正
        t = np.linspace(0.0, 1.0, n)
        correction = delta_left + (delta_right - delta_left) * t

        result += correction
        return result
