"""
基线方法统一接口 (R1)

- 复用仓库已有对比方法（线性/样条/小波/AR）
- FFT 低通频域插值修复（与 E2r_v5_batch.FFTInterpolation 逐行一致，论文中称 "FFT"）
- SSA 与 1D U-Net 直接委托给 ``seisy.core.anomaly_repair.comparison_methods``
  （即论文 Table 1 / Table S1 / 图 6 / 图 8 实际使用的同一份实现），
  本文件不再保留第二套实现 —— 这是 2026-10-10 基线审计后的统一动作。
- DSSRR 包装为统一 .repair(data, start, end) -> repaired
"""

from __future__ import annotations

import os

import numpy as np

from seisy.core.anomaly_repair.comparison_methods import (
    LinearInterpolation,
    CubicSplineInterpolation,
    WaveletRepair,
    SSARepair as _SeisySSARepair,
    UNetRepair as _SeisyUNetRepair,
)
from seisy.core.anomaly_repair.reference_spectrum import ReferenceSpectrumReplacer


class RobustARRepair:
    """Burg 算法 AR 前向后向预测填充（模型驱动预测基线）。

    Burg 比 Yule-Walker/lstsq 对短段更稳定；仍可能对近周期信号发散，
    发散实例按"失败率"如实计入统计，不参与指标聚合。
    """

    def __init__(self, sr: float = 6.625, order: int = 8, fit_len: int = 2000):
        self.sr = sr
        self.order = order
        self.fit_len = fit_len

    @staticmethod
    def _burg(x: np.ndarray, order: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x)
        n = len(x)
        order = min(order, n // 3)
        if order < 1:
            return np.array([0.0])
        ef = x.copy()
        eb = x.copy()
        a = np.zeros(order + 1)
        a[0] = 1.0
        for i in range(1, order + 1):
            ef_i = ef[i:]
            eb_im = eb[:-i]
            den = np.dot(ef_i, ef_i) + np.dot(eb_im, eb_im)
            num = 2.0 * np.dot(ef_i, eb_im)
            k = float(np.clip(num / (den + 1e-15), -0.99, 0.99))
            a_new = a.copy()
            for j in range(1, i):
                a_new[j] = a[j] - k * a[i - j]
            a_new[i] = k
            a = a_new
            ef_new = ef_i - k * eb_im
            eb_new = eb_im - k * ef_i
            ef[i:] = ef_new
            eb[:-i] = eb_new
        return a[1:]

    @staticmethod
    def _predict(a: np.ndarray, hist: np.ndarray, m: int) -> np.ndarray:
        hist = np.asarray(hist, dtype=float).copy()
        order = len(a)
        if len(hist) < order:
            hist = np.concatenate([np.zeros(order - len(hist)), hist])
        out = np.empty(m)
        for i in range(m):
            v = np.dot(a, hist[-order:][::-1])
            out[i] = v
            hist = np.append(hist, v)
        return out

    def repair(self, data: np.ndarray, start: int, end: int) -> np.ndarray:
        x = np.asarray(data, dtype=float).copy()
        n = len(x)
        order = min(self.order, max(2, n // 50))
        m = end - start + 1
        pre = x[max(0, start - self.fit_len):start]
        post = x[end + 1:min(n, end + 1 + self.fit_len)]

        def _safe(seg):
            if seg is None or len(seg) < 4 * order:
                return None
            try:
                a = self._burg(seg, order)
                p = self._predict(a, seg[-order:], m)
                scale = np.std(seg) + 1e-12
                if not np.all(np.isfinite(p)) or np.max(np.abs(p)) > 100 * scale:
                    return None
                return p
            except Exception:
                return None

        pred_f = _safe(pre)
        pred_b = _safe(post[::-1])

        if pred_f is None and pred_b is None:
            v0 = x[max(0, start - 1)]
            v1 = x[min(n - 1, end + 1)]
            x[start:end + 1] = np.linspace(v0, v1, m)
            return x
        if pred_f is None:
            pred_f = pred_b[::-1]
        if pred_b is None:
            pred_b = pred_f[::-1]
        pred_b = pred_b[::-1]

        m_f = np.mean(pre) if len(pre) else 0.0
        m_b = np.mean(post) if len(post) else m_f
        w = np.linspace(1.0, 0.0, m)
        x[start:end + 1] = w * (pred_f + m_f) + (1 - w) * (pred_b + m_b)
        return x


class FFTInterpolation:
    """FFT 低通插值修复（"频域修复"一类方法的代表）。

    实现与 E2r_v5_batch.py 中用于 Table 1 的 ``FFTInterpolation`` **逐行一致**：
    把已知样本拼接后做实数 FFT，只保留最低 1/10 的频段（等效低通），
    再变换回全长时间轴，并用缺口左右各 100 个样本（≈15 s @6.625 Hz）的
    均值把填补段对齐到局部直流基线。

    不使用多锥（multitaper）估计，也不显式保持频谱统计特性。
    ``sr`` / ``nperseg`` 仅为兼容旧调用签名保留，不参与计算。
    """

    def __init__(self, sr: float = 6.625, nperseg: int = 256):
        # 保留参数以兼容 baselines.METHODS 的 lambda sr: ... 调用约定；
        # 该实现本身与采样率无关（边界窗固定 100 样本）。
        self.sr = sr
        self.nperseg = nperseg

    def repair(self, data: np.ndarray, start: int, end: int) -> np.ndarray:
        repaired = data.copy()
        n = len(data)
        known = np.concatenate([data[:start], data[end + 1:]])
        fft_known = np.fft.rfft(known)
        n_low = max(1, len(fft_known) // 10)
        fft_smooth = np.zeros_like(fft_known)
        fft_smooth[:n_low] = fft_known[:n_low]
        smooth_full = np.fft.irfft(fft_smooth, n=n)
        left_mean = np.mean(data[max(0, start - 100):start])
        right_mean = np.mean(data[end + 1:min(n, end + 101)])
        target_mean = (left_mean + right_mean) / 2
        smooth_mean = np.mean(smooth_full[start:end + 1])
        dc_offset = target_mean - smooth_mean
        repaired[start:end + 1] = smooth_full[start:end + 1] + dc_offset
        return repaired


# 向后兼容别名：旧脚本/旧结果以 "STFTRepair" 命名该方法，
# 现统一为与 E2r_v5_batch.FFTInterpolation 相同的实现（论文中统称 "FFT"）。
STFTRepair = FFTInterpolation


class SSARepair:
    """奇异谱分析（SSA）单道缺口迭代填充（模型驱动基线）。

    **实现已统一（2026-10-10）**：本类现在是
    ``seisy.core.anomaly_repair.comparison_methods.SSARepair`` 的薄包装。
    论文 Table 1、Table S1、图 5、图 6、图 8 因此使用同一份 SSA 代码，
    不再存在"两套实现给出不同结果"的隐患。

    构造签名保持向后兼容：``ctx_sec`` / ``energy_frac`` / ``max_rank`` 是
    旧版遗留参数，新实现按缺口长度自适应（不接收这些旋钮），故保留但不再生效；
    ``n_iter`` 直接映射到新实现的 ``max_iter``，默认取 30 以与论文一致
    （``E2r_v5_batch.py`` / ``E9_dl_synthetic.py`` / ``fig08_earth_transfer.py``
    均为 ``SSARepair(max_iter=30)``）。

    依据：Kondrashov & Ghil (2006, Nonlin. Processes Geophys.) 的单变量
    SSA gap-filling：缺失值先用线性插值初始化，嵌入轨迹矩阵 -> 低秩重建
    -> 对角平均 -> 仅更新缺口，迭代至收敛。
    """

    def __init__(self, sr: float = 6.625, ctx_sec: float = 180.0,
                 n_iter: int = 30, energy_frac: float = 0.9,
                 max_rank: int = 80):
        self.sr = sr
        self.ctx_sec = ctx_sec            # 遗留参数：新实现不使用
        self.n_iter = n_iter
        self.energy_frac = energy_frac    # 遗留参数：新实现不使用
        self.max_rank = max_rank          # 遗留参数：新实现不使用
        self._impl = _SeisySSARepair(max_iter=n_iter)

    def repair(self, data: np.ndarray, start: int, end: int) -> np.ndarray:
        return self._impl.repair(data, start, end)


_UNET_CACHE: dict = {}


class UNetRepair:
    """1D U-Net 数据驱动基线（自监督掩码重建，P0-B）。

    **实现已统一（2026-10-10）**：本类现在是
    ``seisy.core.anomaly_repair.comparison_methods.UNetRepair`` 的薄包装，
    与论文 Table 1、Table S1、图 6、图 8 使用同一份代码和同一份预训练权重
    （``experiments/models/unet_synth.pt``）。

    构造签名保持向后兼容：``sr`` / ``ctx_sec`` 是旧版遗留参数，新实现的
    上下文长度由 ``patch_length``（默认 1536，实际以权重 checkpoint 内的
    ``patch`` 配置为准）与缺口长度共同决定，故保留但不再生效。
    """

    def __init__(self, sr: float = 6.625, model_path: str | None = None,
                 ctx_sec: float = 120.0, patch_length: int = 1536,
                 n_epochs: int = 10):
        self.sr = sr
        self.ctx_sec = ctx_sec            # 遗留参数：新实现不使用
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "models", "unet_synth.pt")
        self.model_path = model_path
        self.patch_length = patch_length
        self.n_epochs = n_epochs
        self._impl = _SeisyUNetRepair(
            patch_length=patch_length, n_epochs=n_epochs,
            model_path=model_path)

    def repair(self, data: np.ndarray, start: int, end: int) -> np.ndarray:
        return self._impl.repair(data, start, end)


class DSSRRRepair:
    """DSSRR 统一包装。

    默认参考段长度遵循论文协议：``L = clip(anomaly_duration / 2, 120 s, 600 s)``
    （与 ``E2r_v5_batch.py`` 主实验一致）。传显式 ``before_sec`` / ``after_sec``
    可覆盖为固定长度（仅用于对照实验）。

    ``quantize`` 默认 **True**：Apollo 原始数据是整数 counts，论文协议规定
    DSSRR 的输出取整（正文："DSSRR stays at 0.370 because its output is already
    integral"）。2026-10-05 `ReferenceSpectrumReplacer.replace()` 把原先的
    **无条件取整** 改成了 ``quantize`` 开关（默认 False），本包装器若不显式
    开启就会静默改变长度扫描的 DSSRR 结果——实测加 ``quantize=True`` 后
    80/80 个长度扫描案例与 10-04 归档逐点一致（最大差 0.0000），不加则差 0.5。
    处理浮点物理量（如 10 Hz 地球波形）时应显式传 ``quantize=False``。
    """

    #: 论文默认参考段下限 / 上限（秒）
    REF_MIN_SEC = 120.0
    REF_MAX_SEC = 600.0

    def __init__(self, sr: float, before_sec: float | None = None,
                 after_sec: float | None = None, seed: int = 42,
                 reference_gap_sec: float = 1.0,
                 quantize: bool = True):
        self.sr = sr
        self.before_sec = before_sec
        self.after_sec = after_sec
        self.seed = seed
        self.reference_gap_sec = reference_gap_sec
        self.quantize = quantize
        self.replacer = ReferenceSpectrumReplacer(sr=sr)

    def _ref_len(self, start: int, end: int) -> float:
        if self.before_sec is not None:
            return float(self.before_sec)
        anom_sec = (end - start + 1) / self.sr
        return float(np.clip(anom_sec / 2.0, self.REF_MIN_SEC, self.REF_MAX_SEC))

    def repair(self, data: np.ndarray, start: int, end: int) -> np.ndarray:
        ref_len = self._ref_len(start, end)
        after_len = self.after_sec if self.after_sec is not None else ref_len
        repaired, _ = self.replacer.replace(
            data, start, end,
            reference_before_sec=ref_len,
            reference_after_sec=after_len,
            reference_gap_sec=self.reference_gap_sec,
            random_seed=self.seed,
            quantize=self.quantize,
        )
        return repaired


METHODS = {
    "linear": lambda sr: LinearInterpolation(),
    "spline": lambda sr: CubicSplineInterpolation(),
    "wavelet": lambda sr: WaveletRepair("db4", 4),
    "ar": lambda sr: RobustARRepair(sr=sr),
    # key 仍叫 "stft"（历史命名），实现已统一为 FFT 低通插值；
    # 论文与图注中该方法一律标注为 "FFT"。
    "stft": lambda sr: FFTInterpolation(sr=sr),
    "ssa": lambda sr: SSARepair(sr=sr),
    "unet": lambda sr: UNetRepair(sr=sr),
    "dssrr": lambda sr: DSSRRRepair(sr=sr),
}

METHOD_ORDER = ["linear", "spline", "wavelet", "ar", "stft", "ssa", "unet", "dssrr"]


def make_methods(sr: float) -> dict:
    return {name: METHODS[name](sr) for name in METHOD_ORDER}
