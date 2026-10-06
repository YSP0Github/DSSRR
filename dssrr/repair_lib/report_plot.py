"""
异常修复对比图导出模块
======================
把一次修复的前后波形、频谱和质量指标画成一张可直接放进报告/论文的对比图。

:class:`RepairReportPlotter` 生成的是一个 matplotlib ``Figure``，包含：

- **时域对比**：原始波形、修复后波形、残差，异常段用底色标出，
  前后参考段单独着色。
- **频域对比**：修复段与目标 PSD 的叠加对比（对数谱 + 可选线性谱），
  用于确认重建谱形是否贴合参考。
- **质量指标**：把 :class:`~dssrr.verifier.QualityVerifier` 的
  校验结果（谱一致性、边界连续性等）以通过/失败色块的形式呈现。

既可直接 ``savefig`` 出图，也可嵌入 Qt 对话框做实时预览（见
:mod:`dssrr.gui.export_report_dialog`）。

用法::

    from dssrr.repair_lib.report_plot import RepairReportPlotter

    plotter = RepairReportPlotter(sr=6.625)          # 默认英文
    fig = plotter.plot_report(
        original=original_data,
        repaired=repaired_data,
        anomaly_start=1000,
        anomaly_end=2000,
        ref_before=ref_before,
        ref_after=ref_after,
        verification=verification_report,
        save_path="repair_report.png",
    )

依赖：matplotlib、scipy、numpy。GUI 需要交互式后端，因此本模块在**用户未显式
指定后端**（``MPLBACKEND``）时会尝试切到 ``Qt5Agg``；在无 Qt / 无显示的机器上
（CI、服务器）会自动保留 matplotlib 默认后端（通常为 ``Agg``），不会报错。
中文标签默认使用 ``Microsoft YaHei`` / ``SimHei``，缺失时回退 ``DejaVu Sans``；
字体可在导出对话框里通过图文字批编辑器进一步调整。

报告语言由 :func:`dssrr.i18n.resolve_language` 决定，**默认英文**（与界面一致）；
需要中文报告时设环境变量 ``DSSRR_LANG=zh``，或给构造函数显式传 ``lang="zh"``。
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib

from ..i18n import resolve_language

# GUI 内嵌需要 Qt5Agg，但库本身必须能在 headless 机器上干净导入：
# 仅当用户没有用 MPLBACKEND 显式指定后端时才尝试切换，且 Qt 不可用时静默跳过。
if not os.environ.get("MPLBACKEND"):
    try:
        matplotlib.use("Qt5Agg")
    except Exception:  # noqa: BLE001 - backend availability is environment-specific
        pass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import hilbert

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class RepairReportPlotter:
    """
    异常修复对比图导出器。

    生成包含时域对比、频域对比和质量指标的综合报告图。

    使用示例::

        plotter = RepairReportPlotter(sr=6.625, lang="zh")
        fig = plotter.plot_report(
            original=original_data,
            repaired=repaired_data,
            anomaly_start=1000,
            anomaly_end=2000,
            ref_before=ref_before,
            ref_after=ref_after,
            verification=verification_report,
            save_path="repair_report.png"
        )
    """

    # 颜色方案
    COLORS = {
        "original": "#2c3e50",       # 深灰 - 原始信号
        "repaired": "#e74c3c",       # 红色 - 修复信号
        "residual": "#7f8c8d",       # 灰色 - 残差
        "anomaly_bg": "#ffebee",     # 浅红背景 - 异常区域
        "ref_before": "#3498db",     # 蓝色 - 前参考段
        "ref_after": "#27ae60",      # 绿色 - 后参考段
        "target_psd": "#34495e",     # 深灰虚线 - 目标PSD
        "pass": "#27ae60",           # 绿色 - 通过
        "fail": "#e74c3c",           # 红色 - 失败
        "warning": "#f39c12",        # 黄色 - 警告
        "grid": "#ecf0f1",           # 浅灰网格
    }

    # 国际化文本
    I18N = {
        "zh": {
            "title": "地震数据异常修复报告",
            "time_domain": "(a) 时域对比",
            "spectral_psd": "(b) 归一化功率谱密度对比",
            "spectral_linear": "(c) 归一化频谱对比 (线性)",
            "original": "原始信号",
            "repaired": "修复信号",
            "anomaly_region": "异常区域",
            "anomaly_regions": "异常区段",
            "ref_before": "前参考段",
            "ref_after": "后参考段",
            "inset_title": "异常区域放大",
            "inset_region": "异常 {index}",
            "inset_overflow": "已显示前 {shown} / {total} 处异常",
            "time_axis": "时间 (s)",
            "amplitude": "振幅",
            "freq_axis": "频率 (Hz)",
            "psd": "归一化 PSD",
            "magnitude": "归一化幅度",
            "anomaly_segment": "异常段",
            "samples": "samples",
        },
        "en": {
            "title": "Seismic Data Anomaly Repair Report",
            "time_domain": "(a) Time Domain",
            "spectral_psd": "(b) Normalized Power Spectral Density",
            "spectral_linear": "(c) Normalized Spectrum (Linear)",
            "original": "Original",
            "repaired": "Repaired",
            "anomaly_region": "Anomaly Region",
            "anomaly_regions": "Anomaly Regions",
            "ref_before": "Ref Before",
            "ref_after": "Ref After",
            "inset_title": "Anomaly Zoom",
            "inset_region": "Anomaly {index}",
            "inset_overflow": "Showing the first {shown} of {total} anomalies",
            "time_axis": "Time (s)",
            "amplitude": "Amplitude",
            "freq_axis": "Frequency (Hz)",
            "psd": "Normalized PSD",
            "magnitude": "Normalized Amplitude",
            "anomaly_segment": "Anomaly Segment",
            "samples": "samples",
        },
    }

    def __init__(self, sr: float, timezone_name: str = "UTC",
                 lang: Optional[str] = None):
        """
        Parameters
        ----------
        sr : 采样率 (Hz)
        timezone_name : 时区名称
        lang : 语言代码（``"zh"`` / ``"en"``）。``None`` = 按
            :func:`dssrr.i18n.resolve_language` 解析（默认**英文**，
            可用环境变量 ``DSSRR_LANG=zh`` 改成中文）。
        """
        self.sr = sr
        self.timezone_name = timezone_name
        self.lang = resolve_language(lang)
        self._t = self.I18N[self.lang]

    def plot_report(
        self,
        original: np.ndarray,
        repaired: np.ndarray,
        anomaly_start: int,
        anomaly_end: int,
        anomaly_regions: Optional[List[Tuple[int, int]]] = None,
        ref_before: Optional[np.ndarray] = None,
        ref_after: Optional[np.ndarray] = None,
        verification: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 11),
        dpi: int = 150,
        format: str = "png",
        custom_title: Optional[str] = None,
        freq_range: Optional[Tuple[float, float]] = None,
        show_time_domain: bool = True,
        show_psd: bool = True,
        show_linear_spectrum: bool = True,
        show_quality_metrics: bool = True,
        show_reference_segments: bool = True,
        show_inset_zoom: bool = True,
        fig: Optional[Figure] = None,
    ) -> Figure:
        """
        生成修复报告对比图。

        Parameters
        ----------
        original : 原始波形 (1-D)
        repaired : 修复后波形 (1-D)
        anomaly_start : 异常起始采样点 (闭区间)
        anomaly_end : 异常结束采样点 (闭区间)
        anomaly_regions : 本次修复的全部异常区间。为空时使用单个异常区间。
        ref_before : 前参考段 (可选)
        ref_after : 后参考段 (可选)
        verification : 质量验证报告 (from QualityVerifier.verify())
        metadata : 元数据 {filename, station, channel, starttime, ...}
        save_path : 保存路径 (None 则不保存)
        figsize : 图片尺寸 (width, height)
        dpi : 分辨率
        format : 保存格式 (png, pdf, svg)
        custom_title : 自定义标题 (None 则使用默认标题)
        freq_range : 频率范围 (min_freq, max_freq)，None 则自动
        show_time_domain : 是否显示时域对比图
        show_psd : 是否显示功率谱密度图
        show_linear_spectrum : 是否显示线性频谱图
        show_quality_metrics : 是否显示质量指标表格
        show_reference_segments : 是否显示参考段
        show_inset_zoom : 是否显示放大插图

        Returns
        -------
        fig : matplotlib Figure 对象
        """
        # 验证输入
        original = np.asarray(original, dtype=float)
        repaired = np.asarray(repaired, dtype=float)
        if original.shape != repaired.shape:
            raise ValueError("original 和 repaired 长度必须相同")

        anomaly_regions = self._normalize_anomaly_regions(
            anomaly_regions, anomaly_start, anomaly_end, len(original)
        )

        # 创建时间轴
        time_axis = np.arange(len(original)) / self.sr

        # 确定要显示的子图
        has_spectral = show_psd or show_linear_spectrum
        has_quality = show_quality_metrics and verification is not None

        # 创建图：传入 fig 时复用（供交互式预览画布），否则新建
        if fig is None:
            fig = plt.figure(figsize=figsize, facecolor="white")
        else:
            fig.clear()
            fig.set_size_inches(float(figsize[0]), float(figsize[1]), forward=False)
            fig.set_facecolor("white")

        # 使用 constrained_layout：自动为每个子图标题、轴标签预留空间，
        # 在任何画布宽高比下都不会出现标题与横轴文字重叠（预览窄幅时尤其关键）。
        try:
            fig.set_layout_engine("constrained")
        except AttributeError:  # 兼容较旧 matplotlib
            fig.set_constrained_layout(True)

        # 根据显示选项调整布局：上下分布，每个频域图只占一行
        # （修复前/修复后两条曲线画在同一子图内对比）。
        # 行 0: 时域对比 (a)
        # 行 1..N: 频域对比（PSD (b) + 线性 (c)），每个一图
        # 最后一行: 质量指标 / 图例
        n_spec_rows = int(bool(show_psd)) + int(bool(show_linear_spectrum))
        n_rows = (1 if show_time_domain else 0) + n_spec_rows + (1 if has_quality else 0)
        n_rows = max(n_rows, 1)
        # 高度权重：时域(波形主角)：PSD：线性 = 3:2:2，底部图例最小
        height_ratios = []
        if show_time_domain:
            height_ratios.append(3.0)
        if show_psd:
            height_ratios.append(2.0)
        if show_linear_spectrum:
            height_ratios.append(2.0)
        if has_quality:
            height_ratios.append(0.6)
        if not height_ratios:
            height_ratios = [1.0]
        # 上下分布：单列；constrained_layout 负责标签间距，hspace 只作行距微调。
        gs = GridSpec(n_rows, 1, figure=fig,
                      height_ratios=height_ratios,
                      hspace=0.35, wspace=0.35)

        # 标题
        if custom_title:
            title = custom_title
        else:
            title = self._create_title(metadata, anomaly_regions)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        original_spectrum_data = original
        repair_spectrum_data = repaired

        row = 0

        # ── 图1: 时域对比 (顶部一行) ──
        if show_time_domain:
            ax_time = fig.add_subplot(gs[row, 0])
            self._plot_time_domain(
                ax_time, original, repaired, anomaly_regions,
                time_axis, ref_before, ref_after,
                show_reference_segments=show_reference_segments,
                show_inset_zoom=show_inset_zoom
            )
            row += 1

        # ── 图2: PSD 对比（修复前/修复后同图） ──
        if show_psd:
            ax_psd = fig.add_subplot(gs[row, 0])
            self._plot_spectral(
                ax_psd, ref_before, ref_after,
                original_spectrum_data, repair_spectrum_data,
                freq_range=freq_range,
                show_reference_segments=show_reference_segments
            )
            row += 1

        # ── 图3: 线性频谱对比（修复前/修复后同图） ──
        if show_linear_spectrum:
            ax_lin = fig.add_subplot(gs[row, 0])
            self._plot_linear_spectrum(
                ax_lin, ref_before, ref_after,
                original_spectrum_data, repair_spectrum_data,
                freq_range=freq_range,
                show_reference_segments=show_reference_segments
            )
            row += 1

        # ── 图例和注释 (底部) ──
        if has_quality:
            ax_footer = fig.add_subplot(gs[n_rows - 1, :])
            self._plot_footer(ax_footer, metadata, verification)

        # 保存
        # 注意：不要与 constrained_layout 同时使用 bbox_inches="tight"。
        # 二者组合会让 matplotlib 内部回退调用 tight_layout，触发以下告警
        # 并把布局引擎从 constrained 切换为 tight，反而破坏已计算好的间距：
        #   - "layout engine ... incompatible with subplots_adjust and/or tight_layout"
        #   - "This figure includes Axes that are not compatible with tight_layout"
        #   - "The figure layout has changed to tight"
        # constrained_layout 本身已保证内容不被裁切，故此处使用固定边距保存。
        if save_path:
            fig.savefig(save_path, dpi=dpi, format=format,
                       facecolor="white")
            logger.info("报告图已保存: %s", save_path)

        return fig

    def _create_title(self, metadata: Optional[Dict], anomaly_regions: List[Tuple[int, int]]) -> str:
        """创建报告标题。"""
        # 异常信息
        anomaly_start, anomaly_end = anomaly_regions[0]
        anomaly_len = sum(end - start + 1 for start, end in anomaly_regions)
        duration = anomaly_len / self.sr

        parts = [self._t["title"]]

        if metadata:
            if "station" in metadata and "channel" in metadata:
                parts.append(f"{metadata['station']}.{metadata['channel']}")

        # 时间戳
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        parts.append(now)

        samples_word = self._t["samples"]
        if len(anomaly_regions) == 1:
            segment_text = f"[{anomaly_start}, {anomaly_end}]"
        else:
            segment_text = f"{len(anomaly_regions)} {self._t['anomaly_regions']}"
        parts.append(f"{self._t['anomaly_segment']}: {segment_text} ({anomaly_len} {samples_word}, {duration:.1f}s)")

        return " | ".join(parts)

    def _plot_time_domain(
        self,
        ax: plt.Axes,
        original: np.ndarray,
        repaired: np.ndarray,
        anomaly_regions: List[Tuple[int, int]],
        time_axis: np.ndarray,
        ref_before: Optional[np.ndarray] = None,
        ref_after: Optional[np.ndarray] = None,
        show_reference_segments: bool = True,
        show_inset_zoom: bool = True,
    ):
        """绘制时域对比子图。"""
        ax.set_title(self._t["time_domain"], fontweight="bold", fontsize=11, pad=10)

        # 绘制原始信号
        ax.plot(time_axis, original, color=self.COLORS["original"],
                linewidth=0.6, alpha=0.7, label=self._t["original"], zorder=2)

        # 绘制修复信号
        ax.plot(time_axis, repaired, color=self.COLORS["repaired"],
                linewidth=0.8, alpha=0.9, label=self._t["repaired"], zorder=3)

        # 标记全部修复区间，使主图和放大图涵盖同一次修复操作。
        for index, (anomaly_start, anomaly_end) in enumerate(anomaly_regions):
            t_start = anomaly_start / self.sr
            t_end = anomaly_end / self.sr
            ax.axvspan(
                t_start, t_end, alpha=0.3, color=self.COLORS["anomaly_bg"],
                label=self._t["anomaly_region"] if index == 0 else None, zorder=1,
            )
            ax.axvline(t_start, color=self.COLORS["repaired"], linestyle="--",
                       linewidth=1, alpha=0.5, zorder=4)
            ax.axvline(t_end, color=self.COLORS["repaired"], linestyle="--",
                       linewidth=1, alpha=0.5, zorder=4)

        anomaly_start, anomaly_end = anomaly_regions[0]

        # 标记参考段
        if show_reference_segments:
            if ref_before is not None and len(ref_before) > 0:
                ref_start = max(0, anomaly_start - len(ref_before))
                ref_time = time_axis[ref_start:anomaly_start]
                if len(ref_time) == len(ref_before):
                    ax.plot(ref_time, ref_before, color=self.COLORS["ref_before"],
                            linewidth=1.5, alpha=0.8, label=self._t["ref_before"], zorder=5)

            if ref_after is not None and len(ref_after) > 0:
                ref_end = min(len(original), anomaly_end + 1 + len(ref_after))
                ref_time = time_axis[anomaly_end + 1:ref_end]
                if len(ref_time) == len(ref_after):
                    ax.plot(ref_time, ref_after, color=self.COLORS["ref_after"],
                            linewidth=1.5, alpha=0.8, label=self._t["ref_after"], zorder=5)

        # 添加放大插图
        if show_inset_zoom:
            self._add_inset_zooms(ax, original, repaired, anomaly_regions, time_axis)

        # 设置坐标轴
        ax.set_xlabel(self._t["time_axis"])
        ax.set_ylabel(self._t["amplitude"])
        ax.legend(loc="upper left", fontsize=8, framealpha=0.95,
                  edgecolor="gray", fancybox=True)
        ax.grid(True, alpha=0.3, linestyle="--")

    def _add_inset_zooms(
        self,
        ax_main: plt.Axes,
        original: np.ndarray,
        repaired: np.ndarray,
        anomaly_regions: List[Tuple[int, int]],
        time_axis: np.ndarray,
    ):
        """为每个异常添加缩略放大图，最多显示四处以保留可读性。"""
        visible_regions = anomaly_regions[:4]
        count = len(visible_regions)
        if count == 1:
            positions = [[0.55, 0.15, 0.4, 0.35]]
        elif count == 2:
            positions = [[0.54, 0.15, 0.2, 0.35], [0.75, 0.15, 0.2, 0.35]]
        else:
            positions = [
                [0.54, 0.37, 0.2, 0.19], [0.75, 0.37, 0.2, 0.19],
                [0.54, 0.14, 0.2, 0.19], [0.75, 0.14, 0.2, 0.19],
            ]

        for index, ((anomaly_start, anomaly_end), position) in enumerate(
            zip(visible_regions, positions), start=1
        ):
            self._add_inset_zoom(
                ax_main, original, repaired, anomaly_start, anomaly_end,
                time_axis, position, index, show_connection=(count == 1),
            )

        if len(anomaly_regions) > len(visible_regions):
            ax_main.text(
                0.55, 0.06,
                self._t["inset_overflow"].format(
                    shown=len(visible_regions), total=len(anomaly_regions)
                ),
                transform=ax_main.transAxes, fontsize=7, color="gray",
            )

    def _add_inset_zoom(
        self,
        ax_main: plt.Axes,
        original: np.ndarray,
        repaired: np.ndarray,
        anomaly_start: int,
        anomaly_end: int,
        time_axis: np.ndarray,
        position: List[float],
        index: int,
        show_connection: bool = True,
    ):
        """在指定位置绘制一个异常区域放大插图。"""
        inset_ax = ax_main.inset_axes(position)

        # 异常区域数据
        margin = max(10, (anomaly_end - anomaly_start) // 5)
        plot_start = max(0, anomaly_start - margin)
        plot_end = min(len(original), anomaly_end + margin + 1)

        t_zoom = time_axis[plot_start:plot_end]
        orig_zoom = original[plot_start:plot_end]
        rep_zoom = repaired[plot_start:plot_end]

        # 绘制放大区域
        inset_ax.plot(t_zoom, orig_zoom, color=self.COLORS["original"],
                     linewidth=0.8, label=self._t["original"])
        inset_ax.plot(t_zoom, rep_zoom, color=self.COLORS["repaired"],
                     linewidth=1.2, label=self._t["repaired"])

        # 标记异常区间
        inset_ax.axvspan(anomaly_start / self.sr, anomaly_end / self.sr,
                        alpha=0.3, color=self.COLORS["anomaly_bg"])

        # 设置插图
        inset_ax.set_title(
            self._t["inset_region"].format(index=index), fontsize=7, pad=2
        )
        inset_ax.tick_params(labelsize=6)
        inset_ax.grid(True, alpha=0.2)
        if index == 1:
            inset_ax.legend(loc="upper right", fontsize=5, framealpha=0.9)

        # A single inset benefits from a visual link. With several insets,
        # connector lines obscure the waveform, so their numbered titles are
        # the clearer correspondence cue.
        if show_connection:
            ax_main.indicate_inset_zoom(inset_ax, edgecolor="gray", alpha=0.5)

    @staticmethod
    def _normalize_anomaly_regions(
        anomaly_regions: Optional[List[Tuple[int, int]]],
        fallback_start: int,
        fallback_end: int,
        npts: int,
    ) -> List[Tuple[int, int]]:
        """Validate, sort, and de-duplicate report regions in sample coordinates."""
        normalized = []
        for region in anomaly_regions or [(fallback_start, fallback_end)]:
            if not isinstance(region, (tuple, list)) or len(region) != 2:
                continue
            start, end = sorted((int(region[0]), int(region[1])))
            start = max(0, start)
            end = min(npts - 1, end)
            if end >= start and (start, end) not in normalized:
                normalized.append((start, end))
        return sorted(normalized) or [(max(0, fallback_start), min(npts - 1, fallback_end))]

    @staticmethod
    def _collect_region_samples(data: np.ndarray, regions: List[Tuple[int, int]]) -> np.ndarray:
        """Collect samples from all anomaly regions for consistent report spectra."""
        pieces = []
        npts = len(data)
        for start, end in regions:
            start = max(0, int(start))
            end = min(npts - 1, int(end))
            if end >= start:
                pieces.append(data[start:end + 1])
        if not pieces:
            return np.asarray([], dtype=float)
        return np.concatenate(pieces).astype(float, copy=False)

    def _report_n_fft(self, max_segment_len: int) -> int:
        """Choose a dense but bounded FFT length for report visualization."""
        base_len = max(2, int(max_segment_len))
        if base_len > 65536:
            return self._next_pow2(base_len)
        return max(8192, self._next_pow2(base_len * 2))

    def _plot_spectral(
        self,
        ax: plt.Axes,
        ref_before: Optional[np.ndarray],
        ref_after: Optional[np.ndarray],
        original_segment: np.ndarray,
        repair_segment: np.ndarray,
        freq_range: Optional[Tuple[float, float]] = None,
        show_reference_segments: bool = True,
    ):
        """PSD 对比：修复前 / 修复后 两条曲线画在同一子图中（对数纵轴）。"""
        ax.set_title(
            self._t["spectral_psd"],
            fontweight="bold", fontsize=11, pad=10)

        n_fft = self._report_n_fft(max(
            len(ref_before) if ref_before is not None else 0,
            len(original_segment),
            len(repair_segment),
            len(ref_after) if ref_after is not None else 0
        ))

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        # 原始段是异常输入，与修复结果直接叠放对比。
        psd_original = self._normalize_psd(self._compute_psd(original_segment, n_fft))
        psd_repair = self._normalize_psd(self._compute_psd(repair_segment, n_fft))

        ax.semilogy(freqs, psd_original, color=self.COLORS["original"],
                    linewidth=1.1, alpha=0.9, label=self._t["original"], zorder=3)
        ax.semilogy(freqs, psd_repair, color=self.COLORS["repaired"],
                    linewidth=1.3, alpha=0.95, label=self._t["repaired"], zorder=4)

        # 参考段谱作为背景对照（淡色），便于看修复是否贴近背景水平
        if show_reference_segments:
            if ref_before is not None and len(ref_before) > 0:
                psd_before = self._normalize_psd(self._compute_psd(ref_before, n_fft))
                ax.semilogy(freqs, psd_before, color=self.COLORS["ref_before"],
                            linewidth=1.0, alpha=0.55,
                            label=self._t["ref_before"], zorder=2)

            if ref_after is not None and len(ref_after) > 0:
                psd_after = self._normalize_psd(self._compute_psd(ref_after, n_fft))
                ax.semilogy(freqs, psd_after, color=self.COLORS["ref_after"],
                            linewidth=1.0, alpha=0.55,
                            label=self._t["ref_after"], zorder=2)

        ax.set_ylabel(self._t["psd"], fontsize=9)
        ax.set_xlabel(self._t["freq_axis"], fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--", which="both")
        self._set_freq_xlim(ax, freq_range)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.95,
                  edgecolor="gray", fancybox=True)
        # 对数轴下 PSD 常出现"极低频尖峰 + 其余贴顶"的形态，
        # 按数据分位数收窄纵轴，避免曲线挤在顶部、下方留大片空白。
        self._fit_log_ylim(ax)

    def _fit_log_ylim(self, ax: plt.Axes,
                      low_pct: float = 0.5, high_pct: float = 99.5) -> None:
        """按数据分位数收窄对数纵轴，让曲线铺满子图而非贴顶。"""
        values = []
        for line in ax.get_lines():
            ydata = np.asarray(line.get_ydata(), dtype=float)
            ydata = ydata[np.isfinite(ydata) & (ydata > 0)]
            if ydata.size:
                values.append(ydata)
        if not values:
            return
        all_vals = np.concatenate(values)
        try:
            lo = float(np.percentile(all_vals, low_pct))
            hi = float(np.percentile(all_vals, high_pct))
        except Exception:
            return
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo or lo <= 0:
            return
        # 上下各留半格（对数域）边距
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        span = max(log_hi - log_lo, 1e-3)
        pad = span * 0.08
        ax.set_ylim(10 ** (log_lo - pad), 10 ** (log_hi + pad))

    def _set_freq_xlim(self, ax: plt.Axes, freq_range: Optional[Tuple[float, float]] = None) -> None:
        if freq_range is not None:
            f_min, f_max = freq_range
        else:
            # 默认: 避免DC和Nyquist
            f_min, f_max = 0.0, self.sr / 2 * 0.9
        span = max(float(f_max - f_min), 1e-12)
        pad = span * 0.03
        ax.set_xlim(f_min - pad, f_max + pad)

    def _pad_ylim(self, ax: plt.Axes, pad_frac: float = 0.06) -> None:
        lo, hi = ax.get_ylim()
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return
        # 对数轴用比例；线性轴同样按跨度扩
        if lo > 0 and hi / max(lo, 1e-300) > 50:
            # 近似 log 域：用对数跨度扩边
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            span = log_hi - log_lo
            pad = max(span * pad_frac, 1e-3)
            ax.set_ylim(10 ** (log_lo - pad), 10 ** (log_hi + pad))
        else:
            span = hi - lo
            pad = max(span * pad_frac, abs(hi) * 1e-6, 1e-12)
            ax.set_ylim(lo - pad, hi + pad)

    def _plot_linear_spectrum(
        self,
        ax: plt.Axes,
        ref_before: Optional[np.ndarray],
        ref_after: Optional[np.ndarray],
        original_segment: np.ndarray,
        repair_segment: np.ndarray,
        freq_range: Optional[Tuple[float, float]] = None,
        show_reference_segments: bool = True,
    ):
        """线性频谱对比：修复前 / 修复后 两条曲线画在同一子图中。"""
        ax.set_title(
            self._t["spectral_linear"],
            fontweight="bold", fontsize=11, pad=10)

        n_fft = self._report_n_fft(max(
            len(ref_before) if ref_before is not None else 0,
            len(original_segment),
            len(repair_segment),
            len(ref_after) if ref_after is not None else 0
        ))

        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sr)

        # 计算幅度谱
        def compute_amplitude(segment, n_fft):
            segment = np.asarray(segment, dtype=float)
            segment = segment - np.mean(segment)
            fft_vals = np.fft.rfft(segment, n=n_fft)
            return np.abs(fft_vals) / n_fft

        amp_original = compute_amplitude(original_segment, n_fft)
        amp_repair = compute_amplitude(repair_segment, n_fft)
        amp_original = amp_original / max(float(np.sum(amp_original[1:])), 1e-30)
        amp_repair = amp_repair / max(float(np.sum(amp_repair[1:])), 1e-30)

        ax.plot(freqs, amp_original, color=self.COLORS["original"],
                linewidth=1.1, alpha=0.9, label=self._t["original"], zorder=3)
        ax.plot(freqs, amp_repair, color=self.COLORS["repaired"],
                linewidth=1.3, alpha=0.95, label=self._t["repaired"], zorder=4)

        if show_reference_segments:
            if ref_before is not None and len(ref_before) > 0:
                amp_before = compute_amplitude(ref_before, n_fft)
                amp_before = amp_before / max(float(np.sum(amp_before[1:])), 1e-30)
                ax.plot(freqs, amp_before, color=self.COLORS["ref_before"],
                        linewidth=1.0, alpha=0.55,
                        label=self._t["ref_before"], zorder=2)

            if ref_after is not None and len(ref_after) > 0:
                amp_after = compute_amplitude(ref_after, n_fft)
                amp_after = amp_after / max(float(np.sum(amp_after[1:])), 1e-30)
                ax.plot(freqs, amp_after, color=self.COLORS["ref_after"],
                        linewidth=1.0, alpha=0.55,
                        label=self._t["ref_after"], zorder=2)

        ax.set_ylabel(self._t["magnitude"], fontsize=9)
        ax.set_xlabel(self._t["freq_axis"], fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        self._set_freq_xlim(ax, freq_range)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.95,
                  edgecolor="gray", fancybox=True)
        self._pad_ylim(ax)

    def _plot_quality_metrics(
        self,
        ax: plt.Axes,
        verification: Optional[Dict],
    ):
        """绘制质量指标表格。"""
        ax.set_title("(c) 质量指标", fontweight="bold", fontsize=11, pad=10)
        ax.axis("off")

        if verification is None:
            ax.text(0.5, 0.5, "无质量验证数据",
                   ha="center", va="center", fontsize=12, color="gray")
            return

        # 准备表格数据
        metrics = [
            ("频谱相似度", f"{verification.get('spectral_continuity', 0):.3f}",
             ">= 0.90", verification.get('spectral_continuity', 0) >= 0.90),
            ("包络跳变", f"{verification.get('envelope_continuity', 0):.3f}",
             "<= 0.15", verification.get('envelope_continuity', 0) <= 0.15),
            ("直流跳变", f"{verification.get('dc_continuity', 0):.3f}",
             "<= 0.20", verification.get('dc_continuity', 0) <= 0.20),
            ("自相关差异", f"{verification.get('autocorr_consistency', 0):.3f}",
             "<= 0.30", verification.get('autocorr_consistency', 0) <= 0.30),
            ("能量比", f"{verification.get('energy_ratio', 1):.3f}",
             "[0.5, 1.8]", 0.5 <= verification.get('energy_ratio', 1) <= 1.8),
        ]

        # 创建表格
        table_data = []
        cell_colors = []
        for name, value, threshold, passed in metrics:
            status = "Pass" if passed else "Fail"
            table_data.append([name, value, threshold, status])
            color = "#d5f5e3" if passed else "#fadbd8"
            cell_colors.append([color, color, color, color])

        # 绘制表格
        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "Value", "Threshold", "Status"],
            cellColours=cell_colors,
            colColours=["#d6eaf8"] * 4,
            loc="center",
            cellLoc="center",
        )

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # 调整列宽
        table.auto_set_column_width([0, 1, 2, 3])

        # 综合判定
        passed = verification.get("passed", False)
        verdict = "PASS" if passed else "FAIL"
        verdict_color = self.COLORS["pass"] if passed else self.COLORS["fail"]

        ax.text(0.5, -0.15, f"Verdict: {verdict}",
               ha="center", va="center", fontsize=12,
               fontweight="bold", color=verdict_color,
               transform=ax.transAxes)

    def _plot_footer(
        self,
        ax: plt.Axes,
        metadata: Optional[Dict],
        verification: Optional[Dict],
    ):
        """绘制底部图例和注释。"""
        ax.axis("off")

        # 图例 - 使用更紧凑的布局
        legend_elements = [
            plt.Line2D([0], [0], color=self.COLORS["original"], linewidth=2,
                      label=self._t["original"]),
            plt.Line2D([0], [0], color=self.COLORS["repaired"], linewidth=2,
                      label=self._t["repaired"]),
            plt.Line2D([0], [0], color=self.COLORS["ref_before"], linewidth=2,
                      label=self._t["ref_before"]),
            plt.Line2D([0], [0], color=self.COLORS["ref_after"], linewidth=2,
                      label=self._t["ref_after"]),
            plt.Rectangle((0, 0), 1, 1, facecolor=self.COLORS["anomaly_bg"],
                         alpha=0.5, label=self._t["anomaly_region"]),
        ]

        ax.legend(handles=legend_elements, loc="center", ncol=5,
                 frameon=True, fontsize=9, edgecolor="gray",
                 fancybox=True, shadow=False)

        # 详细信息
        if verification:
            details = verification.get("details", "")
            if details:
                if self.lang == "zh":
                    detail_text = f"验证详情: {details}"
                else:
                    detail_text = f"Verification: {details}"
                ax.text(0.5, -0.4, detail_text,
                       ha="center", va="center", fontsize=8,
                       color="gray", style="italic",
                       transform=ax.transAxes)

    def _compute_psd(self, segment: np.ndarray, n_fft: int) -> np.ndarray:
        """计算功率谱密度。"""
        segment = np.asarray(segment, dtype=float)
        segment = segment - np.mean(segment)
        fft_vals = np.fft.rfft(segment, n=n_fft)
        psd = np.abs(fft_vals) ** 2 / n_fft
        return psd

    @staticmethod
    def _normalize_psd(psd: np.ndarray) -> np.ndarray:
        """Normalize energy so the report compares spectral shape, not scale."""
        psd = np.asarray(psd, dtype=float)
        energy = float(np.sum(np.maximum(psd[1:], 0.0))) if len(psd) > 1 else 0.0
        if not np.isfinite(energy) or energy <= 1e-30:
            return np.zeros_like(psd)
        return np.maximum(psd, 0.0) / energy

    @staticmethod
    def _next_pow2(n: int) -> int:
        """计算下一个2的幂。"""
        p = 1
        while p < n:
            p <<= 1
        return p


def plot_comparison_quick(
    original: np.ndarray,
    repaired: np.ndarray,
    anomaly_start: int,
    anomaly_end: int,
    sr: float,
    save_path: Optional[str] = None,
    lang: Optional[str] = None,
    **kwargs
) -> Figure:
    """
    快速生成对比图的便捷函数。

    Parameters
    ----------
    original : 原始波形
    repaired : 修复后波形
    anomaly_start, anomaly_end : 异常区间
    sr : 采样率
    save_path : 保存路径 (可选)
    lang : 语言代码（``"zh"`` / ``"en"``）；``None`` = 按
        :func:`dssrr.i18n.resolve_language` 解析（默认英文）

    Returns
    -------
    fig : matplotlib Figure
    """
    plotter = RepairReportPlotter(sr=sr, lang=lang)
    return plotter.plot_report(
        original=original,
        repaired=repaired,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end,
        save_path=save_path,
        **kwargs
    )
