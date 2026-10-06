# -*- coding: utf-8 -*-
"""DSSRR 波形修复引擎：把各修复方法应用到 ObsPy ``Stream``/``Trace`` 上。

本模块是「算法 → 数据」的胶水层。上游的修复算法（频谱替换、尖峰检测、
离群点检测……）都只认 numpy 数组，而 GUI 与批量流水线操作的是 ObsPy 的
``Stream``/``Trace``。:class:`StreamRepair` 提供了一组静态方法，把两者接起来，
并统一处理进度回调、取消、以及逐段分派。

方法一览
--------
基础统计方法：
- :meth:`StreamRepair.threshold_truncation`   —— 幅值阈值截断
- :meth:`StreamRepair.mad_based_cleaning`     —— MAD 稳健离群清理
- :meth:`StreamRepair.moving_average_cleaning`—— 滑动均值残差清理
- :meth:`StreamRepair.z_score_cleaning`       —— z 分数清理

月震保护：
- :meth:`StreamRepair.moonquake_protected_spike_cleaning`
  委托 :class:`~dssrr.repair_lib.anomaly_repair.MoonquakeProtectedSpikeReplacer`，
  只修孤立尖峰，不重建长事件。

高级方法（依赖可选库）：
- :meth:`StreamRepair.isolation_forest_cleaning` —— Isolation Forest
- :meth:`StreamRepair.lof_cleaning`              —— Local Outlier Factor
- :meth:`StreamRepair.nn_inpaint_cleaning`       —— 神经网络修补

执行与分派：
- :meth:`StreamRepair._apply_depulse_methods`
  按 ``method_configs`` 列表顺序依次执行，支持 ``progress_callback`` 与
  ``cancel_callback``。
- :meth:`StreamRepair._apply_depulse_segment`
  对单个选段应用一组方法，是手动修复面板的实际执行入口。

依赖：numpy、scipy、obspy。高级方法另需 scikit-learn / torch，缺失时给出
明确的 ``ImportError`` 提示而不会中断其他方法。
"""

import numpy as np
from obspy import Stream, Trace
from scipy.interpolate import PchipInterpolator

from .anomaly_repair import auto_reference_length


class StreamRepair:
    def _apply_depulse_methods(base_stream, method_configs,
                               progress_callback=None, cancel_callback=None):
        if base_stream is None:
            return None
        processed_stream = base_stream.copy()
        total_methods = max(1, len(method_configs))
        for index, cfg in enumerate(method_configs):
            if cancel_callback and cancel_callback():
                return None
            if progress_callback:
                progress_callback(index, total_methods, cfg)
            method_id = cfg["id"]
            params = cfg["params"]
            if method_id == "threshold":
                processed_stream = StreamRepair.threshold_truncation(
                    processed_stream, amplitude_threshold=params["threshold"])
            elif method_id == "mad":
                processed_stream = StreamRepair.mad_based_cleaning(
                    processed_stream, mad_threshold=params["mad_threshold"])
            elif method_id == "moving_avg":
                processed_stream = StreamRepair.moving_average_cleaning(
                    processed_stream,
                    window_size=int(params["window_size"]),
                    amplitude_threshold=params["amplitude_threshold"]
                )
            elif method_id == "moonquake_protected_spike":
                processed_stream = StreamRepair.moonquake_protected_spike_cleaning(
                    processed_stream,
                    threshold_sigma=params.get("threshold_sigma", 8.0),
                    minimum_global_sigma=params.get("minimum_global_sigma", 4.0),
                    detection_window_sec=params.get("detection_window_sec", 0.75),
                    scale_window_sec=params.get("scale_window_sec", 4.0),
                    max_width_samples=params.get("max_width_samples", 2),
                    neighbour_return_ratio=params.get("neighbour_return_ratio", 0.35),
                    strong_peak_rescue=params.get("strong_peak_rescue", True),
                    strong_peak_sigma=params.get("strong_peak_sigma", 16.0),
                    strong_peak_global_sigma=params.get("strong_peak_global_sigma", 6.0),
                    strong_peak_ratio=params.get("strong_peak_ratio", 1.25),
                    rescue_max_width_samples=params.get("rescue_max_width_samples", 4),
                    interpolation_radius_samples=params.get("interpolation_radius_samples", 2),
                )
            elif method_id == "z_score":
                processed_stream = StreamRepair.z_score_cleaning(
                    processed_stream, z_threshold=params["z_threshold"])
            elif method_id == "isolation_forest":
                processed_stream = StreamRepair.isolation_forest_cleaning(
                    processed_stream,
                    contamination=params.get("contamination", 0.05),
                    n_estimators=params.get("n_estimators", 100),
                    fill_method=params.get("fill_method", "interpolate"),
                )
            elif method_id == "lof":
                processed_stream = StreamRepair.lof_cleaning(
                    processed_stream,
                    contamination=params.get("contamination", 0.05),
                    n_neighbors=params.get("n_neighbors", 20),
                    fill_method=params.get("fill_method", "interpolate"),
                )
            elif method_id == "nn_inpaint":
                processed_stream = StreamRepair.nn_inpaint_cleaning(
                    processed_stream,
                    detection=params.get("detection", "mad"),
                    mad_threshold=params.get("mad_threshold", 3.0),
                    z_threshold=params.get("z_threshold", 3.0),
                    contamination=params.get("contamination", 0.05),
                    n_epochs=params.get("n_epochs", 100),
                    lr=params.get("lr", 0.001),
                    preview_mode=params.get("preview_mode", False),
                    progress_callback=params.get("progress_callback", None),
                )
            if progress_callback:
                progress_callback(index + 1, total_methods, cfg)
        return processed_stream

    @staticmethod
    def _apply_depulse_segment(base_stream, method_configs, segment_info):
        if base_stream is None:
            return None
        from .depulse_advanced import AdvancedDePulse
        from .anomaly_repair import (
            LayeredReferenceSpectrumReplacer,
            MoonquakeProtectedSpikeReplacer,
            ReferenceSpectrumReplacer,
        )
        from scipy.signal import savgol_filter

        processed_stream = base_stream.copy()
        apply_all = segment_info.get("apply_all", False)
        apply_savgol = segment_info.get("apply_savgol", False)
        savgol_window = segment_info.get("savgol_window", 9)
        trace_indices = range(len(processed_stream)) if apply_all else [segment_info["trace_index"]]

        # Build list of time ranges to process
        all_selections = segment_info.get("all_selections")
        if all_selections:
            time_ranges = all_selections
        else:
            tr = segment_info.get("time_range")
            time_ranges = [tr] if tr else []

        for tidx in trace_indices:
            if tidx >= len(processed_stream):
                continue
            trace = processed_stream[tidx]
            data = np.asarray(trace.data, dtype=float).copy()
            sr = float(trace.stats.sampling_rate or 1.0)
            # 本次 trace 是否发生整数化修复（用于写回时还原原始整数 dtype）
            quantized_this_trace = False

            for start_time, end_time in time_ranges:
                idx0 = max(0, int(start_time * sr))
                idx1 = min(len(data), int(end_time * sr))
                if idx1 <= idx0 + 2:
                    continue

                # A reference-spectrum method treats the hand-selected range as
                # the complete anomaly core. It must run before point-wise
                # detectors so those detectors cannot shrink the selected range.
                spectral_configs = [
                    cfg for cfg in method_configs
                    if cfg.get("id") in {
                        "reference_spectrum",
                        "layered_reference_spectrum",
                    }
                ]
                spectral_applied = False
                for cfg in spectral_configs:
                    params = cfg.get("params", {})
                    seed = params.get("random_seed", 0)
                    # A suffix shift is a whole-trace operation. With several
                    # selected regions, applying it independently can stack
                    # offsets and create artificial stairs after the last
                    # region. Keep each repair local in that workflow.
                    joining_config = {
                        "baseline_alignment": (
                            "bridge" if len(time_ranges) > 1 else "auto"
                        )
                    }
                    # 整数化开关（双侧/分层参考频谱修复）：月震原始计数为
                    # 整数，频谱重建输出为实数；勾选后仅把恢复区间四舍五入
                    # 回整数，与修复库对比查看器 / 批处理流水线
                    # quantize_to_int 同口径。
                    # 取整由 replacer 内部按 quantize 参数完成（单一真源），
                    # 此处只记录是否发生了整数化，用于写回时还原原始 int dtype。
                    quantize_flag = bool(params.get("quantize_to_int", True))
                    if cfg.get("id") == "layered_reference_spectrum":
                        replacer = LayeredReferenceSpectrumReplacer(
                            sr,
                            config={
                                "synthesis": {"random_seed": seed},
                                "joining": joining_config,
                            },
                        )
                        repaired, _report = replacer.replace(
                            data,
                            idx0,
                            idx1 - 1,
                            reference_before_sec=(auto_reference_length(idx1 - idx0, sr, params.get("reference_min_sec", 120.0)) if params.get("reference_auto", False) else params.get("reference_before_sec", 120.0)),
                            reference_after_sec=params.get("reference_after_sec", 100.0),
                            reference_gap_sec=params.get("reference_gap_sec", 0.5),
                            random_seed=seed,
                            focus_mode=params.get("focus_mode", "ultra_low"),
                            band_min_hz=params.get("band_min_hz", 0.001),
                            band_max_hz=params.get("band_max_hz", 0.08),
                            transition_hz=params.get("transition_hz", 0.01),
                            quantize=quantize_flag,
                        )
                    else:
                        replacer = ReferenceSpectrumReplacer(
                            sr,
                            config={
                                "synthesis": {"random_seed": seed},
                                "joining": joining_config,
                            },
                        )
                        repaired, _report = replacer.replace(
                            data,
                            idx0,
                            idx1 - 1,
                            reference_before_sec=(auto_reference_length(idx1 - idx0, sr, params.get("reference_min_sec", 120.0)) if params.get("reference_auto", False) else params.get("reference_before_sec", 120.0)),
                            reference_after_sec=params.get("reference_after_sec", 100.0),
                            reference_gap_sec=params.get("reference_gap_sec", 0.5),
                            random_seed=seed,
                            quantize=quantize_flag,
                        )
                    data = repaired
                    if quantize_flag:
                        quantized_this_trace = True
                    spectral_applied = True

                # Interpolate only the manually selected core while using the
                # remaining trace as the valid context for existing inpaint.
                manual_interpolation = next(
                    (
                        cfg for cfg in method_configs
                        if cfg.get("id") == "missing_value_interpolation"
                    ),
                    None,
                )
                if manual_interpolation is not None:
                    interpolation_method = (
                        manual_interpolation.get("params", {}).get("method")
                        or "linear"
                    )
                    # Apollo stores missing samples as the sentinel -1.  The
                    # selected interval is only a scope: preserve all real
                    # samples inside it and interpolate sentinel positions.
                    sentinel_mask = np.isfinite(data) & (data == -1.0)
                    fill_mask = np.zeros(len(data), dtype=bool)
                    selected = data[idx0:idx1]
                    fill_mask[idx0:idx1] = np.isfinite(selected) & (selected == -1.0)
                    if not fill_mask.any():
                        continue
                    # Exclude Apollo sentinels outside the selected interval
                    # from the interpolation context without modifying them.
                    if interpolation_method == "pchip":
                        valid = ~sentinel_mask
                        targets = np.flatnonzero(fill_mask)
                        if np.count_nonzero(valid) >= 2:
                            repaired = PchipInterpolator(
                                np.flatnonzero(valid), data[valid], extrapolate=True
                            )(targets)
                            data[fill_mask] = repaired
                        else:
                            data = AdvancedDePulse.median_fill(data, fill_mask)
                    else:
                        min_points = 4 if interpolation_method in {"cubic", "quadratic"} else 2
                        if np.count_nonzero(~sentinel_mask) < min_points:
                            interpolation_method = "linear"
                        repaired = AdvancedDePulse.interpolate_inpaint(
                            data, sentinel_mask, method=interpolation_method
                        )
                        data[fill_mask] = repaired[fill_mask]
                    continue

                segment = data[idx0:idx1].copy()

                # Build outlier mask using the selected detection methods
                mask = np.zeros(len(segment), dtype=bool)
                nn_configs = []
                direct_repair_applied = False
                regular_configs = [
                    cfg for cfg in method_configs
                    if cfg.get("id") not in {
                        "missing_value_interpolation",
                        "reference_spectrum",
                        "layered_reference_spectrum",
                    }
                ]
                for cfg in regular_configs:
                    mid = cfg["id"]
                    p = cfg["params"]
                    if mid == "mad":
                        mask |= AdvancedDePulse.mad_detect(segment, p.get("mad_threshold", 3.0))
                    elif mid == "z_score":
                        mask |= AdvancedDePulse.zscore_detect(segment, p.get("z_threshold", 3.0))
                    elif mid == "threshold":
                        thr = float(p.get("threshold", 3))
                        mask |= (np.abs(segment) > thr)
                    elif mid == "isolation_forest":
                        mask |= AdvancedDePulse.isolation_forest_detect(
                            segment, contamination=p.get("contamination", 0.05),
                            n_estimators=p.get("n_estimators", 100))
                    elif mid == "lof":
                        mask |= AdvancedDePulse.local_outlier_factor_detect(
                            segment, contamination=p.get("contamination", 0.05),
                            n_neighbors=p.get("n_neighbors", 20))
                    elif mid == "moving_avg":
                        window = min(max(1, int(p.get("window_size", 5))), len(segment))
                        thr = p.get("amplitude_threshold", 3.0)
                        mov_avg = np.convolve(segment, np.ones(window) / window, mode='same')
                        mask |= (np.abs(segment - mov_avg) > thr)
                    elif mid == "moonquake_protected_spike":
                        segment, _report = MoonquakeProtectedSpikeReplacer(sr).replace(
                            segment,
                            threshold_sigma=p.get("threshold_sigma", 8.0),
                            minimum_global_sigma=p.get("minimum_global_sigma", 4.0),
                            detection_window_sec=p.get("detection_window_sec", 0.75),
                            scale_window_sec=p.get("scale_window_sec", 4.0),
                            max_width_samples=p.get("max_width_samples", 2),
                            neighbour_return_ratio=p.get("neighbour_return_ratio", 0.35),
                            strong_peak_rescue=p.get("strong_peak_rescue", True),
                            strong_peak_sigma=p.get("strong_peak_sigma", 16.0),
                            strong_peak_global_sigma=p.get("strong_peak_global_sigma", 6.0),
                            strong_peak_ratio=p.get("strong_peak_ratio", 1.25),
                            rescue_max_width_samples=p.get("rescue_max_width_samples", 4),
                            interpolation_radius_samples=p.get("interpolation_radius_samples", 2),
                        )
                        direct_repair_applied = True
                    elif mid == "nn_inpaint":
                        nn_configs.append(cfg)

                if direct_repair_applied:
                    data[idx0:idx1] = segment
                    continue

                if not mask.any() and not nn_configs and not spectral_applied:
                    continue

                # Apply filling
                if nn_configs:
                    p = nn_configs[0]["params"]
                    det = p.get("detection", "mad")
                    mask |= AdvancedDePulse.combined_detect(
                        segment, methods=[det],
                        mad_threshold=p.get("mad_threshold", 3.0),
                        z_threshold=p.get("z_threshold", 3.0),
                        contamination=p.get("contamination", 0.05))
                    n_epochs = p.get("n_epochs", 100)
                    lr = p.get("lr", 0.001)
                    segment = AdvancedDePulse.nn_inpaint(
                        segment, mask, n_epochs=n_epochs, lr=lr, preview_mode=False)
                else:
                    interpolation_method = 'cubic' if np.count_nonzero(~mask) >= 4 else 'linear'
                    segment = AdvancedDePulse.interpolate_inpaint(
                        segment, mask, method=interpolation_method)

                # Optional Savitzky-Golay smoothing
                if apply_savgol:
                    win = savgol_window
                    if win >= len(segment):
                        win = len(segment) - 1 if len(segment) % 2 == 0 else len(segment)
                    if win < 3:
                        win = 3
                    if win % 2 == 0:
                        win += 1
                    if win >= 3:
                        try:
                            segment = savgol_filter(segment, window_length=win, polyorder=min(3, win - 1))
                        except Exception:
                            pass

                data[idx0:idx1] = segment

            if quantized_this_trace and np.issubdtype(
                    trace.data.dtype, np.integer):
                # 整数化生效且原始为整数 dtype（如 int32）：数值已取整，
                # 写回原始 dtype 以贴合月震计数档案；原始数据本身为浮点
                # 时保持 float，避免破坏非整数源数据。
                trace.data = data.astype(trace.data.dtype)
            else:
                trace.data = data

        return processed_stream

    @staticmethod
    def threshold_truncation(stream, amplitude_threshold):
        """
        Function to truncate data by setting values above a threshold to the threshold.

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        - amplitude_threshold: float, threshold value for truncation

        Returns:
        - Stream: new Stream object with truncated data
        """
        cleaned_stream = Stream()
        for trace in stream:
            data = np.asarray(trace.data, dtype=float)
            if isinstance(amplitude_threshold, (int, float)):
                threshold = abs(float(amplitude_threshold))
                outlier_mask = (data < -threshold) | (data > threshold)
            elif isinstance(amplitude_threshold, str):
                value = amplitude_threshold.strip()
                if value.startswith('>'):
                    threshold = float(value[1:])
                    outlier_mask = data > threshold
                elif value.startswith('<'):
                    threshold = float(value[1:])
                    outlier_mask = data < threshold
                else:
                    threshold = abs(float(value))
                    outlier_mask = (data < -threshold) | (data > threshold)
            else:
                outlier_mask = np.zeros(len(data), dtype=bool)

            if outlier_mask.any():
                good = ~outlier_mask
                if np.count_nonzero(good) >= 2:
                    x = np.arange(len(data))
                    data[outlier_mask] = np.interp(
                        x[outlier_mask], x[good], data[good])
                else:
                    data[outlier_mask] = np.median(data[good]) if good.any() else 0.0

            cleaned_trace = Trace(data=data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def mad_based_cleaning(stream, mad_threshold=3):
        """
        Function to remove outliers based on Median Absolute Deviation (MAD).

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        - mad_threshold: float, Multiples of the absolute deviation of the median

        Returns:
        - Stream: new Stream object with outliers removed
        """
        cleaned_stream = Stream()
        for trace in stream:
            data = trace.data
            median = np.median(data)
            mad = np.median(np.abs(data - median)) * 1.4826
            if mad == 0:
                mad = np.std(data) or 1.0
            cleaned_data = np.where(np.abs(data - median) > mad_threshold * mad, median, data)
            cleaned_trace = Trace(data=cleaned_data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def moving_average_cleaning(stream, window_size=700, amplitude_threshold=5):
        """
        Function to remove outliers based on a moving average.

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        - window_size: int, size of the moving average window
        - amplitude_threshold: float, threshold for identifying outliers based on deviation from moving average

        Returns:
        - Stream: new Stream object with outliers removed
        """
        cleaned_stream = Stream()
        for trace in stream:
            data = np.asarray(trace.data, dtype=float)
            if len(data) == 0:
                cleaned_stream.append(Trace(data=data, header=trace.stats))
                continue
            effective_window = min(max(1, int(window_size)), len(data))
            moving_average = np.convolve(
                data, np.ones(effective_window) / effective_window, mode='same')
            cleaned_data = np.where(np.abs(data - moving_average) > amplitude_threshold, moving_average, data)
            cleaned_trace = Trace(data=cleaned_data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def moonquake_protected_spike_cleaning(
            stream,
            threshold_sigma=8.0,
            minimum_global_sigma=4.0,
            detection_window_sec=0.75,
            scale_window_sec=4.0,
            max_width_samples=2,
            neighbour_return_ratio=0.35,
            interpolation_radius_samples=2,
            strong_peak_rescue=True,
            strong_peak_sigma=16.0,
            strong_peak_global_sigma=6.0,
            strong_peak_ratio=1.25,
            rescue_max_width_samples=4):
        """Repair isolated spikes while protecting non-stationary events."""
        from .anomaly_repair import MoonquakeProtectedSpikeReplacer

        cleaned_stream = stream.copy()
        for trace in cleaned_stream:
            data = np.asarray(trace.data, dtype=float)
            repaired, _report = MoonquakeProtectedSpikeReplacer(
                float(trace.stats.sampling_rate or 1.0)).replace(
                    data,
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
                    interpolation_radius_samples=interpolation_radius_samples,
                )
            trace.data = repaired
        return cleaned_stream

    @staticmethod
    def z_score_cleaning(stream, z_threshold=3):
        """
        Function to remove outliers based on Z-scores.

        Parameters:
        - stream: obspy.Stream, input stream of seismic traces
        - z_threshold: float, threshold for identifying outliers based on Z-scores

        Returns:
        - Stream: new Stream object with outliers removed
        """
        cleaned_stream = Stream()
        for trace in stream:
            data = np.asarray(trace.data, dtype=float)
            mean = np.mean(data)
            std = np.std(data)
            if not np.isfinite(std) or std <= np.finfo(float).eps:
                cleaned_data = data.copy()
            else:
                z_scores = (data - mean) / std
                cleaned_data = np.where(np.abs(z_scores) > z_threshold, mean, data)
            cleaned_trace = Trace(data=cleaned_data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def isolation_forest_cleaning(stream, contamination=0.05, n_estimators=100,
                                   fill_method='interpolate'):
        """Detect outliers with Isolation Forest and fill using specified method.

        Parameters
        ----------
        stream : obspy.Stream
        contamination : float
            Expected proportion of outliers (0.01--0.3).
        n_estimators : int
            Number of trees in the forest.
        fill_method : str
            'interpolate' (cubic), 'linear', 'median', or 'nn_inpaint'.

        Returns
        -------
        obspy.Stream
        """
        from .depulse_advanced import AdvancedDePulse
        cleaned_stream = Stream()
        for trace in stream:
            data = np.asarray(trace.data, dtype=float)
            mask = AdvancedDePulse.isolation_forest_detect(
                data, contamination=contamination, n_estimators=n_estimators)
            if mask.any():
                if fill_method == 'nn_inpaint':
                    data = AdvancedDePulse.nn_inpaint(data, mask)
                elif fill_method == 'median':
                    data = AdvancedDePulse.median_fill(data, mask)
                else:
                    data = AdvancedDePulse.interpolate_inpaint(
                        data, mask, method=fill_method if fill_method != 'interpolate' else 'cubic')
            cleaned_trace = Trace(data=data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def lof_cleaning(stream, contamination=0.05, n_neighbors=20,
                      fill_method='interpolate'):
        """Detect outliers with Local Outlier Factor and fill.

        Parameters
        ----------
        stream : obspy.Stream
        contamination : float
            Expected proportion of outliers (0.01--0.3).
        n_neighbors : int
            Number of neighbors for density estimation.
        fill_method : str
            'interpolate' (cubic), 'linear', 'median', or 'nn_inpaint'.

        Returns
        -------
        obspy.Stream
        """
        from .depulse_advanced import AdvancedDePulse
        cleaned_stream = Stream()
        for trace in stream:
            data = np.asarray(trace.data, dtype=float)
            mask = AdvancedDePulse.local_outlier_factor_detect(
                data, contamination=contamination, n_neighbors=n_neighbors)
            if mask.any():
                if fill_method == 'nn_inpaint':
                    data = AdvancedDePulse.nn_inpaint(data, mask)
                elif fill_method == 'median':
                    data = AdvancedDePulse.median_fill(data, mask)
                else:
                    data = AdvancedDePulse.interpolate_inpaint(
                        data, mask, method=fill_method if fill_method != 'interpolate' else 'cubic')
            cleaned_trace = Trace(data=data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

    @staticmethod
    def nn_inpaint_cleaning(stream, detection='mad', mad_threshold=3.0,
                             z_threshold=3.0, contamination=0.05,
                             n_epochs=100, lr=0.001, device='cpu',
                             preview_mode=False, progress_callback=None):
        """Detect outliers then inpaint using a neural network autoencoder.

        Parameters
        ----------
        stream : obspy.Stream
        detection : str
            'mad', 'zscore', 'isolation_forest', 'lof', or 'combined'.
        mad_threshold, z_threshold, contamination :
            Parameters for the chosen detection method.
        n_epochs : int
            Training epochs for the autoencoder.
        lr : float
            Adam learning rate.
        device : str
            'cpu' or 'cuda'.
        preview_mode : bool
            If True, use fewer epochs for fast preview.
        progress_callback : callable or None
            Called with (current, total) for progress reporting.

        Returns
        -------
        obspy.Stream
        """
        from .depulse_advanced import AdvancedDePulse
        cleaned_stream = Stream()
        for trace in stream:
            data = np.asarray(trace.data, dtype=float)
            mask = AdvancedDePulse.combined_detect(
                data, methods=[detection],
                mad_threshold=mad_threshold,
                z_threshold=z_threshold,
                contamination=contamination)
            if mask.any():
                data = AdvancedDePulse.nn_inpaint(
                    data, mask, n_epochs=n_epochs, lr=lr, device=device,
                    preview_mode=preview_mode,
                    progress_callback=progress_callback)
            cleaned_trace = Trace(data=data, header=trace.stats)
            cleaned_stream.append(cleaned_trace)
        return cleaned_stream

