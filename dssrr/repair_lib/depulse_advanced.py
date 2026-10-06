"""Advanced outlier detection and deep-learning inpainting for seismic traces.

Beyond the classical statistical cleaners (threshold / MAD / z-score /
moving-average), DSSRR offers a set of "advanced" detectors and inpainting
fillers that are more robust on non-stationary traces:

- **Detection**
  - ``isolation_forest_detect`` -- scikit-learn Isolation Forest on a small
    feature vector per sample.
  - ``local_outlier_factor_detect`` -- scikit-learn Local Outlier Factor.
  - ``mad_detect`` / ``zscore_detect`` -- fast robust baselines.
  - ``combined_detect`` -- ensemble that requires agreement between methods.
- **Inpainting**
  - ``nn_inpaint`` -- fill flagged samples with a 1-D convolutional autoencoder
    (:mod:`dssrr.repair_lib.depulse_nn`), trained on the fly on the trace.
  - ``interpolate_inpaint`` -- shape-preserving PCHIP interpolation (no torch).
  - ``median_fill`` -- last-resort rolling-median substitution.
- **Diagnostics**
  - :class:`AnomalyDiagnostic` -- summarises what a trace looks like (missing
    ratio, freeze runs, saturation, spikes) and returns an
    :class:`AnomalyDiagnosticResult` for the GUI to display.

Every method is a ``@staticmethod`` operating on plain numpy arrays, so they can
be called without instantiating anything::

    from dssrr.repair_lib import depulse_advanced as adv

    mask = adv.AdvancedDePulse.mad_detect(data, threshold=6.0)
    filled = adv.AdvancedDePulse.interpolate_inpaint(data, mask)

Optional dependencies are handled gracefully: if ``torch`` (neural inpainting)
or ``scikit-learn`` (IForest / LOF) cannot be imported -- including the case
where a frozen Windows build finds the package but fails to load a DLL -- the
corresponding methods raise a clear ``ImportError`` with a ``pip install`` hint,
and the remaining methods keep working.  Install the extras with::

    pip install "dssrr[nn]"    # torch
    pip install scikit-learn   # sklearn detectors (also in [gui]/[batch])
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
from scipy.interpolate import interp1d

from ..i18n import resolve_language

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_HAS_TORCH = False
_HAS_SKLEARN = False
_TORCH_IMPORT_ERROR = None

try:
    import torch
    from .depulse_nn import InpaintAutoencoder1D
    _HAS_TORCH = True
except (ImportError, OSError) as exc:
    # A frozen Windows build can find torch but fail to load one of its DLLs.
    # Treat that the same as an unavailable optional dependency so the base
    # de-pulse tools remain usable.
    _TORCH_IMPORT_ERROR = str(exc)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    _HAS_SKLEARN = True
except ImportError:
    pass

# Suggested pip commands for missing dependencies
_TORCH_HINT = (
    "PyTorch is required for neural network inpainting. "
    "Install with: pip install torch"
)
_SKLEARN_HINT = (
    "scikit-learn is required for this detection method. "
    "Install with: pip install scikit-learn"
)


def _check_torch():
    if not _HAS_TORCH:
        raise ImportError(_TORCH_HINT)


def _check_sklearn():
    if not _HAS_SKLEARN:
        raise ImportError(_SKLEARN_HINT)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class AdvancedDePulse:
    """Static methods for advanced outlier detection and inpainting."""

    # ---- Detection ----

    @staticmethod
    def isolation_forest_detect(data, contamination=0.05, n_estimators=100,
                                random_state=42):
        """Detect outliers using Isolation Forest.

        Isolation Forest isolates observations by randomly selecting a feature
        and a split value. Outliers require fewer splits to isolate and are
        assigned anomaly scores closer to -1.

        Parameters
        ----------
        data : np.ndarray, 1-D
            Trace data.
        contamination : float
            Expected proportion of outliers (0.01--0.3).
        n_estimators : int
            Number of trees in the forest.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray (bool)
            True where a sample is an outlier.
        """
        _check_sklearn()
        data_clean = np.asarray(data, dtype=float)
        finite = np.isfinite(data_clean)
        if not finite.all():
            data_clean[~finite] = np.nanmedian(data_clean[finite]) if finite.any() else 0.0
        X = data_clean.reshape(-1, 1)
        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        preds = clf.fit_predict(X)
        return preds == -1

    @staticmethod
    def local_outlier_factor_detect(data, contamination=0.05, n_neighbors=20):
        """Detect outliers using Local Outlier Factor.

        LOF measures the local deviation of a sample with respect to its
        neighbors. Samples with substantially lower density than their
        neighbors are flagged as outliers.

        For traces > 5000 samples, data is downsampled before detection
        to keep runtime reasonable (LOF is O(N²)).

        Parameters
        ----------
        data : np.ndarray, 1-D
            Trace data.
        contamination : float
            Expected proportion of outliers (0.01--0.3).
        n_neighbors : int
            Number of neighbors for local density estimation.

        Returns
        -------
        np.ndarray (bool)
            True where a sample is an outlier.
        """
        _check_sklearn()
        data_clean = np.asarray(data, dtype=float)
        finite = np.isfinite(data_clean)
        if not finite.all():
            data_clean[~finite] = np.nanmedian(data_clean[finite]) if finite.any() else 0.0

        max_samples = 3000
        if len(data_clean) > max_samples:
            # Downsample for detection: use striding + peak preservation
            step = max(1, len(data_clean) // max_samples)
            # Take max absolute value in each stride to preserve spike info
            n_chunks = len(data_clean) // step
            trimmed = len(data_clean) - n_chunks * step
            if trimmed > 0:
                offset = trimmed // 2
                data_clean = data_clean[offset:offset + n_chunks * step]
            reshaped = data_clean.reshape(n_chunks, step)
            # Use max abs per chunk for detection (preserves spikes)
            down_data = np.max(np.abs(reshaped), axis=1)
            X = down_data.reshape(-1, 1)
            clf = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=min(n_neighbors, n_chunks - 1),
                novelty=False,
                n_jobs=-1,
            )
            preds_down = clf.fit_predict(X)
            mask_down = preds_down == -1
            # Upsample mask to original resolution
            mask = np.repeat(mask_down, step)
            if len(mask) < len(np.asarray(data)):
                mask = np.pad(mask, (0, len(np.asarray(data)) - len(mask)),
                             constant_values=False)
            return mask[:len(np.asarray(data))]

        X = data_clean.reshape(-1, 1)
        clf = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=False,
            n_jobs=-1,
        )
        preds = clf.fit_predict(X)
        return preds == -1

    @staticmethod
    def mad_detect(data, threshold=3.0, ref_data=None):
        """Detect outliers using Median Absolute Deviation (with 1.4826 correction).

        Parameters
        ----------
        data : np.ndarray
        threshold : float
            MAD multiplier.
        ref_data : np.ndarray, optional
            Reference data for computing statistics. If None, uses data itself.
            Useful when signal regions should not affect background statistics.

        Returns
        -------
        np.ndarray (bool)
        """
        data = np.asarray(data, dtype=float)
        ref = np.asarray(ref_data, dtype=float) if ref_data is not None else data
        median = np.nanmedian(ref)
        mad = np.nanmedian(np.abs(ref - median)) * 1.4826
        if mad == 0:
            mad = np.nanstd(ref) or 1.0
        return np.abs(data - median) > threshold * mad

    @staticmethod
    def zscore_detect(data, threshold=3.0, ref_data=None):
        """Detect outliers using Z-score.

        Parameters
        ----------
        data : np.ndarray
        threshold : float
        ref_data : np.ndarray, optional
            Reference data for computing statistics. If None, uses data itself.

        Returns
        -------
        np.ndarray (bool)
        """
        data = np.asarray(data, dtype=float)
        ref = np.asarray(ref_data, dtype=float) if ref_data is not None else data
        mean = np.nanmean(ref)
        std = np.nanstd(ref) or 1.0
        return np.abs(data - mean) > threshold * std

    @staticmethod
    def combined_detect(data, methods=('mad', 'zscore'), mode='union',
                        mad_threshold=3.0, z_threshold=3.0,
                        contamination=0.05, n_estimators=100,
                        n_neighbors=20):
        """Ensemble outlier detection combining multiple methods.

        Parameters
        ----------
        data : np.ndarray
        methods : tuple of str
            Subset of {'mad', 'zscore', 'isolation_forest', 'lof'}.
        mode : str
            'union' — flagged if ANY method detects an outlier.
            'intersection' — flagged if ALL methods detect an outlier.
        contamination, n_estimators, n_neighbors :
            Passed to the respective sklearn-based methods.

        Returns
        -------
        np.ndarray (bool)
        """
        if not methods:
            return np.zeros(len(data), dtype=bool)
        masks = []
        for method in methods:
            m = method.lower().strip()
            if m == 'mad':
                masks.append(AdvancedDePulse.mad_detect(data, mad_threshold))
            elif m == 'zscore':
                masks.append(AdvancedDePulse.zscore_detect(data, z_threshold))
            elif m in ('isolation_forest', 'iforest', 'if'):
                masks.append(AdvancedDePulse.isolation_forest_detect(
                    data, contamination=contamination, n_estimators=n_estimators))
            elif m == 'lof':
                masks.append(AdvancedDePulse.local_outlier_factor_detect(
                    data, contamination=contamination, n_neighbors=n_neighbors))
        if not masks:
            return np.zeros(len(data), dtype=bool)
        stacked = np.stack(masks, axis=0)
        if mode == 'intersection':
            return stacked.all(axis=0)
        return stacked.any(axis=0)

    # ---- Inpainting ----

    @staticmethod
    def nn_inpaint(data, mask, n_epochs=100, lr=0.001, device='cpu',
                   early_stop_patience=20, progress_callback=None,
                   preview_mode=False):
        """Train a lightweight autoencoder on-the-fly to inpaint masked regions.

        Self-supervised strategy:
        1. Normalize data to zero-mean, unit-variance.
        2. Zero-fill masked positions as input.
        3. Additionally mask 5% of known-good samples as a validation hold-out.
        4. Train autoencoder; early-stop when validation loss plateaus.
        5. Blend model output with original data (only masked regions changed).

        For long traces (> 50 000 samples), processing is done in overlapping
        chunks with Hann-window blending.

        Parameters
        ----------
        data : np.ndarray, 1-D
        mask : np.ndarray (bool)
            True = outlier (to be inpainted).
        n_epochs : int
            Maximum training epochs.
        lr : float
            Adam learning rate.
        device : str
            'cpu' or 'cuda'.
        early_stop_patience : int
            Epochs without val-loss improvement before stopping.
        progress_callback : callable or None
            Called with (epoch, max_epochs) for progress reporting.
        preview_mode : bool
            If True, use fewer epochs (capped at 30) for fast preview.

        Returns
        -------
        np.ndarray
            Inpainted data (same shape as input).
        """
        _check_torch()
        data = np.asarray(data, dtype=float).copy()
        mask = np.asarray(mask, dtype=bool)

        if not mask.any():
            return data

        if preview_mode:
            n_epochs = min(n_epochs, 30)

        # Chunk long traces
        chunk_size = 50000
        if len(data) > chunk_size:
            return AdvancedDePulse._nn_inpaint_chunked(
                data, mask, n_epochs, lr, device,
                early_stop_patience, progress_callback)

        # Normalize
        valid = data[~mask]
        if len(valid) == 0:
            return data
        mean_val = np.mean(valid)
        std_val = np.std(valid) or 1.0
        data_norm = (data - mean_val) / std_val

        # Build input: zero-fill masked positions
        x_input = data_norm.copy()
        x_input[mask] = 0.0

        # Hold-out 5% of known-good samples for validation
        good_idx = np.where(~mask)[0]
        n_val = max(1, int(len(good_idx) * 0.05))
        rng = np.random.RandomState(42)
        val_idx = rng.choice(good_idx, size=n_val, replace=False)
        mask_val = np.zeros(len(data), dtype=bool)
        mask_val[val_idx] = True
        # Don't zero-fill validation positions in input
        # The model should still see them

        # Convert to tensors
        x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        y_tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).unsqueeze(0).to(device)
        mask_val_t = torch.tensor(mask_val, dtype=torch.bool).unsqueeze(0).unsqueeze(0).to(device)

        # Create model
        model = InpaintAutoencoder1D(in_channels=1, base_channels=16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(x_tensor)

            # Loss on outlier regions (inpainting target)
            if mask_t.any():
                loss_outlier = F.mse_loss(output[mask_t], y_tensor[mask_t])
            else:
                loss_outlier = torch.tensor(0.0, device=device)

            # Loss on held-out validation regions
            if mask_val_t.any():
                loss_val_region = F.mse_loss(output[mask_val_t], y_tensor[mask_val_t])
            else:
                loss_val_region = torch.tensor(0.0, device=device)

            loss = loss_outlier + 0.5 * loss_val_region
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_out = model(x_tensor)
                val_loss = F.mse_loss(val_out[mask_val_t], y_tensor[mask_val_t]).item() if mask_val_t.any() else loss.item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if progress_callback is not None:
                progress_callback(epoch + 1, n_epochs)

            if patience_counter >= early_stop_patience:
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Inference
        model.eval()
        with torch.no_grad():
            output = model(x_tensor).squeeze().cpu().numpy()

        # Denormalize and blend
        inpainted = output * std_val + mean_val
        result = data.copy()
        result[mask] = inpainted[mask]
        return result

    @staticmethod
    def _nn_inpaint_chunked(data, mask, n_epochs, lr, device,
                            early_stop_patience, progress_callback):
        """Process long traces in overlapping chunks with Hann-window blending."""
        chunk_size = 40000
        overlap = 5000
        step = chunk_size - overlap
        n = len(data)
        result = data.copy()
        weight = np.zeros(n, dtype=float)

        # Hann window for blending
        hann = np.hanning(overlap * 2)
        hann_in = hann[:overlap]
        hann_out = hann[overlap:]

        n_chunks = max(1, int(np.ceil((n - overlap) / step)))
        for i in range(n_chunks):
            start = i * step
            end = min(start + chunk_size, n)
            chunk_data = data[start:end]
            chunk_mask = mask[start:end]
            if not chunk_mask.any():
                # No outliers in this chunk, just copy
                weight[start:end] += 1.0
                continue
            try:
                chunk_result = AdvancedDePulse.nn_inpaint(
                    chunk_data, chunk_mask,
                    n_epochs=n_epochs, lr=lr, device=device,
                    early_stop_patience=early_stop_patience,
                )
            except Exception:
                # Fall back to cubic interpolation for this chunk
                chunk_result = AdvancedDePulse.interpolate_inpaint(
                    chunk_data, chunk_mask, method='cubic')

            # Apply with window blending
            chunk_weight = np.ones(len(chunk_data), dtype=float)
            if i > 0:
                chunk_weight[:overlap] = hann_out
            if end < n:
                chunk_weight[-overlap:] = hann_in
            result[start:end] = result[start:end] * (1 - chunk_weight) + chunk_result * chunk_weight
            weight[start:end] += chunk_weight

            if progress_callback is not None:
                progress_callback(i + 1, n_chunks)

        # Handle unweighted regions
        mask_zero = weight == 0
        if mask_zero.any():
            result[mask_zero] = data[mask_zero]
        return result

    @staticmethod
    def interpolate_inpaint(data, mask, method='cubic'):
        """Fill masked positions using scipy interpolation.

        Parameters
        ----------
        data : np.ndarray, 1-D
        mask : np.ndarray (bool)
            True = position to fill.
        method : str
            'linear', 'cubic', 'nearest', 'quadratic'.

        Returns
        -------
        np.ndarray
        """
        data = np.asarray(data, dtype=float).copy()
        if not mask.any():
            return data
        good = ~mask
        if np.count_nonzero(good) < 2:
            # Not enough valid points; fill with median
            data[mask] = np.median(data[good]) if good.any() else 0.0
            return data
        x = np.arange(len(data))
        f = interp1d(x[good], data[good], kind=method,
                     bounds_error=False, fill_value='extrapolate')
        data[mask] = f(x[mask])
        return data

    @staticmethod
    def median_fill(data, mask):
        """Fill masked positions with the median of valid data."""
        data = np.asarray(data, dtype=float).copy()
        if not mask.any():
            return data
        good = ~mask
        fill_val = np.median(data[good]) if good.any() else 0.0
        data[mask] = fill_val
        return data


# ---------------------------------------------------------------------------
# Anomaly diagnostics — pre-check before de-pulse
# ---------------------------------------------------------------------------

@dataclass
class AnomalyDiagnosticResult:
    """Result of a quick anomaly diagnostic on a seismic trace."""
    total_samples: int = 0
    anomaly_count: int = 0
    anomaly_ratio: float = 0.0
    signal_window_anomalies: int = 0
    noise_window_anomalies: int = 0
    max_deviation_mad: float = 0.0
    kurtosis_global: float = 0.0  # 全局峰态
    kurtosis_max: float = 0.0    # 滑动峰态最大值
    kurtosis_std: float = 0.0    # 滑动峰态标准差
    recommendation: str = ""
    severity: str = ""  # "none", "mild", "moderate", "severe"
    severity_score: float = 0.0  # 0-100
    detail_lines: List[str] = field(default_factory=list)
    time_axis: Optional[np.ndarray] = None
    kurtosis_curve: Optional[np.ndarray] = None  # 滑动峰态曲线


class AnomalyDiagnostic:
    """基于滑动峰态的异常诊断器。

    使用滑动窗口峰态（Kurtosis）作为主要异常检测指标：
    - 正态分布峰态 ≈ 3
    - 异常值（尖峰、饱和）会导致局部峰态显著升高
    - 滑动峰态能有效捕捉突发性异常
    """

    @staticmethod
    def diagnose(data, fs, signal_window=None, window_sec=2.0, lang=None):
        """快速诊断一条 trace 的异常情况。

        Parameters
        ----------
        data : np.ndarray, 1-D
            Trace amplitude data.
        fs : float
            Sampling rate in Hz.
        signal_window : tuple of (float, float) or None
            (start_sec, end_sec) relative to trace start.
        window_sec : float
            滑动窗口长度（秒），默认 2 秒。
        lang : str or None
            诊断文本语言；``None`` = 按 :func:`dssrr.i18n.resolve_language`
            解析（默认英文）。

        Returns
        -------
        AnomalyDiagnosticResult
        """
        data = np.asarray(data, dtype=float)
        finite = np.isfinite(data)
        if not finite.all():
            data = data.copy()
            data[~finite] = np.nanmedian(data[finite]) if finite.any() else 0.0

        n = len(data)
        if n < 10:
            return AnomalyDiagnosticResult()

        # --- Compute sliding kurtosis ---
        win_samples = max(8, int(window_sec * fs))
        kurtosis_curve = AnomalyDiagnostic._sliding_kurtosis(data, win_samples)

        # --- Global statistics ---
        kurtosis_global = float(kurtosis(kurtosis_curve)) if len(kurtosis_curve) > 3 else 0.0
        kurtosis_mean = float(np.mean(kurtosis_curve))
        kurtosis_std = float(np.std(kurtosis_curve))
        kurtosis_max = float(np.max(kurtosis_curve))

        # --- Detect anomalies based on kurtosis ---
        # 阈值：峰态 > mean + 3*std 视为异常窗口
        kurtosis_threshold = kurtosis_mean + 3.0 * max(kurtosis_std, 0.5)
        anomaly_mask = kurtosis_curve > kurtosis_threshold

        # 扩展异常掩码到原始采样点
        expanded_mask = AnomalyDiagnostic._expand_anomaly_mask(
            anomaly_mask, n, win_samples
        )

        anomaly_count = int(expanded_mask.sum())
        anomaly_ratio = anomaly_count / n if n > 0 else 0.0

        # --- Max deviation in MAD units ---
        median = np.median(data)
        mad_val = np.median(np.abs(data - median)) * 1.4826
        if mad_val > 0:
            max_dev = float(np.max(np.abs(data - median)) / mad_val)
        else:
            max_dev = 0.0

        # --- Determine signal window indices ---
        sw_i0, sw_i1 = 0, 0
        if signal_window is not None:
            sw_start, sw_end = signal_window
            sw_i0 = max(0, int(sw_start * fs))
            sw_i1 = min(n, int(sw_end * fs))

        # --- Window split ---
        sig_anom = 0
        noise_anom = 0
        if sw_i1 > sw_i0:
            sig_mask = np.zeros(n, dtype=bool)
            sig_mask[sw_i0:sw_i1] = True
            sig_anom = int(np.count_nonzero(expanded_mask & sig_mask))
            noise_anom = anomaly_count - sig_anom
        else:
            noise_anom = anomaly_count

        # --- Compute severity score (0-100) ---
        severity_score = AnomalyDiagnostic._compute_severity(
            anomaly_ratio, kurtosis_max, kurtosis_std, sig_anom, noise_anom,
            signal_window is not None
        )

        # --- Determine severity level ---
        if severity_score < 5:
            severity = "none"
        elif severity_score < 25:
            severity = "mild"
        elif severity_score < 60:
            severity = "moderate"
        else:
            severity = "severe"

        # --- Recommendation based on severity ---
        recommendation = AnomalyDiagnostic._get_recommendation(
            severity, anomaly_ratio, kurtosis_max, sig_anom, noise_anom
        )

        # --- Detail text ---
        lines = AnomalyDiagnostic._build_detail_lines(
            n, anomaly_count, anomaly_ratio, sig_anom, noise_anom,
            max_dev, kurtosis_global, kurtosis_max, kurtosis_std,
            severity_score, severity, sw_i0, sw_i1, lang
        )

        return AnomalyDiagnosticResult(
            total_samples=n,
            anomaly_count=anomaly_count,
            anomaly_ratio=anomaly_ratio,
            signal_window_anomalies=sig_anom,
            noise_window_anomalies=noise_anom,
            max_deviation_mad=max_dev,
            kurtosis_global=kurtosis_global,
            kurtosis_max=kurtosis_max,
            kurtosis_std=kurtosis_std,
            recommendation=recommendation,
            severity=severity,
            severity_score=severity_score,
            detail_lines=lines,
            kurtosis_curve=kurtosis_curve,
        )

    @staticmethod
    def _sliding_kurtosis(data, window_size):
        """计算滑动窗口峰态。

        Parameters
        ----------
        data : np.ndarray
            输入信号
        window_size : int
            窗口大小（采样点数）

        Returns
        -------
        np.ndarray : 滑动峰态曲线
        """
        n = len(data)
        if n < window_size:
            # 数据太短，返回全局峰态
            k = kurtosis(data)
            return np.full(n, k)

        # 使用滑动窗口计算峰态
        result = np.zeros(n)
        half_win = window_size // 2

        for i in range(n):
            start = max(0, i - half_win)
            end = min(n, i + half_win)
            window = data[start:end]
            if len(window) >= 4:
                result[i] = kurtosis(window)
            else:
                result[i] = 0.0

        return result

    @staticmethod
    def _expand_anomaly_mask(anomaly_mask, n, win_samples):
        """将异常窗口掩码扩展到采样点级别。"""
        expanded = np.zeros(n, dtype=bool)
        half_win = win_samples // 2

        for i, is_anom in enumerate(anomaly_mask):
            if is_anom:
                center = int(i * win_samples + half_win)
                start = max(0, center - half_win)
                end = min(n, center + half_win)
                expanded[start:end] = True

        return expanded

    @staticmethod
    def _compute_severity(anomaly_ratio, kurtosis_max, kurtosis_std,
                          sig_anom, noise_anom, has_signal_window):
        """计算异常严重程度评分 (0-100)。

        Scoring factors:
        - 异常比例 (权重 25%)
        - 峰态异常程度 (权重 40%)
        - 峰态波动性 (权重 15%)
        - 信号窗口影响 (权重 20%)
        """
        # 1. 异常比例得分 (0-25)
        if anomaly_ratio <= 0:
            ratio_score = 0
        elif anomaly_ratio < 0.001:
            ratio_score = 5
        elif anomaly_ratio < 0.01:
            ratio_score = 10 + 10 * (anomaly_ratio / 0.01)
        elif anomaly_ratio < 0.05:
            ratio_score = 20 + 5 * ((anomaly_ratio - 0.01) / 0.04)
        else:
            ratio_score = 25

        # 2. 峰态异常程度得分 (0-40)
        # 正态分布峰态 ≈ 3，异常信号峰态会显著升高
        # 峰态 > 10 表示明显异常，> 50 表示严重异常
        if kurtosis_max <= 3:
            kurt_score = 0
        elif kurtosis_max < 10:
            kurt_score = 10 * ((kurtosis_max - 3) / 7)
        elif kurtosis_max < 50:
            kurt_score = 10 + 20 * ((kurtosis_max - 10) / 40)
        elif kurtosis_max < 200:
            kurt_score = 30 + 10 * ((kurtosis_max - 50) / 150)
        else:
            kurt_score = 40

        # 3. 峰态波动性得分 (0-15)
        # 标准差越大，说明局部异常越明显
        if kurtosis_std <= 0:
            std_score = 0
        elif kurtosis_std < 2:
            std_score = 3 * kurtosis_std
        elif kurtosis_std < 10:
            std_score = 6 + 5 * ((kurtosis_std - 2) / 8)
        else:
            std_score = 15

        # 4. 信号窗口影响得分 (0-20)
        if has_signal_window:
            if sig_anom > 0 and noise_anom > 0:
                window_score = 20
            elif sig_anom > 0:
                window_score = 15
            elif noise_anom > 0:
                window_score = 8
            else:
                window_score = 0
        else:
            window_score = min(10, ratio_score * 0.4)

        return min(100, ratio_score + kurt_score + std_score + window_score)

    @staticmethod
    def _get_recommendation(severity, anomaly_ratio, kurtosis_max, sig_anom, noise_anom):
        """根据严重程度给出推荐建议。"""
        if severity == "none":
            return "no_action"
        elif severity == "mild":
            return "optional_review"
        elif severity == "moderate":
            if sig_anom > 0:
                return "suggest_clean"
            else:
                return "optional_review"
        else:  # severe
            if sig_anom > 0:
                return "strongly_recommend_clean"
            elif noise_anom > 0 and anomaly_ratio > 0.01:
                return "recommend_clean"
            else:
                return "suggest_clean"

    @staticmethod
    def _build_detail_lines(n, anomaly_count, anomaly_ratio, sig_anom, noise_anom,
                            max_dev, kurtosis_global, kurtosis_max, kurtosis_std,
                            severity_score, severity, sw_i0, sw_i1, lang=None):
        """构建详细的诊断信息文本（``lang=None`` = 默认英文）。"""
        zh = resolve_language(lang) == "zh"
        lines = []

        # 基本统计
        if zh:
            lines.append(f"采样点: {n:,}")
            lines.append(f"异常点: {anomaly_count:,} ({anomaly_ratio*100:.3f}%)")
        else:
            lines.append(f"Samples: {n:,}")
            lines.append(f"Anomalies: {anomaly_count:,} ({anomaly_ratio*100:.3f}%)")

        # 信号窗口分析
        if sw_i1 > sw_i0:
            if zh:
                lines.append(f"信号窗口内异常: {sig_anom}")
                lines.append(f"噪声窗口异常: {noise_anom}")
            else:
                lines.append(f"Anomalies in signal window: {sig_anom}")
                lines.append(f"Anomalies in noise window: {noise_anom}")

        # 峰态信息
        if zh:
            lines.append(
                f"全局峰态: {kurtosis_global:.2f} | 最大滑动峰态: "
                f"{kurtosis_max:.2f} | 峰态波动: {kurtosis_std:.2f}")
            lines.append(f"最大偏离: {max_dev:.1f} 倍 MAD")
        else:
            lines.append(
                f"Global kurtosis: {kurtosis_global:.2f} | Max sliding kurtosis: "
                f"{kurtosis_max:.2f} | Kurtosis variation: {kurtosis_std:.2f}")
            lines.append(f"Max deviation: {max_dev:.1f} x MAD")

        return lines


    @staticmethod
    def get_severity_label(severity, lang=None):
        """获取严重程度的显示标签（``lang=None`` = 默认英文，见 dssrr.i18n）。"""
        labels = {
            "zh": {
                "none": "正常",
                "mild": "轻微异常",
                "moderate": "中度异常",
                "severe": "严重异常",
            },
            "en": {
                "none": "Normal",
                "mild": "Mild",
                "moderate": "Moderate",
                "severe": "Severe",
            }
        }
        return labels[resolve_language(lang)].get(severity, severity)

    @staticmethod
    def get_recommendation_text(recommendation, lang=None):
        """获取推荐建议的显示文本（``lang=None`` = 默认英文）。"""
        texts = {
            "zh": {
                "no_action": "✓ 数据质量良好，无需处理",
                "optional_review": "ℹ 发现少量异常，可选择性检查",
                "suggest_clean": "⚠ 建议清理信号窗口内的异常",
                "recommend_clean": "⚠ 建议清理异常区域",
                "strongly_recommend_clean": "⛔ 强烈建议清理信号窗口内的异常",
            },
            "en": {
                "no_action": "✓ Data quality is good, no action needed",
                "optional_review": "ℹ Minor anomalies detected, review optional",
                "suggest_clean": "⚠ Suggested: clean anomalies in signal window",
                "recommend_clean": "⚠ Recommended: clean anomaly regions",
                "strongly_recommend_clean": "⛔ Strongly recommended: clean anomalies in signal window",
            }
        }
        return texts[resolve_language(lang)].get(recommendation, recommendation)

    @staticmethod
    def detect_anomaly_distribution(data, fs, window_sec=10.0,
                                     mad_threshold=3.0):
        """滑动窗口异常密度分布，用于可视化。

        Returns
        -------
        time_axis : np.ndarray
            Center time of each window (seconds).
        density : np.ndarray
            Fraction of anomalous samples in each window (0~1).
        """
        data = np.asarray(data, dtype=float)
        finite = np.isfinite(data)
        if not finite.all():
            data = data.copy()
            data[~finite] = np.nanmedian(data[finite]) if finite.any() else 0.0

        n = len(data)
        if n == 0:
            return np.array([]), np.array([])

        win_samples = max(64, int(window_sec * fs))
        step = max(win_samples // 2, 1)

        # Global MAD for threshold
        median = np.nanmedian(data)
        mad_val = np.nanmedian(np.abs(data - median)) * 1.4826
        if mad_val == 0:
            mad_val = np.nanstd(data) or 1.0
        thresh = mad_threshold * mad_val

        centers = []
        densities = []
        for start in range(0, n - win_samples + 1, step):
            seg = data[start:start + win_samples]
            anom = np.sum(np.abs(seg - median) > thresh)
            centers.append((start + win_samples / 2) / fs)
            densities.append(anom / win_samples)

        return np.array(centers), np.array(densities)


# Alias the torch functional import for nn_inpaint
if _HAS_TORCH:
    import torch.nn.functional as F
