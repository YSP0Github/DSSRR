# -*- coding: utf-8 -*-
"""
Shared statistical-fidelity metric library for DSSRR experiments
(Task 6 Earth transfer & Task 7 DL synthetic).

All metrics operate on the anomaly segment and are "lower is better"
except psd_cos (higher is better). Formulas match E2r_v5_batch.py,
parameterized by sampling rate.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import wasserstein_distance


def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _acf(x, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    full = np.correlate(x, x, mode="full")[n - 1:]
    full = full / (full[0] + 1e-12)
    lags = np.arange(min(max_lag + 1, n))
    return full[lags]


def acf_l2(truth, rep, max_lag: int) -> float:
    a_t = _acf(truth, max_lag)
    a_r = _acf(rep, max_lag)
    L = min(len(a_t), len(a_r))
    return float(np.sqrt(np.mean((a_t[:L] - a_r[:L]) ** 2)))


def envelope_dist(truth, rep) -> float:
    e_t = np.abs(signal.hilbert(np.asarray(truth, dtype=float)))
    e_r = np.abs(signal.hilbert(np.asarray(rep, dtype=float)))
    return float(wasserstein_distance(e_t, e_r))


def _spec_entropy(x, fs: float, nperseg: int) -> float:
    x = np.asarray(x, dtype=float)
    nperseg = int(min(nperseg, max(8, len(x) // 2)))
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, scaling="density")
    P = Pxx / (Pxx.sum() + 1e-12)
    P = np.clip(P, 1e-12, None)
    return float(-np.sum(P * np.log(P)) / np.log(len(P)))


def spec_entropy_err(truth, rep, fs: float, nperseg: int = 512) -> float:
    return abs(_spec_entropy(truth, fs, nperseg)
               - _spec_entropy(rep, fs, nperseg))


def psd_cosine(truth, rep, fs: float, nperseg: int = 1024) -> float:
    t = np.asarray(truth, dtype=float)
    r = np.asarray(rep, dtype=float)
    nperseg = int(min(nperseg, max(8, min(len(t), len(r)) // 2)))
    _, p_t = signal.welch(t, fs=fs, nperseg=nperseg)
    _, p_r = signal.welch(r, fs=fs, nperseg=nperseg)
    return float(np.dot(p_t, p_r)
                 / (np.linalg.norm(p_t) * np.linalg.norm(p_r) + 1e-12))


def psd_welch(x, fs: float, nperseg: int = 1024):
    x = np.asarray(x, dtype=float)
    nperseg = int(min(nperseg, max(8, len(x) // 2)))
    f, p = signal.welch(x, fs=fs, nperseg=nperseg)
    return f, p


def all_metrics(truth_seg, rep_seg, fs: float, acf_max_lag: int,
                entropy_nperseg: int = 512, psd_nperseg: int = 1024) -> dict:
    truth_seg = np.asarray(truth_seg, dtype=float)
    rep_seg = np.asarray(rep_seg, dtype=float)
    return {
        "rmse_anom": rmse(truth_seg, rep_seg),
        "wasserstein": float(wasserstein_distance(truth_seg, rep_seg)),
        "acf_l2": acf_l2(truth_seg, rep_seg, acf_max_lag),
        "envelope": envelope_dist(truth_seg, rep_seg),
        "spec_ent": spec_entropy_err(truth_seg, rep_seg, fs,
                                     entropy_nperseg),
        "psd_cos": psd_cosine(truth_seg, rep_seg, fs, psd_nperseg),
    }
