"""E7 去异常流水线参数设置（异常检测 + 修复分级 + DSSRR 参考/能量保护）。

持久化到用户目录 JSON（默认 ``~/.dssrr_settings.json``，路径统一由
:mod:`dssrr.paths` 管理），供：

- ``DbRepairViewerWindow`` 的 ``RepairSettingsDialog`` 编辑保存；
- ``ManualRepairDialog`` 的 auto 模式（手动重处理）读取应用。

.. note::
   旧版本（以及本包从中抽取出来的 ``seisy`` 桌面程序）把文件命名为
   ``~/.seisy_repair_settings.json``。读取时仍兼容该旧名，写入一律使用新名，
   因此升级不会丢失已有设置。

批处理流水线（``dssrr.repair_lib.e7_repair_build``）默认**不读**用户设置，
始终使用默认参数以保证论文结果可复现；用户自定义参数只影响 GUI 手动重处理，
除非显式加 ``--settings``。
"""

import json
import os

from ..paths import settings_read_path, user_settings_path

# 写入位置。读取位置请用 settings_read_path()，它会在新文件不存在时回退到
# 旧版文件名，从而平滑迁移已有安装。
SETTINGS_FILE = user_settings_path()

# 每组: key -> (默认值, 最小值, 最大值, 小数位数, 单位标签)
_DETECT_SPEC = {
    "spike_window_sec": (2.0, 0.5, 60.0, 1, "s"),
    "spike_k": (8.0, 2.0, 30.0, 1, ""),
    "step_k": (12.0, 3.0, 50.0, 1, ""),
    "freeze_min_sec": (30.0, 5.0, 300.0, 1, "s"),
    "rms_short_sec": (10.0, 1.0, 60.0, 1, "s"),
    "rms_bg_sec": (120.0, 10.0, 600.0, 1, "s"),
    "rms_ratio": (5.0, 2.0, 30.0, 1, "x"),
    "min_burst_sec": (5.0, 1.0, 60.0, 1, "s"),
    "low_dynamic_rms": (4.0, 0.5, 20.0, 1, ""),
    "low_dynamic_gain": (3.0, 1.0, 20.0, 1, "x"),
    "merge_gap_sec": (120.0, 10.0, 600.0, 1, "s"),
    "short_pad_sec": (5.0, 0.0, 60.0, 1, "s"),
    "long_pad_sec": (30.0, 0.0, 300.0, 1, "s"),
    "max_repair_sec": (1800.0, 300.0, 7200.0, 1, "s"),
    "event_morph_frac": (0.03, 0.0, 0.2, 3, ""),
    "event_jitter_frac": (0.04, 0.0, 0.2, 3, ""),
    "event_min_sec": (30.0, 5.0, 600.0, 1, "s"),
    "event_coda_ratio": (100.0, 10.0, 500.0, 1, "x"),
    "event_rms_smooth": (0.12, 0.0, 1.0, 2, ""),
    "event_head_mid_ratio": (1.15, 0.5, 3.0, 2, "x"),
    "event_missing_frac": (0.05, 0.0, 0.5, 2, ""),
}

_REPAIR_SPEC = {
    "lin_max_run": (2, 1, 5, 0, "samples"),
    "z_max_run": (20, 3, 100, 0, "samples"),
    "miss_frac_lo": (0.30, 0.0, 1.0, 2, ""),
    "miss_frac_hi": (0.50, 0.0, 1.0, 2, ""),
    "z_thr": (4.0, 2.0, 10.0, 1, "σ"),
    "z_bg_sec": (30.0, 5.0, 300.0, 1, "s"),
    "spike_k_sigma": (8.0, 3.0, 20.0, 1, "σ"),
    "ref_ratio": (0.5, 0.1, 1.0, 2, "xL"),
    # 论文口径：参考段 120–600 s（DSSRR 推荐默认）；旧默认 30 s 过短
    "ref_min_sec": (120.0, 30.0, 300.0, 1, "s"),
    "ref_max_sec": (600.0, 60.0, 1800.0, 1, "s"),
    "bkg_sec": (300.0, 30.0, 1200.0, 1, "s"),
    "osc_lo": (0.2, 0.05, 2.0, 2, "x"),
    "osc_hi": (4.0, 1.0, 20.0, 2, "x"),
    "std_raw_factor": (1.1, 1.0, 5.0, 2, "x"),
    "std_bkg_factor": (2.0, 1.0, 10.0, 2, "x"),
}

# 布尔开关（不适用数值 spec）：key -> 默认值
_REPAIR_BOOL_SPEC = {
    "quantize_to_int": True,
}

DETECT_KEYS = list(_DETECT_SPEC)
REPAIR_KEYS = list(_REPAIR_SPEC) + list(_REPAIR_BOOL_SPEC)


def default_settings():
    """返回默认设置副本：{'detect': {...}, 'repair': {...}}。"""
    return {
        "detect": {k: v[0] for k, v in _DETECT_SPEC.items()},
        "repair": {k: v[0] for k, v in _REPAIR_SPEC.items()}
                 | dict(_REPAIR_BOOL_SPEC),
    }


def load_settings():
    """读取设置文件；文件缺失/损坏时回退默认值。值类型按 spec 校验。

    读取路径由 :func:`dssrr.paths.settings_read_path` 决定：优先新文件名，
    其次旧版 ``.seisy_repair_settings.json``（兼容升级前的安装）。
    """
    base = default_settings()
    try:
        with open(settings_read_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return base
    for grp, spec in (("detect", _DETECT_SPEC), ("repair", _REPAIR_SPEC)):
        if not isinstance(data.get(grp), dict):
            continue
        for key, (dflt, vmin, vmax, _dec, _unit) in spec.items():
            if key in data[grp]:
                try:
                    val = type(dflt)(data[grp][key])
                    base[grp][key] = float(np_clip(val, vmin, vmax)) if isinstance(dflt, float) \
                        else int(np_clip(val, vmin, vmax))
                except (TypeError, ValueError):
                    pass
    if isinstance(data.get("repair"), dict):
        for key, dflt in _REPAIR_BOOL_SPEC.items():
            if key in data["repair"]:
                base["repair"][key] = bool(data["repair"][key])
    return base


def np_clip(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def save_settings(settings):
    """持久化设置到用户目录 JSON。"""
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def detect_spec():
    return dict(_DETECT_SPEC)


def repair_spec():
    return dict(_REPAIR_SPEC)
