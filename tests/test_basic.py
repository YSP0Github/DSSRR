# -*- coding: utf-8 -*-
"""DSSRR 公开 API 的基础单元测试。

覆盖最小可用的契约，全部只依赖 numpy + pytest（不需要 obspy / PyQt5），
因此可以在任何环境下快速运行：

- :func:`test_basic_repair` —— ``DSSRR.repair`` 能把一段置零的缺口填回非零值，
  且**异常段以外**的样本逐位保持不变（DSSRR 的硬约束之一）。
- :func:`test_auto_length` —— ``auto_reference_length`` 的取值符合论文约定
  （``L = clip(anomaly/2, 120 s, 600 s)``）。
- :func:`test_invalid_sr` —— 非法采样率抛出 ``ValueError``。

运行::

    pytest tests/test_basic.py -q

更全面的契约测试见 ``tests/test_single_source_of_truth.py``（单一真源一致性）
与 ``tests/test_repair_engine.py``（ObsPy 应用层，缺失 obspy 时自动跳过）。
"""

import numpy as np
import pytest
from dssrr import DSSRR, auto_reference_length


def make_test_data(sr=6.625, dur=1800, seed=0):
    """构造一条「常数 + 高斯噪声 + 慢正弦」的合成台站记录。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    noise = rng.normal(0, 1, len(t))
    return 450 + 2 * noise + np.sin(2 * np.pi * 0.1 * t)


def test_basic_repair():
    sr = 6.625
    data = make_test_data(sr)
    # 把 [800s, 860s] 置零，模拟一段缺失
    start, end = int(800 * sr), int(860 * sr)
    corrupted = data.copy()
    corrupted[start:end + 1] = 0

    model = DSSRR(sr, reference_sec=200, seed=42)
    repaired = model.repair(corrupted, start, end)

    # 形状不变，缺口被填回非零值
    assert repaired.shape == data.shape
    assert not np.allclose(repaired[start:end + 1], 0)
    # 异常段以外的样本必须逐位保持不变
    assert np.allclose(repaired[:start], corrupted[:start])
    assert np.allclose(repaired[end + 1:], corrupted[end + 1:])


def test_auto_length():
    sr = 6.625
    # L = clip(anomaly/2, 120 s, 600 s)
    assert auto_reference_length(int(60 * sr), sr) == 120.0
    assert auto_reference_length(int(2000 * sr), sr) == 600.0


def test_invalid_sr():
    with pytest.raises(ValueError):
        DSSRR(sr=-1)
