"""DSSRR 内置的异常段「频谱替换」修复模块。

本子包提供三种可直接调用的修复器（Replacer），它们是 DSSRR 算法层的
应用侧封装，供手动修复面板与批量修复流水线使用：

- :class:`ReferenceSpectrumReplacer`
  双侧参考频谱修复：取异常段前后的正常段估计目标功率谱，据此合成替代波形。
  这是 DSSRR 的核心方法，实现位于 :mod:`dssrr.core`。
- :class:`LayeredReferenceSpectrumReplacer`
  分层参考频谱修复：在双侧参考结果之上，再叠加一层用户指定频段的
  参考引导层（例如让超低频月震自由振荡成分主导修复），实现见本目录
  ``layered_reference_spectrum.py``。
- :class:`MoonquakeProtectedSpikeReplacer`
  月震保护型局部尖峰修复：只修孤立尖峰/短削顶平台，绝不重建长事件段，
  实现见本目录 ``moonquake_protected_spike.py``。

关于「单一真源」：本子包中的 :mod:`~dssrr.repair_lib.anomaly_repair.spectral`、
:mod:`~dssrr.repair_lib.anomaly_repair.verifier`、
:mod:`~dssrr.repair_lib.anomaly_repair.synthesizer`、
:mod:`~dssrr.repair_lib.anomaly_repair.config`、
:mod:`~dssrr.repair_lib.anomaly_repair.reference_spectrum` 都是**转出（re-export）
薄壳**，真正的实现统一在公开库层（``dssrr/core.py``、``dssrr/spectral.py`` 等），
以保证公开 API 与应用层调用得到逐位一致的结果。

用法::

    from dssrr.repair_lib.anomaly_repair import ReferenceSpectrumReplacer

    replacer = ReferenceSpectrumReplacer(sr=6.625)
    repaired, info = replacer.replace(data, start, end)
"""

from .reference_spectrum import (
    ReferenceSpectrumReplacer,
    replace_by_reference_spectrum,
    auto_reference_length,
)
from .layered_reference_spectrum import (
    LayeredReferenceSpectrumReplacer,
    replace_by_layered_reference_spectrum,
)
from .moonquake_protected_spike import (
    MoonquakeProtectedSpikeReplacer,
    replace_by_moonquake_protected_spikes,
)

__all__ = [
    "ReferenceSpectrumReplacer",
    "replace_by_reference_spectrum",
    "auto_reference_length",
    "LayeredReferenceSpectrumReplacer",
    "replace_by_layered_reference_spectrum",
    "MoonquakeProtectedSpikeReplacer",
    "replace_by_moonquake_protected_spikes",
]
