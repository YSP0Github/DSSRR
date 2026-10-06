# -*- coding: utf-8 -*-
"""用户可见文本的语言必须是**英文**（开源发行版的默认语言）。

背景
----
DSSRR 的界面是英文的：包内没有加载 Qt 翻译文件（``.qm``），``self.tr()`` 恒等于
源码里的英文原文。但报告出图那一块早期按**系统区域**猜语言，于是在中文 Windows
上出现了"界面英文、导出设置对话框和对比图整篇中文"的割裂。

现在语言统一由 :func:`dssrr.i18n.resolve_language` 解析：显式参数 →
环境变量 ``DSSRR_LANG`` → **英文**。这些测试锁住这条规则，防止有人再往
"猜系统区域"的方向改回去。

注意：这些测试**不导入 PyQt5**（GUI 那部分只查源码里的兜底分支），因此可以在
headless / 无 PyQt5 的环境里跑。
"""
import io
import os

import pytest

from dssrr import i18n
from dssrr.repair_lib import depulse_advanced, report_plot


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """每个用例都从"用户没设过 DSSRR_LANG"的干净状态开始。

    还要清掉**进程级语言**：GUI 的语言切换按钮会通过 ``set_language()`` 写入，
    它优先级高于环境变量。别的测试文件（test_gui_i18n.py）跑过之后会把它钉住，
    不清掉的话下面这些"环境变量优先"的断言会全部失败。
    """
    monkeypatch.delenv(i18n.ENV_VAR, raising=False)
    i18n.set_language(None)
    yield
    i18n.set_language(None)


# ---------------------------------------------------------------------------
# i18n 解析规则
# ---------------------------------------------------------------------------
def test_default_is_english():
    assert i18n.DEFAULT_LANGUAGE == "en"
    assert i18n.resolve_language() == "en"


def test_explicit_argument_wins():
    assert i18n.resolve_language("zh") == "zh"
    assert i18n.resolve_language("en") == "en"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv(i18n.ENV_VAR, "zh")
    assert i18n.resolve_language() == "zh"
    # 显式参数优先级仍高于环境变量
    assert i18n.resolve_language("en") == "en"


def test_env_var_accepts_common_spellings(monkeypatch):
    for code in ("zh", "zh_CN", "zh-Hans-CN", "Chinese", "ZH"):
        monkeypatch.setenv(i18n.ENV_VAR, code)
        assert i18n.resolve_language() == "zh", code
    for code in ("en", "en_US", "English"):
        monkeypatch.setenv(i18n.ENV_VAR, code)
        assert i18n.resolve_language() == "en", code


def test_unknown_language_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(i18n.ENV_VAR, "de")
    assert i18n.resolve_language() == "en"
    assert i18n.resolve_language("klingon") == "en"


def test_resolver_ignores_system_locale(monkeypatch):
    """核心回归点：不再读系统区域，否则中文 Windows 又会出中文。"""
    import locale

    monkeypatch.setattr(locale, "getdefaultlocale", lambda: ("zh_CN", "cp936"),
                        raising=False)
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
    assert i18n.resolve_language() == "en"


# ---------------------------------------------------------------------------
# 报告出图 / 诊断文本的默认语言
# ---------------------------------------------------------------------------
def test_report_plotter_defaults_to_english():
    plotter = report_plot.RepairReportPlotter(sr=6.625)
    assert plotter.lang == "en"
    assert plotter._t["title"] == "Seismic Data Anomaly Repair Report"


def test_report_plotter_honours_explicit_lang():
    plotter = report_plot.RepairReportPlotter(sr=6.625, lang="zh")
    assert plotter.lang == "zh"
    assert plotter._t["title"] == "地震数据异常修复报告"


def test_quick_plot_helper_defaults_to_english():
    import inspect

    sig = inspect.signature(report_plot.plot_comparison_quick)
    assert sig.parameters["lang"].default is None


def test_severity_and_recommendation_default_to_english():
    assert depulse_advanced.AnomalyDiagnostic.get_severity_label("severe") == "Severe"
    assert depulse_advanced.AnomalyDiagnostic.get_severity_label(
        "severe", lang="zh") == "严重异常"
    assert depulse_advanced.AnomalyDiagnostic.get_recommendation_text(
        "no_action").startswith("✓ Data quality")
    assert depulse_advanced.AnomalyDiagnostic.get_recommendation_text(
        "no_action", lang="zh").startswith("✓ 数据质量")


def test_detail_lines_default_to_english():
    args = (1000, 12, 0.012, 3, 9, 4.2, 3.1, 8.8, 1.2, 50, "mild", 0, 100)
    en = depulse_advanced.AnomalyDiagnostic._build_detail_lines(*args)
    zh = depulse_advanced.AnomalyDiagnostic._build_detail_lines(*args, lang="zh")
    assert en[0].startswith("Samples:")
    assert zh[0].startswith("采样点:")


# ---------------------------------------------------------------------------
# 导出设置对话框的源码级约定（避免为了跑测试而强依赖 PyQt5）
# ---------------------------------------------------------------------------
def _gui_source(name):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "dssrr", "gui", name)
    with io.open(path, encoding="utf-8") as fh:
        return fh.read()


def test_export_dialog_defaults_to_resolved_language():
    src = _gui_source("export_report_dialog.py")
    # 默认参数必须是 None（= 交给 i18n 解析），不能写死 "zh"
    assert 'lang: Optional[str] = None' in src
    assert 'lang: str = "zh"' not in src
    # 漏翻的键要回退英文，否则英文界面上会冒中文
    assert 'translations.get(self.lang, translations["en"])' in src


def test_depulse_report_language_delegates_to_i18n():
    src = _gui_source("depulse_dialog.py")
    assert "from ..i18n import resolve_language" in src
    # 旧的"猜系统区域"实现必须彻底消失
    assert "QLocale.system()" not in src
    assert "getdefaultlocale" not in src


# ---------------------------------------------------------------------------
# 旧 catalog 的"Method"列也必须是英文
# ---------------------------------------------------------------------------
def test_legacy_inferred_method_names_render_in_english():
    """老 CSV 没有 method 列时 ``_infer_method()`` 会合成中文描述，
    英文界面的 Method 列不能因此冒出中文（需要 PyQt5）。"""
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])      # noqa: F841

    from dssrr.gui.viewer import _infer_method, method_name

    inferred = [
        _infer_method("event", None, 10, "osc"),
        _infer_method("event", None, 10, "det"),
        _infer_method("event", None, 10, "moon"),
        _infer_method("gap", None, 10),
        _infer_method("repair", "spike", 0.2),
        _infer_method("repair", "spike", 1.0),
        _infer_method("repair", "zero", 0.2),
        _infer_method("repair", "zero", 10.0),
        _infer_method("repair", "freeze", 10.0),
        _infer_method("repair", "step", 10.0),
    ]
    for raw in inferred:
        shown = method_name(raw)
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in shown), \
            "Method 列出现中文: %r -> %r" % (raw, shown)
