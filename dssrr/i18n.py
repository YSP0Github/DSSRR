# -*- coding: utf-8 -*-
"""DSSRR 文本语言解析（GUI / 报告出图共用的唯一真源）。

为什么需要单独一个模块
----------------------
DSSRR 的**界面**是英文的：包内没有加载 Qt 翻译文件（``.qm``），
``self.tr()`` 因此恒等于源码里的英文原文。但**报告出图**（导出设置对话框、
对比图标题与轴标签、质量指标表）早期按**系统区域**猜语言，于是中文 Windows
上导出对话框和报告整篇都是中文，与界面语言对不上——这正是"开源版 GUI 里
导出设置默认中文"的原因。

现在统一按下面的顺序解析，**不再读取系统区域**：

1. 调用方显式传入的 ``lang`` 参数（例如查看器的 ``set_tool_language``）；
2. 进程内已设置的当前语言（:func:`set_language`，GUI 的语言切换按钮走这里）；
3. 环境变量 ``DSSRR_LANG``（``zh`` / ``en`` / ``zh_CN`` / ``Chinese`` 等写法都认）；
4. 兜底 ``DEFAULT_LANGUAGE``，即 **英文**，与界面保持一致。

想让报告出中文，显式指定即可，不必改代码::

    DSSRR_LANG=zh python -m dssrr.gui          # Linux / macOS
    set DSSRR_LANG=zh && python -m dssrr.gui   # Windows cmd

也可以直接在代码里切::

    from dssrr.i18n import set_language
    set_language("zh")        # 之后所有 resolve_language() 都返回 "zh"
    set_language(None)        # 清除，回到"环境变量 → 英文"
"""

from __future__ import annotations

import os
from typing import Optional

#: 兜底语言。与界面（未加载 Qt 翻译的英文原文）保持一致。
DEFAULT_LANGUAGE = "en"

#: 覆盖语言用的环境变量名。
ENV_VAR = "DSSRR_LANG"

#: 内置翻译表支持的语言代码。
SUPPORTED = ("en", "zh")

#: 进程内当前语言（``None`` = 未显式设置，走环境变量/兜底）。
#: 由 GUI 的语言切换按钮通过 :func:`set_language` 写入。
_active_language: Optional[str] = None

#: 各种写法的别名 → 规范代码。
_ALIASES = {
    "en": "en", "eng": "en", "english": "en", "en-us": "en", "en-gb": "en",
    "c": "en", "posix": "en",
    "zh": "zh", "cn": "zh", "chi": "zh", "zho": "zh",
    "chinese": "zh", "zh-cn": "zh", "zh-sg": "zh", "zh-hans": "zh",
    "zh-tw": "zh", "zh-hk": "zh", "zh-hant": "zh",
}


def normalize_language(code: Optional[str]) -> Optional[str]:
    """把用户给的任意语言写法规范成 ``"en"`` / ``"zh"``；认不出返回 ``None``。

    >>> normalize_language("zh_CN")      # 下划线写法
    'zh'
    >>> normalize_language("English")    # 大小写无关
    'en'
    >>> normalize_language("de") is None
    True
    """
    if not code:
        return None
    text = str(code).strip().lower().replace("_", "-")
    if not text:
        return None
    if text in _ALIASES:
        return _ALIASES[text]
    # 形如 "zh-Hans-CN" / "en-US" 的带地区后缀写法
    head = text.split("-")[0]
    return _ALIASES.get(head)


def set_language(code: Optional[str]) -> Optional[str]:
    """设置进程内当前语言（``None`` = 清除，回到环境变量/兜底）。

    GUI 的语言切换按钮调用它；设置之后所有 :func:`resolve_language` 都会返回
    这个语言（除非调用方显式传了别的）。

    Returns
    -------
    str or None
        实际生效的语言代码；``code`` 无法识别时返回 ``None``（表示"已清除"）。
    """
    global _active_language
    _active_language = normalize_language(code)
    return _active_language


def get_language(default: str = DEFAULT_LANGUAGE) -> str:
    """当前生效的语言：进程内设置 → ``DSSRR_LANG`` → ``default`` → 英文。"""
    return (_active_language
            or normalize_language(os.environ.get(ENV_VAR))
            or normalize_language(default)
            or DEFAULT_LANGUAGE)


def resolve_language(explicit: Optional[str] = None,
                     default: str = DEFAULT_LANGUAGE) -> str:
    """解析出本次要用的语言代码（``"en"`` 或 ``"zh"``）。

    Parameters
    ----------
    explicit : str or None
        调用方显式指定的语言；``None`` / 空串表示"没指定"，走
        :func:`get_language`。
    default : str
        环境变量也无效时的兜底，默认英文。

    Returns
    -------
    str
        ``"en"`` 或 ``"zh"``。
    """
    return normalize_language(explicit) or get_language(default)
