# -*- coding: utf-8 -*-
"""``resolve_paths`` / ``_find_repaired_path`` 修复库路径解析的回归测试。

背景
----
批量去异常 / CLI 都把修复后文件写到 ``<fix_root>/YYYY/MM/<filename>``，与原始
库的目录结构无关。查看器的 ``resolve_paths`` 早期用 ``os.path.relpath`` 直接镜像
``raw_root`` 的目录结构——当原始库是扁平目录而修复库是 ``YYYY/MM/`` 镜像（或者
反过来）时，拼接出来的路径根本不存在，界面就显示"该台站/日期没有文件（尚未修复）"。

修复后的解析顺序：

1. 同布局镜像（``<fix_root>/<rel>``），最常见且最快。
2. 批量 ``YYYY/MM/`` 约定（``<fix_root>/YYYY/MM/<filename>``），兼容扁平原始库。
3. 递归扫描（兜底），按 ``parse_mseed_name`` 解析的 ``(station, day)`` 匹配；
   任意目录结构都能找到。

所有路径只用 ``tmp_path`` 临时构造，不污染仓库。
"""
import os

import pytest

pytest.importorskip("PyQt5")  # viewer.py 顶层会 import PyQt5 widgets

from PyQt5.QtWidgets import QApplication

# 单实例 QApplication，让 viewer 顶层导入不爆
_app = QApplication.instance() or QApplication([])                          # noqa: F841

from dssrr.discovery import find_counterpart  # noqa: E402
from dssrr.gui.viewer import resolve_paths  # noqa: E402 - 必须在 QApplication 之后


def _touch(path):
    """``mkdir -p`` + ``touch`` 一个文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def _index_of(raw_root):
    """构造一个像查看器 _index 那样的 ``{(stream, day): raw_path}`` 映射。"""
    index = {}
    for root, _dirs, files in os.walk(raw_root):
        for fn in files:
            if not fn.endswith(".mseed"):
                continue
            from dssrr.discovery import parse_mseed_name
            info = parse_mseed_name(fn)
            if info is None:
                continue
            net, sta, loc, cha, day = info
            label = f"{net}.{sta}"
            if loc or cha:
                label += f".{loc}.{cha}" if loc and cha else (f".{loc}" if loc else f".{cha}")
            index.setdefault((label, day), os.path.join(root, fn))
    return index


def test_resolve_paths_flat_raw_vs_mirror_fix(tmp_path):
    """用户的精确场景：原始库扁平 + 修复库 ``YYYY/MM/`` 镜像。"""
    raw = tmp_path / "raw"
    fix = tmp_path / "fix"
    raw.mkdir()
    # 原始库：3 个片段文件直接放在 raw 根目录下（扁平）
    raw_paths = [
        "XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed",
        "XA.S15.01.MHZ.19760113_070152-19760113_190149.mseed",
        "XA.S16.01.MHZ.19760113_070152-19760113_190149.mseed",
    ]
    for fn in raw_paths:
        _touch(raw / fn)
    # 修复库：同 3 个文件 + 31 个日文件，按 YYYY/MM/ 镜像
    (fix / "1976" / "01").mkdir(parents=True)
    for fn in raw_paths:
        _touch(fix / "1976" / "01" / fn)
    # 跑
    index = _index_of(str(raw))
    for fn in raw_paths:
        stream, day = "XA." + fn.split(".")[1] + "." + fn.split(".")[2] + "." + fn.split(".")[3], fn.split(".")[4].split("_")[0]
        raw_p, fix_p, err = resolve_paths(stream, day, str(raw), str(fix), index)
        assert err is None, (stream, day, err)
        assert os.path.exists(fix_p), fix_p


def test_resolve_paths_same_layout_mirror(tmp_path):
    """历史行为：raw 与 fix 同布局（都是 ``YYYY/MM/`` 或都是扁平），应走快路径。"""
    raw = tmp_path / "raw"
    fix = tmp_path / "fix"
    (raw / "1976" / "01").mkdir(parents=True)
    (fix / "1976" / "01").mkdir(parents=True)
    fn = "XA.S12.19760113.mseed"
    _touch(raw / "1976" / "01" / fn)
    _touch(fix / "1976" / "01" / fn)
    index = _index_of(str(raw))
    # 实际界面下 dropdown 里出现的完整标签是 "XA.S12"
    raw_p, fix_p, err = resolve_paths("XA.S12", "19760113", str(raw), str(fix), index)
    assert err is None
    assert os.path.exists(fix_p)
    # 快路径走的是同布局镜像（rel = "1976/01/<filename>"），不存在才是 YYYY/MM 兜底
    assert fix_p.replace("\\", "/").endswith("1976/01/" + fn)


def test_resolve_paths_completely_arbitrary_layout(tmp_path):
    """兜底：fix_root 的目录布局完全任意，也能找到。"""
    raw = tmp_path / "raw"
    fix = tmp_path / "fix" / "deep" / "nested" / "dir"
    raw.mkdir()
    _touch(raw / "XA.S12.19760113.mseed")
    _touch(fix / "XA.S12.19760113.mseed")
    index = _index_of(str(raw))
    raw_p, fix_p, err = resolve_paths("XA.S12", "19760113", str(raw), str(fix), index)
    assert err is None
    assert fix_p == str(fix / "XA.S12.19760113.mseed")


def test_resolve_paths_returns_fix_missing_when_really_missing(tmp_path):
    """修复文件真的不存在时，应仍报 FIX_MISSING 并带回"以为在那"的路径。"""
    raw = tmp_path / "raw"
    fix = tmp_path / "fix"
    raw.mkdir()
    _touch(raw / "XA.S12.19760113.mseed")
    # 修复库里只放一个别的日期的文件
    _touch(fix / "XA.S12.19760114.mseed")
    index = _index_of(str(raw))
    raw_p, fix_p, err = resolve_paths("XA.S12", "19760113", str(raw), str(fix), index)
    assert err == "FIX_MISSING"
    assert raw_p == str(raw / "XA.S12.19760113.mseed")
    # fix_path 是"以为应该在那"的路径，不是真正找到的（不要求存在）
    assert "XA.S12.19760113.mseed" in fix_p


def test_resolve_paths_raw_missing(tmp_path):
    """原始库缺文件：返回 RAW_MISSING，fix_path 为 None。"""
    raw = tmp_path / "raw"
    fix = tmp_path / "fix"
    raw.mkdir()
    fix.mkdir()
    raw_p, fix_p, err = resolve_paths("XA.S12", "19760113", str(raw), str(fix), {})
    assert err == "RAW_MISSING"
    assert raw_p is None and fix_p is None


def test_resolve_paths_fix_root_nonexistent(tmp_path):
    """修复库目录根本不存在：返回 FIX_MISSING。"""
    raw = tmp_path / "raw"
    raw.mkdir()
    _touch(raw / "XA.S12.19760113.mseed")
    index = _index_of(str(raw))
    raw_p, fix_p, err = resolve_paths(
        "XA.S12", "19760113", str(raw), str(tmp_path / "no_such_dir"), index)
    assert err == "FIX_MISSING"


def test_find_counterpart_does_not_pick_source_file(tmp_path):
    """兜底扫描必须排除源文件本身，否则会自指。"""
    root = tmp_path / "db"
    root.mkdir()
    fn = "XA.S12.19760113.mseed"
    path = str(root / fn)
    # from_root 与 to_root 相同时，递归扫描命中时必须过滤掉自指
    assert find_counterpart(path, str(root), str(root)) is None


def test_find_counterpart_reverse_direction(tmp_path):
    """反向映射（fix → raw）同样成立：verify_month 走的就是这个方向。"""
    raw = tmp_path / "raw"          # 扁平
    fix = tmp_path / "fix" / "1976" / "01"   # YYYY/MM 镜像
    raw.mkdir()
    fix.mkdir(parents=True)
    fn = "XA.S12.19760113.mseed"
    _touch(raw / fn)
    _touch(fix / fn)
    # fix → raw（verify_month 的方向）
    found = find_counterpart(str(fix / fn), str(tmp_path / "fix"), str(raw))
    assert found == str(raw / fn)
    # raw → fix（查看器的方向）
    found = find_counterpart(str(raw / fn), str(raw), str(tmp_path / "fix"))
    assert found == str(fix / fn)
