"""MiniSEED 库扫描：从目录树里发现台站、日期与文件。

把"文件名怎么解析、目录怎么遍历"集中到这一处，GUI 批量去异常
(``dssrr.gui.batch_repair``) 与命令行 (``dssrr.cli``) 都从这里取用，
避免两边各写一份、规则还悄悄不一致。

**只依赖标准库**，因此库层（numpy/scipy/obspy）和 GUI 层都能安全引用，
导入本模块不会拉起 PyQt5。

支持的命名（目录结构不限，扁平目录 / 按年 / 按月 / 带站点子目录都可以）：

- **日文件** ``NET.STA.YYYYMMDD.mseed``
  （如 ``XA.S12.19760113.mseed``）——文件名里没有定位码/通道；
- **片段文件** ``NET.STA.LOC.CHA.START-END.mseed``
  （如 ``XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed``）
  ——日期取起始时间的前 8 位。

历史包袱：早期实现把目录写死成 ``raw_root/YYYY/MM/``，台站从 ``fn[3:6]``
取、日期从 ``parts[2]`` 取。于是像 ``examples/data`` 这种**扁平目录 +
片段命名**的数据一个文件都扫不到。现在只看文件名。
"""
from __future__ import annotations

import os

__all__ = ["parse_mseed_name", "scan_stations_years", "iter_scope_files",
           "diagnose_scan", "find_counterpart"]


def parse_mseed_name(fn):
    """从 MiniSEED 文件名解析出 ``(net, sta, loc, cha, day)``；识别不了返回 None。

    ``day`` 统一为 ``YYYYMMDD`` 字符串。日文件没有定位码/通道，对应位置返回
    空串（而不是猜一个值——猜出来的信息会污染界面标签）。
    """
    if not fn or not fn.lower().endswith(".mseed"):
        return None
    parts = fn.split(".")
    if len(parts) < 4 or parts[-1].lower() != "mseed":
        return None
    net, sta = parts[0], parts[1]
    if not net or not sta:
        return None
    if len(parts) == 4:
        # 日文件：NET.STA.DAY.mseed
        loc, cha, day = "", "", parts[2]
    elif len(parts) >= 6:
        # 片段：NET.STA.LOC.CHA.START-END.mseed
        loc, cha = parts[2], parts[3]
        day = parts[4].split("-")[0].replace("_", "")[:8]
    else:
        return None
    if len(day) != 8 or not day.isdigit():
        return None
    return net, sta, loc, cha, day


def find_counterpart(path, from_root, to_root):
    """把 ``from_root`` 下的一个文件，映射到 ``to_root`` 下的对应文件。

    为什么需要它
    ------------
    DSSRR 的**写**和**读**两端对"目录布局"的约定不一致：

    - 批量去异常 / CLI 写修复库时固定写到 ``<fix_root>/YYYY/MM/<filename>``，
      **与原始库的目录结构无关**；
    - 查看器读修复库时早期直接 ``os.path.relpath(raw_path, raw_root)`` **镜像
      原始库的目录结构**。

    于是只要两端布局不同（典型情况：原始库是**扁平**目录、修复库是
    ``YYYY/MM/`` 镜像），映射就会失败——查看器报"该台站/日期没有文件（尚未修复）"，
    ``verify_month`` 则把所有文件静默 ``continue`` 掉，最后报"验收通过"但
    **一个文件都没验**（假阳性，比前者更危险）。

    本函数按代价递增依次尝试，**任意布局都能对上**：

    1. **同布局镜像**：``<to_root>/<relpath(path, from_root)>``——结构一致时最快。
    2. **``YYYY/MM/`` 约定**：``<to_root>/YYYY/MM/<filename>``——批量修复的写法。
    3. **递归扫描**（兜底）：按 :func:`parse_mseed_name` 解析出的
       ``(station, day)`` 匹配；目录结构完全任意也能找到。

    Parameters
    ----------
    path : str
        位于 ``from_root`` 下的文件。
    from_root, to_root : str
        两个库的根目录。

    Returns
    -------
    str or None
        在 ``to_root`` 下找到的对应文件路径；找不到返回 ``None``。
    """
    if not path or not to_root or not os.path.isdir(to_root):
        return None
    base = os.path.basename(path)
    # 1) 同布局镜像
    try:
        rel = os.path.relpath(path, from_root)
    except Exception:
        rel = base
    if rel and not rel.startswith(".."):
        candidate = os.path.join(to_root, rel)
        if os.path.exists(candidate):
            return candidate
    info = parse_mseed_name(base)
    # 2) 批量约定的 YYYY/MM/
    if info:
        _net, _sta, _loc, _cha, day8 = info
        candidate = os.path.join(to_root, day8[:4], day8[4:6], base)
        if os.path.exists(candidate):
            return candidate
    # 3) 递归扫描兜底：按 (station, day) 匹配
    if info:
        target_sta, target_day = info[1], info[4]
        for root, _dirs, files in os.walk(to_root):
            for fn in files:
                if not fn.endswith(".mseed"):
                    continue
                pinfo = parse_mseed_name(fn)
                if not pinfo:
                    continue
                _net, sta, _loc, _cha, day = pinfo
                if sta == target_sta and day == target_day:
                    candidate = os.path.join(root, fn)
                    if os.path.abspath(candidate) != os.path.abspath(path):
                        return candidate          # 排除自指
    return None


def scan_stations_years(raw_root):
    """递归扫描原始根目录，返回 ``(sorted stations, sorted years)``。

    只依据文件名，不要求任何目录结构。
    """
    stations, years = set(), set()
    if not raw_root or not os.path.isdir(raw_root):
        return [], []
    for root, _dirs, files in os.walk(raw_root):
        for fn in files:
            info = parse_mseed_name(fn)
            if not info:
                continue
            _net, sta, _loc, _cha, day = info
            stations.add(sta)
            years.add(int(day[:4]))
    return sorted(stations), sorted(years)


def iter_scope_files(raw_root, stations, d0, d1):
    """按日期范围递归产出 ``(file_path, year, month)``。

    ``d0`` / ``d1`` 为 int ``YYYYMMDD``。先按台站集合过滤（空集合 = 全要），
    再按日期区间过滤。
    """
    if not raw_root or not os.path.isdir(raw_root):
        return
    for root, _dirs, files in os.walk(raw_root):
        for fn in sorted(files):
            info = parse_mseed_name(fn)
            if not info:
                continue
            _net, sta, _loc, _cha, day = info
            if stations and sta not in stations:
                continue
            fdate = int(day)
            if fdate < d0 or fdate > d1:
                continue
            yield os.path.join(root, fn), int(day[:4]), int(day[4:6])


def diagnose_scan(raw_root, stations, d0, d1):
    """扫描结果为 0 时统计"卡在哪一步"，供界面给出可读提示。

    Returns
    -------
    dict
        ``seen`` 看到的 ``.mseed`` 总数、``bad_name`` 文件名不合规数、
        ``bad_sta`` 台站不在勾选范围数、``bad_date`` 日期落在区间外数、
        ``stations_seen`` 实际出现的台站列表。
    """
    r = {"seen": 0, "bad_name": 0, "bad_sta": 0, "bad_date": 0,
         "stations_seen": set()}
    if not raw_root or not os.path.isdir(raw_root):
        r["stations_seen"] = []
        return r
    for _root, _dirs, files in os.walk(raw_root):
        for fn in files:
            if not fn.lower().endswith(".mseed"):
                continue
            r["seen"] += 1
            info = parse_mseed_name(fn)
            if not info:
                r["bad_name"] += 1
                continue
            _net, sta, _loc, _cha, day = info
            r["stations_seen"].add(sta)
            if stations and sta not in stations:
                r["bad_sta"] += 1
                continue
            if not (d0 <= int(day) <= d1):
                r["bad_date"] += 1
    r["stations_seen"] = sorted(r["stations_seen"])
    return r
