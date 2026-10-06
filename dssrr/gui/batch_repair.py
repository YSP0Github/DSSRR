# -*- coding: utf-8 -*-
"""
BatchRepairDialog — 批量去异常工具（嵌入 DbRepairViewerWindow）
=================================================================
在 GUI 中完成"导入数据 → 选择范围 → 检查数据 → 批量修复 → 自动验收"
的完整流程，取代命令行脚本的直接运行：

- 数据源：原始库根目录、修复库根目录（输出镜像 YYYY/MM 结构）。
  **输入目录结构不限**，只要文件名是 ``NET.STA.YYYYMMDD.mseed`` 或
  ``NET.STA.LOC.CHA.START-END.mseed``（扫描规则见 :mod:`dssrr.discovery`）。
  两个根目录**不硬编码**，按以下优先级取值：
  环境变量 ``DSSRR_RAW_ROOT`` / ``DSSRR_FIX_ROOT`` → 上次在界面中填写的值
  → 空（界面提示用户选择）。
- 月震目录：读取 ``e7_detector`` 旁的 ``moonquake_catalog.json``
  （由 ``apollo_catalog/*.csv`` 合并而来），默认启用保护；可关闭
  （此时与"无目录记录日"口径一致：检测器事件不构成保护依据）
- 范围：台站多选（从原始根自动扫描）、年份/月份区间
- 参数：论文默认（可复现）或用户设置（``~/.dssrr_settings.json``，
  由查看器 Settings 对话框写入）；修复层与检测层参数均生效
- 预检：快速统计（文件数/台站/预计耗时）+ 深度扫描（MHZ 覆盖、坏文件）
- 运行：后台 QThread，实时进度 + 日志，可随时停止；
  每（年,月）处理完自动写明细 CSV（catalog 目录/E7_catalog_{YYYYMM}.csv），
  全部完成后自动验收（verify），结果写 verify_{YYYYMM}.json 并汇总展示

依赖：复用 e7_repair_build / e7_detector 的修复流水线（与论文口径一致），
import 由本模块延迟加载（_ensure_paths），开发/安装包环境均可运行。
"""
from __future__ import annotations

import os
import sys
import time
import json

import numpy as np

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QCheckBox, QSpinBox, QPlainTextEdit,
    QDateEdit,
    QProgressBar, QFileDialog, QMessageBox, QRadioButton, QScrollArea,
)

from .. import discovery as _discovery
from .viewer import (_qss, _EXPERIMENTS_DIR, RAW_ROOT_HINT, FIX_ROOT_HINT,
                     run_modal_dialog)

SR = 6.625

# 默认数据根目录。故意留空而不是写死某台机器上的路径：用户可以通过环境变量
# 预置，也可以直接在界面上选择，选择结果会被记住（见 _LAST_SCOPE）。
DEFAULT_RAW_ROOT = os.environ.get("DSSRR_RAW_ROOT", "")
DEFAULT_FIX_ROOT = os.environ.get("DSSRR_FIX_ROOT", "")

# 1976-01 实测基准：93 文件约 60 s（v5 全月重跑），用于预计耗时外推
SEC_PER_FILE = 60.0 / 93.0

# 批量去异常范围记忆（当前进程运行期间有效；对话框重开时恢复上次设置）
_LAST_SCOPE = {}


def _ensure_paths():
    """把 E7 修复流水线目录加入 sys.path（延迟加载，避免启动开销）。
    独立包内统一指向随包分发的 dssrr/repair_lib。"""
    from .viewer import _REPAIR_LIB_DIR
    for d in (_EXPERIMENTS_DIR, _REPAIR_LIB_DIR):
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)


# ----------------------------------------------------------------------
# 范围扫描（纯文件系统，秒级）
# ----------------------------------------------------------------------
def parse_mseed_name(fn):
    """兼容性别名 —— 实现已移到 :mod:`dssrr.discovery`，CLI 与 GUI 共用一份。

    保留这个名字是为了不动 GUI 内部既有的调用点，也方便老代码继续 import。
    """
    return _discovery.parse_mseed_name(fn)


def scan_stations_years(raw_root):
    """递归扫描原始根目录，返回 ``(sorted stations, sorted years)``。"""
    return _discovery.scan_stations_years(raw_root)


def _iter_scope_files(raw_root, stations, d0, d1):
    """按日期范围递归产出 ``(file_path, year, month)``（见 dssrr.discovery）。"""
    return _discovery.iter_scope_files(raw_root, stations, d0, d1)


def diagnose_scan(raw_root, stations, d0, d1):
    """扫描结果为 0 时统计"卡在哪一步"（见 dssrr.discovery）。"""
    return _discovery.diagnose_scan(raw_root, stations, d0, d1)


def quick_scan(raw_root, stations, d0, d1):
    """快速统计（不读文件内容）：文件数、按年月、总字节、预计耗时。"""
    n = 0
    nbytes = 0
    by_ym = {}
    for f, y, m in _iter_scope_files(raw_root, stations, d0, d1):
        n += 1
        ym = f"{y:04d}-{m:02d}"
        by_ym[ym] = by_ym.get(ym, 0) + 1
        try:
            nbytes += os.path.getsize(f)
        except OSError:
            pass
    return {
        "files": n, "bytes": nbytes, "by_ym": by_ym,
        "est_sec": n * SEC_PER_FILE,
    }


class DeepScanWorker(QThread):
    """深度扫描：逐文件 headonly 检查 MHZ 覆盖与读取失败（进度信号）。"""
    progress = pyqtSignal(int, int, str)
    done = pyqtSignal(dict)

    def __init__(self, files, parent=None):
        super().__init__(parent)
        self._files = files
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from obspy import read
        n_mhz = n_nomhz = n_err = 0
        errs = []
        total = len(self._files)
        for i, f in enumerate(self._files):
            if self._stop:
                break
            try:
                st = read(f, headonly=True)
                if any(tr.stats.channel == "MHZ" for tr in st):
                    n_mhz += 1
                else:
                    n_nomhz += 1
            except Exception as ex:
                n_err += 1
                errs.append((os.path.basename(f), str(ex)[:80]))
            if i % 10 == 0 or i == total - 1:
                self.progress.emit(i + 1, total, os.path.basename(f))
        self.done.emit({"mhz": n_mhz, "nomhz": n_nomhz, "err": n_err,
                        "errs": errs})


# ----------------------------------------------------------------------
# 批量修复 worker
# ----------------------------------------------------------------------
class BatchWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, files, raw_root, fix_root, catalog_dir, stations,
                 det_kwargs, params, use_catalog, sr=SR, parent=None):
        super().__init__(parent)
        self._files = files
        self._raw_root = raw_root
        self._fix_root = fix_root
        self._catalog_dir = catalog_dir
        self._stations = stations
        self._det_kwargs = det_kwargs or {}
        self._params = params
        self._use_catalog = bool(use_catalog)
        self._sr = sr
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        _ensure_paths()
        try:
            from dssrr.repair_lib.e7_repair_build import (repair_one_file,
                                                         write_catalog_csv,
                                                         verify_month,
                                                         verify_file)
            from dssrr.repair_lib.e7_detector import E7Detector
        except Exception as ex:
            self.failed.emit(f"import E7 pipeline failed: {ex}")
            return
        det = E7Detector(sr=self._sr, **self._det_kwargs)

        total = len(self._files)
        done = 0
        t0 = time.time()
        summary = {
            "files": total, "repaired": 0, "gaps": 0, "errors": [],
            "months": [], "verify": None, "elapsed": 0,
        }
        groups = {}
        for f, y, m in self._files:
            groups.setdefault(f"{y:04d}{m:02d}", []).append(f)

        for ym in sorted(groups):
            if self._stop:
                break
            out_ym = os.path.join(self._fix_root, ym[:4], ym[4:6])
            rows = []
            for f in sorted(groups[ym]):
                if self._stop:
                    break
                try:
                    nrep, ngap = repair_one_file(
                        f, det, out_ym, verbose=False, catalog_rows=rows,
                        params=self._params, use_catalog=self._use_catalog)
                    summary["repaired"] += nrep
                    summary["gaps"] += ngap
                except Exception as ex:
                    summary["errors"].append(
                        (os.path.basename(f), str(ex)[:120]))
                    self.log.emit(f"ERR {os.path.basename(f)}: {ex}")
                done += 1
                self.progress.emit(done, total, os.path.basename(f))
            if rows and self._catalog_dir:
                try:
                    os.makedirs(self._catalog_dir, exist_ok=True)
                    cpath = os.path.join(
                        self._catalog_dir, f"E7_catalog_{ym}.csv")
                    write_catalog_csv(cpath, rows)
                    self.log.emit(
                        f"catalog {ym}: {len(rows)} rows -> {cpath}")
                except Exception as ex:
                    self.log.emit(f"catalog {ym} ERR: {ex}")
            summary["months"].append(ym)

        # 自动验收：只验收本次处理过的文件（不扫整个月目录）
        self.log.emit("Verifying repaired files...")
        try:
            ver = {"ok": True, "n_files": 0, "n_fail": 0,
                   "n_skip": 0, "by_month": {}}
            for ym in sorted(groups):
                r_raw = os.path.join(self._raw_root, ym[:4], ym[4:6])
                r_fix = os.path.join(self._fix_root, ym[:4], ym[4:6])
                ym_files = 0
                ym_fail = 0
                for f in sorted(groups[ym]):
                    fb = os.path.basename(f)
                    fix_path = os.path.join(r_fix, fb)
                    if not os.path.exists(fix_path):
                        ver["n_skip"] += 1
                        self.log.emit(f"VERIFY SKIP {fb}: no repaired file")
                        continue
                    try:
                        rep = verify_file(f, fix_path, det,
                                          use_catalog=self._use_catalog)
                        ver["n_files"] += 1
                        ym_files += 1
                        if not rep.get("ok"):
                            ver["n_fail"] += 1
                            ym_fail += 1
                            ver["ok"] = False
                            self.log.emit(
                                f"VERIFY FAIL {fb}: "
                                f"{'; '.join(rep.get('issues', []))}")
                    except Exception as ex:
                        ver["n_fail"] += 1
                        ym_fail += 1
                        ver["ok"] = False
                        self.log.emit(f"VERIFY ERR {fb}: {ex}")
                ver["by_month"][ym] = {
                    "ok": ym_fail == 0, "n_files": ym_files,
                    "n_fail": ym_fail}
            if self._catalog_dir:
                try:
                    vpath = os.path.join(
                        self._catalog_dir,
                        "verify_" + "_".join(sorted(summary["months"]))[:12]
                        + ".json")
                    with open(vpath, "w", encoding="utf-8") as f:
                        json.dump(ver, f, ensure_ascii=False, indent=1)
                    self.log.emit(f"verify json -> {vpath}")
                except Exception as ex:
                    self.log.emit(f"verify json ERR: {ex}")
            summary["verify"] = ver
        except Exception as ex:
            summary["verify"] = {"ok": False, "error": str(ex)[:200]}

        summary["elapsed"] = time.time() - t0
        self.finished_ok.emit(summary)


# ----------------------------------------------------------------------
# 对话框
# ----------------------------------------------------------------------
class BatchRepairDialog(QDialog):
    """批量去异常对话框（嵌入查看器，入口按钮 Batch Repair）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewer = parent
        self._dark = bool(getattr(parent, "_dark", False))
        self._theme = dict(getattr(parent, "_theme", {}) or {})
        self._station_boxes = []
        self._files_cache = None  # 预检后的文件清单（供运行复用）
        self._scan_worker = None
        self._batch_worker = None
        self._running = False
        self.setWindowTitle(self.tr("Batch Repair"))
        self.setMinimumSize(860, 640)
        self.resize(980, 760)
        self._build_ui()
        self._reload_scope()

    # ---------------- UI ----------------
    def _build_ui(self):
        self.setStyleSheet(_qss(self._dark, self._theme))
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # 1) 数据源
        src = QGroupBox(self.tr("Data Sources"))
        g = QGridLayout(src)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        g.setColumnStretch(1, 1)
        g.addWidget(QLabel(self.tr("Raw DB root")), 0, 0)
        self.ed_raw = QLineEdit(DEFAULT_RAW_ROOT)
        self.ed_raw.setToolTip(self.tr(
            "Root of the raw MiniSEED DB. Any folder layout is accepted — "
            "files just have to be named\n"
            "NET.STA.YYYYMMDD.mseed or NET.STA.LOC.CHA.START-END.mseed."))
        b = QPushButton(self.tr("Browse"))
        b.clicked.connect(lambda: self._browse("raw"))
        r = QHBoxLayout(); r.addWidget(self.ed_raw, 1); r.addWidget(b)
        g.addLayout(r, 0, 1, 1, 2)
        g.addWidget(QLabel(self.tr("Repaired DB root")), 1, 0)
        self.ed_fix = QLineEdit(DEFAULT_FIX_ROOT)
        b = QPushButton(self.tr("Browse"))
        b.clicked.connect(lambda: self._browse("fix"))
        r = QHBoxLayout(); r.addWidget(self.ed_fix, 1); r.addWidget(b)
        g.addLayout(r, 1, 1, 1, 2)
        g.addWidget(QLabel(self.tr("Catalog CSV dir")), 2, 0)
        self.ed_cat = QLineEdit(os.path.join(DEFAULT_FIX_ROOT, "catalogs"))
        b = QPushButton(self.tr("Browse"))
        b.clicked.connect(lambda: self._browse("cat"))
        r = QHBoxLayout(); r.addWidget(self.ed_cat, 1); r.addWidget(b)
        g.addLayout(r, 2, 1, 1, 2)
        self.chk_cat = QCheckBox(self.tr("Enable moonquake-catalog protection"))
        self.chk_cat.setChecked(True)
        g.addWidget(self.chk_cat, 3, 0, 1, 3)
        self.lbl_cat = QLabel("")
        self.lbl_cat.setWordWrap(True)
        self.lbl_cat.setStyleSheet("color: #888; font-size: 11px;")
        g.addWidget(self.lbl_cat, 4, 0, 1, 3)
        # 目录保护与 CSV 路径说明（避免用户找不到 catalogs 目录）
        self.lbl_cat_help = QLabel(self.tr(
            "Catalog protection: moonquake windows from apollo_catalog/*.csv "
            "are kept unchanged during repair (event/coda not rebuilt). "
            "Detail CSVs are written to the Catalog CSV dir on first batch "
            "run (auto-created)."))
        self.lbl_cat_help.setWordWrap(True)
        self.lbl_cat_help.setStyleSheet("color: #888; font-size: 11px;")
        g.addWidget(self.lbl_cat_help, 5, 0, 1, 3)
        root.addWidget(src)
        self.ed_raw.textChanged.connect(lambda *_: self._reload_scope())

        # 2) 范围
        scope = QGroupBox(self.tr("Scope"))
        g = QGridLayout(scope)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        g.addWidget(QLabel(self.tr("Stations")), 0, 0)
        self.box_stations = QWidget()
        self.lay_sta = QHBoxLayout(self.box_stations)
        self.lay_sta.setContentsMargins(0, 0, 0, 0)
        self.lay_sta.setSpacing(6)
        g.addWidget(self.box_stations, 0, 1, 1, 3)
        from PyQt5.QtCore import QDate
        g.addWidget(QLabel(self.tr("Date")), 1, 0)
        cal_qss = (
            "QCalendarWidget QWidget { background-color: #ffffff; color: #222; }"
            "QCalendarWidget QAbstractItemView:enabled {"
            " background-color: #ffffff; color: #222; selection-background-color: #4a90d9;"
            " selection-color: white; }"
            "QCalendarWidget QToolButton { background-color: #f0f0f0; color: #222;"
            " border: none; padding: 4px; }"
            "QCalendarWidget QMenu { background-color: #ffffff; color: #222; }"
        )
        self.dt0 = QDateEdit()
        self.dt0.setDisplayFormat("yyyy-MM-dd")
        self.dt0.setCalendarPopup(True)
        self.dt0.calendarWidget().setStyleSheet(cal_qss)
        self.dt1 = QDateEdit()
        self.dt1.setDisplayFormat("yyyy-MM-dd")
        self.dt1.setCalendarPopup(True)
        self.dt1.calendarWidget().setStyleSheet(cal_qss)
        self.dt0.setDate(QDate(1976, 1, 1))
        self.dt1.setDate(QDate(1976, 2, 29))
        r = QHBoxLayout()
        r.addWidget(self.dt0); r.addWidget(QLabel("~")); r.addWidget(self.dt1)
        g.addLayout(r, 1, 1, 1, 3)
        root.addWidget(scope)

        # 3) 参数
        prm = QGroupBox(self.tr("Repair Parameters"))
        g = QGridLayout(prm)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        self.rad_default = QRadioButton(self.tr("Paper defaults (reproducible)"))
        self.rad_user = QRadioButton(
            self.tr("User settings (~/.dssrr_settings.json)"))
        self.rad_default.setChecked(True)
        g.addWidget(self.rad_default, 0, 0, 1, 2)
        g.addWidget(self.rad_user, 0, 2, 1, 2)
        btn_set = QPushButton(self.tr("Open Settings..."))
        btn_set.clicked.connect(self._open_settings)
        g.addWidget(btn_set, 1, 0)
        self.lbl_params = QLabel("")
        self.lbl_params.setWordWrap(True)
        self.lbl_params.setStyleSheet("color: #888; font-size: 11px;")
        g.addWidget(self.lbl_params, 1, 1, 1, 3)
        root.addWidget(prm)
        self._refresh_params_label()

        # 4) 预检
        pre = QGroupBox(self.tr("Pre-check"))
        g = QGridLayout(pre)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        btn_scan = QPushButton(self.tr("Check Data"))
        btn_scan.clicked.connect(self._precheck)
        btn_deep = QPushButton(self.tr("Deep Scan (MHZ / broken files)"))
        btn_deep.clicked.connect(self._deep_scan)
        self.btn_deep = btn_deep
        g.addWidget(btn_scan, 0, 0)
        g.addWidget(btn_deep, 0, 1)
        self.lbl_pre = QLabel("")
        self.lbl_pre.setWordWrap(True)
        g.addWidget(self.lbl_pre, 1, 0, 1, 4)
        root.addWidget(pre)

        # 5) 运行
        run = QGroupBox(self.tr("Run"))
        g = QGridLayout(run)
        g.setHorizontalSpacing(8)
        g.setVerticalSpacing(6)
        self.btn_start = QPushButton(self.tr("Start Batch Repair"))
        self.btn_start.clicked.connect(self._start)
        self.btn_stop = QPushButton(self.tr("Stop"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        g.addWidget(self.btn_start, 0, 0)
        g.addWidget(self.btn_stop, 0, 1)
        self.pbar = QProgressBar()
        self.pbar.setRange(0, 1)
        self.lbl_prog = QLabel("")
        r = QHBoxLayout()
        r.addWidget(self.pbar, 1); r.addWidget(self.lbl_prog)
        g.addLayout(r, 0, 2, 1, 2)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        g.addWidget(self.log, 1, 0, 1, 4)
        root.addWidget(run, 1)

    # ---------------- 交互 ----------------
    def _browse(self, kind):
        d = run_modal_dialog(
            self,
            lambda host: QFileDialog.getExistingDirectory(
                host, self.tr("Select directory"),
                getattr(self, f"ed_{kind}").text().strip()
                or os.path.expanduser("~")))
        if d:
            getattr(self, f"ed_{kind}").setText(os.path.normpath(d))
            if kind == "raw":
                self._reload_scope()

    def _reload_scope(self):
        raw = self.ed_raw.text().strip()
        stations, years = scan_stations_years(raw)
        # 台站复选框
        for cb in self._station_boxes:
            self.lay_sta.removeWidget(cb)
            cb.deleteLater()
        self._station_boxes = []
        for stn in (stations or ["S12", "S14", "S15", "S16"]):
            cb = QCheckBox(stn)
            cb.setChecked(True)
            self.lay_sta.addWidget(cb)
            self._station_boxes.append(cb)
        self.lay_sta.addStretch(1)
        # 日期范围（按扫描到的年份设置默认边界）
        if years:
            from PyQt5.QtCore import QDate
            lo, hi = min(years), max(years)
            self.dt0.setDate(QDate(lo, 1, 1))
            self.dt1.setDate(QDate(hi, 12, 31))
        # 月震目录自动探测
        self._update_catalog_label()
        # 恢复上次范围记忆（年份/月份/台站/目录保护开关）
        self._restore_scope()

    def _save_scope(self):
        """保存当前范围设置到模块级记忆（开始批量时调用）。"""
        _LAST_SCOPE.clear()
        _LAST_SCOPE.update({
            "d0": self.dt0.date().toString("yyyyMMdd"),
            "d1": self.dt1.date().toString("yyyyMMdd"),
            "stations": [cb.text() for cb in self._station_boxes
                         if cb.isChecked()],
            "catalog": self.chk_cat.isChecked(),
        })

    def _restore_scope(self):
        """从记忆恢复范围设置（对话框重开/数据源刷新后调用）。"""
        if not _LAST_SCOPE:
            return
        try:
            from PyQt5.QtCore import QDate
            self.dt0.setDate(QDate.fromString(_LAST_SCOPE["d0"], "yyyyMMdd"))
            self.dt1.setDate(QDate.fromString(_LAST_SCOPE["d1"], "yyyyMMdd"))
            saved_st = set(_LAST_SCOPE.get("stations", []))
            for cb in self._station_boxes:
                cb.setChecked(cb.text() in saved_st)
            self.chk_cat.setChecked(_LAST_SCOPE.get("catalog", True))
        except Exception:
            pass

    def _update_catalog_label(self):
        """显示系统内置月震目录信息（JSON 合并版，非扫描文件路径）。"""
        try:
            import json
            from dssrr.repair_lib.e7_detector import _CATALOG_JSON, _user_catalog_json
            # 用户覆盖文件优先（不存在则为 None）
            user = _user_catalog_json()
            active = user if user else _CATALOG_JSON
            with open(active, "r", encoding="utf-8") as f:
                data = json.load(f)
            n = data.get("event_count", len(data.get("events", [])))
            sources = data.get("sources", [])
            tag = self.tr("(user override)") if user else ""
            self.lbl_cat.setText(
                self.tr("Moonquake catalog: {n} events from {src} {tag}").format(
                    n=n, src="+".join(s.split("_")[0] for s in sources), tag=tag))
        except Exception:
            self.lbl_cat.setText(self.tr("Moonquake catalog: built-in"))

    def _open_settings(self):
        if self._viewer is not None and hasattr(self._viewer,
                                                "open_repair_settings"):
            self._viewer.open_repair_settings()
            self._refresh_params_label()

    def _refresh_params_label(self):
        try:
            from dssrr.repair_lib import repair_settings as RS
            from .viewer import RepairSettingsDialog
            from PyQt5.QtCore import QCoreApplication
            s = RS.load_settings()
            rep = s.get("repair", {})
            names = RepairSettingsDialog._PARAM_NAMES
            def _n(k):
                return QCoreApplication.translate("RepairSettingsDialog",
                                                  names.get(k, k))
            txt = (self.tr("Repair: ") + ", ".join(
                f"{_n(k)}={v}" for k, v in list(rep.items())[:6])
                + (", ..." if len(rep) > 6 else ""))
        except Exception:
            txt = self.tr("(no user settings)")
        self.lbl_params.setText(txt)

    def _stations_selected(self):
        return [cb.text() for cb in self._station_boxes if cb.isChecked()]

    def _scope_files(self):
        raw = self.ed_raw.text().strip()
        stations = self._stations_selected()
        d0 = int(self.dt0.date().toString("yyyyMMdd"))
        d1 = int(self.dt1.date().toString("yyyyMMdd"))
        if d0 > d1:
            d0, d1 = d1, d0
        return raw, stations, d0, d1

    # ---------------- 预检 ----------------
    def _no_files_hint(self, raw, stations, d0, d1):
        """扫不到文件时，把"卡在哪一步"讲清楚，而不是只报一句 0 文件。

        依次区分：目录里根本没有 .mseed / 文件名不符合命名约定 / 台站没勾选 /
        日期区间没覆盖。用户最常踩的是后两个（例如把 examples/data 指进来，
        里面只有 1976-01 的数据，而日期框默认是 1976-01-01 ~ 1976-02-29，
        这时台站列表会被自动识别成 S12/S15/S16，只要日期区间对得上就能扫到）。
        """
        dg = diagnose_scan(raw, stations, d0, d1)
        if dg["seen"] == 0:
            return self.tr("No .mseed file found under: ") + raw
        if dg["seen"] - dg["bad_name"] == 0:
            return self.tr(
                "Found {n} .mseed file(s), but none matches the naming "
                "convention NET.STA.YYYYMMDD.mseed or "
                "NET.STA.LOC.CHA.START-END.mseed.").format(n=dg["seen"])
        if dg["bad_sta"] and not dg["bad_date"]:
            return self.tr(
                "All {n} matching file(s) belong to stations outside the "
                "selection ({sel}). Stations found: {seen}.").format(
                    n=dg["bad_sta"], sel=", ".join(stations) or "-",
                    seen=", ".join(dg["stations_seen"]) or "-")
        if dg["bad_date"]:
            return self.tr(
                "{n} file(s) matched but their dates fall outside "
                "{d0} ~ {d1}. Widen the date range or check the file names."
            ).format(n=dg["bad_date"], d0=d0, d1=d1)
        return self.tr("No mseed files in the selected scope.")

    def _precheck(self):
        raw, stations, d0, d1 = self._scope_files()
        if not raw or not os.path.isdir(raw):
            QMessageBox.warning(self, self.tr("Batch Repair"),
                                self.tr("Raw DB root not found: ") + raw)
            return
        r = quick_scan(raw, stations, d0, d1)
        self._files_cache = [(f, y, m)
                             for f, y, m in _iter_scope_files(
                                 raw, stations, d0, d1)]
        if r["files"] == 0:
            self.lbl_pre.setText(self._no_files_hint(raw, stations, d0, d1))
            return
        est = r["est_sec"]
        txt = (self.tr("Files: {n}  ({gb:.2f} GB)  |  {months} months\n"
                       "Estimated repair time: {est:.0f} min "
                       "({per:.1f} s/file, 1976-01 measured base)").format(
                           n=r["files"], gb=r["bytes"] / 1024**3,
                           months=len(r["by_ym"]), est=est / 60, per=SEC_PER_FILE))
        self.lbl_pre.setText(txt)

    def _deep_scan(self):
        if not self._files_cache:
            self._precheck()
        files = [f for f, _, _ in (self._files_cache or [])]
        if not files:
            return
        self.btn_deep.setEnabled(False)
        self.pbar.setRange(0, len(files))
        self.pbar.setValue(0)
        self._scan_worker = DeepScanWorker(files)
        self._scan_worker.progress.connect(
            lambda d, t, fn: (self.pbar.setValue(d),
                              self.lbl_prog.setText(f"{d}/{t}")))
        self._scan_worker.done.connect(self._deep_done)
        self._scan_worker.start()

    def _deep_done(self, r):
        self.btn_deep.setEnabled(True)
        self.pbar.setRange(0, 1)
        self.lbl_prog.setText("")
        txt = (self.tr("Deep scan: MHZ={mhz}  no-MHZ={nomhz}  "
                       "unreadable={err}").format(**r))
        if r["errs"]:
            txt += "\n" + self.tr("Broken files: ") + \
                "; ".join(f"{n}({m})" for n, m in r["errs"][:8])
            if len(r["errs"]) > 8:
                txt += f" ... (+{len(r['errs'])-8})"
        self.lbl_pre.setText(txt)

    # ---------------- 运行 ----------------
    def _collect_params(self):
        """返回 (det_kwargs, params)。radio_user 时读用户设置。"""
        if self.rad_user.isChecked():
            try:
                from dssrr.repair_lib import repair_settings as RS
                s = RS.load_settings()
                return dict(s.get("detect", {}) or {}), \
                    dict(s.get("repair", {}) or {})
            except Exception:
                return {}, None
        return {}, None

    def _start(self):
        if self._running:
            return
        self._save_scope()
        raw, stations, d0, d1 = self._scope_files()
        if not stations:
            QMessageBox.warning(self, self.tr("Batch Repair"),
                                self.tr("Select at least one station."))
            return
        if not self._files_cache:
            self._precheck()
        files = [(f, y, m) for f, y, m in (self._files_cache or [])]
        if not files:
            return
        fix_root = self.ed_fix.text().strip() or DEFAULT_FIX_ROOT
        cat_dir = self.ed_cat.text().strip() or os.path.join(fix_root, "catalogs")
        det_kw, params = self._collect_params()
        # 只保留对 E7Detector 有意义的检测键（防御设置文件含未知键）
        try:
            from dssrr.repair_lib.e7_detector import E7Detector
            import inspect
            sig = inspect.signature(E7Detector.__init__)
            known = set(sig.parameters) - {"self", "sr"}
            det_kw = {k: v for k, v in det_kw.items() if k in known}
        except Exception:
            det_kw = {}
        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log.clear()
        self.log.appendPlainText(
            self.tr("Start: {n} files, stations={st}, {d0}~{d1}, "
                    "catalog={cat}").format(
                        n=len(files), st=",".join(stations), d0=d0, d1=d1,
                        cat=self.tr("on") if self.chk_cat.isChecked()
                        else self.tr("off")))
        self.pbar.setRange(0, len(files))
        self.pbar.setValue(0)
        self._batch_worker = BatchWorker(
            files, raw, fix_root, cat_dir, stations, det_kw, params,
            self.chk_cat.isChecked())
        self._batch_worker.progress.connect(self._on_progress)
        self._batch_worker.log.connect(self.log.appendPlainText)
        self._batch_worker.finished_ok.connect(self._on_finished)
        self._batch_worker.failed.connect(self._on_failed)
        self._batch_worker.start()

    def _on_progress(self, done, total, name):
        self.pbar.setValue(done)
        self.lbl_prog.setText(f"{done}/{total}  {name}")

    def _stop(self):
        if self._batch_worker is not None:
            self._batch_worker.stop()
            self.btn_stop.setEnabled(False)
            self.log.appendPlainText(self.tr("Stop requested..."))

    def _on_finished(self, summary):
        self._running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.pbar.setRange(0, 1)
        self.lbl_prog.setText("")
        ver = summary.get("verify") or {}
        msg = (self.tr("Done in {m:.1f} min.\n"
                       "Processed: {n} files   repaired segments: {r}   gaps: {g}\n"
                       "Verify: {vo}  ({vf} checked, {vfail} FAIL, {vskip} skipped)").format(
                           m=summary.get("elapsed", 0) / 60,
                           n=summary.get("files", 0),
                           r=summary.get("repaired", 0),
                           g=summary.get("gaps", 0),
                           vo=self.tr("PASS") if ver.get("ok")
                           else self.tr("FAIL"),
                           vf=ver.get("n_files", 0),
                           vfail=ver.get("n_fail", 0),
                           vskip=ver.get("n_skip", 0)))
        if summary.get("errors"):
            msg += "\n" + self.tr("Repair errors: {n}").format(
                n=len(summary["errors"]))
        self.log.appendPlainText("---- " + self.tr("Summary") + " ----")
        self.log.appendPlainText(msg)
        QMessageBox.information(self, self.tr("Batch Repair"), msg)
        # 若查看器指向同一修复库 → 刷新
        if (self._viewer is not None
                and hasattr(self._viewer, "refresh")
                and hasattr(self._viewer, "ed_fix")
                and self._viewer.ed_fix.text().strip() ==
                self.ed_fix.text().strip()):
            try:
                self._viewer.refresh()
            except Exception:
                pass

    def _on_failed(self, err):
        self._running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log.appendPlainText("ERR: " + err)
        QMessageBox.critical(self, self.tr("Batch Repair"), err)

    # ---------------- 主题 ----------------
    def _apply_style(self):
        self.setStyleSheet(_qss(self._dark, self._theme))

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("Batch Repair"))
