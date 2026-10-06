# -*- coding: utf-8 -*-
"""简体中文界面词条表（英文原文 → 中文）。

为什么用"字典 + QTranslator"而不是 Qt 的 .ts/.qm
------------------------------------------------
Qt 的标准做法是 ``pylupdate5`` 抽字符串、``lrelease`` 编译成 ``.qm``，运行时用
``QTranslator.load()`` 加载。那套流程有两个问题：

1. **需要额外工具链**：``lrelease`` 在 PyQt5 的 pip 轮子里并不总是提供，
   开源发行版让用户为了加一句翻译去装 Qt Linguist 太重；
2. **二进制资源不可读**：``.qm`` 是编译产物，改一个错别字要重走全流程。

所以这里把词条直接写成 Python 字典，由 :class:`dssrr.gui.i18n.DssrrTranslator`
（一个 ``QTranslator`` 子类）在 ``translate()`` 里查表返回。好处是**所有已有的
``self.tr(...)`` 调用点一行都不用改**就自动生效，改词条也不需要任何编译步骤。

维护约定
--------
- **key 必须是源码里的英文原文**，与 ``self.tr("...")`` 里写的**逐字符一致**
  （含标点、空格、``\\n``）。
- **占位符必须原样保留**：``{n}``、``{path}``、``{s:.2f}``、``%p%``、``\\n`` 等。
  只翻译占位符之外的文字。
- 词条里带 HTML 的（帮助文档片段）只翻译文字，标签与 ``style`` 原样保留。
- 英文原文与中文相同时（如 ``"DSSRR"``、``"PNG"``）**不必收录**，查不到会
  自动回退成原文。

词条覆盖范围由 ``tools`` 里的抽取脚本统计：``dssrr/gui`` 下全部
``self.tr()`` / ``QCoreApplication.translate()`` 的字符串常量。
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# 通用 / 对话框按钮
# ---------------------------------------------------------------------------
_COMMON: Dict[str, str] = {
    "OK": "确定",
    "Cancel": "取消",
    "Close": "关闭",
    "Apply": "应用",
    "Save": "保存",
    "Run": "运行",
    "Stop": "停止",
    "Refresh": "刷新",
    "Browse": "浏览",
    "Help": "帮助",
    "Settings": "设置",
    "Preview": "预览",
    "Export": "导出",
    "Exported": "已导出",
    "Imported": "已导入",
    "Loaded": "已导入",
    "Saved": "已保存",
    "Ready": "就绪",
    "Detected": "已检出",
    "Detect failed": "检测失败",
    "Pipeline done": "流水线完成",
    "Processing...": "正在处理…",
    "Cancelling...": "正在取消…",
    "Stop requested...": "已请求停止…",
    "Success": "成功",
    "Warning": "警告",
    "Error": "错误",
    "Information": "提示",
    "Preview Failed": "预览失败",
    "Write failed": "写入失败",
    "File missing": "文件缺失",
    "No data loaded": "未载入数据",
    "No axes found": "未找到坐标轴",
    "PASS": "通过",
    "FAIL": "未通过",
    "on": "开",
    "off": "关",
    "Normal": "常规",
    "Light": "细体",
    "Medium": "中等",
    "Semibold": "半粗",
    "Bold": "粗体",
    "Heavy": "特粗",
    "Light theme": "浅色主题",
    "Dark theme": "深色主题",
    "Light mode": "浅色模式",
    "Dark mode": "深色模式",
    # 语言切换（侧边栏按钮 + 状态栏提示）
    "Language": "语言",
    "Interface language": "界面语言",
}

# ---------------------------------------------------------------------------
# 主窗口 / 侧边栏
# ---------------------------------------------------------------------------
_MAIN_WINDOW: Dict[str, str] = {
    "DSSRR — DB Repair Suite": "DSSRR — DB 修复套件",
    "DB Repair Suite": "DB 修复套件",
    "Repaired DB Viewer": "修复库查看器",
    "Manual Repair": "手动修复",
    "Exit": "退出",
    "Switch interface language (English / 中文)": "切换界面语言（English / 中文）",
}

# ---------------------------------------------------------------------------
# 查看器页（Data Selection / View / Repair Details / 帮助）
# ---------------------------------------------------------------------------
_VIEWER: Dict[str, str] = {
    "DSSRR Repaired DB Viewer": "DSSRR 修复库查看器",
    "Data Selection": "数据选择",
    "Raw DB": "原始库",
    "Repaired DB": "修复库",
    "Raw DB root": "原始库根目录",
    "Repaired DB root": "修复库根目录",
    "Stream": "数据流",
    "Day": "日",
    "Start (min)": "开始（分钟）",
    "Window (min)": "窗口（分钟）",
    "Date": "日期",
    "Range": "范围",
    "Scope": "范围",
    "View": "视图",
    "Time · Split": "时间 · 分栏",
    "Time · Overlay": "时间 · 叠加",
    "Frequency · PSD": "频率 · 功率谱",
    "Frequency · Amplitude": "频率 · 振幅",
    "Combo 2×2": "组合 2×2",
    "Export PNG": "导出 PNG",
    "Batch Repair": "批量修复",
    "Repair Details": "修复明细",
    "Summary": "汇总",
    "Detected events (double-click to locate):": "检出的事件（双击定位）：",
    "Selected Segment": "选中的段",
    "Time": "时间",
    "Anomaly": "异常",
    "Method": "方法",
    "Length": "长度",
    "Len (s)": "时长（秒）",
    "Description": "说明",
    "Parameter": "参数",
    "Effect": "作用",
    "Amplitude": "振幅",
    "Residual": "残差",
    "PSD": "功率谱",
    "Raw": "原始",
    "Repaired": "修复后",
    "Title": "标题",
    "repair": "修复",
    "event": "事件",
    "gap": "缺口",
    "segments": "段",
    "window segments": "窗口内段",
    "All day": "全天",
    "Time (s)": "时间（秒）",
    "Seconds since file start": "距文件起始的秒数",
    "Seconds since trace start": "距道起始的秒数",
    "Modified samples: {count:,} ({ratio:.3f}%)": "修改样点：{count:,}（{ratio:.3f}%）",
    "Repair: ": "修复：",
    "Repair errors: {n}": "修复错误：{n}",
    "All traces found under the raw DB root.\nNET.STA (day files) or NET.STA.LOC.CHA (segment files).":
        "原始库根目录下找到的全部道。\nNET.STA（日文件）或 NET.STA.LOC.CHA（片段文件）。",
    "Root of the raw MiniSEED DB. Any folder layout is accepted — files just have to be named\n"
    "NET.STA.YYYYMMDD.mseed or NET.STA.LOC.CHA.START-END.mseed.":
        "原始 MiniSEED 库的根目录。目录结构不限 —— 文件名只要符合\n"
        "NET.STA.YYYYMMDD.mseed 或 NET.STA.LOC.CHA.START-END.mseed 即可。",
    "Select raw/repaired DB and station/day": "请选择原始库/修复库与台站/日期",
    "Select raw/repaired DB and station/day in the viewer first":
        "请先在查看器里选择原始库/修复库与台站/日期",
    "No MHZ channel in repaired file": "修复文件中没有 MHZ 通道",
    "Raw DB has no day file for this station/day": "原始库中该台站/日期没有日文件",
    "Repaired DB has no day file for this station/day (not repaired yet)":
        "修复库中该台站/日期没有日文件（尚未修复）",
    "Raw DB is a segment-file library for this day (auto-merge not supported yet). "
    "Please point Raw DB at a day-file database matching the repaired DB":
        "原始库这一天是片段文件库（暂不支持自动合并）。请把原始库指向与修复库匹配的日文件库。",
    "Raw DB root not found: ": "未找到原始库根目录：",
    "No .mseed file found under: ": "未在以下路径找到 .mseed 文件：",
    "Configure anomaly-detection and repair parameters (applied to manual re-processing)":
        "配置异常检测与修复参数（应用到手动重处理）",
    "Manually re-repair a segment of the current file with real-time preview":
        "对当前文件的某一段手动重新修复，带实时预览",
    "Batch de-anomaly: select raw/fix roots, stations and date range, pre-check the data, "
    "then run the full pipeline with progress and auto verification":
        "批量去异常：选择原始库/修复库、台站与日期范围，先检查数据，再跑完整流水线，"
        "带进度显示与自动验收",
    "Open the viewer and parameter help": "打开查看器与参数帮助",
    "Open the parameter explanation help": "打开参数说明帮助",
    "Viewer Help": "查看器帮助",
    "Viewer usage": "查看器用法",
    "Usage Guide": "使用指南",
    "Quick Start": "快速上手",
    "Basic Steps": "基本步骤",
    "Legend & Concepts": "图例与概念",
    "Methods Overview": "方法总览",
    "Parameter Reference": "参数说明",
    "Recommended Settings": "推荐设置",
    "Tips": "小贴士",
    "Step 1 — Point Raw DB and Repaired DB at the two database roots (Browse buttons).":
        "第 1 步 — 用「浏览」把原始库与修复库指向两个库根目录。",
    "Step 2 — Pick Stream and Day; the stream list holds every trace in the raw DB and "
    "the day list comes from the database folders automatically.":
        "第 2 步 — 选择数据流与日期；数据流列表包含原始库里的每一条道，"
        "日期列表由数据库目录自动生成。",
    "Step 3 — Set Start (min) and Window (min); the preset buttons 30m/1h/6h/12h/1d change "
    "the window in one click and refresh immediately.":
        "第 3 步 — 设置开始（分钟）与窗口（分钟）；30m/1h/6h/12h/1d 预设按钮一键切换窗口并立即刷新。",
    "Step 4 — Choose a view: Time · Split (raw above, repaired below), Time · Overlay, "
    "Frequency · PSD, Frequency · Amplitude, or Combo 2×2.":
        "第 4 步 — 选择视图：时间 · 分栏（上原始、下修复）、时间 · 叠加、频率 · 功率谱、"
        "频率 · 振幅，或组合 2×2。",
    "Step 5 — The Repair Details table at the bottom lists every anomaly of the day. "
    "Click a row to highlight that segment on the canvas without changing the zoom.":
        "第 5 步 — 底部「修复明细」表列出当天所有异常。点击某一行会在画布上高亮该段，"
        "且不会改变缩放。",
    "Step 6 — Manual Repair re-processes the selected segment with a live preview; "
    "Apply writes the result back to the repaired DB and updates the details.":
        "第 6 步 — 「手动修复」对选中的段重新处理并实时预览；「应用」把结果写回修复库并更新明细。",
    "Step 7 — Settings opens the parameter editor; changes apply to the next manual "
    "re-processing. Reset defaults restores the paper v5 values.":
        "第 7 步 — 「设置」打开参数编辑器，改动应用到下一次手动重处理；"
        "「恢复默认」恢复论文 v5 参数。",
    "Step 8 — Export PNG saves the current canvas.": "第 8 步 — 「导出 PNG」保存当前画布。",
    "• <span style='color:#27ae60'>green</span> — event segment "
    "(moonquake, protected by the catalog; not modified)":
        "• <span style='color:#27ae60'>绿色</span> — 事件段（月震，受目录保护；不修改）",
    "• <span style='color:#7f8c8d'>gray</span> — gap segment (no data, kept as-is)":
        "• <span style='color:#7f8c8d'>灰色</span> — 缺口段（无数据，保持原样）",
    "• gold fill with red dashed edges — the segment selected in the details table":
        "• 金色填充 + 红色虚线边框 — 明细表中选中的那一段",
    "• repair segments — each colored by its repair method (see the color legend on the "
    "canvas and the method list below)":
        "• 修复段 — 按修复方法着色（见画布上的颜色图例与下方的方法列表）",
}

# ---------------------------------------------------------------------------
# 手动去异常面板（DePulseDialog）
# ---------------------------------------------------------------------------
_DEPULSE: Dict[str, str] = {
    "Anomaly Repair": "异常修复",
    "Segment Selection": "分段选择",
    "Enable Segment Selection": "启用分段选择",
    "Trace:": "道：",
    "Apply to all traces": "应用到所有道",
    "Import": "导入",
    "Import one or more waveform files.": "导入一个或多个波形文件。",
    "Import files...": "导入文件…",
    "Import files…": "导入文件…",
    "Import folder...": "导入文件夹…",
    "Import folder…": "导入文件夹…",
    "Import folder": "导入文件夹",
    "Import folder (recursive *.mseed)": "导入文件夹（递归 *.mseed）",
    "Import raw DB files": "导入原始库文件",
    "Import waveform files": "导入波形文件",
    "Imported files; pick one to work on": "已导入的文件；选一个开始处理",
    "Imported {count} trace(s).": "已导入 {count} 条道。",
    "Import a raw file first (Import files… / Import folder…)":
        "请先导入原始文件（导入文件… / 导入文件夹…）",
    "Import a raw file to visualize": "导入原始文件以绘图",
    "Select one or more XA.*.mseed files to add to the list":
        "选择一个或多个 XA.*.mseed 文件加入列表",
    "Scan a folder (recursively) for *.mseed files and add them all to the list":
        "递归扫描文件夹里的 *.mseed 并全部加入列表",
    "Select folder": "选择文件夹",
    "Select directory": "选择目录",
    "No *.mseed files found in this folder.": "该文件夹下没有 *.mseed 文件。",
    "No supported waveform files found in this folder.": "该文件夹下没有支持的波形文件。",
    "Could not read {name}: {error}": "无法读取 {name}：{error}",
    "Left drag: replace. Right drag: add. Double-click: clear.":
        "左键拖动：替换。右键拖动：追加。双击：清空。",
    # 导航工具栏处于平移/缩放模式时，SpanSelector 被 matplotlib 的 widgetlock
    # 挡住，框选会失效——这句提示负责把"为什么选不中"讲清楚。
    "Zoom/pan is active — segment selection is disabled. Click Zoom or Pan again to turn it off, then drag to select.":
        "缩放/平移模式已开启，此时无法框选异常段。再次点击「缩放」或「平移」按钮退出该模式，"
        "即可拖动选择。",
    "Selection: None": "选择：无",
    "Selection: {n} regions": "选择：{n} 个区段",
    "Selection: {s:.2f}s — {e:.2f}s": "选择：{s:.2f}s — {e:.2f}s",
    "Drag across the waveform to select a range.": "在波形上拖动以选择范围。",
    "Enable segment selection, drag left-click to select the anomaly region. "
    "Right-click to add more regions. Double-click to clear.":
        "启用分段选择后，左键拖动即可框选异常区。右键拖动可追加区域，双击清空。",
    "Seconds since trace start": "距道起始的秒数",
    "De-pulse Preview": "去脉冲预览",
    "Before": "修复前",
    "After": "修复后",
    "raw": "原始",
    "current repaired DB": "当前修复库",
    "new result": "新结果",
    "manual repair preview": "手动修复预览",
    "preview": "预览",
    "segment": "段",
    "method": "方法",
    "counts": "计数",
    "Choose one method to suppress spikes/outliers. Use Preview to compare before and after.":
        "选择一种方法来压制尖峰/离群点。用「预览」对比修复前后。",
    "Choose one method, preview, then switch methods if the result is not satisfactory.":
        "先选一种方法预览，结果不满意再换方法。",
    "Check one method on the left. Selecting another method automatically clears the previous one.":
        "在左侧勾选一种方法。选中另一种会自动清除上一种。",
    "Please select at least one method.": "请至少选择一种方法。",
    "This repair method requires a selected segment.": "该方法需要先选中一个段。",
    "Always preview before confirming. The preview shows exactly what will change.":
        "确认前务必先预览。预览展示的就是实际会发生的改动。",
    "Click \"Preview\" to see before/after comparison. The blue line shows the result.":
        "点「预览」查看修复前后对比，蓝线是修复结果。",
    "Click \"OK\" to apply. The result replaces the original data. You can undo from the main window.":
        "点「确定」应用。结果会替换原始数据，之后可在主窗口撤销。",
    "Results can be undone from the main window toolbar (Undo button).":
        "结果可从主窗口工具栏（撤销按钮）撤回。",
    "Preview a repair before saving it.": "保存前先预览修复结果。",
    "Preview a repair first, then confirm to export.": "请先预览一次修复，再确认导出。",
    "Preview a repair first, then export the report.": "请先预览一次修复，再导出报告。",
    "Please wait for the preview to finish.": "请等待预览完成。",
    "No preview available. Run de-pulse first.": "没有可用预览，请先执行去脉冲。",
    "No data available for export.": "没有可用于导出的数据。",
    "No data available for preview.": "没有可用于预览的数据。",
    "Switched to another trace. Run Preview again.": "已切换到另一条道，请重新运行预览。",
    "Run & Preview": "运行并预览",
    "Processing {method}": "正在处理 {method}",
    "Processing {method}: %p%": "正在处理 {method}：%p%",
    "Processing: %p%": "正在处理：%p%",
    "Save current repair": "保存当前修复",
    "Save repaired data": "保存修复数据",
    "Current repair saved.": "当前修复已保存。",
    "Last saved repair undone.": "已撤销上次保存的修复。",
    "Nothing to undo.": "没有可撤销的操作。",
    "Undo last repair": "撤销上次修复",
    "Undo the latest repair saved inside this dialog.": "撤销本对话框内最近保存的一次修复。",
    "Commit the current preview inside this dialog. It will not change the main window.":
        "在本对话框内提交当前预览，不会改动主窗口。",
    "Applied. Repaired DB updated; click Refresh in viewer to see the new data.":
        "已应用。修复库已更新；在查看器里点「刷新」即可看到新数据。",
    "Repaired data saved to:\n{path}": "修复数据已保存到：\n{path}",
    "Report saved to:\n{path}": "报告已保存到：\n{path}",
    "Failed to save: {error}": "保存失败：{error}",
    "Failed to load repair library": "加载修复库失败",
    "Manual Repair synced with viewer": "手动修复已与查看器同步",
    "This overwrites the segment in the repaired DB file:\n\n{path}\n\n"
    "[{s}-{e}] s = [{sm}-{em}] min\n\nContinue?":
        "这会覆盖修复库文件中的该段：\n\n{path}\n\n[{s}-{e}] 秒 = [{sm}-{em}] 分钟\n\n继续？",
    "Export Report...": "导出报告…",
    "Export Waveform...": "导出波形…",
    "Export a graphical repair report: configure title, size and frequency range, "
    "preview it, then save as PNG/PDF/SVG.":
        "导出图形化修复报告：配置标题、尺寸与频带范围，先预览，再保存为 PNG/PDF/SVG。",
    "Write the repaired waveform to a file (MiniSEED/SAC/TXT/ASCII).":
        "把修复后的波形写入文件（MiniSEED/SAC/TXT/ASCII）。",
    "Save Report Image": "保存报告图片",
    "Screenshot": "截图",
    "Copy preview to clipboard": "复制预览到剪贴板",
    "Preview copied to clipboard.": "预览已复制到剪贴板。",
    "Failed to copy to clipboard: {}": "复制到剪贴板失败：{}",
    "Failed to save report: {}": "保存报告失败：{}",
    "Failed to import report module: {}": "导入报告模块失败：{}",
    "Failed to generate preview: {}": "生成预览失败：{}",
    "Open usage guide": "打开使用指南",
    "Open Settings...": "打开设置…",
    "Open repaired DB file": "打开修复库文件",
    "Open the matching repaired DB file (optional; auto-derived from the raw file when possible)":
        "打开匹配的修复库文件（可选；能自动从原始文件推导时会自动填）",
    "Load a repaired DB file first (Load repaired…).": "请先载入修复库文件（载入修复…）。",
    "Load repaired…": "载入修复…",
    "Selected Segment": "选中的段",
    "Segment range (min)": "分段范围（分钟）",
    "Step 1: Select segment": "第 1 步：选择段",
    "Step 2: Choose method": "第 2 步：选择方法",
    "Step 3: Preview": "第 3 步：预览",
    "Step 4: Confirm": "第 4 步：确认",
    "Apply to repaired DB": "写入修复库",
    "Keep current": "保持当前",
    "Close": "关闭",
}

# ---------------------------------------------------------------------------
# 修复方法 / 检测算法说明
# ---------------------------------------------------------------------------
_METHODS: Dict[str, str] = {
    "Auto (by rules)": "自动（按规则分级）",
    "Threshold truncation": "阈值截断",
    "MAD-based cleaning": "基于 MAD 的清理",
    "Moving-average cleaning": "滑动均值清理",
    "Moonquake-protected local spike repair": "月震保护型局部尖峰修复",
    "Z-score cleaning": "Z 分数清理",
    "Missing-value interpolation": "缺失值插值",
    "Two-sided reference spectrum repair": "双侧参考频谱修复",
    "Layered reference spectrum repair": "分层参考频谱修复",
    "Reference-spectrum replacement": "参考频谱替换",
    "Isolated spike repair": "孤立尖峰修复",
    "Auto reference length (from anomaly)": "自动参考长度（按异常段推算）",
    "Paper defaults (reproducible)": "论文默认参数（可复现）",
    "User settings (~/.dssrr_settings.json)": "用户设置（~/.dssrr_settings.json）",
    "DSSRR with missing-run fallback: leftover zero/NaN points are linearly closed":
        "DSSRR + 缺失段兜底：残留的 0/NaN 点用线性插值补齐",
    "Frozen-clock segment: only the missing samples inside it are repaired":
        "冻结段：只修复段内缺失的样点",
    "Kept as-is (gap / too little information)": "保持原样（缺口 / 信息不足）",
    "Moonquake catalog protection (event left untouched)": "月震目录保护（事件保持原样）",
    "PSD-characteristics rebuild from both-side references (long gaps / anomalies)":
        "基于双侧参考段的 PSD 特征重建（长缺口 / 异常）",
    "Short isolated missing runs": "短的孤立缺失段",
    "DSSRR reference and energy guard": "DSSRR 参考段与能量守护",
    "DSSRR reference & energy guard": "DSSRR 参考段与能量守护",
    "DSSRR & Energy Guard": "DSSRR + 能量守护",
    "Amplitude threshold": "振幅阈值",
    "Amplitude threshold:": "振幅阈值：",
    "MAD threshold": "MAD 阈值",
    "MAD threshold:": "MAD 阈值：",
    "Z threshold": "Z 阈值",
    "Z threshold:": "Z 阈值：",
    "Random seed": "随机种子",
    "Random seed (-1 = random):": "随机种子（-1 = 随机）：",
    "Safety gap (s)": "安全间隔（秒）",
    "Safety gap (s):": "安全间隔（秒）：",
    "Reference before (s):": "前参考段（秒）：",
    "Reference after (s):": "后参考段（秒）：",
    "Auto min reference (s):": "自动最小参考长度（秒）：",
    "Before reference (s)": "前参考段长度（秒）",
    "After reference (s)": "后参考段长度（秒）",
    "Window size": "窗口大小",
    "Window size:": "窗口大小：",
    "Band minimum (Hz):": "频带下限（Hz）：",
    "Band maximum (Hz):": "频带上限（Hz）：",
    "Transition width (Hz):": "过渡带宽（Hz）：",
    "Frequency focus:": "频率侧重：",
    "Ultra-low-frequency priority": "超低频优先",
    "High-frequency priority": "高频优先",
    "Custom band": "自定义频带",
    "Interpolation method:": "插值方法：",
    "Detection window (s):": "检测窗口（秒）：",
    "Scale window (s):": "尺度窗口（秒）：",
    "Minimum global amplitude (sigma):": "全局最小振幅（σ）：",
    "Maximum spike width (samples):": "尖峰最大宽度（样点）：",
    "Robust threshold (sigma):": "稳健阈值（σ）：",
    "Neighbour return ratio:": "邻点回归比：",
    "Strong-peak local threshold (sigma):": "强尖峰局部阈值（σ）：",
    "Strong-peak global floor (sigma):": "强尖峰全局下限（σ）：",
    "Strong-peak neighbour ratio:": "强尖峰邻点比：",
    "Enable strong-peak rescue": "启用强尖峰救援",
    "Enable moonquake-catalog protection": "启用月震目录保护",
    "Quantize to int": "量化为整数",
    "Reset defaults": "恢复默认",
    "Repair methods": "修复方法",
    "Repair method": "修复方法",
    "Repair grading": "修复分级",
    "Repair grading and output": "修复分级与输出",
    "Repair Parameters": "修复参数",
    "Repair & Output": "修复与输出",
    "Detection Parameters": "检测参数",
    "Anomaly detection": "异常检测",
    "Output": "输出",
    "Verification": "验收",
    "Data Sources": "数据源",
    "Catalog CSV dir": "目录 CSV 输出目录",
    "Anomaly detection: spike/step/freeze/burst/event detectors and boundary padding "
    "(Auto pipeline). Hover any field in Settings for the same explanation.":
        "异常检测：尖峰/阶跃/冻结/突发/事件检测器与边界填充（Auto 流水线）。"
        "把鼠标停在设置里的任一字段上可看到同样的说明。",
    "Spike/step/freeze/burst/event detection thresholds and boundary padding; "
    "applied in Auto pipeline.":
        "尖峰/阶跃/冻结/突发/事件的检测阈值与边界填充；在 Auto 流水线中生效。",
    "Catalog protection: moonquake windows from apollo_catalog/*.csv are kept unchanged "
    "during repair (event/coda not rebuilt). Detail CSVs are written to the Catalog CSV "
    "dir on first batch run (auto-created).":
        "月震目录保护：apollo_catalog/*.csv 中的月震窗口在修复期间保持原样"
        "（事件/尾波不重建）。明细 CSV 在首次批量运行时写入「目录 CSV 输出目录」（自动创建）。",
    "Missing-run grading: runs ≤ lin_max_run samples are linearly interpolated, "
    "≤ z_max_run are Z-score cleaned, longer runs or energy/step segments go to DSSRR.":
        "缺失段分级：≤ lin_max_run 个样点用线性插值；≤ z_max_run 用 Z 分数清理；"
        "更长的段或能量/阶跃异常交给 DSSRR 重建。",
    "Missing runs are graded: ≤ lin_max_run samples → linear interpolation; "
    "3–z_max_run samples → Z-score cleaning; longer runs and energy/step anomalies → "
    "DSSRR rebuild.":
        "缺失段分级：≤ lin_max_run 个样点 → 线性插值；3–z_max_run 个样点 → Z 分数清理；"
        "更长的段与能量/阶跃异常 → DSSRR 重建。",
    "DSSRR reference length ≥ L×ref_ratio (clamped to [ref_min, ref_max] s); energy guard "
    "backs off repair when the segment is part of active oscillation background "
    "(osc_lo..osc_hi) or the rebuilt std exceeds raw×std_raw_factor / bg×std_bkg_factor.":
        "DSSRR 参考长度 ≥ L×ref_ratio（并钳制到 [ref_min, ref_max] 秒）；当该段属于"
        "活跃振荡背景（osc_lo..osc_hi），或重建后的标准差超过 raw×std_raw_factor / "
        "bg×std_bkg_factor 时，能量守护会放弃修复。",
    "DSSRR synthesizes the missing segment from PSD characteristics of reference segments "
    "on both sides. The energy guard backs off repair when the segment sits inside active "
    "oscillation background or the rebuilt std would exceed the caps.":
        "DSSRR 用异常段两侧参考段的 PSD 特征合成缺失信号。当该段位于活跃振荡背景内，"
        "或重建后的标准差会超过上限时，能量守护会放弃修复。",
    "DSSRR segment too long (max 180 min). Pick 'Auto' for pipeline repair of a long range, "
    "or narrow the range.":
        "DSSRR 段过长（上限 180 分钟）。长范围请选「自动」走流水线修复，或缩小范围。",
    "DSSRR needs at least one healthy reference segment: the range touches the data "
    "boundary or its reference windows are all missing/anomalous. Narrow the range or "
    "pick another method.":
        "DSSRR 至少需要一个健康参考段：所选范围触到了数据边界，或其参考窗口全部缺失/异常。"
        "请缩小范围或换一种方法。",
    "Synthesizes replacement signal from spectral features of healthy segments before/after "
    "the anomaly. Best for complex anomalies with specific frequency content.":
        "用异常段前后健康段的谱特征合成替代信号。适合具有特定频率成分的复杂异常。",
    "Clips values exceeding the threshold to the threshold level. Good for hard saturation "
    "(e.g., ADC max).":
        "把超过阈值的数值截断到阈值。适合硬饱和（如 ADC 满量程）。",
    "Compares each point to its local moving average. Deviations above threshold are flagged.":
        "把每个点与它的局部滑动均值比较，偏离超过阈值的点被标记。",
    "Detects outliers using Median Absolute Deviation. Robust to non-Gaussian noise.":
        "用中位数绝对偏差（MAD）检测离群点，对非高斯噪声稳健。",
    "Detects outliers using Z-score (standard deviations from mean). Fast but sensitive to "
    "extreme values.":
        "用 Z 分数（相对均值的标准差倍数）检测离群点。速度快，但对极端值敏感。",
    "For complex waveforms: reference-spectrum replacement preserves signal character "
    "better than simple interpolation.":
        "复杂波形：参考频谱替换比简单插值更能保留信号特征。",
    "For saturation spikes: Threshold truncation is fastest. Reference-spectrum gives "
    "smoother results.":
        "饱和尖峰：阈值截断最快；参考频谱替换结果更平滑。",
    "Maximum allowed absolute value. Points above are clipped.":
        "允许的最大绝对值，超过的点会被截断。",
    "Moving average window length in samples.": "滑动均值窗口长度（样点）。",
    "Outlier threshold in MAD units (1 MAD ≈ 0.6745 σ).": "离群阈值，单位 MAD（1 MAD ≈ 0.6745 σ）。",
    "Outlier threshold in standard deviations.": "离群阈值，单位标准差。",
    "Threshold: set to instrument max (e.g., 1023 for 10-bit ADC).":
        "阈值：设为仪器满量程（如 10 位 ADC 为 1023）。",
    "Window: 5–20 samples. Amplitude threshold: 2.0–5.0.": "窗口：5–20 样点。振幅阈值：2.0–5.0。",
    "MAD threshold: 3.0 (default) catches ~0.3% outliers. Increase to 5.0 for fewer false "
    "positives.":
        "MAD 阈值 3.0（默认）约能抓出 0.3% 的离群点。调到 5.0 可减少误报。",
    "Z threshold: 3.0 catches ~0.3%. Use 4.0–5.0 for conservative cleaning.":
        "Z 阈值 3.0 约能抓出 0.3%。保守清理用 4.0–5.0。",
    "Length of reference segment before anomaly for spectral estimation.":
        "异常段之前用于谱估计的参考段长度。",
    "Length of reference segment after anomaly.": "异常段之后参考段的长度。",
    "Minimum distance between reference segment and anomaly boundary.":
        "参考段与异常边界之间的最小距离。",
    "Minimum auto reference length (s). Keep >=120 for a stable Welch PSD; the paper "
    "default range is 120--600 s.":
        "自动参考长度的下限（秒）。为保证 Welch PSD 稳定，请保持 ≥120；论文默认范围 120–600 秒。",
    "Controls reproducibility of synthesized signal. Fixed value = reproducible. -1 = random "
    "each time.":
        "控制合成信号的可重复性。固定值 = 可复现；-1 = 每次随机。",
    "When enabled, reference length = clip(anomaly/2, min, 600) s. Manual before/after "
    "fields are disabled.":
        "启用后，参考长度 = clip(异常长度/2, 下限, 600) 秒；手动的前/后参考段输入框会被禁用。",
    "Before/After reference: 50–200s. Safety gap: 0.5–5s. Random seed: use fixed value for "
    "debugging, -1 for production.":
        "前后参考段：50–200 秒。安全间隔：0.5–5 秒。随机种子：调试用固定值，生产用 -1。",
    "Quantize repaired values back to integer counts (round-half-even) to match the raw "
    "int32 archives. Raw counts are integers; PSD-rebuilt values are real, ±0.5 quantization "
    "noise is negligible vs DSSRR rebuild uncertainty. Disable to keep float precision for "
    "high-resolution spectral analysis.":
        "把修复值量化回整数计数（四舍六入五成双），与原始 int32 存档一致。原始计数是整数，"
        "而 PSD 重建值是实数；±0.5 的量化噪声相对 DSSRR 重建本身的不确定度可以忽略。"
        "如需高分辨率谱分析，可关闭以保留浮点精度。",
    "Round repaired values to integer counts on write-back. Uncheck to keep float precision "
    "in this run.":
        "写回时把修复值四舍五入为整数计数。取消勾选可在本次运行中保留浮点精度。",
}

# ---------------------------------------------------------------------------
# 批量修复对话框
# ---------------------------------------------------------------------------
_BATCH: Dict[str, str] = {
    "Start Batch Repair": "开始批量修复",
    "Check Data": "检查数据",
    "Pre-check": "预检查",
    "Deep Scan (MHZ / broken files)": "深度扫描（MHZ / 损坏文件）",
    "Stations": "台站",
    "Select at least one station.": "请至少选择一个台站。",
    "Broken files: ": "损坏文件：",
    "Deep scan: MHZ={mhz}  no-MHZ={nomhz}  unreadable={err}":
        "深度扫描：MHZ={mhz}  无MHZ={nomhz}  不可读={err}",
    "Moonquake catalog: built-in": "月震目录：内置",
    "Moonquake catalog: {n} events from {src} {tag}": "月震目录：来自 {src} 的 {n} 个事件 {tag}",
    "(no user settings)": "（无用户设置）",
    "(user override)": "（用户自定义）",
    "Previous year": "前一年",
    "Next year": "后一年",
    "Previous month": "前一月",
    "Next month": "后一月",
    "Previous day": "前一天",
    "Next day": "后一天",
    "End (min)": "结束（分钟）",
    "Files: {n}  ({gb:.2f} GB)  |  {months} months\nEstimated repair time: {est:.0f} min "
    "({per:.1f} s/file, 1976-01 measured base)":
        "文件：{n} 个（{gb:.2f} GB）  |  {months} 个月\n预计修复耗时：{est:.0f} 分钟"
        "（{per:.1f} 秒/文件，按 1976-01 实测基准）",
    "No mseed files in the selected scope.": "所选范围内没有 mseed 文件。",
    "No *.mseed files found in this folder.": "该文件夹下没有 *.mseed 文件。",
    "Found {n} .mseed file(s), but none matches the naming convention "
    "NET.STA.YYYYMMDD.mseed or NET.STA.LOC.CHA.START-END.mseed.":
        "找到 {n} 个 .mseed 文件，但没有一个符合命名约定 "
        "NET.STA.YYYYMMDD.mseed 或 NET.STA.LOC.CHA.START-END.mseed。",
    "All {n} matching file(s) belong to stations outside the selection ({sel}). "
    "Stations found: {seen}.":
        "匹配到的 {n} 个文件全部属于未勾选的台站（{sel}）。实际发现的台站：{seen}。",
    "{n} file(s) matched but their dates fall outside {d0} ~ {d1}. Widen the date range or "
    "check the file names.":
        "匹配到 {n} 个文件，但日期都在 {d0} ~ {d1} 之外。请放宽日期范围或检查文件名。",
    "Start: {n} files, stations={st}, {d0}~{d1}, catalog={cat}":
        "开始：{n} 个文件，台站={st}，{d0}~{d1}，目录={cat}",
    "Each day is verified after processing: protected moonquake segments must stay "
    "unchanged, modified-sample ratios are reported, and residual statistics are checked "
    "before the day is accepted.":
        "每天处理完都会验收：受保护的月震段必须保持原样，报告修改样点比例，"
        "并在通过前检查残差统计。",
    "Done in {m:.1f} min.\nProcessed: {n} files   repaired segments: {r}   gaps: {g}\n"
    "Verify: {vo}  ({vf} checked, {vfail} FAIL, {vskip} skipped)":
        "耗时 {m:.1f} 分钟。\n已处理：{n} 个文件   修复段：{r}   缺口：{g}\n"
        "验收：{vo}（检查 {vf}，未通过 {vfail}，跳过 {vskip}）",
    "Invalid range: end <= start": "范围无效：结束 ≤ 开始",
    "Parameters saved and will be applied to the next manual re-processing (Auto pipeline).":
        "参数已保存，将应用到下一次手动重处理（Auto 流水线）。",
}

# ---------------------------------------------------------------------------
# 导出设置 / 报告图文字编辑器
# ---------------------------------------------------------------------------
_EXPORT: Dict[str, str] = {
    "Export Settings": "导出设置",
    "Basic Settings": "基本设置",
    "Frequency Range": "频带范围",
    "Display Options": "显示选项",
    "Title:": "标题：",
    "Default title": "使用默认标题",
    "Leave empty to use the default report title.": "留空则使用默认报告标题。",
    "Resolution:": "分辨率：",
    "Format:": "格式：",
    "Size:": "尺寸：",
    "Min frequency:": "最小频率：",
    "Max frequency:": "最大频率：",
    "0 = Auto (Nyquist × 0.9)": "0 = 自动（奈奎斯特频率 × 0.9）",
    "Time domain": "时域对比图",
    "PSD spectrum": "功率谱密度图",
    "Linear spectrum": "线性频谱图",
    "Quality metrics": "质量指标表格",
    "Reference segments": "参考段",
    "Inset zoom": "放大插图",
    "Click 'Preview' to generate": "点击「预览」生成报告图",
    "Edit Text": "编辑文字",
    "Batch edit titles, axis labels, and legends": "批量编辑标题、轴标签和图例",
    "Figure Title:": "图标题：",
    "Axes": "坐标轴",
    "Axes Text": "坐标轴文字",
    "Legend Title": "图例标题",
    "X Label": "X 轴标签",
    "Y Label": "Y 轴标签",
    "Font": "字体",
    "Font Family:": "字体：",
    "Font Size:": "字号：",
    "Font Weight:": "字重：",
    "Text Color:": "文字颜色：",
    "Choose Text Color": "选择文字颜色",
    "Choose a color for edited text.": "为编辑的文字选择颜色。",
    "Choose the font family for edited text.": "为编辑的文字选择字体。",
    "Set the font size in points; zero keeps the current size.":
        "设置字号（磅）；0 表示保持当前字号。",
    "Set the weight for edited text.": "设置编辑文字的字重。",
    "Apply the text and style changes.": "应用文字与样式改动。",
    "Reload text from the current figure.": "从当前图形重新载入文字。",
    "Canvas colors": "画布配色",
    "Optional figure-level title": "可选的整图标题",
    "3, >3, or <-3": "3、>3 或 <-3",
}

#: 完整的英文 → 中文词条表。
ZH: Dict[str, str] = {}
for _part in (_COMMON, _MAIN_WINDOW, _VIEWER, _DEPULSE, _METHODS, _BATCH, _EXPORT):
    ZH.update(_part)
del _part

__all__ = ["ZH"]
