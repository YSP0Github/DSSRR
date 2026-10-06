# DSSRR 批量去异常命令行工具（`dssrr-batch`）

跨平台批量去异常工具：Linux / macOS / Windows PowerShell 通用。与桌面 GUI 的
Batch Repair 使用**同一套修复流水线**（E7 检测 → 修复入库 → 月震目录保护 →
catalog CSV → 自动验收），但不依赖 PyQt，可在无图形环境的服务器上运行。

## 1. 安装

```bash
# 从源码目录安装（任选其一）
pip install -e .                            # 在 DSSRR 仓库根目录下
pip install -e /path/to/DSSRR               # Linux / macOS
pip install -e "G:/path/to/DSSRR"           # Windows（PowerShell 同样适用）

# 或直接通过包入口运行（无需安装，需在 DSSRR 目录内）
python -m dssrr.cli --help
```

安装后获得两个命令：`dssrr-gui`（桌面版）和 `dssrr-batch`（命令行版）。
命令行模式**不需要 PyQt5**，核心依赖仅 numpy / scipy / obspy。

## 2. 数据目录约定

扫描器只识别原始库的标准结构：

```
<raw_root>/
└── 1976/
    └── 01/
        ├── XA.S12.19760113.mseed
        └── XA.S15.19760113.mseed
```

- 文件名必须为 `XA.{station}.{YYYYMMDD}.mseed`（第 2 段台站、第 3 段日期）
- 修复库输出保持同样结构：`<fix_root>/1976/01/XA.S12.19760113.mseed`

> `--raw-root` / `--fix-root` 是**必填参数**，DSSRR 不在代码里硬编码任何
> 机器上的路径。GUI 的 Batch Repair 页面则从环境变量 `DSSRR_RAW_ROOT` /
> `DSSRR_FIX_ROOT` 读取默认值（未设置时留空由用户选择）。

## 3. 基本用法

```bash
# Linux / macOS
dssrr-batch \
  --raw-root /data/apollo/raw \
  --fix-root /data/apollo/repair \
  --stations S12,S15 \
  --d0 19760101 --d1 19760131

# Windows PowerShell（反引号换行）
dssrr-batch `
  --raw-root "D:\apollo\raw" `
  --fix-root "D:\apollo\repair" `
  --stations S12,S15 `
  --d0 19760101 --d1 19760131
```

参数说明：

| 参数 | 必填 | 说明 |
|------|------|------|
| `--raw-root` | 是 | 原始库根目录；目录结构不限，文件名需为 `NET.STA.YYYYMMDD.mseed` 或 `NET.STA.LOC.CHA.START-END.mseed` |
| `--fix-root` | 是 | 修复库输出根目录（自动创建） |
| `--stations` | 否 | 逗号分隔台站，如 `S12,S15`；缺省 = 全部 |
| `--d0` / `--d1` | 否 | 日期范围 YYYYMMDD（缺省 19760101~19760201） |
| `--catalog-dir` | 否 | catalog/verify 输出目录（缺省 `<fix-root>/catalogs`） |
| `--no-catalog` | 否 | 关闭月震目录保护（事件全部转 repair） |
| `--settings` | 否 | 读取用户设置 `~/.dssrr_settings.json` |
| `--sr` | 否 | 采样率 Hz（缺省 6.625） |
| `--summary-json` | 否 | 汇总 JSON 路径（缺省 `<catalog-dir>/batch_summary.json`） |

## 4. 运行流程与产出

命令执行后依次完成：

1. **扫描** — 统计范围内文件数 / 体积 / 月份数 / 预计耗时
2. **逐月修复** — 每个文件经 E7 检测后修复，输出到 `<fix_root>/YYYY/MM/`
3. **catalog** — 每月修复明细写 `<catalog-dir>/E7_catalog_{YYYYMM}.csv`
4. **自动验收** — 逐文件检查缺失残留 / 0 坑 / 非修复区零改动 / 保护段
   - 结果写 `<catalog-dir>/verify_{YYYYMM}.json`
   - 汇总写 `<catalog-dir>/batch_summary.json`
5. **退出码** — `0` = verify 全部通过；`1` = 有失败文件或参数错误

终端输出示例（S12 单日）：

```
[scan] 1 file(s), 0.00 GB, 1 month(s), est. 0.0 min
[scope] 1 file(s), stations=S12, 19760113~19760113
[1/1] month 197601: 1 file(s)
  repaired XA.S12.19760113.mseed  (rep=5 gap=0)
[catalog] 197601: 5 row(s) -> .../fix/catalogs/E7_catalog_197601.csv
[verify] verifying repaired files...
[verify] json -> .../fix/catalogs/verify_197601.json
[done] repaired=5 gaps=0 errors=0 elapsed=1.2s verify_ok=True
[summary] -> .../fix/catalogs/batch_summary.json
[RESULT] VERIFY OK
```

## 5. 进阶用法

### 5.1 使用自定义检测 / 修复参数

先在任意机器上用 GUI（Batch Repair → 设置）调好参数，参数会保存到
`~/.dssrr_settings.json`；命令行加 `--settings` 即读取同一份配置：

```bash
dssrr-batch --raw-root ... --fix-root ... --stations S12 --settings
```

> 升级兼容：若旧安装里存在 `~/.seisy_repair_settings.json`，读取时会自动回退
> 到该旧文件；新的写入一律使用 `~/.dssrr_settings.json`。

### 5.2 关闭月震目录保护（仅检测器规则）

```bash
dssrr-batch --raw-root ... --fix-root ... --no-catalog
```

### 5.3 脚本中调用并判断结果

```bash
# Linux bash
if dssrr-batch --raw-root ... --fix-root ... --d0 19760101 --d1 19760131; then
  echo "repair + verify OK"
else
  echo "verify failed, see batch_summary.json"
fi

# Windows PowerShell
dssrr-batch --raw-root ... --fix-root ... --d0 19760101 --d1 19760131
if ($LASTEXITCODE -eq 0) { "OK" } else { "FAILED - see batch_summary.json" }
```

### 5.4 直接以 Python API 调用（嵌入自己的流程）

```python
from dssrr.cli import batch_repair, quick_scan

r = quick_scan("/data/raw", ["S12"], 19760101, 19760131)
print(r["files"], r["est_sec"])

summary = batch_repair(
    raw_root="/data/raw", fix_root="/data/repair",
    stations=["S12"], d0=19760101, d1=19760131,
    use_catalog=True)
print(summary["verify"]["ok"], summary["repaired"])
```

## 6. 注意事项

- **片段文件名不识别**：扫描器只认 `XA.{sta}.{YYYYMMDD}.mseed` 日文件；
  若原始库是片段文件（`XA.S12.01.MHZ.19760113_032409-....mseed`），需先
  合并/重命名为日文件再运行。
- **采样率**：Apollo 原始库采样率为 6.625 Hz；若处理其他数据，请用 `--sr`
  显式指定。
- **中断**：`Ctrl+C` 安全退出（退出码 130）；已完成的文件不受影响，重跑会
  覆盖同名单文件。
- **验证失败排查**：先看 `<catalog-dir>/verify_*.json` 中 `by_month` 的
  `n_fail` 与日志里的 `FAIL` 行，常见原因是数据本身只覆盖部分时段（非全天）
  或修复库缺失。
- **月震目录**：随包分发一份合并月震目录（`dssrr/repair_lib/moonquake_catalog.json`，
  约 9500 个事件）。如需覆盖，把同结构的 JSON 放到
  `~/.dssrr_moonquake_catalog.json` 即可（用户文件优先）。`--no-catalog`
  会完全关闭目录保护，此时所有事件段都按普通异常修复。
- **运行环境**：命令行模式不需要 PyQt5；核心依赖为 numpy / scipy / obspy，
  写 catalog CSV 还需 pandas。
