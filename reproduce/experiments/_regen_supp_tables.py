# -*- coding: utf-8 -*-
"""按最新的 results_stats/dispersion_length_scan.csv 重生成补充材料 Table S2 / S3。

背景
----
SSA 实现统一后（2026-10-10）图 5 的 SSA 曲线变了，Table S2--S3 是从同一份
离散度统计生成的，必须同步更新，否则补充材料与正文图 5 自相矛盾。

口径（与补充材料 caption 完全一致）
------------------------------------
每格 = ``median ± sd``（n = 4 条记录）。
**n = 4 时"截尾平均"（去掉最好与最差后取中两项均值）恰好等于中位数**，
所以 CSV 的 ``median`` 列就是 caption 里说的 trimmed mean，``sd`` 是全部 4 条的样本标准差。
长度 12.137540 min 按 Figure 5 的 ``EXCLUDE_LENGTHS`` 剔除。

用法：G:/miniconda3/envs/python_3.11/python.exe _regen_supp_tables.py [--write]
不带 --write 只打印，带 --write 才改 .tex（改前自动备份）。
"""
from __future__ import annotations

import os
import re
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SUB_DIR = os.path.normpath(os.path.join(HERE, "..", "submission_srl", "supplement"))
TEX = os.path.join(SUB_DIR, "SRL-DSSRR-supplement.tex")
DISP = os.path.join(HERE, "results_stats", "dispersion_length_scan.csv")

EXCLUDE_LENGTHS = [12.137540]
METHODS = ["DSSRR", "Linear", "FFT", "SSA"]


def load():
    d = pd.read_csv(DISP)
    for L in EXCLUDE_LENGTHS:
        d = d[~d["length_min"].round(6).eq(round(L, 6))]
    return d


def rows_for(d, metric):
    out = []
    lens = sorted(d["length_min"].unique())
    for L in lens:
        cells = []
        for m in METHODS:
            r = d[(d["method"] == m) & (d["metric"] == metric)
                  & (d["length_min"] == L)]
            if r.empty:
                cells.append("--")
            else:
                cells.append(f"${r['median'].iloc[0]:.2f} \\pm {r['sd'].iloc[0]:.2f}$")
        out.append(f"{L:.2f} & " + " & ".join(cells) + r" \\")
    return out


def build_tabS2(d):
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Gap (min) & DSSRR & Linear & FFT & SSA \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Wasserstein distance}} \\",
    ]
    lines += rows_for(d, "wasserstein")
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{ACF $L_2$ distance}} \\",
    ]
    lines += rows_for(d, "acf_l2")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def build_tabS3(d):
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Gap (min) & DSSRR & Linear & FFT & SSA \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Energy ratio $\mathrm{std(repaired)}/\mathrm{std(truth)}$}} \\",
    ]
    lines += rows_for(d, "std_ratio")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    d = load()
    new2, new3 = build_tabS2(d), build_tabS3(d)
    print("=== Table S2 (new) ===")
    print(new2)
    print("\n=== Table S3 (new) ===")
    print(new3)

    if "--write" not in sys.argv:
        print("\n[DRY-RUN] 未加 --write，未修改任何文件。")
        return

    with open(TEX, encoding="utf-8") as fh:
        tex = fh.read()

    pat2 = re.compile(
        r"(\\caption\{Dispersion of the gap-length sweep.*?\\label\{tabS:lengthdisp\}\n)"
        r"\\begin\{tabular\}\{lcccc\}.*?\\end\{tabular\}",
        re.S)
    pat3 = re.compile(
        r"(\\caption\{Energy fidelity across the gap-length sweep.*?\\label\{tabS:lengthenergy\}\n)"
        r"\\begin\{tabular\}\{lcccc\}.*?\\end\{tabular\}",
        re.S)

    n2 = len(pat2.findall(tex))
    n3 = len(pat3.findall(tex))
    if n2 != 1 or n3 != 1:
        print(f"[ABORT] 定位失败：S2 命中 {n2} 次，S3 命中 {n3} 次")
        sys.exit(1)

    bak = TEX + ".bak_pre_ssa_unify"
    shutil.copy2(TEX, bak)
    tex = pat2.sub(lambda m: m.group(1) + new2, tex, count=1)
    tex = pat3.sub(lambda m: m.group(1) + new3, tex, count=1)
    with open(TEX, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"\n[OK] 已写入 {TEX}\n[BAK] {bak}")


if __name__ == "__main__":
    main()
