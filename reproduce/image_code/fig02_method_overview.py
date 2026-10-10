# -*- coding: utf-8 -*-
"""合成 Figure 2: method overview（双行垂直拼接）。

fig02_method_overview.png = _panels/fig02a_workflow.png              (上, panel a)
                          + _panels/fig02bc_repair_before_after.png  (下, panel b/c)

上面两个面板由以下脚本生成（须先运行，输出在 manuscript/figures/_panels/）：
    fig02a_workflow.png             <- fig02a_workflow_panel.py
    fig02bc_repair_before_after.png <- fig02bc_repair_panel.py

拼接前先裁掉四周纯白边、压缩内部超长空白带，再**不缩放**地用白边补齐到同一宽度后白底拼接，
最终图写入 manuscript/figures/。一键顺序出图见 build_all_figures.py。
"""
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.normpath(os.path.join(HERE, "..", "manuscript", "figures"))
PANEL_DIR = os.path.join(FIG_DIR, "_panels")
GAP = 160         # 两行之间的白色间隔（像素，按统一宽度计）
MIN_GAP = 500     # 只压缩长于该值的内部空白带
KEEP = 90         # 压缩后保留的空白高度
# 统一输出宽度：两个面板来自同一 figsize（7.09 in @1200 dpi）的图，原始像素/英寸完全相同，
# 所以**绝不能各自 resize 到同一宽度** —— 裁白边后两者宽度略有差异（8425 vs 8502 px），
# 分别缩放到同一宽度会引入 0.9% 的缩放差，两个面板的字号就不再一致。
# 正确做法：不缩放，把较窄的一张用白边补齐到最宽者。2026-10-09 修。


def trim_white(im, thresh=250, pad=8):
    """裁掉四周接近纯白的边。"""
    a = np.asarray(im.convert('L'))
    mask = a < thresh
    if not mask.any():
        return im
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, a.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, a.shape[1])
    return im.crop((c0, r0, c1, r1))


def squeeze_white(im, min_gap=MIN_GAP, keep=KEEP, thresh=250):
    """把内部超长纯白横带压缩为 keep 像素。"""
    a = np.asarray(im.convert('L'))
    rows = (a < thresh).any(axis=1)
    idx = np.where(rows)[0]
    if len(idx) == 0:
        return im
    segs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            segs.append((start, prev + 1))
            start = i
        prev = i
    segs.append((start, prev + 1))
    if len(segs) == 1:
        return im
    crops = [im.crop((0, r0, im.size[0], r1)) for r0, r1 in segs]
    H = sum(c.size[1] for c in crops) + keep * (len(crops) - 1)
    canvas = Image.new('RGB', (im.size[0], H), 'white')
    y = 0
    for c in crops:
        canvas.paste(c, (0, y))
        y += c.size[1] + keep
    return canvas


def prep(path):
    """读入一张面板图并做裁白边 + 压空白带。"""
    im = Image.open(path).convert('RGB')
    before = im.size[1]
    im = squeeze_white(trim_white(im))
    if im.size[1] != before:
        print('   squeeze %s: %d -> %d px' % (os.path.basename(path), before, im.size[1]))
    return im


def stack(top_path, bot_path, out_name):
    """上下拼接两张面板图，输出到 figures/<out_name>。

    两张面板都不缩放：只做白边裁剪 + 长空白带压缩，再把较窄者用白边居中补齐到
    最宽者，这样两行的像素/英寸严格相同，字号一致（见文件头 GAP 处说明）。
    """
    ims = [prep(p) for p in (top_path, bot_path)]
    W = max(im.size[0] for im in ims)
    padded = []
    for im in ims:
        w, h = im.size
        canvas = Image.new('RGB', (W, h), 'white')
        canvas.paste(im, ((W - w) // 2, 0))
        padded.append(canvas)
    H = padded[0].size[1] + GAP + padded[1].size[1]
    canvas = Image.new('RGB', (W, H), 'white')
    canvas.paste(padded[0], (0, 0))
    canvas.paste(padded[1], (0, padded[0].size[1] + GAP))
    canvas.save(os.path.join(FIG_DIR, out_name), dpi=(1200, 1200))
    print('   -> %s  %s  ratio=%.3f' % (out_name, canvas.size, H / W))


def main():
    print('fig02_method_overview:')
    stack(os.path.join(PANEL_DIR, 'fig02a_workflow.png'),
          os.path.join(PANEL_DIR, 'fig02bc_repair_before_after.png'),
          'fig02_method_overview.png')
    print('done')


if __name__ == '__main__':
    main()
