#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图1：研究背景概览
(a) Apollo PSE 台站分布（引用 Nunn et al., 2020）
(b) 长时程记录中的典型异常（真实 IRIS 数据）
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# 设置学术风格
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

client = Client("IRIS")

# 创建图形：1行2列
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.5),
                                  gridspec_kw={'width_ratios': [1.0, 1.1]})
fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.13, wspace=0.20)

# ============================================================
# (a) Nunn 2020 台站分布图
# ============================================================
nunn_img_path = r'G:\SeisY\docs\dssrr_paper\submission_jgr_planets\figures\_page1.png'
nunn_img = Image.open(nunn_img_path)

# 在 axes 中显示图片
ax_a.imshow(nunn_img)
ax_a.set_title('(a) Apollo PSE station distribution',
               fontsize=10.5, fontweight='bold', loc='left', pad=8)
ax_a.axis('off')  # 不显示坐标轴

# 引用标注
ax_a.text(0.5, -0.05, 'Adapted from Nunn et al. (2020)',
          transform=ax_a.transAxes, ha='center', va='top',
          fontsize=8, style='italic', color='#666666')

# ============================================================
# (b) 典型异常长连续记录波形
# ============================================================
# 下载 S12 1976 数据
t0 = UTCDateTime('1976-01-13T07:00:00')
t1 = UTCDateTime('1976-01-13T12:00:00')
print(f"Downloading XA.S12.01.MHZ from {t0} to {t1} ...")
st = client.get_waveforms(network='XA', station='S12', location='01', channel='MHZ',
                          starttime=t0, endtime=t1)
tr = st[0]

# 获取时间和数据
times = [t0 + t for t in tr.times()]
times_mdates = mdates.date2num([t.datetime for t in times])
data = tr.data

# 绘制波形
ax_b.plot(times_mdates, data, color='#2c3e50', linewidth=0.5)


# 标注典型异常
# Spikes: 两个尖峰簇，中间一个标签，连线指向各自区域
spikes1_sd = mdates.date2num(UTCDateTime('1976-01-13T07:23:54').datetime)
spikes1_ed = mdates.date2num(UTCDateTime('1976-01-13T07:26:00').datetime)
spikes2_sd = mdates.date2num(UTCDateTime('1976-01-13T10:14:00').datetime)
spikes2_ed = mdates.date2num(UTCDateTime('1976-01-13T10:14:33').datetime)
ax_b.axvspan(spikes1_sd, spikes1_ed, alpha=0.2, color='#e67e22', zorder=0)
ax_b.axvspan(spikes2_sd, spikes2_ed, alpha=0.2, color='#e67e22', zorder=0)

# 标签位置：两个区域上方中间
spikes_label_x = (spikes1_ed + spikes2_sd) / 2
spikes_label_y = ax_b.get_ylim()[1] * 0.85
# 文字
ax_b.text(spikes_label_x, spikes_label_y, 'Spikes',
          ha='center', va='bottom', fontsize=8, color='#d35400', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))
# 从两个区域顶部各画一条斜线指向文字
for region_sd, region_ed in [(spikes1_sd, spikes1_ed), (spikes2_sd, spikes2_ed)]:
    region_center = (region_sd + region_ed) / 2
    ax_b.plot([region_center, spikes_label_x],
              [ax_b.get_ylim()[1] * 0.98, spikes_label_y],
              color='#d35400', lw=0.7, alpha=0.7, zorder=1)

# Dropout (10:24-10:27)
drop_sd = mdates.date2num(UTCDateTime('1976-01-13T10:24:00').datetime)
drop_ed = mdates.date2num(UTCDateTime('1976-01-13T10:27:00').datetime)
ax_b.axvspan(drop_sd, drop_ed, alpha=0.18, color='#e74c3c', zorder=0)
ax_b.text((drop_sd + drop_ed) / 2, ax_b.get_ylim()[1] * 0.93,
          'Dropout', ha='center', va='top', fontsize=8, color='#e74c3c',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

# Spike Cluster (11:05-11:20)
spike_cluster_sd = mdates.date2num(UTCDateTime('1976-01-13T11:05:00').datetime)
spike_cluster_ed = mdates.date2num(UTCDateTime('1976-01-13T11:20:00').datetime)
ax_b.axvspan(spike_cluster_sd, spike_cluster_ed, alpha=0.2, color='#e67e22', zorder=0)
ax_b.text((spike_cluster_sd + spike_cluster_ed) / 2 + 0.005, ax_b.get_ylim()[1] * 0.80,
          'Spike Cluster', ha='left', va='top', fontsize=8, color='#d35400',
          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))

# 格式化x轴
ax_b.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax_b.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax_b.set_xlabel('Time (HH:MM)', fontsize=10)
ax_b.set_ylabel('Amplitude (DU)', fontsize=10)
ax_b.set_title('(b) Typical anomalies in long-duration record',
               fontsize=10.5, fontweight='bold', loc='left', pad=8)

# 台站信息
ax_b.text(0.02, 0.95, f'{tr.id}',
          transform=ax_b.transAxes, va='top', fontsize=8, color='#666666',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

# 保存
output_path = r'G:\SeisY\docs\dssrr_paper\submission_srl\supplement\figures\figS1_background.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图S1已保存到: {output_path}")

plt.close()
