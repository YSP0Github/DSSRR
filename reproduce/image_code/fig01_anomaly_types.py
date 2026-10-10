"""图1：四类典型月震异常形态（IRIS/FDSN 原始数据，Signal Red 高清版）

数据通过 IRIS FDSN 客户端实时下载 Apollo PSE 台站（XA 网络）原始 MHZ 波形，
不使用 experiments/results_obs 下可能被处理过的本地文件。
首次下载后把 IRIS 原始字节缓存到 image_code/_iris_cache，重跑直接读缓存；
2x2 布局，异常填充统一浅红，1200 dpi。
事件选择/时段/异常位置与历史 draw_fig2_from_iris.py 一致。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime, read
from obspy.clients.fdsn import Client
import sys
sys.path.insert(0, os.path.dirname(__file__))
from srl_style import apply_srl_style, DOUBLE_COL_WIDTH, ANOMALY_FILL, ANOMALY_ALPHA, PANEL_LABEL_KW, SAVE_KW

apply_srl_style()
FIG_DIR = r'G:\SeisY\docs\dssrr_paper\submission_srl\manuscript\figures'
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_iris_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

client = Client("IRIS")

# 四个案例配置（全部 IRIS 实时下载）
cases = [
    {
        'network': 'XA', 'station': 'S12', 'location': '01', 'channel': 'MHZ',
        'start': '1976-01-13T09:30:00',
        'end':   '1976-01-13T12:30:00',
        'anomalies': [
            ('1976-01-13T10:14:26', '1976-01-13T10:14:33'),
            ('1976-01-13T10:24:41', '1976-01-13T10:27:49'),
            ('1976-01-13T11:05:06', '1976-01-13T11:20:08'),
        ],
        'panel': '(a) Long dropout + spike clusters'
    },
    {
        'network': 'XA', 'station': 'S15', 'location': '00', 'channel': 'MHZ',
        'start': '1972-09-17T13:30:00',
        'end':   '1972-09-17T17:00:00',
        'anomalies': [
            ('1972-09-17T14:30:38', '1972-09-17T15:35:03'),
            ('1972-09-17T15:51:45', '1972-09-17T15:58:35'),
        ],
        'panel': '(b) Stepped long dropout'
    },
    {
        'network': 'XA', 'station': 'S15', 'location': '01', 'channel': 'MHZ',
        'start': '1976-01-13T10:30:00',
        'end':   '1976-01-13T13:00:00',
        'anomalies': [
            ('1976-01-13T11:47:07', '1976-01-13T12:09:56'),
        ],
        'panel': '(c) Dense spike cluster'
    },
    {
        'network': 'XA', 'station': 'S12', 'location': '00', 'channel': 'MHZ',
        'start': '1972-01-04T05:30:00',
        'end':   '1972-01-04T07:30:00',
        'anomalies': [
            ('1972-01-04T05:57:06', '1972-01-04T06:00:05'),
        ],
        'panel': '(d) Single large spike'
    },
]


def fetch(case):
    """从 IRIS 下载（首次），否则读 IRIS 原始字节缓存。返回 (stream, t0)。"""
    t0 = UTCDateTime(case['start'])
    t1 = UTCDateTime(case['end'])
    net, sta, loc, cha = case['network'], case['station'], case['location'], case['channel']
    cache = os.path.join(
        CACHE_DIR,
        f"{net}.{sta}.{loc}.{cha}.{t0.strftime('%Y%m%dT%H%M%S')}_{t1.strftime('%Y%m%dT%H%M%S')}.mseed")
    if os.path.exists(cache):
        st = read(cache)
    else:
        print(f"Downloading {net}.{sta}.{loc}.{cha}  {case['start']} -> {case['end']} ...")
        st = client.get_waveforms(net, sta, loc, cha, t0, t1)
        st.write(cache, format='MSEED')
        print(f"  cached IRIS original -> {os.path.basename(cache)}")
    return st, t0


# 标准双栏 2x2 尺寸
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.7))
axes = axes.flatten()

for i, case in enumerate(cases):
    ax = axes[i]
    st, t0 = fetch(case)
    tr = st[0]
    times = [t0 + t for t in tr.times()]
    times_md = mdates.date2num([t.datetime for t in times])
    ax.plot(times_md, tr.data, color="#000000", lw=0.6)
    for s_str, e_str in case['anomalies']:
        s_md = mdates.date2num(UTCDateTime(s_str).datetime)
        e_md = mdates.date2num(UTCDateTime(e_str).datetime)
        ax.axvspan(s_md, e_md, color=ANOMALY_FILL, alpha=ANOMALY_ALPHA, zorder=0, lw=0)
    ax.set_title(case['panel'], **PANEL_LABEL_KW)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.set_xlabel('Time (HH:MM)', fontsize=8)
    ax.set_ylabel('Amplitude (DU)', fontsize=8)
    ax.text(0.02, 0.92, tr.id, transform=ax.transAxes, va='top',
            fontsize=6.8, color='#666666',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))
    ax.tick_params(labelsize=7)

plt.tight_layout(h_pad=1.0, w_pad=1.0, pad=0.1)
out_path = f'{FIG_DIR}/fig01_anomaly_types.png'
plt.savefig(out_path, **SAVE_KW)
plt.close()
print(f"IRIS-sourced Figure 1 saved to {out_path}")
