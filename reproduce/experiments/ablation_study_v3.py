"""
DSSRR 消融实验 v3 — 核心组件贡献量化

消融配置（从完整模型逐步去掉组件）：
1. 完整 DSSRR（baseline）
2. 单侧参考（仅前参考）
3. 无相位模板（纯随机相位）
4. 关趋势对齐（baseline alignment off）
5. 关接缝混合（boundary blend off）
6. 线性域融合（vs log域融合）

实验设计：
- 4个事件 × 60min dropout
- 3个核心指标：Wasserstein距离、ACF L2、边界跳变
- 输出：CSV结果 + 汇总条形图（带baseline虚线）
"""
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from obspy import read
from scipy import signal
from scipy.stats import wasserstein_distance

matplotlib.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, r'G:\SeisY')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import results_io as ri  # noqa: E402
from seisy.core.anomaly_repair.reference_spectrum import ReferenceSpectrumReplacer
from seisy.core.anomaly_repair.spectral import SpectralEstimator


SR = 6.625
SEED = 42
CLEAN_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\clean_data_12h'
OUT_DIR = r'G:\SeisY\docs\dssrr_paper\experiments\results_ablation_v3'

# 4个事件，跨3个台站
EVENTS = [
    ('S12', '19760113', 'XA.S12.01.MHZ.19760113_070152-19760113_190149.mseed'),
    ('S15', '19760125', 'XA.S15.01.MHZ.19760125_160000-19760126_045957.mseed'),
    ('S16', '19760125', 'XA.S16.01.MHZ.19760125_160000-19760126_045957.mseed'),
    ('S12', '19761114', 'XA.S12.01.MHZ.19761114_230000-19761115_115957.mseed'),
]

ANOM_LEN_SEC = 3600  # 60 min dropout


# ============================================================
# 自定义谱估计器：线性域融合（消融用）
# ============================================================
class LinearFusionSpectralEstimator(SpectralEstimator):
    """线性域平均融合（替代默认的log域融合）"""

    def _fuse_spectra(self, psd_before, psd_after, ratio):
        if psd_before is None and psd_after is None:
            raise ValueError("at least one PSD is required for fusion")
        if psd_before is None:
            return np.asarray(psd_after, dtype=float)
        if psd_after is None:
            return np.asarray(psd_before, dtype=float)
        # 线性域等权平均
        return (1.0 - ratio) * psd_before + ratio * psd_after


class LinearFusionDSSRR(ReferenceSpectrumReplacer):
    """使用线性域融合的DSSRR"""

    def __init__(self, sr, config=None):
        super().__init__(sr, config)
        self.estimator = LinearFusionSpectralEstimator(sr, self.config)


# ============================================================
# 单侧参考 DSSRR
# ============================================================
class OneSidedDSSRR(ReferenceSpectrumReplacer):
    """仅使用前参考段的DSSRR"""

    def replace(self, data, start, end, **kwargs):
        kwargs['reference_after_sec'] = 0.0
        return super().replace(data, start, end, **kwargs)


# ============================================================
# 工具函数
# ============================================================
def load_clean_data(filename):
    """加载干净数据，返回 (data, sr, starttime)"""
    st = read(os.path.join(CLEAN_DIR, filename))
    tr = st[0]
    return tr.data.astype(float), tr.stats.sampling_rate, tr.stats.starttime


def inject_dropout(data, len_sec):
    """注入60min dropout，放在数据后段（平稳背景上）"""
    n = len(data)
    corrupted = data.copy()
    len_pts = int(len_sec * SR)
    center = int(n * 0.75)  # 放在后段，避开开头可能的瞬态
    start = center - len_pts // 2
    end = start + len_pts - 1
    start = max(int(1200 * SR), start)  # 至少留前20分钟做参考
    end = min(n - int(1200 * SR) - 1, end)
    corrupted[start:end + 1] = 0
    return corrupted, start, end


def compute_metrics(original, repaired, start, end, sr):
    """计算核心指标"""
    anom_orig = original[start:end + 1]
    anom_rep = repaired[start:end + 1]

    # Wasserstein 距离（幅值分布）
    wd = wasserstein_distance(np.abs(anom_orig), np.abs(anom_rep))

    # ACF L2 距离
    max_lag = int(5 * sr)
    def acf(x):
        x = x - np.mean(x)
        c = np.correlate(x, x, mode='full')
        c = c[len(c)//2:len(c)//2 + max_lag]
        return c / (c[0] + 1e-10)
    acf_o = acf(anom_orig)
    acf_r = acf(anom_rep)
    acf_l2 = np.sqrt(np.mean((acf_o - acf_r) ** 2))

    # 边界跳变（归一化）
    left_jump = abs(repaired[start] - original[start - 1])
    right_jump = abs(repaired[end] - original[end + 1])
    local_rms = np.std(original[max(0,start-100):start+100])
    boundary_jump = max(left_jump, right_jump) / (local_rms + 1e-10)

    return {
        'wasserstein': wd,
        'acf_l2': acf_l2,
        'boundary_jump': boundary_jump,
    }


def run_config(config_name, data, start, end, config_overrides=None):
    """运行一个消融配置"""
    if config_overrides is None:
        config_overrides = {}

    if config_name == 'full':
        replacer = ReferenceSpectrumReplacer(SR, config_overrides)
    elif config_name == 'one_sided':
        replacer = OneSidedDSSRR(SR, config_overrides)
    elif config_name == 'no_phase_template':
        # 纯随机相位：用 spectral_shaping 模式
        cfg = {'synthesis': {'method': 'spectral_shaping'}}
        cfg.update(config_overrides)
        replacer = ReferenceSpectrumReplacer(SR, cfg)
    elif config_name == 'no_trend_align':
        cfg = {'joining': {'baseline_alignment': 'off'}}
        cfg.update(config_overrides)
        replacer = ReferenceSpectrumReplacer(SR, cfg)
    elif config_name == 'no_seam_blend':
        cfg = {'joining': {'boundary_blend_fraction': 0.0}}
        cfg.update(config_overrides)
        replacer = ReferenceSpectrumReplacer(SR, cfg)
    elif config_name == 'linear_fusion':
        cfg = dict(config_overrides)
        replacer = LinearFusionDSSRR(SR, cfg)
    else:
        raise ValueError(f"Unknown config: {config_name}")

    repaired, report = replacer.replace(
        data, start, end,
        reference_before_sec=600,
        reference_after_sec=600,
        random_seed=SEED,
    )
    return repaired, report


# ============================================================
# 主函数
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []

    configs = [
        'full',
        'one_sided',
        'no_phase_template',
        'no_trend_align',
        'no_seam_blend',
        'linear_fusion',
    ]

    config_labels = {
        'full': 'Full DSSRR',
        'one_sided': 'One-sided ref',
        'no_phase_template': 'No phase template',
        'no_trend_align': 'No trend align',
        'no_seam_blend': 'No seam blend',
        'linear_fusion': 'Linear fusion',
    }

    # 文件名安全的配置名（用于 after_<X>.png / repaired_<X>.mseed）
    config_slugs = {
        'full': 'Full',
        'one_sided': 'OneSided',
        'no_phase_template': 'NoPhaseTemplate',
        'no_trend_align': 'NoTrendAlign',
        'no_seam_blend': 'NoSeamBlend',
        'linear_fusion': 'LinearFusion',
    }
    method_order = [config_slugs[c] for c in configs]
    slug2cfg = {v: k for k, v in config_slugs.items()}

    writer = ri.ResultsWriter(OUT_DIR, experiment='ablation')

    for station, date, filename in EVENTS:
        print(f"\nProcessing {station} {date}...")
        original, sr, starttime = load_clean_data(filename)
        corrupted, start, end = inject_dropout(original, ANOM_LEN_SEC)
        intervals = [(int(start), int(end))]

        repaired_by_method = {}
        case_metrics = {}
        for cfg_name in configs:
            print(f"  {cfg_name}...")
            try:
                repaired, _ = run_config(cfg_name, corrupted, start, end)
                slug = config_slugs[cfg_name]
                repaired_by_method[slug] = repaired
                # 论文消融图沿用的原始指标口径（|幅值| Wasserstein + 5 s ACF）
                case_metrics[slug] = compute_metrics(original, repaired,
                                                     start, end, sr)
            except Exception as e:
                print(f"    ERROR: {e}")

        # 统一结构的产物：before/after PNG + corrupted/repaired mseed + CSV
        case_rows = writer.save_case(
            event=f"{station}_{date}", station=station,
            scenario='ablation', anom_type='dropout', level='large',
            truth=original, corrupted=corrupted, intervals=intervals,
            repaired_by_method=repaired_by_method, method_order=method_order,
            starttime=starttime, sr=sr,
        )

        # 追加消融专属列（boundary_jump），并把原始口径指标另存旧格式
        for row in case_rows:
            cfg = slug2cfg[row['method']]
            m = case_metrics.get(row['method'], {})
            row['boundary_jump'] = m.get('boundary_jump', np.nan)
            row['ablation_wasserstein_abs'] = m.get('wasserstein', np.nan)
            row['ablation_acf_l2_5s'] = m.get('acf_l2', np.nan)
            all_results.append({
                'config': cfg, 'station': station, 'date': date,
                'wasserstein': m.get('wasserstein', np.nan),
                'acf_l2': m.get('acf_l2', np.nan),
                'boundary_jump': m.get('boundary_jump', np.nan),
            })

    # 保存结果（旧格式，供论文消融图脚本继续使用）
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUT_DIR, 'ablation_v3_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # 汇总统计
    print("\n" + "=" * 70)
    print("Summary (mean across 4 events)")
    print("=" * 70)
    summary = df.groupby('config')[['wasserstein', 'acf_l2', 'boundary_jump']].mean()
    print(summary.to_string())
    summary.to_csv(os.path.join(OUT_DIR, 'ablation_v3_summary.csv'))

    # 统一结构的汇总表 + 溯源
    writer.write_summary()
    writer.write_provenance(script_path=os.path.abspath(__file__))
    # 重新落一遍 detail（补上 boundary_jump 等消融专属列）
    writer.rewrite_detail()

    # ============================================================
    # 画汇总图：横向条形图，带baseline虚线
    # ============================================================
    print("\nPlotting ablation summary figure...")

    # 按贡献大小排序（去掉full本身）
    baseline_wd = summary.loc['full', 'wasserstein']
    baseline_acf = summary.loc['full', 'acf_l2']
    baseline_bj = summary.loc['full', 'boundary_jump']

    plot_configs = [c for c in configs if c != 'full']
    labels = [config_labels[c] for c in plot_configs]

    # 计算相对完整模型的退化比例
    wd_degradation = [(summary.loc[c, 'wasserstein'] - baseline_wd) / baseline_wd * 100 for c in plot_configs]
    acf_degradation = [(summary.loc[c, 'acf_l2'] - baseline_acf) / baseline_acf * 100 for c in plot_configs]
    bj_degradation = [(summary.loc[c, 'boundary_jump'] - baseline_bj) / baseline_bj * 100 for c in plot_configs]

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    metrics_data = [
        ('Wasserstein distance', wd_degradation, baseline_wd),
        ('ACF $L_2$ distance', acf_degradation, baseline_acf),
        ('Boundary jump', bj_degradation, baseline_bj),
    ]

    y_pos = np.arange(len(labels))
    colors = '#e74c3c'

    for ax, (title, degradations, baseline_val) in zip(axes, metrics_data):
        bars = ax.barh(y_pos, degradations, color=colors, alpha=0.7, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Degradation vs. full model (%)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.grid(axis='x', alpha=0.3)

        # 在柱子上标注数值
        for i, (bar, deg) in enumerate(zip(bars, degradations)):
            width = bar.get_width()
            ax.text(width + max(degradations) * 0.02 if width > 0 else width - max(degradations) * 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f'+{deg:.1f}%',
                    va='center', ha='left' if width > 0 else 'right',
                    fontsize=9)

    plt.suptitle('DSSRR Component Ablation (60-min dropout, 4 events)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, 'ablation_v3_summary.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {fig_path}")
    plt.close()

    # 同时画一个绝对指标的图（更直观）
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    all_configs_ordered = ['full'] + plot_configs
    all_labels = ['Full DSSRR'] + labels
    y_pos2 = np.arange(len(all_labels))

    for ax, (title, _, baseline_val) in zip(axes, metrics_data):
        if title == 'Wasserstein distance':
            vals = [summary.loc[c, 'wasserstein'] for c in all_configs_ordered]
        elif title == 'ACF $L_2$ distance':
            vals = [summary.loc[c, 'acf_l2'] for c in all_configs_ordered]
        else:
            vals = [summary.loc[c, 'boundary_jump'] for c in all_configs_ordered]

        bar_colors = ['#2c3e50'] + ['#e74c3c'] * len(plot_configs)
        bars = ax.barh(y_pos2, vals, color=bar_colors, alpha=0.8, height=0.6)
        ax.set_yticks(y_pos2)
        ax.set_yticklabels(all_labels, fontsize=10)
        ax.set_xlabel(title, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axvline(x=baseline_val, color='#2c3e50', linewidth=1.2, linestyle='--', alpha=0.7)
        ax.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, vals):
            width = bar.get_width()
            ax.text(width + max(vals) * 0.02,
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}',
                    va='center', fontsize=9)

    plt.suptitle('DSSRR Component Ablation — Absolute Metrics\n(60-min dropout, mean of 4 events)',
                 fontsize=12, fontweight='bold', y=1.05)
    plt.tight_layout()

    fig2_path = os.path.join(OUT_DIR, 'ablation_v3_absolute.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Absolute figure saved to {fig2_path}")
    plt.close()

    print("\nAblation v3 completed!")


if __name__ == '__main__':
    main()
