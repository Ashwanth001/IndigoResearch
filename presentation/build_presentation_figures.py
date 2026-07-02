"""
Build evaluation-results figures for the Trade Complexity 2.0 slide deck.

These are the standard results charts for slides:
  - one large bar chart PER metric (one-metric-per-slide friendly)
  - a grouped 2x2 bar panel of the four primary metrics
  - a radar / "pentagon" trade-off chart
  - a heatmap of all metrics x all methods
Both the sampled test set and the full universe are covered.

Reads:
  internal_benchmarking/full_sampled_results.csv        (sampled, 2 years)
  full_universe_eval/full_universe_combined_results.csv  (full universe, 2 years)

Writes to presentation/images/  (150 dpi PNG, white background).

Run:  python3.14 presentation/build_presentation_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import zscore

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG  = os.path.join(ROOT, 'presentation', 'images')
os.makedirs(IMG, exist_ok=True)

# Family colours (consistent everywhere)
C_CLASSICAL = '#6baed6'   # blue
C_EMBED     = '#74c476'   # green
C_XGB       = '#fd8d3c'   # orange
C_GNN       = '#9e9ac8'   # purple
C_GNN_BEST  = '#54278f'   # dark purple (best GNN)
INK         = '#222222'

SHORT = {
    'RCA Persistence': 'RCA-Persist', 'Density': 'Density', 'ECI': 'ECI',
    'ECI + Density': 'ECI+Dens', 'KNN (LLM embeddings)': 'KNN-LLM', 'XGBoost': 'XGBoost',
    'GNN-4F': 'GNN-4F', 'GNN-11F (BACI+WDI)': 'GNN-11F', 'GNN-11F+LLM': 'GNN-11F+LLM',
    'GNN-LLM v2 (GAT+Focal)': 'GNN-v2-GAT', 'GNN-LLM v2 Unopt': 'GNN-v2-Unopt',
    'GNN-LLM PCA-A (SAGE)': 'PCA-A', 'GNN-LLM PCA-B (GCN+EW)': 'PCA-B',
    'GNN-LLM PCA-C (GAT opt)': 'PCA-C', 'GNN-LLM PCA-D (GAT+EW opt)': 'PCA-D',
}
METHOD_ORDER = [
    'RCA Persistence', 'Density', 'ECI', 'ECI + Density', 'KNN (LLM embeddings)', 'XGBoost',
    'GNN-4F', 'GNN-11F (BACI+WDI)', 'GNN-11F+LLM', 'GNN-LLM v2 (GAT+Focal)', 'GNN-LLM v2 Unopt',
    'GNN-LLM PCA-A (SAGE)', 'GNN-LLM PCA-B (GCN+EW)', 'GNN-LLM PCA-C (GAT opt)',
    'GNN-LLM PCA-D (GAT+EW opt)',
]
ALL_METRICS = ['PR-AUC', 'AUROC', 'NDCG@20', 'Prec@20', 'CWR', 'Best F1', 'P@1000', 'mAP@10']
PRIMARY     = ['PR-AUC', 'NDCG@20', 'CWR', 'P@1000']

# Higher-is-better direction note per metric (all are higher-better here)
METRIC_SUBTITLE = {
    'PR-AUC':  'Overall discrimination on imbalanced data (primary metric)',
    'AUROC':   'Ranking quality (less reliable under heavy imbalance)',
    'NDCG@20': 'Per-country top-20 ranking quality',
    'Prec@20': 'Precision in each country\'s top-20 predictions',
    'CWR':     'Complexity-Weighted Recall — catching the hard, valuable transitions',
    'Best F1': 'Best F1 across thresholds (literature-comparable)',
    'P@1000':  'Precision in the global top-1000 predictions',
    'mAP@10':  'Mean average precision @10 per country (literature-comparable)',
}

def fam_colour(m):
    if m in ('RCA Persistence', 'Density', 'ECI', 'ECI + Density'):
        return C_CLASSICAL
    if m == 'KNN (LLM embeddings)':
        return C_EMBED
    if m == 'XGBoost':
        return C_XGB
    if m == 'GNN-LLM PCA-B (GCN+EW)':
        return C_GNN_BEST
    return C_GNN

LEGEND = [
    mpatches.Patch(color=C_CLASSICAL, label='Classical / network baseline'),
    mpatches.Patch(color=C_EMBED,     label='Embedding KNN'),
    mpatches.Patch(color=C_XGB,       label='XGBoost (tabular ML)'),
    mpatches.Patch(color=C_GNN,       label='GNN'),
    mpatches.Patch(color=C_GNN_BEST,  label='Best GNN (PCA-B)'),
]

def save(fig, name):
    p = os.path.join(IMG, name)
    fig.savefig(p, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', os.path.relpath(p, ROOT))

def load(path):
    df = pd.read_csv(path)
    d15 = df[df['Year'] == 2015].set_index('Method')
    d16 = df[df['Year'] == 2016].set_index('Method')
    methods = [m for m in METHOD_ORDER if m in d15.index]
    return d15, d16, methods


# ─────────────────────────────────────────────────────────────────────────────
# Individual bar chart for ONE metric (2015 vs 2016), one-metric-per-slide
# ─────────────────────────────────────────────────────────────────────────────
def fig_single_metric(d15, d16, methods, metric, tag, regime_label):
    vals15 = np.array([d15.loc[m, metric] if m in d15.index and pd.notna(d15.loc[m, metric])
                       and d15.loc[m, metric] != 'N/A' else np.nan for m in methods], dtype=float)
    vals16 = np.array([d16.loc[m, metric] if m in d16.index and pd.notna(d16.loc[m, metric])
                       and d16.loc[m, metric] != 'N/A' else np.nan for m in methods], dtype=float)
    # order by 2015 value (desc), NaN last
    order = np.argsort(-np.nan_to_num(vals15, nan=-1))
    ms  = [methods[i] for i in order]
    v15 = vals15[order]
    v16 = vals16[order]
    labels  = [SHORT.get(m, m) for m in ms]
    colours = [fam_colour(m) for m in ms]

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(ms))
    w = 0.4
    ax.bar(x - w/2, v15, w, color=colours, zorder=3, label='2015')
    ax.bar(x + w/2, v16, w, color=colours, alpha=0.55, edgecolor=colours, hatch='//',
           zorder=3, label='2016')
    for xi, a in zip(x, v15):
        if not np.isnan(a):
            ax.text(xi - w/2, a + 0.008, f'{a:.3f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel(metric, fontsize=13)
    top = np.nanmax([np.nanmax(v15), np.nanmax(v16)])
    ax.set_ylim(0, top * 1.15)
    ax.set_title(f'{metric} — {regime_label}\n{METRIC_SUBTITLE.get(metric, "")}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', ls='--', alpha=0.4, zorder=0)
    # solid=2015, hatched=2016 note + family legend
    yr_handles = [mpatches.Patch(facecolor='#888', label='2015 (solid)'),
                  mpatches.Patch(facecolor='#888', alpha=0.55, hatch='//', label='2016 (hatched)')]
    leg1 = ax.legend(handles=yr_handles, loc='upper right', fontsize=9, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=LEGEND, loc='upper right', bbox_to_anchor=(1.0, 0.82), fontsize=8.5,
              framealpha=0.9)
    save(fig, f'metric_{tag}_{metric.replace("@","").replace(" ","").replace("-","").lower()}.png')


# ─────────────────────────────────────────────────────────────────────────────
# 2x2 grouped panel of primary metrics
# ─────────────────────────────────────────────────────────────────────────────
def fig_primary_panel(d15, d16, methods, tag, regime_label):
    labels  = [SHORT.get(m, m) for m in methods]
    colours = [fam_colour(m) for m in methods]
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle(f'Primary Metrics — {regime_label}', fontsize=15, fontweight='bold', y=1.0)
    x = np.arange(len(methods)); w = 0.4
    for ax, metric in zip(axes.flat, PRIMARY):
        v15 = [float(d15.loc[m, metric]) if m in d15.index else np.nan for m in methods]
        v16 = [float(d16.loc[m, metric]) if m in d16.index else np.nan for m in methods]
        ax.bar(x - w/2, v15, w, color=colours, zorder=3)
        ax.bar(x + w/2, v16, w, color=colours, alpha=0.55, edgecolor=colours, hatch='//', zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
        ax.set_title(metric, fontweight='bold', fontsize=13)
        ax.set_ylabel(metric)
        ax.grid(axis='y', ls='--', alpha=0.4, zorder=0)
    fig.legend(handles=LEGEND + [mpatches.Patch(facecolor='#888', alpha=0.55, hatch='//',
               label='2016 (hatched) vs 2015 (solid)')],
               loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.03), frameon=False)
    fig.tight_layout()
    save(fig, f'panel_primary_metrics_{tag}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Radar / pentagon trade-off chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_radar(d15, d16, methods, tag, regime_label):
    RADAR_METRICS = ['PR-AUC', 'NDCG@20', 'CWR', 'P@1000', 'AUROC']
    RADAR_METHODS = ['XGBoost', 'RCA Persistence', 'GNN-11F+LLM',
                     'GNN-LLM PCA-B (GCN+EW)', 'Density']
    RADAR_METHODS = [m for m in RADAR_METHODS if m in d15.index]
    colour_map = {'XGBoost': C_XGB, 'RCA Persistence': C_CLASSICAL,
                  'GNN-11F+LLM': C_GNN, 'GNN-LLM PCA-B (GCN+EW)': C_GNN_BEST,
                  'Density': C_EMBED}
    davg = pd.DataFrame({metric: (d15[metric].astype(float) + d16[metric].astype(float)) / 2
                         for metric in RADAR_METRICS})
    N = len(RADAR_METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); angles += angles[:1]
    # normalise each metric across all methods for readability
    norm = {}
    for metric in RADAR_METRICS:
        col = davg[metric].reindex(methods).astype(float)
        lo, hi = col.min(), col.max()
        norm[metric] = {m: (davg.loc[m, metric] - lo) / (hi - lo + 1e-9) for m in RADAR_METHODS}
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})
    for m in RADAR_METHODS:
        vals = [norm[metric][m] for metric in RADAR_METRICS]; vals += vals[:1]
        c = colour_map.get(m, C_GNN)
        ax.plot(angles, vals, 'o-', lw=2.2, color=c, label=SHORT.get(m, m))
        ax.fill(angles, vals, alpha=0.10, color=c)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_METRICS, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title(f'Method Trade-off Profiles — {regime_label}\n'
                 '(normalised per metric; further out = better)',
                 fontsize=13.5, fontweight='bold', pad=24)
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12), fontsize=10)
    ax.grid(True, ls='--', alpha=0.5)
    save(fig, f'radar_{tag}.png')


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap: all metrics x all methods
# ─────────────────────────────────────────────────────────────────────────────
def fig_heatmap(d15, d16, methods, tag, regime_label):
    labels = [SHORT.get(m, m) for m in methods]
    def num(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return np.nan
    m15 = np.array([[num(d15.loc[m, met]) for met in ALL_METRICS] for m in methods])
    m16 = np.array([[num(d16.loc[m, met]) if m in d16.index else np.nan for met in ALL_METRICS]
                    for m in methods])
    mat = np.nanmean(np.stack([m15, m16]), axis=0)
    matz = np.apply_along_axis(lambda c: zscore(c, nan_policy='omit'), 0, mat)
    fig, ax = plt.subplots(figsize=(15, 6.5))
    im = ax.imshow(matz.T, aspect='auto', cmap='RdYlGn', vmin=-2, vmax=2)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=10)
    ax.set_yticks(range(len(ALL_METRICS)))
    ax.set_yticklabels(ALL_METRICS, fontsize=10)
    ax.set_title(f'All Methods x All Metrics — {regime_label}\n'
                 'green = best in row, red = worst (avg of 2015 & 2016)',
                 fontsize=14, fontweight='bold')
    for i in range(len(methods)):
        for j in range(len(ALL_METRICS)):
            v = mat[i, j]
            txt = f'{v:.3f}' if not np.isnan(v) else 'N/A'
            col = 'black' if (np.isnan(matz[i, j]) or abs(matz[i, j]) < 1.2) else 'white'
            ax.text(i, j, txt, ha='center', va='center', fontsize=7, color=col)
    plt.colorbar(im, ax=ax, label='Z-score (per metric)', shrink=0.8)
    fig.tight_layout()
    save(fig, f'heatmap_{tag}.png')


def build_for(csv_path, tag, regime_label):
    if not os.path.exists(csv_path):
        print('Skip (missing):', csv_path); return
    d15, d16, methods = load(csv_path)
    for metric in ALL_METRICS:
        fig_single_metric(d15, d16, methods, metric, tag, regime_label)
    fig_primary_panel(d15, d16, methods, tag, regime_label)
    fig_radar(d15, d16, methods, tag, regime_label)
    fig_heatmap(d15, d16, methods, tag, regime_label)


if __name__ == '__main__':
    build_for(os.path.join(ROOT, 'internal_benchmarking', 'full_sampled_results.csv'),
              'sampled', 'sampled test set (~14.5% positive)')
    build_for(os.path.join(ROOT, 'full_universe_eval', 'full_universe_combined_results.csv'),
              'universe', 'full universe (~1.7% positive)')
    print('\nAll evaluation figures written to presentation/images/')
