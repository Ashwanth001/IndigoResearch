"""
Plot full-universe evaluation results and save to full_universe_eval/plots/.

Figures:
  1. Grouped bar chart  — 4 primary metrics side-by-side per method (both years overlaid)
  2. Heatmap            — all 8 metrics × all methods (2015 avg shown, colour = z-score)
  3. Cross-year delta   — horizontal bar chart of 2016-2015 change per method per metric
  4. Radar / spider     — top-5 methods on 5 key metrics to show trade-off profiles
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib import gridspec

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

OUT_DIR  = 'full_universe_eval'
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUT_DIR, 'full_universe_combined_results.csv'))
df15 = df[df['Year'] == 2015].set_index('Method')
df16 = df[df['Year'] == 2016].set_index('Method')

# Canonical method order (drop any not present)
METHOD_ORDER = [
    'RCA Persistence', 'Density', 'ECI', 'ECI + Density',
    'KNN (LLM embeddings)', 'XGBoost',
    'GNN-4F', 'GNN-11F (BACI+WDI)', 'GNN-11F+LLM',
    'GNN-LLM v2 (GAT+Focal)', 'GNN-LLM v2 Unopt',
    'GNN-LLM PCA-A (SAGE)', 'GNN-LLM PCA-B (GCN+EW)',
    'GNN-LLM PCA-D (GAT+EW opt)',
]
methods = [m for m in METHOD_ORDER if m in df15.index]

# Short labels for readability
SHORT = {
    'RCA Persistence':         'RCA-Persist',
    'Density':                 'Density',
    'ECI':                     'ECI',
    'ECI + Density':           'ECI+Dens',
    'KNN (LLM embeddings)':    'KNN-LLM',
    'XGBoost':                 'XGBoost',
    'GNN-4F':                  'GNN-4F',
    'GNN-11F (BACI+WDI)':      'GNN-11F',
    'GNN-11F+LLM':             'GNN-11F+LLM',
    'GNN-LLM v2 (GAT+Focal)':  'GNN-v2-GAT',
    'GNN-LLM v2 Unopt':        'GNN-v2-Unopt',
    'GNN-LLM PCA-A (SAGE)':    'PCA-A',
    'GNN-LLM PCA-B (GCN+EW)':  'PCA-B',
    'GNN-LLM PCA-D (GAT+EW opt)': 'PCA-D',
}
labels = [SHORT.get(m, m) for m in methods]

# Colour by method family
COLOURS = {}
for m in methods:
    if m in ('RCA Persistence', 'Density', 'ECI', 'ECI + Density'):
        COLOURS[m] = '#6baed6'   # blue — classical baselines
    elif m == 'KNN (LLM embeddings)':
        COLOURS[m] = '#74c476'   # green — embedding baseline
    elif m == 'XGBoost':
        COLOURS[m] = '#fd8d3c'   # orange — tabular ML
    else:
        COLOURS[m] = '#9e9ac8'   # purple — GNN


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Grouped bar: 4 primary metrics, 2015 vs 2016
# ══════════════════════════════════════════════════════════════════════════════
PRIMARY = ['PR-AUC', 'NDCG@20', 'CWR', 'P@1000']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Full-Universe Evaluation — Primary Metrics (all methods)', fontsize=13, fontweight='bold', y=1.01)

x = np.arange(len(methods))
w = 0.35

for ax, metric in zip(axes.flat, PRIMARY):
    v15 = [df15.loc[m, metric] if m in df15.index else np.nan for m in methods]
    v16 = [df16.loc[m, metric] if m in df16.index else np.nan for m in methods]
    bars15 = ax.bar(x - w/2, v15, w, label='2015', color=[COLOURS[m] for m in methods], alpha=0.9, zorder=3)
    bars16 = ax.bar(x + w/2, v16, w, label='2016', color=[COLOURS[m] for m in methods], alpha=0.5,
                    edgecolor=[COLOURS[m] for m in methods], linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel(metric)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.legend(fontsize=8)

# Legend patches for families
legend_patches = [
    mpatches.Patch(color='#6baed6', label='Classical baseline'),
    mpatches.Patch(color='#74c476', label='KNN-LLM'),
    mpatches.Patch(color='#fd8d3c', label='XGBoost'),
    mpatches.Patch(color='#9e9ac8', label='GNN'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.tight_layout()
path1 = os.path.join(PLOT_DIR, 'fig1_primary_metrics.png')
fig.savefig(path1, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {path1}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Heatmap: all 8 metrics × all methods (mean of 2015+2016)
# ══════════════════════════════════════════════════════════════════════════════
ALL_METRICS = ['PR-AUC', 'AUROC', 'NDCG@20', 'Prec@20', 'CWR', 'Best F1', 'P@1000', 'mAP@10']

df_avg = (df15 + df16) / 2   # mean across years
mat = df_avg.loc[methods, ALL_METRICS].values.astype(float)

# Z-score per column so all metrics are on the same colour scale
from scipy.stats import zscore
mat_z = np.apply_along_axis(lambda col: zscore(col, nan_policy='omit'), 0, mat)

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(mat_z.T, aspect='auto', cmap='RdYlGn', vmin=-2, vmax=2)

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(len(ALL_METRICS)))
ax.set_yticklabels(ALL_METRICS, fontsize=9)
ax.set_title('All Methods × All Metrics — Z-score heatmap (avg 2015+2016)\n'
             'Green = best in column, Red = worst', fontweight='bold')

# Annotate with raw values
for i, m in enumerate(methods):
    for j, met in enumerate(ALL_METRICS):
        val = df_avg.loc[m, met]
        txt = f'{val:.3f}' if not np.isnan(val) else 'N/A'
        col = 'black' if abs(mat_z[i, j]) < 1.2 else 'white'
        ax.text(i, j, txt, ha='center', va='center', fontsize=7, color=col)

plt.colorbar(im, ax=ax, label='Z-score', shrink=0.8)
fig.tight_layout()
path2 = os.path.join(PLOT_DIR, 'fig2_heatmap_all_metrics.png')
fig.savefig(path2, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {path2}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Cross-year delta (2016 - 2015) per method
# ══════════════════════════════════════════════════════════════════════════════
DELTA_METRICS = ['PR-AUC', 'NDCG@20', 'CWR', 'P@1000']
n_metrics = len(DELTA_METRICS)
n_methods = len(methods)

fig, axes = plt.subplots(1, n_metrics, figsize=(16, 5), sharey=True)
fig.suptitle('Cross-Year Stability: Δ = 2016 − 2015\n(positive = improves in 2016, negative = degrades)',
             fontsize=12, fontweight='bold')

for ax, metric in zip(axes, DELTA_METRICS):
    deltas = [df16.loc[m, metric] - df15.loc[m, metric]
              if m in df15.index and m in df16.index else np.nan
              for m in methods]
    colours_bar = ['#2ca02c' if d >= 0 else '#d62728' for d in deltas]
    ax.barh(labels, deltas, color=colours_bar, alpha=0.85, zorder=3)
    ax.axvline(0, color='black', linewidth=0.8, zorder=4)
    ax.set_title(metric, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    ax.tick_params(axis='y', labelsize=8.5)

fig.tight_layout()
path3 = os.path.join(PLOT_DIR, 'fig3_cross_year_delta.png')
fig.savefig(path3, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {path3}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Radar chart: top methods on 5 key metrics
# ══════════════════════════════════════════════════════════════════════════════
RADAR_METRICS  = ['PR-AUC', 'NDCG@20', 'CWR', 'P@1000', 'AUROC']
RADAR_METHODS  = ['RCA Persistence', 'XGBoost', 'GNN-11F+LLM',
                  'GNN-LLM PCA-B (GCN+EW)', 'Density']
RADAR_LABELS   = [SHORT[m] for m in RADAR_METHODS if m in df_avg.index]
RADAR_METHODS  = [m for m in RADAR_METHODS if m in df_avg.index]
RADAR_COLOURS  = ['#6baed6', '#fd8d3c', '#9e9ac8', '#756bb1', '#41ab5d']

N = len(RADAR_METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]   # close the polygon

# Normalise each metric to [0,1] across all methods for radar readability
norm_vals = {}
for met in RADAR_METRICS:
    col = df_avg[met].reindex(methods).values.astype(float)
    lo, hi = np.nanmin(col), np.nanmax(col)
    norm_vals[met] = {m: (df_avg.loc[m, met] - lo) / (hi - lo + 1e-9)
                     if m in df_avg.index else 0.0
                     for m in RADAR_METHODS}

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})
for method, label, colour in zip(RADAR_METHODS, RADAR_LABELS, RADAR_COLOURS):
    vals = [norm_vals[met][method] for met in RADAR_METRICS]
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, label=label, color=colour)
    ax.fill(angles, vals, alpha=0.08, color=colour)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(RADAR_METRICS, fontsize=10)
ax.set_yticklabels([])
ax.set_title('Method Trade-off Profiles\n(normalised per metric, avg 2015+2016)',
             fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)

path4 = os.path.join(PLOT_DIR, 'fig4_radar_top_methods.png')
fig.savefig(path4, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {path4}')

print(f'\nAll plots saved to {PLOT_DIR}/')
