"""
Plot internal-benchmarking evaluation results and save to internal_benchmarking/plots/.

Reads:
  internal_benchmarking/full_sampled_results.csv
  internal_benchmarking/filtered_rca025_results.csv

Figures (saved as PNG, 150 dpi):
  fig1_primary_metrics.png    — 2x2 grouped bar: PR-AUC / NDCG@20 / CWR / P@1000
                                 2015 (solid) vs 2016 (faded), bars coloured by family
  fig2_heatmap_all_metrics.png — Z-score heatmap: all 8 metrics x all methods (avg 2015+2016)
  fig3_cross_year_delta.png   — Horizontal delta bars per metric (2016 - 2015)
  fig4_radar_top_methods.png  — Radar chart: top-5 representative methods on 5 key metrics
  fig5_filtered_vs_full.png   — Side-by-side PR-AUC comparison: full-sampled vs RCA>0.25 filtered
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import zscore

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})

OUT_DIR  = 'internal_benchmarking'
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Load CSVs ─────────────────────────────────────────────────────────────────
df_full = pd.read_csv(os.path.join(OUT_DIR, 'full_sampled_results.csv'))
df_filt = pd.read_csv(os.path.join(OUT_DIR, 'filtered_rca025_results.csv'))

df15 = df_full[df_full['Year'] == 2015].set_index('Method')
df16 = df_full[df_full['Year'] == 2016].set_index('Method')
df15f = df_filt[df_filt['Year'] == 2015].set_index('Method')
df16f = df_filt[df_filt['Year'] == 2016].set_index('Method')

METHOD_ORDER = [
    'RCA Persistence', 'Density', 'ECI', 'ECI + Density',
    'KNN (LLM embeddings)', 'XGBoost',
    'GNN-4F', 'GNN-11F (BACI+WDI)', 'GNN-11F+LLM',
    'GNN-LLM v2 (GAT+Focal)', 'GNN-LLM v2 Unopt',
    'GNN-LLM PCA-A (SAGE)', 'GNN-LLM PCA-B (GCN+EW)',
    'GNN-LLM PCA-C (GAT opt)', 'GNN-LLM PCA-D (GAT+EW opt)',
]
methods = [m for m in METHOD_ORDER if m in df15.index]

SHORT = {
    'RCA Persistence':              'RCA-Persist',
    'Density':                      'Density',
    'ECI':                          'ECI',
    'ECI + Density':                'ECI+Dens',
    'KNN (LLM embeddings)':         'KNN-LLM',
    'XGBoost':                      'XGBoost',
    'GNN-4F':                       'GNN-4F',
    'GNN-11F (BACI+WDI)':           'GNN-11F',
    'GNN-11F+LLM':                  'GNN-11F+LLM',
    'GNN-LLM v2 (GAT+Focal)':       'GNN-v2-GAT',
    'GNN-LLM v2 Unopt':             'GNN-v2-Unopt',
    'GNN-LLM PCA-A (SAGE)':         'PCA-A',
    'GNN-LLM PCA-B (GCN+EW)':       'PCA-B',
    'GNN-LLM PCA-C (GAT opt)':      'PCA-C',
    'GNN-LLM PCA-D (GAT+EW opt)':   'PCA-D',
}
labels = [SHORT.get(m, m) for m in methods]

COLOURS = {}
for m in methods:
    if m in ('RCA Persistence', 'Density', 'ECI', 'ECI + Density'):
        COLOURS[m] = '#6baed6'
    elif m == 'KNN (LLM embeddings)':
        COLOURS[m] = '#74c476'
    elif m == 'XGBoost':
        COLOURS[m] = '#fd8d3c'
    else:
        COLOURS[m] = '#9e9ac8'

legend_patches = [
    mpatches.Patch(color='#6baed6', label='Classical baseline'),
    mpatches.Patch(color='#74c476', label='KNN-LLM'),
    mpatches.Patch(color='#fd8d3c', label='XGBoost'),
    mpatches.Patch(color='#9e9ac8', label='GNN'),
]


# ── Figure 1: Grouped bar — 4 primary metrics, 2015 vs 2016 ──────────────────
PRIMARY = ['PR-AUC', 'NDCG@20', 'CWR', 'P@1000']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Internal Benchmarking — Primary Metrics (full sampled test set)',
             fontsize=13, fontweight='bold', y=1.01)

x = np.arange(len(methods))
w = 0.35

for ax, metric in zip(axes.flat, PRIMARY):
    v15 = [df15.loc[m, metric] if m in df15.index else np.nan for m in methods]
    v16 = [df16.loc[m, metric] if m in df16.index else np.nan for m in methods]
    ax.bar(x - w/2, v15, w, label='2015', color=[COLOURS[m] for m in methods], alpha=0.9, zorder=3)
    ax.bar(x + w/2, v16, w, label='2016', color=[COLOURS[m] for m in methods], alpha=0.5,
           edgecolor=[COLOURS[m] for m in methods], linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8.5)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylabel(metric)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.legend(fontsize=8)

fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.tight_layout()
path1 = os.path.join(PLOT_DIR, 'fig1_primary_metrics.png')
fig.savefig(path1, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {path1}')


# ── Figure 2: Heatmap — all 8 metrics x all methods (mean 2015+2016) ─────────
ALL_METRICS = ['PR-AUC', 'AUROC', 'NDCG@20', 'Prec@20', 'CWR', 'Best F1', 'P@1000', 'mAP@10']

df_avg = (df15 + df16) / 2
mat = df_avg.loc[methods, ALL_METRICS].values.astype(float)
mat_z = np.apply_along_axis(lambda col: zscore(col, nan_policy='omit'), 0, mat)

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(mat_z.T, aspect='auto', cmap='RdYlGn', vmin=-2, vmax=2)

ax.set_xticks(range(len(methods)))
ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(len(ALL_METRICS)))
ax.set_yticklabels(ALL_METRICS, fontsize=9)
ax.set_title('All Methods x All Metrics — Z-score heatmap (avg 2015+2016)\n'
             'Green = best in column, Red = worst', fontweight='bold')

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


# ── Figure 3: Cross-year delta (2016 - 2015) ──────────────────────────────────
DELTA_METRICS = ['PR-AUC', 'NDCG@20', 'CWR', 'P@1000']

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
fig.suptitle('Cross-Year Stability: Delta = 2016 - 2015\n'
             '(positive = improves in 2016, negative = degrades)',
             fontsize=12, fontweight='bold')

for ax, metric in zip(axes, DELTA_METRICS):
    deltas = [df16.loc[m, metric] - df15.loc[m, metric]
              if m in df15.index and m in df16.index else np.nan
              for m in methods]
    colours_bar = ['#2ca02c' if (d is not np.nan and d >= 0) else '#d62728' for d in deltas]
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


# ── Figure 4: Radar chart — top representative methods ────────────────────────
RADAR_METRICS = ['PR-AUC', 'NDCG@20', 'CWR', 'P@1000', 'AUROC']
RADAR_METHODS = ['RCA Persistence', 'XGBoost', 'GNN-11F+LLM',
                 'GNN-LLM PCA-B (GCN+EW)', 'Density']
RADAR_METHODS = [m for m in RADAR_METHODS if m in df_avg.index]
RADAR_LABELS  = [SHORT[m] for m in RADAR_METHODS]
RADAR_COLOURS = ['#6baed6', '#fd8d3c', '#9e9ac8', '#756bb1', '#41ab5d']

N = len(RADAR_METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

norm_vals = {}
for met in RADAR_METRICS:
    col = df_avg[met].reindex(methods).values.astype(float)
    lo, hi = np.nanmin(col), np.nanmax(col)
    norm_vals[met] = {
        m: (df_avg.loc[m, met] - lo) / (hi - lo + 1e-9) if m in df_avg.index else 0.0
        for m in RADAR_METHODS
    }

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


# ── Figure 5: Full sampled vs RCA>0.25 filtered — PR-AUC side-by-side ─────────
# Shows the effect of the RCA>0.25 filter on each method's PR-AUC in both years.
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig.suptitle('Full-Sampled vs RCA>0.25 Filtered — PR-AUC Comparison',
             fontsize=12, fontweight='bold')

for ax, (yr, df_yr, df_yf) in zip(axes, [(2015, df15, df15f), (2016, df16, df16f)]):
    mf = [m for m in methods if m in df_yf.index]
    labels_f = [SHORT.get(m, m) for m in mf]
    x_f = np.arange(len(mf))
    v_full = [df_yr.loc[m, 'PR-AUC'] if m in df_yr.index else np.nan for m in mf]
    v_filt = [df_yf.loc[m, 'PR-AUC'] if m in df_yf.index else np.nan for m in mf]
    cols = [COLOURS[m] for m in mf]
    ax.bar(x_f - w/2, v_full, w, label='Full sampled', color=cols, alpha=0.9, zorder=3)
    ax.bar(x_f + w/2, v_filt, w, label='RCA>0.25 filtered', color=cols, alpha=0.45,
           edgecolor=cols, linewidth=1.2, hatch='//', zorder=3)
    ax.set_xticks(x_f)
    ax.set_xticklabels(labels_f, rotation=40, ha='right', fontsize=8.5)
    ax.set_title(f't={yr} — PR-AUC', fontweight='bold')
    ax.set_ylabel('PR-AUC')
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.legend(fontsize=8)

fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.05), frameon=False)
fig.tight_layout()
path5 = os.path.join(PLOT_DIR, 'fig5_filtered_vs_full.png')
fig.savefig(path5, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f'Saved: {path5}')

print(f'\nAll plots saved to {PLOT_DIR}/')
