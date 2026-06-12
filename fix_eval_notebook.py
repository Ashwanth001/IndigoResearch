"""Apply four fixes to evaluation.ipynb."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('evaluation.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
changes = []

def replace_line(cell, old, new):
    for i, line in enumerate(cell['source']):
        if old in line:
            cell['source'][i] = line.replace(old, new)
            return True
    return False

def insert_after_line(cell, marker, new_lines):
    for i, line in enumerate(cell['source']):
        if marker in line:
            for j, nl in enumerate(new_lines):
                cell['source'].insert(i + 1 + j, nl)
            return True
    return False

def insert_before_line(cell, marker, new_lines):
    for i, line in enumerate(cell['source']):
        if marker in line:
            for j, nl in enumerate(new_lines):
                cell['source'].insert(i + j, nl)
            return True
    return False

for idx, c in enumerate(cells):
    src = ''.join(c['source'])

    # Fix 1a: GNN-4F path
    if 'CKPT_4F' in src and "models', 'gnn', 'checkpoints', 'gnn_4f.pt" in src:
        ok = replace_line(c,
            "os.path.join(DATA_DIR, 'models', 'gnn', 'checkpoints', 'gnn_4f.pt')",
            "os.path.join(DATA_DIR, 'checkpoints', 'gnn_4f.pt')")
        if ok:
            changes.append(f'Cell {idx}: Fixed CKPT_4F path -> data/checkpoints/gnn_4f.pt')

    # Fix 1b: GNN-11F path
    if 'CKPT_11F' in src and "models', 'gnn', 'checkpoints', 'gnn_11f.pt" in src:
        ok = replace_line(c,
            "os.path.join(DATA_DIR, 'models', 'gnn', 'checkpoints', 'gnn_11f.pt')",
            "os.path.join(DATA_DIR, 'checkpoints', 'gnn_11f.pt')")
        if ok:
            changes.append(f'Cell {idx}: Fixed CKPT_11F path -> data/checkpoints/gnn_11f.pt')

    # Fix 2a: makedirs before torch.save in train_gnn()
    if 'def train_gnn(' in src and 'torch.save({' in src and 'makedirs(os.path.dirname' not in src:
        ok = insert_before_line(c, '        torch.save({',
            ['        os.makedirs(os.path.dirname(save_path), exist_ok=True)\n'])
        if ok:
            changes.append(f'Cell {idx}: Added makedirs before torch.save in train_gnn()')

    # Fix 2b: dropped-pairs log in load_gnn_scores()
    if 'def load_gnn_scores(' in src and 'n_mapped' not in src:
        ok = insert_after_line(c,
            "    te   = build_sample(TEST_YEAR, test_lbl, c_x)",
            [
                "    n_mapped = len(te['countries_raw'])\n",
                "    print(f'GNN: {n_mapped}/{len(test_lbl)} test pairs have valid mapping "
                "({len(test_lbl)-n_mapped} get score=0)')\n",
            ])
        if ok:
            changes.append(f'Cell {idx}: Added mapping stats log to load_gnn_scores()')

    # Fix 3a: INVALID_METRICS dict in setup cell
    if 'ALL_RESULTS = {}' in src and 'INVALID_METRICS' not in src:
        ok = insert_after_line(c, 'ALL_RESULTS = {}',
            ['INVALID_METRICS = {}  # method -> set of metrics that are not meaningful\n'])
        if ok:
            changes.append(f'Cell {idx}: Added INVALID_METRICS dict to setup cell')

    # Fix 3b: Mark ECI metrics as invalid
    if "res_eci = evaluate('ECI'" in src and 'INVALID_METRICS' not in src:
        ok = insert_after_line(c, "res_eci = evaluate('ECI', eci_scores)",
            [
                "\n",
                "INVALID_METRICS['ECI'] = {'NDCG@20', 'Prec@20'}  # ECI is per-country, no product-level ranking\n",
            ])
        if ok:
            changes.append(f'Cell {idx}: Added INVALID_METRICS for ECI after evaluate()')

print('Line-level changes:')
for ch in changes:
    print(' ', ch)

# Fix 4: Rewrite comparison cell with N/A handling
comp_idx = None
for idx, c in enumerate(cells):
    src = ''.join(c['source'])
    if 'METHOD_ORDER' in src and 'PR Curves' in src:
        comp_idx = idx
        break

print(f'\nComparison cell index: {comp_idx}')

if comp_idx is not None:
    cells[comp_idx]['source'] = r"""# ── Comparison Table + Plots ─────────────────────────────────────────────────
import matplotlib.gridspec as gridspec

METRICS = ['PR-AUC', 'NDCG@20', 'Prec@20', 'CWR', 'AUROC']
METHOD_ORDER = [
    'RCA Persistence', 'Density', 'ECI', 'ECI + Density',
    'KNN (LLM embeddings)', 'GNN-4F', 'GNN-11F (BACI+WDI)'
]
methods_run = [m for m in METHOD_ORDER if m in ALL_RESULTS]

# ── Print table (N/A for invalid metrics) ─────────────────────────────────────
print('=' * 78)
print(f'  {"Method":<28} {"PR-AUC":>7} {"NDCG@20":>8} {"Prec@20":>8} {"CWR":>7} {"AUROC":>7}')
print('-' * 78)
for m in methods_run:
    r   = ALL_RESULTS[m]
    inv = INVALID_METRICS.get(m, set())
    ndcg_s = '     N/A' if 'NDCG@20' in inv else f'{r["NDCG@20"]:>8.4f}'
    prec_s = '     N/A' if 'Prec@20'  in inv else f'{r["Prec@20"]:>8.4f}'
    gnn = ' <--' if 'GNN' in m else ''
    print(f'  {m:<28} {r["PR-AUC"]:>7.4f} {ndcg_s} {prec_s}'
          f' {r["CWR"]:>7.4f} {r["AUROC"]:>7.4f}{gnn}')
print('=' * 78)
print('N/A = ECI has no product-level ranking signal (all products in a country get the same score).')
print('CWR uses percentile-ranked scores (top 50% = predicted positive) for fair comparison.')

# ── Figure 1: PR Curves ───────────────────────────────────────────────────────
colors = ['#e41a1c', '#ff7f00', '#f0c040', '#4daf4a', '#984ea3', '#377eb8', '#a65628']
styles = ['-', '-', '--', '--', '-.', '-', '-']

fig1, ax1 = plt.subplots(figsize=(10, 7))
for i, m in enumerate(methods_run):
    if m not in PR_CURVES:
        continue
    prec, rec, pr_auc = PR_CURVES[m]
    ax1.plot(rec, prec, color=colors[i % len(colors)],
             linestyle=styles[i % len(styles)], lw=2,
             label=f'{m}  (AUC={pr_auc:.4f})')

ax1.set_xlabel('Recall', fontsize=13)
ax1.set_ylabel('Precision', fontsize=13)
ax1.set_title('Precision-Recall Curves — All Methods', fontsize=14)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('evaluation_pr_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 2: Grouped bar chart ───────────────────────────────────────────────
# NDCG@20 and Prec@20 bars are omitted (shown as N/A) for methods in INVALID_METRICS.
PLOT_METRICS = ['PR-AUC', 'NDCG@20', 'Prec@20', 'CWR']
n_methods = len(methods_run)
n_metrics = len(PLOT_METRICS)
x = np.arange(n_metrics)
width = 0.8 / n_methods

fig2, ax2 = plt.subplots(figsize=(13, 6))
for i, m in enumerate(methods_run):
    inv  = INVALID_METRICS.get(m, set())
    vals = [np.nan if met in inv else ALL_RESULTS[m][met] for met in PLOT_METRICS]
    ax2.bar(x + i * width - (n_methods - 1) * width / 2,
            vals, width * 0.9, label=m,
            color=colors[i % len(colors)], alpha=0.85)
    for j, (val, met) in enumerate(zip(vals, PLOT_METRICS)):
        if np.isnan(val):
            bx = x[j] + i * width - (n_methods - 1) * width / 2
            ax2.text(bx, 0.02, 'N/A', ha='center', va='bottom',
                     fontsize=6, color='gray', rotation=90)

ax2.set_xticks(x)
ax2.set_xticklabels(PLOT_METRICS, fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Method Comparison — All Metrics\n(CWR percentile-ranked; N/A = no product-level ranking for ECI)', fontsize=13)
ax2.set_ylim(0, 1.05)
ax2.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=2)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(0.5, color='gray', linestyle=':', lw=1, alpha=0.5)
plt.tight_layout()
plt.savefig('evaluation_bar_chart.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Figure 3: Radar chart ─────────────────────────────────────────────────────
# Methods with invalid metrics are plotted as 0 on those axes; marked (*) in legend.
RADAR_METRICS = ['PR-AUC', 'NDCG@20', 'Prec@20', 'CWR', 'AUROC']
N = len(RADAR_METRICS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig3, ax3 = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for i, m in enumerate(methods_run):
    inv  = INVALID_METRICS.get(m, set())
    vals = [0.0 if met in inv else ALL_RESULTS[m][met] for met in RADAR_METRICS]
    vals += vals[:1]
    lbl  = f'{m} (*)' if inv else m
    ax3.plot(angles, vals, color=colors[i % len(colors)],
             linestyle=styles[i % len(styles)], lw=2, label=lbl)
    ax3.fill(angles, vals, color=colors[i % len(colors)], alpha=0.07)

ax3.set_thetagrids(np.degrees(angles[:-1]), RADAR_METRICS, fontsize=11)
ax3.set_ylim(0, 1)
ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
ax3.set_title('Radar Chart — All Methods & Metrics\n(*) = NDCG@20/Prec@20 not applicable, shown as 0', fontsize=13, pad=20)
ax3.legend(loc='upper left', bbox_to_anchor=(1.15, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig('evaluation_radar.png', dpi=150, bbox_inches='tight')
plt.show()

print('Plots saved: evaluation_pr_curves.png, evaluation_bar_chart.png, evaluation_radar.png')

# Save results table with N/A for invalid metrics
rows = []
for m in methods_run:
    inv = INVALID_METRICS.get(m, set())
    row = {met: ('N/A' if met in inv else ALL_RESULTS[m][met]) for met in METRICS}
    row['Method'] = m
    rows.append(row)
df_res = pd.DataFrame(rows).set_index('Method')[METRICS]
df_res.to_csv(os.path.join(DATA_DIR, 'full_evaluation_results.csv'))
print('Results saved to data/full_evaluation_results.csv')
""".splitlines(keepends=True)
    changes.append(f'Cell {comp_idx}: Rewrote comparison cell with N/A table + chart handling')

print(f'  Cell {comp_idx}: Rewrote comparison cell with N/A handling')

with open('evaluation.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('\nNotebook saved successfully.')
