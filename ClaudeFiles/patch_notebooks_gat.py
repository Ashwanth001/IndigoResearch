"""
Patch internal_benchmarking.ipynb and full_universe_eval.ipynb to add
Variants C (GAT+PCA, Optuna) and D (GAT+PCA+EdgeWeights, Optuna).
"""
import json, sys

sys.stdout.reconfigure(encoding='utf-8')

def get_src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s

def set_src(c, text):
    c['source'] = text

def find_cell(nb, cell_id):
    for i, c in enumerate(nb['cells']):
        if c['id'] == cell_id:
            return i, c
    return None, None

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PATCH internal_benchmarking.ipynb
# ═══════════════════════════════════════════════════════════════════════════════
print('Patching internal_benchmarking.ipynb...')
nb_ib = json.load(open('internal_benchmarking.ipynb', encoding='utf-8'))

# ── 1a. Update ib-gnn-arch: add _GATBlock_PCA, BipartiteEncoderGAT_PCA,
#        TemporalGNN_GAT_PCA, gnn_scores_sampled_gat_pca, CKPT_PCA_C/D ─────────
idx_arch, cell_arch = find_cell(nb_ib, 'ib-gnn-arch')
src = get_src(cell_arch)

# Insert new arch classes + scorer before the checkpoint paths block
OLD_CKPT_BLOCK = "# ── Checkpoint paths ────────────────────────────────────────────────────────"
GAT_PCA_BLOCK = r"""# ── Variants C & D: GATConv + PCA features (Methods 13 & 14) ─────────────────
class _GATBlock_PCA(nn.Module):
    def __init__(self, hidden, heads, drop):
        super().__init__()
        self.gat1 = GATConv(hidden, hidden, heads=heads, concat=True,
                             dropout=drop, edge_dim=1, add_self_loops=False, fill_value='mean')
        self.proj = nn.Linear(hidden * heads, hidden)
        self.gat2 = GATConv(hidden, hidden, heads=1, concat=False,
                             dropout=drop, edge_dim=1, add_self_loops=False, fill_value='mean')
        self.drop = drop
    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(self.proj(self.gat1(x, edge_index, edge_attr=edge_attr).relu()),
                      p=self.drop, training=self.training)
        return self.gat2(x, edge_index, edge_attr=edge_attr)

class BipartiteEncoderGAT_PCA(nn.Module):
    def __init__(self, c_in, p_in, hidden, heads, drop, meta):
        super().__init__()
        self.country_lin = nn.Linear(c_in, hidden)
        self.product_lin = nn.Linear(p_in, hidden)
        self.gnn = to_hetero(_GATBlock_PCA(hidden, heads, drop), meta)
    def forward(self, x_dict, ei_dict, ea_dict=None):
        x_proj = {'country': self.country_lin(x_dict['country']),
                  'product': self.product_lin(x_dict['product'])}
        return self.gnn(x_proj, ei_dict, ea_dict) if ea_dict else self.gnn(x_proj, ei_dict)

class TemporalGNN_GAT_PCA(nn.Module):
    def __init__(self, enc, hidden):
        super().__init__()
        self.enc   = enc
        self.gru_c = nn.GRU(hidden, hidden)
        self.gru_p = nn.GRU(hidden, hidden)
    def forward(self, snaps):
        cs, ps = [], []
        for s, ea in snaps:
            z = self.enc(s.x_dict, s.edge_index_dict, ea)
            cs.append(z['country']); ps.append(z['product'])
        z_c, _ = self.gru_c(torch.stack(cs))
        z_p, _ = self.gru_p(torch.stack(ps))
        return {'country': z_c[-1], 'product': z_p[-1]}


# ── Score builder: Optimized GAT+PCA models (Variants C & D, Methods 13 & 14) ─
@torch.no_grad()
def gnn_scores_sampled_gat_pca(ckpt_path, t, df_lbl, variant='C'):
    ckpt   = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    hidden = ckpt.get('hidden', 128)
    heads  = ckpt.get('heads', 2)
    drop   = ckpt.get('dropout', 0.3)
    p_in   = ckpt.get('p_in', P_IN_PCA)

    use_ew = (variant == 'D')
    raw_snaps = [build_snap(y, c_x_11feat, p_x_dict=p_x_with_pca,
                             with_cap=True, use_gat_pca=use_ew)
                 for y in range(t - 4, t + 1)]
    snaps = [move_snap_to_dev(sp, DEVICE) for sp in raw_snaps]

    meta = snaps[0][0].metadata()
    enc  = BipartiteEncoderGAT_PCA(C_IN, p_in, hidden, heads, drop, meta).to(DEVICE)
    mdl  = TemporalGNN_GAT_PCA(enc, hidden).to(DEVICE)
    pred = LinkPredictor(hidden).to(DEVICE)
    mdl.load_state_dict({k: v.to(DEVICE) for k, v in ckpt['mdl_state'].items()})
    pred.load_state_dict({k: v.to(DEVICE) for k, v in ckpt['pred_state'].items()})
    mdl.eval(); pred.eval()

    ci_s = df_lbl['country'].map(c_map['to_idx'])
    pi_s = df_lbl['product'].map(p_map['to_idx'])
    ok   = ci_s.notna() & pi_s.notna()
    ei_t = torch.tensor([ci_s[ok].astype(int).values, pi_s[ok].astype(int).values],
                        dtype=torch.long).to(DEVICE)

    z   = mdl(snaps)
    raw = torch.sigmoid(pred(z['country'], z['product'], ei_t)).cpu().numpy()

    gnn_df = pd.DataFrame({'country': df_lbl.loc[ok, 'country'].values,
                           'product': df_lbl.loc[ok, 'product'].values,
                           'score':   raw})
    return df_lbl.merge(gnn_df[['country', 'product', 'score']],
                        on=['country', 'product'], how='left').fillna(0)['score'].values


"""

if OLD_CKPT_BLOCK not in src:
    print('[1a] WARNING: checkpoint block marker not found in ib-gnn-arch')
else:
    src = src.replace(OLD_CKPT_BLOCK, GAT_PCA_BLOCK + OLD_CKPT_BLOCK)
    print('[1a] Added BipartiteEncoderGAT_PCA + gnn_scores_sampled_gat_pca')

# Update build_snap to support use_gat_pca flag (insert after use_v2 block)
OLD_BUILD_SNAP = "        else:\n            ea = cos_weights_1d   # scalar weights for GCNConv (Method 12)"
NEW_BUILD_SNAP = """        elif use_gat_pca:
            # GATConv PCA needs edge_attr shape [E, 1]
            d['product', 'capability', 'product'].edge_attr = cos_weights.unsqueeze(1)
            ea = {
                ('country', 'exports',     'product'): None,
                ('product', 'rev_exports', 'country'): None,
                ('product', 'capability',  'product'): cos_weights.unsqueeze(1),
            }
        else:
            ea = cos_weights_1d   # scalar weights for GCNConv (Method 12)"""

if OLD_BUILD_SNAP not in src:
    print('[1a-snap] WARNING: build_snap else-clause not found for use_gat_pca insertion')
else:
    src = src.replace(OLD_BUILD_SNAP, NEW_BUILD_SNAP)
    print('[1a-snap] Updated build_snap with use_gat_pca flag')

# Update build_snap signature
OLD_SIG = "def build_snap(year, c_x, p_x_dict=None, with_cap=False, use_v2=False):"
NEW_SIG  = "def build_snap(year, c_x, p_x_dict=None, with_cap=False, use_v2=False, use_gat_pca=False):"
if OLD_SIG in src:
    src = src.replace(OLD_SIG, NEW_SIG)
    print('[1a-sig] Updated build_snap signature')
else:
    print('[1a-sig] WARNING: build_snap signature not found')

# Add CKPT_PCA_C and CKPT_PCA_D checkpoint paths
OLD_CKPT_END = "CKPT_PCA_B   = os.path.join(CKPT_DIR, 'gnn_11f_llm_pca_ew.pt')"
NEW_CKPT_END = """CKPT_PCA_B   = os.path.join(CKPT_DIR, 'gnn_11f_llm_pca_ew.pt')
CKPT_PCA_C   = os.path.join(CKPT_DIR, 'gnn_11f_llm_pca_gat.pt')
CKPT_PCA_D   = os.path.join(CKPT_DIR, 'gnn_11f_llm_pca_gat_ew.pt')"""
if OLD_CKPT_END in src:
    src = src.replace(OLD_CKPT_END, NEW_CKPT_END)
    print('[1a-ckpt] Added CKPT_PCA_C and CKPT_PCA_D')
else:
    print('[1a-ckpt] WARNING: CKPT_PCA_B line not found')

# Update the status print at the bottom
OLD_PRINT = "                   ('PCA-A (SAGE+PCA)', CKPT_PCA_A), ('PCA-B (GCN+EW)', CKPT_PCA_B)]:"
NEW_PRINT = """                   ('PCA-A (SAGE)', CKPT_PCA_A), ('PCA-B (GCN+EW)', CKPT_PCA_B),
                   ('PCA-C (GAT opt)', CKPT_PCA_C), ('PCA-D (GAT+EW opt)', CKPT_PCA_D)]:"""
if OLD_PRINT in src:
    src = src.replace(OLD_PRINT, NEW_PRINT)
    print('[1a-print] Updated checkpoint status print')
else:
    print('[1a-print] WARNING: status print line not found')

set_src(cell_arch, src)

# ── 1b. Update ib-2015-gnns: add Methods 13 & 14 ────────────────────────────
idx_2015, cell_2015 = find_cell(nb_ib, 'ib-2015-gnns')
src_2015 = get_src(cell_2015)

OLD_END_2015 = """# ── Method 12: GNN-11F + LLM-PCA + EdgeWeights (GCNConv with cosine weights) ─
if os.path.exists(CKPT_PCA_B):
    print(f'Loading GNN-LLM PCA-B (GCNConv+EW)...')
    evaluate(T, 'GNN-LLM PCA-B (GCN+EW)', gnn_scores_sampled_pca(CKPT_PCA_B, T, df_t, variant='B'))
else:
    print(f'GNN-LLM PCA-B checkpoint NOT FOUND ({CKPT_PCA_B}) — run new_gnn_training_fixed.ipynb first')"""

NEW_END_2015 = """# ── Method 12: GNN-11F + LLM-PCA + EdgeWeights (GCNConv with cosine weights) ─
if os.path.exists(CKPT_PCA_B):
    print(f'Loading GNN-LLM PCA-B (GCNConv+EW)...')
    evaluate(T, 'GNN-LLM PCA-B (GCN+EW)', gnn_scores_sampled_pca(CKPT_PCA_B, T, df_t, variant='B'))
else:
    print(f'GNN-LLM PCA-B checkpoint NOT FOUND ({CKPT_PCA_B}) — run new_gnn_training_fixed.ipynb first')

# ── Method 13: GNN-LLM PCA-C (GATConv + Focal + Optuna, no edge weights) ─────
if os.path.exists(CKPT_PCA_C):
    print(f'Loading GNN-LLM PCA-C (GAT+PCA, Optuna)...')
    evaluate(T, 'GNN-LLM PCA-C (GAT opt)', gnn_scores_sampled_gat_pca(CKPT_PCA_C, T, df_t, variant='C'))
else:
    print(f'GNN-LLM PCA-C checkpoint NOT FOUND ({CKPT_PCA_C}) — run new_gnn_training_fixed.ipynb first')

# ── Method 14: GNN-LLM PCA-D (GATConv + Focal + Optuna + cosine edge_attr) ───
if os.path.exists(CKPT_PCA_D):
    print(f'Loading GNN-LLM PCA-D (GAT+EW, Optuna)...')
    evaluate(T, 'GNN-LLM PCA-D (GAT+EW opt)', gnn_scores_sampled_gat_pca(CKPT_PCA_D, T, df_t, variant='D'))
else:
    print(f'GNN-LLM PCA-D checkpoint NOT FOUND ({CKPT_PCA_D}) — run new_gnn_training_fixed.ipynb first')"""

if OLD_END_2015 in src_2015:
    set_src(cell_2015, src_2015.replace(OLD_END_2015, NEW_END_2015))
    print('[1b] Methods 13 & 14 added to ib-2015-gnns')
else:
    print('[1b] WARNING: ib-2015-gnns Method 12 block not found')

# ── 1c. Update ib-2016-gnns: add Methods 13 & 14 ────────────────────────────
idx_2016, cell_2016 = find_cell(nb_ib, 'ib-2016-gnns')
src_2016 = get_src(cell_2016)

OLD_END_2016 = """# ── Method 12: GNN-11F + LLM-PCA + EdgeWeights (GCNConv with cosine weights) ─
if os.path.exists(CKPT_PCA_B):
    print(f'Loading GNN-LLM PCA-B for t={T}...')
    evaluate(T, 'GNN-LLM PCA-B (GCN+EW)', gnn_scores_sampled_pca(CKPT_PCA_B, T, df_t, variant='B'))
else:
    print(f'GNN-LLM PCA-B checkpoint NOT FOUND ({CKPT_PCA_B}) — run new_gnn_training_fixed.ipynb first')"""

NEW_END_2016 = """# ── Method 12: GNN-11F + LLM-PCA + EdgeWeights (GCNConv with cosine weights) ─
if os.path.exists(CKPT_PCA_B):
    print(f'Loading GNN-LLM PCA-B for t={T}...')
    evaluate(T, 'GNN-LLM PCA-B (GCN+EW)', gnn_scores_sampled_pca(CKPT_PCA_B, T, df_t, variant='B'))
else:
    print(f'GNN-LLM PCA-B checkpoint NOT FOUND ({CKPT_PCA_B}) — run new_gnn_training_fixed.ipynb first')

# ── Method 13: GNN-LLM PCA-C (GATConv + Focal + Optuna, no edge weights) ─────
if os.path.exists(CKPT_PCA_C):
    print(f'Loading GNN-LLM PCA-C (GAT+PCA, Optuna) for t={T}...')
    evaluate(T, 'GNN-LLM PCA-C (GAT opt)', gnn_scores_sampled_gat_pca(CKPT_PCA_C, T, df_t, variant='C'))
else:
    print(f'GNN-LLM PCA-C checkpoint NOT FOUND ({CKPT_PCA_C}) — run new_gnn_training_fixed.ipynb first')

# ── Method 14: GNN-LLM PCA-D (GATConv + Focal + Optuna + cosine edge_attr) ───
if os.path.exists(CKPT_PCA_D):
    print(f'Loading GNN-LLM PCA-D (GAT+EW, Optuna) for t={T}...')
    evaluate(T, 'GNN-LLM PCA-D (GAT+EW opt)', gnn_scores_sampled_gat_pca(CKPT_PCA_D, T, df_t, variant='D'))
else:
    print(f'GNN-LLM PCA-D checkpoint NOT FOUND ({CKPT_PCA_D}) — run new_gnn_training_fixed.ipynb first')"""

if OLD_END_2016 in src_2016:
    set_src(cell_2016, src_2016.replace(OLD_END_2016, NEW_END_2016))
    print('[1c] Methods 13 & 14 added to ib-2016-gnns')
else:
    print('[1c] WARNING: ib-2016-gnns Method 12 block not found')

# ── 1d. Update ib-tables METHOD_ORDER ────────────────────────────────────────
idx_tbl, cell_tbl = find_cell(nb_ib, 'ib-tables')
src_tbl = get_src(cell_tbl)
OLD_ORDER = "    'GNN-LLM PCA-A (SAGE)', 'GNN-LLM PCA-B (GCN+EW)',\n]"
NEW_ORDER = "    'GNN-LLM PCA-A (SAGE)', 'GNN-LLM PCA-B (GCN+EW)',\n    'GNN-LLM PCA-C (GAT opt)', 'GNN-LLM PCA-D (GAT+EW opt)',\n]"
if OLD_ORDER in src_tbl:
    set_src(cell_tbl, src_tbl.replace(OLD_ORDER, NEW_ORDER))
    print('[1d] METHOD_ORDER updated in ib-tables')
else:
    print('[1d] WARNING: METHOD_ORDER tail not found in ib-tables')

json.dump(nb_ib, open('internal_benchmarking.ipynb', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'\nSaved internal_benchmarking.ipynb  ({len(nb_ib["cells"])} cells)')

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PATCH full_universe_eval.ipynb
# ═══════════════════════════════════════════════════════════════════════════════
print('\nPatching full_universe_eval.ipynb...')
nb_fu = json.load(open('full_universe_eval.ipynb', encoding='utf-8'))

# ── 2a. Add GAT-PCA arch + scorer cell after pca-arch-fu ────────────────────
idx_pca_arch, _ = find_cell(nb_fu, 'pca-arch-fu')

gat_pca_arch_src = r"""# ── Variants C & D: GATConv + PCA features (Optuna-optimized) ───────────────
class _GATBlock_PCA(nn.Module):
    def __init__(self, hidden, heads, drop):
        super().__init__()
        self.gat1 = GATConv(hidden, hidden, heads=heads, concat=True,
                             dropout=drop, edge_dim=1, add_self_loops=False, fill_value='mean')
        self.proj = nn.Linear(hidden * heads, hidden)
        self.gat2 = GATConv(hidden, hidden, heads=1, concat=False,
                             dropout=drop, edge_dim=1, add_self_loops=False, fill_value='mean')
        self.drop = drop
    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(self.proj(self.gat1(x, edge_index, edge_attr=edge_attr).relu()),
                      p=self.drop, training=self.training)
        return self.gat2(x, edge_index, edge_attr=edge_attr)

class BipartiteEncoderGAT_PCA(nn.Module):
    def __init__(self, c_in, p_in, hidden, heads, drop, meta):
        super().__init__()
        self.country_lin = nn.Linear(c_in, hidden)
        self.product_lin = nn.Linear(p_in, hidden)
        self.gnn = to_hetero(_GATBlock_PCA(hidden, heads, drop), meta)
    def forward(self, x_dict, ei_dict, ea_dict=None):
        x_proj = {'country': self.country_lin(x_dict['country']),
                  'product': self.product_lin(x_dict['product'])}
        return self.gnn(x_proj, ei_dict, ea_dict) if ea_dict else self.gnn(x_proj, ei_dict)

class TemporalGNN_GAT_PCA(nn.Module):
    def __init__(self, enc, hidden):
        super().__init__()
        self.enc   = enc
        self.gru_c = nn.GRU(hidden, hidden)
        self.gru_p = nn.GRU(hidden, hidden)
    def forward(self, snaps):
        cs, ps = [], []
        for s, ea in snaps:
            z = self.enc(s.x_dict, s.edge_index_dict, ea)
            cs.append(z['country']); ps.append(z['product'])
        z_c, _ = self.gru_c(torch.stack(cs))
        z_p, _ = self.gru_p(torch.stack(ps))
        return {'country': z_c[-1], 'product': z_p[-1]}


@torch.no_grad()
def gnn_scores_universe_gat_pca(ckpt_path, t, df_yr, variant='C'):
    ckpt   = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    hidden = ckpt.get('hidden', 128)
    heads  = ckpt.get('heads', 2)
    drop   = ckpt.get('dropout', 0.3)
    p_in   = ckpt.get('p_in', P_IN_PCA)

    cap_ei_dev = cap_ei.to(DEVICE)
    ew_dev     = cos_weights_1d.unsqueeze(1).to(DEVICE)   # [E, 1] for GATConv

    snaps = []
    for y in range(t - 4, t + 1):
        d = HeteroData()
        d['country'].x = c_x_11feat[y].to(DEVICE)
        d['product'].x = p_x_with_pca[y].to(DEVICE)
        ei = edge_idx_by_yr[y].long().to(DEVICE)
        d['country', 'exports',     'product'].edge_index = ei
        d['product', 'rev_exports', 'country'].edge_index = ei.flip(0)
        d['product', 'capability',  'product'].edge_index = cap_ei_dev
        if variant == 'D':
            d['product', 'capability', 'product'].edge_attr = ew_dev
            ea = {
                ('country', 'exports',     'product'): None,
                ('product', 'rev_exports', 'country'): None,
                ('product', 'capability',  'product'): ew_dev,
            }
        else:
            ea = None
        snaps.append((d, ea))

    meta = snaps[0][0].metadata()
    enc  = BipartiteEncoderGAT_PCA(11, p_in, hidden, heads, drop, meta).to(DEVICE)
    mdl  = TemporalGNN_GAT_PCA(enc, hidden).to(DEVICE)
    pred = LinkPredictor(hidden).to(DEVICE)
    mdl.load_state_dict({k: v.to(DEVICE) for k, v in ckpt['mdl_state'].items()})
    pred.load_state_dict({k: v.to(DEVICE) for k, v in ckpt['pred_state'].items()})
    mdl.eval(); pred.eval()

    ci_gnn = df_yr['country'].map(c_map['to_idx'])
    pi_gnn = df_yr['product'].map(p_map['to_idx'])
    ok     = ci_gnn.notna() & pi_gnn.notna()
    ci_v   = ci_gnn[ok].astype(int).values
    pi_v   = pi_gnn[ok].astype(int).values
    ei_t   = torch.tensor([ci_v, pi_v], dtype=torch.long).to(DEVICE)

    z   = mdl(snaps)
    raw = torch.sigmoid(pred(z['country'], z['product'], ei_t)).cpu().numpy()

    score_series = pd.Series(0.0, index=df_yr.index)
    score_series[ok[ok].index] = raw
    n_unmapped = (~ok).sum()
    if n_unmapped > 0:
        print(f'  Warning: {n_unmapped} pairs had no GNN mapping (scored 0)')
    return score_series.values


CKPT_PCA_C = os.path.join(CKPT_DIR, 'gnn_11f_llm_pca_gat.pt')
CKPT_PCA_D = os.path.join(CKPT_DIR, 'gnn_11f_llm_pca_gat_ew.pt')

for name, path_ck in [('PCA-C (GAT opt)', CKPT_PCA_C), ('PCA-D (GAT+EW opt)', CKPT_PCA_D)]:
    status = 'OK' if os.path.exists(path_ck) else 'NOT FOUND — run new_gnn_training_fixed.ipynb first'
    print(f'  {name}: {status}')
print('GAT-PCA architecture + scorers ready.')
"""

gat_pca_arch_cell = {
    "id":             "gat-pca-arch-fu",
    "cell_type":      "code",
    "metadata":       {},
    "source":         gat_pca_arch_src,
    "outputs":        [],
    "execution_count": None,
}
nb_fu['cells'].insert(idx_pca_arch + 1, gat_pca_arch_cell)
print(f'[2a] GAT-PCA arch cell inserted after pca-arch-fu (index {idx_pca_arch+1})')

# ── 2b. Add GAT-PCA eval calls after pca-eval-fu ────────────────────────────
idx_pca_eval, _ = find_cell(nb_fu, 'pca-eval-fu')

gat_pca_eval_src = r"""# ── Methods 13 & 14: GNN-LLM PCA-C and PCA-D (GAT + Focal + Optuna) ─────────
for yr, df_yr in [(2015, df_2015), (2016, df_2016)]:
    if os.path.exists(CKPT_PCA_C):
        print(f'  Loading GNN-LLM PCA-C (GAT+PCA, Optuna) for t={yr}...')
        evaluate(yr, 'GNN-LLM PCA-C (GAT opt)', gnn_scores_universe_gat_pca(CKPT_PCA_C, yr, df_yr, variant='C'))
    else:
        print(f'  GNN-LLM PCA-C NOT FOUND — run new_gnn_training_fixed.ipynb first')
        break

for yr, df_yr in [(2015, df_2015), (2016, df_2016)]:
    if os.path.exists(CKPT_PCA_D):
        print(f'  Loading GNN-LLM PCA-D (GAT+EW, Optuna) for t={yr}...')
        evaluate(yr, 'GNN-LLM PCA-D (GAT+EW opt)', gnn_scores_universe_gat_pca(CKPT_PCA_D, yr, df_yr, variant='D'))
    else:
        print(f'  GNN-LLM PCA-D NOT FOUND — run new_gnn_training_fixed.ipynb first')
        break
"""

gat_pca_eval_cell = {
    "id":             "gat-pca-eval-fu",
    "cell_type":      "code",
    "metadata":       {},
    "source":         gat_pca_eval_src,
    "outputs":        [],
    "execution_count": None,
}
nb_fu['cells'].insert(idx_pca_eval + 1, gat_pca_eval_cell)
print(f'[2b] GAT-PCA eval cell inserted after pca-eval-fu (index {idx_pca_eval+1})')

# ── 2c. Update METHOD_ORDER in fu-table ──────────────────────────────────────
idx_tbl, cell_tbl = find_cell(nb_fu, 'fu-table')
src_tbl = get_src(cell_tbl)
OLD_M = "    'GNN-LLM PCA-A (SAGE)', 'GNN-LLM PCA-B (GCN+EW)',\n]"
NEW_M  = "    'GNN-LLM PCA-A (SAGE)', 'GNN-LLM PCA-B (GCN+EW)',\n    'GNN-LLM PCA-C (GAT opt)', 'GNN-LLM PCA-D (GAT+EW opt)',\n]"
if OLD_M in src_tbl:
    set_src(cell_tbl, src_tbl.replace(OLD_M, NEW_M))
    print('[2c] METHOD_ORDER updated in fu-table')
else:
    print('[2c] WARNING: METHOD_ORDER tail not found in fu-table')

json.dump(nb_fu, open('full_universe_eval.ipynb', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'\nSaved full_universe_eval.ipynb  ({len(nb_fu["cells"])} cells)')
print('\nDone.')
