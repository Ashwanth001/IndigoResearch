"""Patch full_universe_eval.ipynb to add PCA-based GNN models (Variants A & B)."""
import json, sys

path = r'evaluation\full_universe_eval.ipynb'
nb = json.load(open(path, encoding='utf-8'))

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

# ── 1. Update fu-setup: add PCA_DIM, GCNConv import, PCA computation ──────────
idx, cell = find_cell(nb, 'fu-setup')
src = get_src(cell)

src = src.replace(
    "from torch_geometric.nn import SAGEConv, GATConv, to_hetero",
    "from sklearn.decomposition import PCA\nfrom torch_geometric.nn import SAGEConv, GATConv, GCNConv, to_hetero"
)
src = src.replace(
    "DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'\nos.makedirs(OUT_DIR, exist_ok=True)",
    "DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'\nPCA_DIM   = 32\nos.makedirs(OUT_DIR, exist_ok=True)"
)

# Add cos_weights_1d + PCA block before the final print block
old_print_block = "print('\\nAll artifacts loaded.')"
new_pca_block = """# Scalar cosine weights for GCNConv (Variant B): [144192]
cos_weights_1d = cos_weights.squeeze(1)

# ── PCA-compressed LLM features (Variants A & B) ─────────────────────────────
pca = PCA(n_components=PCA_DIM, random_state=42)
pca.fit(emb_np)
explained = pca.explained_variance_ratio_.sum()
llm_pca_np = pca.transform(emb_np).astype('float32')
llm_pca_np /= (llm_pca_np ** 2).sum(axis=1, keepdims=True) ** 0.5
llm_pca_np = llm_pca_np.clip(-1e8, 1e8)  # safety
llm_pca_t   = torch.from_numpy(llm_pca_np)

P_IN_PCA = 3 + PCA_DIM   # 35
p_x_with_pca = {yr: torch.cat([base, llm_pca_t], dim=1)
                for yr, base in p_x_by_yr.items()}

print('\\nAll artifacts loaded.')"""
src = src.replace(old_print_block, new_pca_block)

# Add PCA line to the final print statements
src = src.replace(
    "print(f'LLM embeddings: {llm_emb.shape}  |  cos weight range [{cos_weights.min():.3f}, {cos_weights.max():.3f}]')",
    "print(f'LLM embeddings: {llm_emb.shape}  |  cos weight range [{cos_weights.min():.3f}, {cos_weights.max():.3f}]')\nprint(f'PCA({PCA_DIM}d) explains {explained*100:.1f}% variance  |  P_IN_PCA={P_IN_PCA}')"
)

set_src(cell, src)
print('[1] fu-setup updated')

# ── 2. Insert PCA architecture + scorer cell after 7ad40d8f ───────────────────
idx_v2arch, _ = find_cell(nb, '7ad40d8f')

pca_arch_src = r"""# ── Variant A: SAGEConv + PCA features ─────────────────────────────────────
class _HomoGNN_SAGE(nn.Module):
    def __init__(self, hidden, drop=0.3):
        super().__init__()
        self.c1   = SAGEConv(hidden, hidden)
        self.c2   = SAGEConv(hidden, hidden)
        self.drop = drop
    def forward(self, x, edge_index):
        x = F.dropout(self.c1(x, edge_index).relu(), p=self.drop, training=self.training)
        return self.c2(x, edge_index)

class BipartiteEncoderSAGE_PCA(nn.Module):
    def __init__(self, c_in, p_in, hidden, drop, meta):
        super().__init__()
        self.country_lin = nn.Linear(c_in, hidden)
        self.product_lin = nn.Linear(p_in, hidden)
        self.gnn = to_hetero(_HomoGNN_SAGE(hidden, drop), meta)
    def forward(self, snap, _ew=None):
        x_proj = {'country': self.country_lin(snap['country'].x),
                  'product': self.product_lin(snap['product'].x)}
        return self.gnn(x_proj, snap.edge_index_dict)

class TemporalGNN_PCA(nn.Module):
    # Accepts list of (HeteroData, ew|None) tuples
    def __init__(self, enc, hidden):
        super().__init__()
        self.enc   = enc
        self.gru_c = nn.GRU(hidden, hidden)
        self.gru_p = nn.GRU(hidden, hidden)
    def forward(self, snaps):
        cs, ps = [], []
        for s, ew in snaps:
            z = self.enc(s, ew)
            cs.append(z['country']); ps.append(z['product'])
        z_c, _ = self.gru_c(torch.stack(cs))
        z_p, _ = self.gru_p(torch.stack(ps))
        return {'country': z_c[-1], 'product': z_p[-1]}

# ── Variant B: Mixed SAGEConv + GCNConv with cosine edge weights ─────────────
class MixedBipartiteEncoder(nn.Module):
    def __init__(self, c_in, p_in, hidden, drop):
        super().__init__()
        self.country_lin = nn.Linear(c_in, hidden)
        self.product_lin = nn.Linear(p_in, hidden)
        self.drop = drop
        self.sage_exp_1  = SAGEConv(hidden, hidden)
        self.sage_rexp_1 = SAGEConv(hidden, hidden)
        self.gcn_cap_1   = GCNConv(hidden, hidden, add_self_loops=False, normalize=False)
        self.sage_exp_2  = SAGEConv(hidden, hidden)
        self.sage_rexp_2 = SAGEConv(hidden, hidden)
        self.gcn_cap_2   = GCNConv(hidden, hidden, add_self_loops=False, normalize=False)

    def _layer(self, xc, xp, snap, cap_ew, sage_exp, sage_rexp, gcn_cap):
        ei_exp  = snap['country', 'exports',     'product'].edge_index
        ei_rexp = snap['product', 'rev_exports', 'country'].edge_index
        ei_cap  = snap['product', 'capability',  'product'].edge_index
        xc_new  = sage_rexp((xp, xc), ei_rexp)
        xp_new  = sage_exp((xc, xp), ei_exp) + gcn_cap(xp, ei_cap, edge_weight=cap_ew)
        return xc_new, xp_new

    def forward(self, snap, cap_ew):
        xc = self.country_lin(snap['country'].x)
        xp = self.product_lin(snap['product'].x)
        xc, xp = self._layer(xc, xp, snap, cap_ew,
                              self.sage_exp_1, self.sage_rexp_1, self.gcn_cap_1)
        xc = F.dropout(xc.relu(), p=self.drop, training=self.training)
        xp = F.dropout(xp.relu(), p=self.drop, training=self.training)
        xc, xp = self._layer(xc, xp, snap, cap_ew,
                              self.sage_exp_2, self.sage_rexp_2, self.gcn_cap_2)
        return {'country': xc, 'product': xp}

# ── Full-universe scorer for PCA-based models ─────────────────────────────────
@torch.no_grad()
def gnn_scores_universe_pca(ckpt_path, t, df_yr, variant='A'):
    # variant='A': BipartiteEncoderSAGE_PCA + TemporalGNN_PCA (no edge weights)
    # variant='B': MixedBipartiteEncoder + TemporalGNN_PCA (cosine weights via GCNConv)
    ckpt   = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    hidden = ckpt.get('hidden', 128)
    p_in   = ckpt.get('p_in', P_IN_PCA)

    cap_ei_dev = cap_ei.to(DEVICE)
    ew_dev     = cos_weights_1d.to(DEVICE)

    snaps = []
    for y in range(t - 4, t + 1):
        d = HeteroData()
        d['country'].x = c_x_11feat[y].to(DEVICE)
        d['product'].x = p_x_with_pca[y].to(DEVICE)
        ei = edge_idx_by_yr[y].long().to(DEVICE)
        d['country', 'exports',     'product'].edge_index = ei
        d['product', 'rev_exports', 'country'].edge_index = ei.flip(0)
        d['product', 'capability',  'product'].edge_index = cap_ei_dev
        snaps.append((d, ew_dev if variant == 'B' else None))

    meta = snaps[0][0].metadata()
    if variant == 'A':
        enc = BipartiteEncoderSAGE_PCA(11, p_in, hidden, drop=0.3, meta=meta).to(DEVICE)
    else:
        enc = MixedBipartiteEncoder(11, p_in, hidden, drop=0.3).to(DEVICE)

    mdl  = TemporalGNN_PCA(enc, hidden).to(DEVICE)
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


CKPT_PCA_A = os.path.join(CKPT_DIR, 'gnn_11f_llm_pca.pt')
CKPT_PCA_B = os.path.join(CKPT_DIR, 'gnn_11f_llm_pca_ew.pt')

for name, path_ck in [('PCA-A (SAGE+PCA)', CKPT_PCA_A), ('PCA-B (GCN+EW)', CKPT_PCA_B)]:
    status = 'OK' if os.path.exists(path_ck) else 'NOT FOUND — run new_gnn_training_fixed.ipynb first'
    print(f'  {name}: {status}')
print('PCA architecture + scorers ready.')
"""

pca_arch_cell = {
    "id": "pca-arch-fu",
    "cell_type": "code",
    "metadata": {},
    "source": pca_arch_src,
    "outputs": [],
    "execution_count": None,
}

nb['cells'].insert(idx_v2arch + 1, pca_arch_cell)
print(f'[2] PCA arch cell inserted at index {idx_v2arch+1}')

# ── 3. Insert PCA eval calls after the v2_unopt eval cell (43240e48) ──────────
idx_unopt, _ = find_cell(nb, '43240e48')

pca_eval_src = r"""# ── Method 11 & 12: GNN-LLM PCA-A and PCA-B ─────────────────────────────────
for yr, df_yr in [(2015, df_2015), (2016, df_2016)]:
    if os.path.exists(CKPT_PCA_A):
        print(f'  Loading GNN-LLM PCA-A (SAGEConv+PCA) for t={yr}...')
        evaluate(yr, 'GNN-LLM PCA-A (SAGE)', gnn_scores_universe_pca(CKPT_PCA_A, yr, df_yr, variant='A'))
    else:
        print(f'  GNN-LLM PCA-A NOT FOUND — run new_gnn_training_fixed.ipynb first')
        break

for yr, df_yr in [(2015, df_2015), (2016, df_2016)]:
    if os.path.exists(CKPT_PCA_B):
        print(f'  Loading GNN-LLM PCA-B (GCNConv+EW) for t={yr}...')
        evaluate(yr, 'GNN-LLM PCA-B (GCN+EW)', gnn_scores_universe_pca(CKPT_PCA_B, yr, df_yr, variant='B'))
    else:
        print(f'  GNN-LLM PCA-B NOT FOUND — run new_gnn_training_fixed.ipynb first')
        break
"""

pca_eval_cell = {
    "id": "pca-eval-fu",
    "cell_type": "code",
    "metadata": {},
    "source": pca_eval_src,
    "outputs": [],
    "execution_count": None,
}

nb['cells'].insert(idx_unopt + 1, pca_eval_cell)
print(f'[3] PCA eval cell inserted at index {idx_unopt+1}')

# ── 4. Update METHOD_ORDER in fu-table ───────────────────────────────────────
idx_table, cell_table = find_cell(nb, 'fu-table')
old_order = """METHOD_ORDER = [
    'RCA Persistence', 'Density', 'ECI', 'ECI + Density',
    'KNN (LLM embeddings)', 'GNN-4F', 'GNN-11F (BACI+WDI)',
    'GNN-11F+LLM', 'GNN-LLM v2 (GAT+Focal)', 'GNN-LLM v2 Unopt',
]"""
new_order = """METHOD_ORDER = [
    'RCA Persistence', 'Density', 'ECI', 'ECI + Density',
    'KNN (LLM embeddings)', 'GNN-4F', 'GNN-11F (BACI+WDI)',
    'GNN-11F+LLM', 'GNN-LLM v2 (GAT+Focal)', 'GNN-LLM v2 Unopt',
    'GNN-LLM PCA-A (SAGE)', 'GNN-LLM PCA-B (GCN+EW)',
]"""
new_table_src = get_src(cell_table).replace(old_order, new_order)
if new_table_src == get_src(cell_table):
    print('[4] WARNING: METHOD_ORDER not found — check fu-table cell content')
else:
    set_src(cell_table, new_table_src)
    print('[4] METHOD_ORDER updated in fu-table')

# ── 5. Verify filtered loop uses METHOD_ORDER ─────────────────────────────────
idx_filt, cell_filt = find_cell(nb, '57dd9e2b')
filt_src = get_src(cell_filt)
if 'METHOD_ORDER' in filt_src:
    print('[5] 57dd9e2b already uses METHOD_ORDER — no change needed')
else:
    print('[5] WARNING: 57dd9e2b does not use METHOD_ORDER — check manually')

json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('\nSaved full_universe_eval.ipynb')
print(f'Total cells now: {len(nb["cells"])}')
