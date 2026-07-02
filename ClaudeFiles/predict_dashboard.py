"""
Predict which products USA / India / China can gain RCA >= 1 in, using the
trained XGBoost and GNN-11F+LLM models. Ranks candidates by ubiquity ascending
(rarest / most "precious" first) and renders a self-contained HTML dashboard.

Run:  python3.14 ClaudeFiles/predict_dashboard.py
Output: ClaudeFiles/rca_dashboard.html
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

warnings.filterwarnings('ignore')

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, 'data')
CKPT_DIR  = os.path.join(DATA_DIR, 'models', 'gnn', 'checkpoints')
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_CUTOFF = 2012
PCA_DIM   = 32
OBS_YEAR  = 2019            # observation snapshot: predict RCA>=1 gains ~5yr ahead
TARGETS   = {842: 'USA', 356: 'India', 156: 'China'}
TOP_N     = 40             # products shown per country in the dashboard

print(f'Device: {DEVICE}  |  observation year t={OBS_YEAR}')

# ============================================================ load raw artifacts
smooth = pd.read_csv(os.path.join(DATA_DIR, 'M_cpt_smoothed.csv'))
rca_df = pd.read_csv(os.path.join(DATA_DIR, 'rca_cpt.csv'))

countries = sorted(smooth['country'].unique())
products  = sorted(smooth['product'].unique())
C, P = len(countries), len(products)
c_idx = {c: i for i, c in enumerate(countries)}
p_idx = {p: i for i, p in enumerate(products)}
idx_to_product = {i: p for p, i in p_idx.items()}

def build_M(year):
    M = np.zeros((C, P), dtype=np.float32)
    yr = smooth[smooth['year'] == year]
    M[yr['country'].map(c_idx).values, yr['product'].map(p_idx).values] = 1.0
    return M

# proximity matrix (train years only) for density
co_exp  = np.zeros((P, P), dtype=np.float32)
any_exp = np.zeros((P, P), dtype=np.float32)
for yr in sorted(y for y in smooth['year'].unique() if y <= TRAIN_CUTOFF):
    M = build_M(yr)
    co = M.T @ M
    ex = M.sum(axis=0)
    co_exp  += co
    any_exp += ex[:, None] + ex[None, :] - co
phi = np.where(any_exp > 0, co_exp / (any_exp + 1e-9), 0.0)
np.fill_diagonal(phi, 0.0)
phi_row_sum = phi.sum(axis=1)

def build_year_structures(t):
    M_t = build_M(t)
    kc  = M_t.sum(axis=1); kp = M_t.sum(axis=0)
    kc_s = np.where(kc > 0, kc, 1.0); kp_s = np.where(kp > 0, kp, 1.0)
    kc_n, kp_n = kc.astype(float), kp.astype(float)
    for _ in range(20):
        kc_n = (1.0 / kc_s) * (M_t   @ kp_n)
        kp_n = (1.0 / kp_s) * (M_t.T @ kc_n)
    eci = (kc_n - kc_n.mean()) / (kc_n.std() + 1e-9)
    dens_mat = (M_t @ phi) / (phi_row_sum[None, :] + 1e-9)
    return M_t, dens_mat, eci

# GNN tensor artifacts
edge_idx_raw   = torch.load(os.path.join(DATA_DIR, 'edge_index_by_year.pt'), weights_only=False)
edge_idx_by_yr = {k: v.long() for k, v in edge_idx_raw.items()}
p_x_by_yr      = torch.load(os.path.join(DATA_DIR, 'product_x_by_year.pt'), weights_only=False)
c_x_11feat     = torch.load(os.path.join(DATA_DIR, 'country_x_by_year.pt'), weights_only=False)
cap_ei         = torch.load(os.path.join(DATA_DIR, 'capability_edge_index.pt'), weights_only=False).long()

with open(os.path.join(DATA_DIR, 'country_mapping.pkl'), 'rb') as f: c_map = pickle.load(f)
with open(os.path.join(DATA_DIR, 'product_mapping.pkl'), 'rb') as f: p_map = pickle.load(f)

# LLM embeddings (for capability cosine weights, though SAGE cap uses no weights)
llm_emb_t = torch.load(os.path.join(DATA_DIR, 'product_llm_embeddings.pt'),
                       weights_only=False, map_location='cpu').float()
llm_emb   = llm_emb_t.numpy()

print(f'Countries: {C}  Products: {P}  |  LLM emb: {llm_emb.shape}')

# ============================================================ product metadata
pc = pd.read_csv(os.path.join(DATA_DIR, '..', 'datasets', 'BACIDataset1995',
                              'product_codes_HS92_V202601.csv'))
pc['code_int'] = pc['code'].astype(str).str.lstrip('0').replace('', '0').astype(int)
code_to_desc = dict(zip(pc['code_int'], pc['description']))

# ubiquity reference (2010, same as CWR weighting): # countries with RCA>=1
rca_ref = rca_df[rca_df['year'] == 2010]
ubiq_series = rca_ref.groupby('product')['rca'].apply(lambda x: int((x >= 1).sum()))
ubiq_dict = ubiq_series.to_dict()
ubiq_median = int(ubiq_series.median())

# ============================================================ GNN architecture
class _HomoGNN(nn.Module):
    def __init__(self, hidden, drop=0.3):
        super().__init__()
        self.c1 = SAGEConv(hidden, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.drop = drop
    def forward(self, x, edge_index):
        x = F.dropout(self.c1(x, edge_index).relu(), p=self.drop, training=self.training)
        return self.c2(x, edge_index)

class BipartiteEncoder(nn.Module):
    def __init__(self, c_in, hidden, meta):
        super().__init__()
        self.country_lin = nn.Linear(c_in, hidden)
        self.product_lin = nn.Linear(3, hidden)
        self.gnn = to_hetero(_HomoGNN(hidden), meta)
    def forward(self, x_dict, ei_dict):
        return self.gnn({'country': self.country_lin(x_dict['country']),
                         'product': self.product_lin(x_dict['product'])}, ei_dict)

class TemporalGNN(nn.Module):
    def __init__(self, enc, hidden):
        super().__init__()
        self.enc = enc
        self.gru_c = nn.GRU(hidden, hidden)
        self.gru_p = nn.GRU(hidden, hidden)
    def forward(self, snaps):
        cs, ps = [], []
        for s in snaps:
            z = self.enc(s.x_dict, s.edge_index_dict)
            cs.append(z['country']); ps.append(z['product'])
        z_c, _ = self.gru_c(torch.stack(cs))
        z_p, _ = self.gru_p(torch.stack(ps))
        return {'country': z_c[-1], 'product': z_p[-1]}

class LinkPredictor(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
    def forward(self, zc, zp, ei):
        return self.mlp(torch.cat([zc[ei[0]], zp[ei[1]]], -1)).view(-1)

def build_snap(year, with_cap=True):
    d = HeteroData()
    d['country'].x = c_x_11feat[year]
    d['product'].x = p_x_by_yr[year]
    ei = edge_idx_by_yr[year].long()
    d['country', 'exports',     'product'].edge_index = ei
    d['product', 'rev_exports', 'country'].edge_index = ei.flip(0)
    if with_cap:
        d['product', 'capability', 'product'].edge_index = cap_ei
    return d

def to_dev(s, dev):
    s['country'].x = s['country'].x.to(dev)
    s['product'].x = s['product'].x.to(dev)
    for et in s.edge_types:
        s[et].edge_index = s[et].edge_index.to(device=dev, dtype=torch.long)
    return s

@torch.no_grad()
def gnn_full_scores(t):
    """Return [C, P] sigmoid link scores for the GNN-11F+LLM model."""
    ckpt = torch.load(os.path.join(CKPT_DIR, 'gnn_11f_llm.pt'),
                      weights_only=False, map_location=DEVICE)
    enc  = BipartiteEncoder(ckpt['c_in'], ckpt['hidden'], ckpt['meta']).to(DEVICE)
    mdl  = TemporalGNN(enc, ckpt['hidden']).to(DEVICE)
    pred = LinkPredictor(ckpt['hidden']).to(DEVICE)
    mdl.load_state_dict({k: v.to(DEVICE) for k, v in ckpt['mdl_state'].items()})
    pred.load_state_dict({k: v.to(DEVICE) for k, v in ckpt['pred_state'].items()})
    mdl.eval(); pred.eval()

    snaps = [to_dev(build_snap(y, with_cap=True), DEVICE) for y in range(t - 4, t + 1)]
    z = mdl(snaps)
    zc, zp = z['country'], z['product']

    scores = np.zeros((C, P), dtype=np.float32)
    for ci in [c_idx[c] for c in TARGETS]:
        ei = torch.tensor([[ci] * P, list(range(P))], dtype=torch.long, device=DEVICE)
        scores[ci] = torch.sigmoid(pred(zc, zp, ei)).cpu().numpy()
    return scores

# ============================================================ XGBoost inference
import xgboost as xgb
with open(os.path.join(DATA_DIR, 'models', 'xgboost', 'xgb_model.pkl'), 'rb') as f:
    _bundle = pickle.load(f)
xgb_model = _bundle['model']
xgb_pca   = _bundle['pca']

xgb_llm_pca = xgb_pca.transform(llm_emb).astype('float32')
xgb_llm_pca /= np.linalg.norm(xgb_llm_pca, axis=1, keepdims=True).clip(min=1e-8)

_c_enrich = pd.read_csv(os.path.join(DATA_DIR, 'country_features_enriched.csv'))
_p_feat   = pd.read_csv(os.path.join(DATA_DIR, 'product_features.csv'))
_CCOLS = ['log_export', 'n_products', 'avg_rca', 'max_rca',
          'gdp_pc', 'capital_formation', 'tertiary_enrollment',
          'fdi_inflows', 'manufacturing_va', 'internet_users', 'population']
_PCOLS = ['log_world_export', 'ubiquity', 'avg_rca']

def _build_lookup(feat_df, id_col, idx_map, cols, n_ids):
    arr = np.zeros((n_ids, len(cols)), dtype=np.float32)
    avail_yrs = sorted(feat_df['year'].unique())
    yr = max(y for y in avail_yrs if y <= TRAIN_CUTOFF)
    sub = feat_df[feat_df['year'] == yr].set_index(id_col)[cols]
    for eid, ei in idx_map.items():
        if eid in sub.index:
            arr[ei] = sub.loc[eid].values.astype(np.float32)
    return arr

_c_arr = _build_lookup(_c_enrich, 'country', c_idx, _CCOLS, C)
_p_arr = _build_lookup(_p_feat,   'product', p_idx, _PCOLS, P)

def build_xgb_features(ci_arr, pi_arr, t, dens_t, eci_t):
    c_batch   = _c_arr[ci_arr]
    p_batch   = _p_arr[pi_arr]
    llm_batch = xgb_llm_pca[pi_arr]
    dens_batch = dens_t[ci_arr, pi_arr].reshape(-1, 1)
    eci_batch  = eci_t[ci_arr].reshape(-1, 1)

    hist_yrs = [t - 2, t - 1, t]
    rca_sub  = rca_df[rca_df['year'].isin(hist_yrs)].copy()
    rca_sub['ci2'] = rca_sub['country'].map(c_idx)
    rca_sub['pi2'] = rca_sub['product'].map(p_idx)
    rca_sub  = rca_sub.dropna(subset=['ci2', 'pi2'])
    rca_sub['ci2'] = rca_sub['ci2'].astype(int)
    rca_sub['pi2'] = rca_sub['pi2'].astype(int)
    rca_arr  = np.zeros((C, P, 3), dtype=np.float32)
    for k, yr in enumerate(hist_yrs):
        rows = rca_sub[rca_sub['year'] == yr]
        rca_arr[rows['ci2'].values, rows['pi2'].values, k] = rows['rca'].values.astype(np.float32)
    rca_batch = rca_arr[ci_arr, pi_arr]

    return np.hstack([c_batch, p_batch, llm_batch, dens_batch, eci_batch, rca_batch]).astype(np.float32)

# ============================================================ run inference
M_t, dens_mat, eci = build_year_structures(OBS_YEAR)
print('Running GNN-11F+LLM full inference...')
gnn_scores = gnn_full_scores(OBS_YEAR)

records = []
for code, name in TARGETS.items():
    ci = c_idx[code]
    # candidates = products this country does NOT currently export with RCA>=1
    cand_pi = np.where(M_t[ci] == 0)[0]

    Xc = build_xgb_features(np.full(len(cand_pi), ci), cand_pi, OBS_YEAR, dens_mat, eci)
    xgb_p = xgb_model.inplace_predict(Xc).astype(np.float64)
    gnn_p = gnn_scores[ci, cand_pi]

    for j, pi in enumerate(cand_pi):
        prod = idx_to_product[pi]
        records.append({
            'country': name,
            'product_code': int(prod),
            'description': code_to_desc.get(int(prod), f'HS {prod}'),
            'ubiquity': ubiq_dict.get(int(prod), ubiq_median),
            'xgb': float(xgb_p[j]),
            'gnn': float(gnn_p[j]),
            'ensemble': float((xgb_p[j] + gnn_p[j]) / 2),
        })

df = pd.DataFrame(records)
df['agree'] = (df['xgb'] >= 0.5) & (df['gnn'] >= 0.5)

# for each country: keep products both models call likely (ensemble >= 0.5),
# then order by ubiquity ascending (rarest first)
out_frames = []
for name in TARGETS.values():
    sub = df[(df['country'] == name) & (df['ensemble'] >= 0.5)].copy()
    sub = sub.sort_values(['ubiquity', 'ensemble'], ascending=[True, False]).head(TOP_N)
    out_frames.append(sub)
    print(f'{name}: {len(df[df.country==name])} candidate products, '
          f'{(df[(df.country==name)]["ensemble"]>=0.5).sum()} above 0.5 ensemble')

top = pd.concat(out_frames, ignore_index=True)
top.to_csv(os.path.join(os.path.dirname(__file__), 'rca_predictions.csv'), index=False)
print('Saved rca_predictions.csv')

# ============================================================ HTML dashboard
def rows_html(name):
    sub = top[top['country'] == name]
    trs = []
    for rank, (_, r) in enumerate(sub.iterrows(), 1):
        rarity = ('🔴 Ultra-rare' if r['ubiquity'] <= 5 else
                  '🟠 Rare'       if r['ubiquity'] <= 20 else
                  '🟡 Uncommon'   if r['ubiquity'] <= 50 else
                  '🟢 Common')
        def bar(v, cls):
            return (f'<div class="bar"><div class="fill {cls}" style="width:{v*100:.0f}%"></div>'
                    f'<span>{v:.2f}</span></div>')
        trs.append(f"""<tr>
          <td class="rank">{rank}</td>
          <td class="ubiq"><b>{int(r['ubiquity'])}</b><br><span class="tag">{rarity}</span></td>
          <td class="desc"><b>{r['description']}</b><br><span class="code">HS {r['product_code']}</span></td>
          <td>{bar(r['xgb'],'xgb')}</td>
          <td>{bar(r['gnn'],'gnn')}</td>
          <td>{bar(r['ensemble'],'ens')}</td>
        </tr>""")
    return '\n'.join(trs)

tab_buttons = '\n'.join(
    f'<button class="tab {"active" if i==0 else ""}" onclick="show(\'{n}\')">{n}</button>'
    for i, n in enumerate(TARGETS.values()))
panels = '\n'.join(f"""
  <div class="panel" id="panel-{n}" style="display:{'block' if i==0 else 'none'}">
    <table>
      <thead><tr>
        <th>#</th><th>Ubiquity<br>(rarity)</th><th>Product (HS92)</th>
        <th>XGBoost</th><th>GNN+LLM</th><th>Ensemble</th>
      </tr></thead>
      <tbody>{rows_html(n)}</tbody>
    </table>
  </div>""" for i, n in enumerate(TARGETS.values()))

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>RCA Gain Predictions - USA / India / China</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 0;
         background: #0f1117; color: #e6e6e6; }}
  header {{ padding: 28px 40px 12px; border-bottom: 1px solid #262a35; }}
  h1 {{ margin: 0 0 6px; font-size: 24px; }}
  .sub {{ color: #9aa4b2; font-size: 13.5px; line-height: 1.5; max-width: 900px; }}
  .sub code {{ background:#1c2029; padding:1px 6px; border-radius:4px; color:#8ecdf7; }}
  .tabs {{ padding: 18px 40px 0; }}
  .tab {{ background: #1c2029; color: #cdd3dd; border: 1px solid #2c313d; padding: 9px 22px;
          font-size: 15px; border-radius: 8px 8px 0 0; cursor: pointer; margin-right: 6px; }}
  .tab.active {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
  .panel {{ padding: 0 40px 60px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 0; }}
  thead th {{ position: sticky; top: 0; background: #171b23; color: #9aa4b2; text-align: left;
             font-size: 12px; text-transform: uppercase; letter-spacing: .4px;
             padding: 12px 10px; border-bottom: 2px solid #2c313d; }}
  tbody td {{ padding: 11px 10px; border-bottom: 1px solid #1e222b; font-size: 14px; vertical-align: top; }}
  tbody tr:hover {{ background: #151a22; }}
  .rank {{ color: #6b7280; font-weight: 700; width: 34px; }}
  .ubiq {{ width: 120px; }}
  .tag {{ font-size: 11px; color: #9aa4b2; }}
  .desc {{ max-width: 420px; }}
  .code {{ color: #6b7280; font-size: 12px; }}
  .bar {{ position: relative; background: #1c2029; border-radius: 5px; height: 22px; width: 130px; }}
  .fill {{ height: 100%; border-radius: 5px; }}
  .fill.xgb {{ background: linear-gradient(90deg,#7c3aed,#a855f7); }}
  .fill.gnn {{ background: linear-gradient(90deg,#0891b2,#22d3ee); }}
  .fill.ens {{ background: linear-gradient(90deg,#16a34a,#4ade80); }}
  .bar span {{ position: absolute; right: 6px; top: 2px; font-size: 12px; color:#e6e6e6;
              text-shadow:0 1px 2px #000; }}
</style></head><body>
<header>
  <h1>🌐 Which products can they gain a comparative advantage in?</h1>
  <div class="sub">
    Predicted new products where <b>USA</b>, <b>India</b>, and <b>China</b> are likely to reach
    <b>RCA&nbsp;&ge;&nbsp;1</b> within ~5 years, from an observation snapshot of <b>t={OBS_YEAR}</b>.
    Scores from the trained <code>XGBoost</code> and <code>GNN-11F+LLM</code> models; shown products are
    those <b>both</b> models flag (ensemble&nbsp;&ge;&nbsp;0.50) that the country does <b>not</b> yet export with RCA&nbsp;&ge;&nbsp;1.
    Ordered by <b>ubiquity ascending</b> — the rarest products (exported competitively by the fewest countries) are the most
    precious opportunities and appear first.
  </div>
</header>
<div class="tabs">{tab_buttons}</div>
{panels}
<script>
function show(n){{
  document.querySelectorAll('.panel').forEach(p=>p.style.display='none');
  document.getElementById('panel-'+n).style.display='block';
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  event.target.classList.add('active');
}}
</script>
</body></html>"""

out_path = os.path.join(os.path.dirname(__file__), 'rca_dashboard.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'\nDashboard written: {out_path}')
