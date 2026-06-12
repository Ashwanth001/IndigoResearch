# === CELL 2 ===
import os, glob, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')

# -- Paths ----------------------------------------------------------------------
BACI_DIR         = os.path.join('datasets', 'BACIDataset1995')
WDI_CSV          = os.path.join('datasets', 'WDI_csv', 'WDICSV.csv')
COUNTRY_CODES    = os.path.join(BACI_DIR, 'country_codes_V202601.csv')
PRODUCT_CODES    = os.path.join(BACI_DIR, 'product_codes_HS92_V202601.csv')
DATA_DIR         = 'data'

# Train cutoff year: normalization stats and temporal split are based on this
TRAIN_CUTOFF     = 2012
VAL_YEAR         = 2013
TEST_YEAR        = 2015
LABEL_HORIZON    = 5      # predict RCA transition h years ahead
NEG_RATIO        = 5      # negatives sampled per positive

os.makedirs(DATA_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
print(f'BACI files found: {len(glob.glob(os.path.join(BACI_DIR, "BACI_HS92_Y*_V202601.csv")))}')
print(f'WDI file exists:  {os.path.exists(WDI_CSV)}')
print(f'Output dir:       {os.path.abspath(DATA_DIR)}')
# === CELL 22 ===
# Load all pre-built tensors
edge_idx_by_yr = torch.load(os.path.join(DATA_DIR, 'edge_index_by_year.pt'), weights_only=False)
c_x_by_yr      = torch.load(os.path.join(DATA_DIR, 'country_x_by_year.pt'),  weights_only=False)
p_x_by_yr      = torch.load(os.path.join(DATA_DIR, 'product_x_by_year.pt'),  weights_only=False)

with open(os.path.join(DATA_DIR, 'country_mapping.pkl'), 'rb') as f:
    c_map = pickle.load(f)
with open(os.path.join(DATA_DIR, 'product_mapping.pkl'), 'rb') as f:
    p_map = pickle.load(f)

train_labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
val_labels   = pd.read_csv(os.path.join(DATA_DIR, 'val_labels.csv'))
test_labels  = pd.read_csv(os.path.join(DATA_DIR, 'test_labels.csv'))

def build_snapshot(year):
    """One HeteroData object = one year's graph."""
    data = HeteroData()
    data['country'].x = c_x_by_yr[year]
    data['product'].x = p_x_by_yr[year]
    data['country', 'exports', 'product'].edge_index = edge_idx_by_yr[year]
    return data

def build_temporal_sample(obs_year, labels_df):
    """5-year window ending at obs_year + labeled edges predicting obs_year+5."""
    snapshots = [build_snapshot(y) for y in range(obs_year - 4, obs_year + 1)]
    ldf = labels_df[labels_df['year'] == obs_year].copy()
    ci  = ldf['country'].map(c_map['to_idx']).values
    pi  = ldf['product'].map(p_map['to_idx']).values
    lv  = ldf['label'].values
    # Drop any rows where mapping returned NaN (safety check)
    valid = ~(np.isnan(ci.astype(float)) | np.isnan(pi.astype(float)))
    ci, pi, lv = ci[valid].astype(int), pi[valid].astype(int), lv[valid]
    return {
        'snapshots': snapshots,
        'labels': {
            'edge_label_index': torch.tensor([ci, pi], dtype=torch.long),
            'edge_label':       torch.tensor(lv,       dtype=torch.float32)
        },
        'year': int(obs_year)
    }

# Build training samples (one per observation year)
print('Building training samples...')
train_samples = [build_temporal_sample(y, train_labels)
                 for y in sorted(train_labels['year'].unique())]

print(f'Building validation sample (year {VAL_YEAR})...')
val_sample  = build_temporal_sample(VAL_YEAR,  val_labels)

print(f'Building test sample (year {TEST_YEAR})...')
test_sample = build_temporal_sample(TEST_YEAR, test_labels)

torch.save(train_samples, os.path.join(DATA_DIR, 'train_data.pt'))
torch.save(val_sample,    os.path.join(DATA_DIR, 'val_data.pt'))
torch.save(test_sample,   os.path.join(DATA_DIR, 'test_data.pt'))

# -- Final Validation ------------------------------------------------------------
print(f'\n=== STEP 10 — FINAL VALIDATION ===')
print(f'Training samples:  {len(train_samples)} (one per observation year)')
print(f'Val year:          {val_sample["year"]}')
print(f'Test year:         {test_sample["year"]}')

# Inspect one training sample in detail
s = train_samples[-1]  # most recent training year
print(f'\nInspecting sample (year {s["year"]}):')
print(f'  Number of snapshots:       {len(s["snapshots"])} (years {s["year"]-4}–{s["year"]})')
snap = s['snapshots'][0]
print(f'  Snapshot 0 — country.x:   {snap["country"].x.shape}')
print(f'  Snapshot 0 — product.x:   {snap["product"].x.shape}')
print(f'  Snapshot 0 — edge_index:  {snap["country", "exports", "product"].edge_index.shape}')
print(f'  Labeled edges:            {s["labels"]["edge_label"].shape[0]}')
print(f'  Positive labels:          {s["labels"]["edge_label"].sum().int().item()}')
print(f'  Negative labels:          {(s["labels"]["edge_label"] == 0).sum().item()}')

# Sanity: labeled edge indices in valid range
eli = s['labels']['edge_label_index']
n_c = snap['country'].x.shape[0]
n_p = snap['product'].x.shape[0]
assert eli[0].max().item() < n_c, 'Labeled country index out of range!'
assert eli[1].max().item() < n_p, 'Labeled product index out of range!'
print('OK Labeled edge indices in valid range')

# Pipeline summary visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1: snapshot sizes over years
snap_sizes = [(s['year'], s['snapshots'][-1]['country', 'exports', 'product'].edge_index.shape[1])
              for s in train_samples]
sy, se = zip(*snap_sizes)
axes[0].bar(sy, se, color='steelblue', alpha=0.8)
axes[0].set_title('Trade Edges in Most-Recent Snapshot\n(per training sample)')
axes[0].set_xlabel('Observation year t')
axes[0].set_ylabel('Edges in snapshot t')

# 2: labeled edges per training year
label_counts = [(s['year'], s['labels']['edge_label'].shape[0]) for s in train_samples]
ly, lc = zip(*label_counts)
axes[1].bar(ly, lc, color='darkorange', alpha=0.8)
axes[1].set_title('Labeled Edges per Training Year')
axes[1].set_xlabel('Observation year t')
axes[1].set_ylabel('Total labeled (pos + neg) edges')

# 3: positive rate per training year
pos_rates = [(s['year'], s['labels']['edge_label'].mean().item() * 100) for s in train_samples]
ry, rr = zip(*pos_rates)
axes[2].plot(ry, rr, 'o-', markersize=6, color='coral')
axes[2].axhline(100 / (NEG_RATIO + 1), color='gray', linestyle='--',
                label=f'Expected {100/(NEG_RATIO+1):.1f}% (ratio {NEG_RATIO}:1)')
axes[2].set_title('Positive Rate per Training Sample')
axes[2].set_xlabel('Observation year t')
axes[2].set_ylabel('% positive labels')
axes[2].legend()
plt.suptitle('Final Data Pipeline Summary', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print(f'\n===== PIPELINE COMPLETE =====')
print(f'All outputs saved to: {os.path.abspath(DATA_DIR)}/')
for f in ['exports_cpt.csv', 'rca_cpt.csv', 'M_cpt_smoothed.csv', 'labels_h5.csv',
           'country_features.csv', 'product_features.csv', 'country_features_enriched.csv',
           'wdi_features.csv', 'edge_index_by_year.pt', 'country_x_by_year.pt',
           'product_x_by_year.pt', 'train_labels.csv', 'val_labels.csv', 'test_labels.csv',
           'country_mapping.pkl', 'product_mapping.pkl',
           'train_data.pt', 'val_data.pt', 'test_data.pt']:
    exists = os.path.exists(os.path.join(DATA_DIR, f))
    status = 'OK' if exists else '✗ MISSING'
    print(f'  {status}  {f}')
# === CELL 27 ===
# ====================================================================
#  STEP 12 — Train & Evaluate GNN: BACI-only (4F) vs BACI+WDI (11F)
# ====================================================================

import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, ndcg_score
from torch_geometric.nn import SAGEConv, to_hetero
from copy import deepcopy
import warnings; warnings.filterwarnings('ignore')
torch.manual_seed(42); np.random.seed(42)

HIDDEN   = 128
EPOCHS   = 80
LR       = 1e-3
WD       = 1e-5
PATIENCE = 15

# -- A: Load pipeline artifacts ------------------------------------
edge_idx_by_yr_raw = torch.load(os.path.join(DATA_DIR, 'edge_index_by_year.pt'), weights_only=False)
edge_idx_by_yr     = {k: v.long() for k, v in edge_idx_by_yr_raw.items()}  # cast to torch.long for PyG
p_x_by_yr      = torch.load(os.path.join(DATA_DIR, 'product_x_by_year.pt'),  weights_only=False)
c_x_11feat     = torch.load(os.path.join(DATA_DIR, 'country_x_by_year.pt'),  weights_only=False)  # 11F from Step 9

with open(os.path.join(DATA_DIR, 'country_mapping.pkl'), 'rb') as f: c_map12 = pickle.load(f)
with open(os.path.join(DATA_DIR, 'product_mapping.pkl'), 'rb') as f: p_map12 = pickle.load(f)

train_lbl = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
val_lbl   = pd.read_csv(os.path.join(DATA_DIR, 'val_labels.csv'))
test_lbl  = pd.read_csv(os.path.join(DATA_DIR, 'test_labels.csv'))

# 4F country tensors: BACI only (re-read from country_features.csv, train-set z-scored per year)
c_feat_4  = pd.read_csv(os.path.join(DATA_DIR, 'country_features.csv'))
BACI_COLS = ['log_export', 'n_products', 'avg_rca', 'max_rca']
c_x_4feat = {}
for yr in sorted(c_feat_4['year'].unique()):
    yd = c_feat_4[c_feat_4['year'] == yr].copy()
    yd['idx'] = yd['country'].map(c_map12['to_idx'])
    yd = yd.dropna(subset=['idx']).sort_values('idx')
    c_x_4feat[int(yr)] = torch.tensor(yd[BACI_COLS].values, dtype=torch.float32)

# PCI proxy: -ubiquity(2010) — less ubiquitous products are more complex and get higher weight
rca_ref  = pd.read_csv(os.path.join(DATA_DIR, 'rca_cpt.csv'), usecols=['year', 'product', 'rca'])
ubiq     = rca_ref[rca_ref['year'] == 2010].groupby('product')['rca'].apply(lambda x: (x >= 1).sum())
max_ubiq = ubiq.max()
pci_proxy_dict = {int(p): float(-u / max_ubiq) for p, u in ubiq.items()}  # range [-1, 0]
print(f'PCI proxy: {len(pci_proxy_dict)} products  |  range [{min(pci_proxy_dict.values()):.3f}, 0.000]')

# -- B: Snapshot + temporal sample builders ------------------------
def build_snap12(year, c_x):
    d = HeteroData()
    d['country'].x = c_x[year]
    d['product'].x = p_x_by_yr[year]
    ei = edge_idx_by_yr[year].long()
    d['country', 'exports',     'product'].edge_index = ei
    d['product', 'rev_exports', 'country'].edge_index = ei.flip(0)
    return d

def build_sample12(obs_yr, ldf, c_x):
    snaps = [build_snap12(y, c_x) for y in range(obs_yr - 4, obs_yr + 1)]
    row   = ldf[ldf['year'] == obs_yr].copy().reset_index(drop=True)
    ci_s  = row['country'].map(c_map12['to_idx'])
    pi_s  = row['product'].map(p_map12['to_idx'])
    ok    = ci_s.notna() & pi_s.notna()
    ci, pi = ci_s[ok].astype(int).values, pi_s[ok].astype(int).values
    lv     = row.loc[ok, 'label'].values
    return {
        'snapshots':     snaps,
        'labels': {'edge_label_index': torch.tensor([ci, pi], dtype=torch.long),
                   'edge_label':       torch.tensor(lv,       dtype=torch.float32)},
        'year':          int(obs_yr),
        'countries_raw': row.loc[ok, 'country'].values,
        'products_raw':  row.loc[ok, 'product'].values,
    }

print('Building 4F datasets (BACI only)...')
tr4 = [build_sample12(y, train_lbl, c_x_4feat) for y in sorted(train_lbl['year'].unique())]
va4 = build_sample12(VAL_YEAR,  val_lbl,  c_x_4feat)
te4 = build_sample12(TEST_YEAR, test_lbl, c_x_4feat)

print('Building 11F datasets (BACI + WDI)...')
tr11 = [build_sample12(y, train_lbl, c_x_11feat) for y in sorted(train_lbl['year'].unique())]
va11 = build_sample12(VAL_YEAR,  val_lbl,  c_x_11feat)
te11 = build_sample12(TEST_YEAR, test_lbl, c_x_11feat)

print(f'OK {len(tr4)} training years  |  4F country dim={tr4[0]["snapshots"][0]["country"].x.shape[1]}'
      f'  |  11F country dim={tr11[0]["snapshots"][0]["country"].x.shape[1]}')

# -- C: Model architecture (matches oldResearch/models.py) ---------
class _HomoGNN(nn.Module):
    def __init__(self, hidden, drop=0.3):
        super().__init__()
        self.c1 = SAGEConv(hidden, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.drop = drop
    def forward(self, x, edge_index):
        x = F.dropout(self.c1(x, edge_index).relu(), p=self.drop, training=self.training)
        return self.c2(x, edge_index)

class BipartiteEncoder12(nn.Module):
    def __init__(self, c_in, hidden, meta):
        super().__init__()
        self.country_lin = nn.Linear(c_in, hidden)
        self.product_lin = nn.Linear(3, hidden)
        self.gnn = to_hetero(_HomoGNN(hidden), meta)
    def forward(self, x_dict, ei_dict):
        return self.gnn({'country': self.country_lin(x_dict['country']),
                         'product': self.product_lin(x_dict['product'])}, ei_dict)

class TemporalGNN12(nn.Module):
    def __init__(self, enc, hidden):
        super().__init__()
        self.enc   = enc
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

class LinkPredictor12(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))
    def forward(self, zc, zp, ei):
        return self.mlp(torch.cat([zc[ei[0]], zp[ei[1]]], -1)).view(-1)

# -- D: Training helpers -------------------------------------------
def to_dev12(samp, dev):
    for s in samp['snapshots']:
        s['country'].x = s['country'].x.to(dev)
        s['product'].x = s['product'].x.to(dev)
        for et in s.edge_types:
            s[et].edge_index = s[et].edge_index.to(device=dev, dtype=torch.long)
    samp['labels']['edge_label_index'] = samp['labels']['edge_label_index'].to(dev)
    samp['labels']['edge_label']       = samp['labels']['edge_label'].to(dev)

@torch.no_grad()
def get_scores12(mdl, pred, samp, dev):
    mdl.eval(); pred.eval()
    to_dev12(samp, dev)
    z = mdl(samp['snapshots'])
    return torch.sigmoid(pred(z['country'], z['product'],
                              samp['labels']['edge_label_index'])).cpu().numpy()

def train_gnn12(name, tr, va, c_in, hidden=HIDDEN, epochs=EPOCHS, patience=PATIENCE, dev=DEVICE):
    meta = tr[0]['snapshots'][0].metadata()
    enc  = BipartiteEncoder12(c_in, hidden, meta).to(dev)
    mdl  = TemporalGNN12(enc, hidden).to(dev)
    pred = LinkPredictor12(hidden).to(dev)

    # Correct pos_weight for class imbalance
    all_lv  = torch.cat([s['labels']['edge_label'] for s in tr])
    n_pos   = all_lv.sum().item()
    n_neg   = (all_lv == 0).sum().item()
    pw      = torch.tensor([n_neg / max(n_pos, 1)], device=dev)
    crit    = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt     = torch.optim.Adam(list(mdl.parameters()) + list(pred.parameters()), lr=LR, weight_decay=WD)

    best_vpa, best_state, no_imp = -1.0, None, 0
    hist = {'loss': [], 'val_pr_auc': []}

    print(f'\nTraining {name}  |  c_in={c_in}  |  pos_weight={n_neg/n_pos:.1f}x')
    for ep in range(1, epochs + 1):
        mdl.train(); pred.train(); ep_loss = 0.0
        for samp in tr:
            to_dev12(samp, dev)
            opt.zero_grad()
            z    = mdl(samp['snapshots'])
            loss = crit(pred(z['country'], z['product'], samp['labels']['edge_label_index']),
                        samp['labels']['edge_label'])
            loss.backward()
            nn.utils.clip_grad_norm_(list(mdl.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            ep_loss += loss.item()

        vscores = get_scores12(mdl, pred, va, dev)
        vlabels = va['labels']['edge_label'].cpu().numpy()
        p_, r_, _ = precision_recall_curve(vlabels, vscores)
        vpa = auc(r_, p_)
        hist['loss'].append(ep_loss / len(tr))
        hist['val_pr_auc'].append(vpa)

        if vpa > best_vpa:
            best_vpa   = vpa
            best_state = ({k: v.cpu().clone() for k, v in mdl.state_dict().items()},
                          {k: v.cpu().clone() for k, v in pred.state_dict().items()})
            no_imp = 0
        else:
            no_imp += 1

        if ep % 10 == 0:
            print(f'  Ep {ep:3d}  loss={ep_loss/len(tr):.4f}  val_PR-AUC={vpa:.4f}  best={best_vpa:.4f}')
        if no_imp >= patience:
            print(f'  Early stop at epoch {ep}  (best val PR-AUC={best_vpa:.4f})')
            break

    mdl.load_state_dict({k: v.to(dev) for k, v in best_state[0].items()})
    pred.load_state_dict({k: v.to(dev) for k, v in best_state[1].items()})
    return mdl, pred, hist

# -- E: 3-tier evaluation (matches evaluator.py logic exactly) -----
def tier_eval12(mdl, pred, samp, pci_dict, dev, k=20):
    scores = get_scores12(mdl, pred, samp, dev)
    labels = samp['labels']['edge_label'].cpu().numpy()
    ctry   = samp['countries_raw']
    prod   = samp['products_raw']

    # Tier 3: PR-AUC (primary metric for imbalanced data)
    prec_, rec_, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(rec_, prec_)
    auroc  = roc_auc_score(labels, scores) if 0 < labels.mean() < 1 else 0.0

    # Tier 1: macro NDCG@k and Precision@k (per-country, then averaged)
    df = pd.DataFrame({'country': ctry, 'product': prod, 'score': scores, 'label': labels})
    ndcg_vals, prec_vals = [], []
    for _, grp in df.groupby('country'):
        if grp['label'].sum() == 0:
            continue
        yt, ys = grp['label'].values, grp['score'].values
        try:
            ndcg_vals.append(ndcg_score([yt], [ys], k=k))
        except Exception:
            pass
        prec_vals.append(grp.sort_values('score', ascending=False).head(k)['label'].mean())
    ndcg_k = float(np.nanmean(ndcg_vals)) if ndcg_vals else 0.0
    prec_k = float(np.nanmean(prec_vals)) if prec_vals else 0.0

    # Tier 2: complexity-weighted recall at threshold 0.5 (matches evaluator.py)
    df['pci'] = df['product'].map(pci_dict).fillna(df['product'].map(pci_dict).mean())
    df['pci'] = df['pci'].fillna(0.0)
    df['w']   = df['pci'] - df['pci'].min()  # shift to non-negative
    tot_w     = df.loc[df['label'] == 1, 'w'].sum()
    df['score_pct'] = df['score'].rank(pct=True)  # normalise to [0,1] so threshold is method-agnostic
    hit_w     = df.loc[(df['label'] == 1) & (df['score_pct'] >= 0.5), 'w'].sum()
    cwr       = float(hit_w / tot_w) if tot_w > 0 else 0.0

    return {'PR-AUC':    round(pr_auc, 4),
            'AUROC':     round(auroc,  4),
            f'NDCG@{k}': round(ndcg_k, 4),
            f'Prec@{k}': round(prec_k, 4),
            'CWR':       round(cwr,    4)}

# -- F: Train both configurations ---------------------------------
c_in_4  = tr4[0]["snapshots"][0]["country"].x.shape[1]
c_in_11 = tr11[0]["snapshots"][0]["country"].x.shape[1]
print(f"Detected feature dims — 4F={c_in_4}  11F={c_in_11}")
if c_in_11 == c_in_4:
    print(f"WARNING: country_x_by_year.pt only has {c_in_11} features — WDI step may not have run yet.")
mdl4,  pred4,  hist4  = train_gnn12('GNN-4F  (BACI only)',  tr4,  va4,  c_in=c_in_4)
mdl11, pred11, hist11 = train_gnn12('GNN-11F (BACI + WDI)', tr11, va11, c_in=c_in_11)

# -- G: Evaluate on test set (year 2015 -> predicts 2020 outcomes) -
res4  = tier_eval12(mdl4,  pred4,  te4,  pci_proxy_dict, DEVICE)
res11 = tier_eval12(mdl11, pred11, te11, pci_proxy_dict, DEVICE)

# Save results
gnn_df = pd.DataFrame([{'Method': 'GNN-4F (BACI only)',  **res4},
                        {'Method': 'GNN-11F (BACI+WDI)', **res11}])
gnn_df.to_csv(os.path.join(DATA_DIR, 'gnn_tiered_results.csv'), index=False)

# -- H: Unified comparison table -----------------------------------
# Baseline numbers from oldResearch/data/baseline_tiered_results.csv
# and oldResearch/data/persistence_tiered_results.csv
baselines = [
    ('RCA Persistence',  0.525, 0.510, 0.528, 0.3356),
    ('Density',          0.349, 0.487, 0.434, 0.8575),
    ('ECI',              0.231, 0.146, 0.869, 0.4637),
    ('ECI + Density',    0.349, 0.487, 0.434, 0.8575),
]

print('\n' + '='*72)
print(f'  {"Model":<26} {"PR-AUC":>8} {"NDCG@20":>8} {"Prec@20":>8} {"CWR":>8}')
print('-'*72)
for name, prauc, ndcg, prec, cwr in baselines:
    print(f'  {name:<26} {prauc:>8.3f} {ndcg:>8.3f} {prec:>8.3f} {cwr:>8.3f}')
print('  ' + '.'*68)
for label, res in [('GNN-4F  (BACI only) <-', res4), ('GNN-11F (BACI+WDI)  <-', res11)]:
    print(f'  {label:<26} {res["PR-AUC"]:>8.3f} {res["NDCG@20"]:>8.3f} {res["Prec@20"]:>8.3f} {res["CWR"]:>8.3f}')
print('='*72)
print('  Tiers: PR-AUC = global (Tier 3) | NDCG@20 & Prec@20 = per-country (Tier 1) | CWR = complexity-weighted (Tier 2)')
print(f'  AUROC:  GNN-4F={res4["AUROC"]:.3f}  GNN-11F={res11["AUROC"]:.3f}')
print(f'\nOK Saved -> data/gnn_tiered_results.csv')

# -- I: Training curves --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for hist, label, color in [(hist4, '4F (no WDI)', 'steelblue'), (hist11, '11F (with WDI)', 'darkorange')]:
    axes[0].plot(hist['loss'],       label=label, color=color)
    axes[1].plot(hist['val_pr_auc'], label=label, color=color)
axes[0].set(title='Training BCE Loss (pos_weight corrected)', xlabel='Epoch', ylabel='Loss')
axes[1].set(title='Validation PR-AUC', xlabel='Epoch', ylabel='PR-AUC')
for ax in axes: ax.legend()
plt.suptitle('GNN Training Curves — BACI-only vs BACI+WDI', fontsize=13)
plt.tight_layout()
plt.show()