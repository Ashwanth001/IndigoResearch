"""Train GNN-4F and GNN-11F and save checkpoints to data/checkpoints/."""
import os, sys, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

DATA_DIR = 'data'
CKPT_DIR = os.path.join(DATA_DIR, 'checkpoints')
TEST_YEAR = 2015
VAL_YEAR  = 2013
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN, EPOCHS, LR, WD, PATIENCE = 128, 80, 1e-3, 1e-5, 15

os.makedirs(CKPT_DIR, exist_ok=True)
print(f'Device: {DEVICE}')

# ── Load artifacts ────────────────────────────────────────────────────────────
train_lbl = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
val_lbl   = pd.read_csv(os.path.join(DATA_DIR, 'val_labels.csv'))
test_lbl  = pd.read_csv(os.path.join(DATA_DIR, 'test_labels.csv'))

edge_idx_raw = torch.load(os.path.join(DATA_DIR, 'edge_index_by_year.pt'), weights_only=False)
edge_idx_by_yr = {k: v.long() for k, v in edge_idx_raw.items()}
p_x_by_yr  = torch.load(os.path.join(DATA_DIR, 'product_x_by_year.pt'),  weights_only=False)
c_x_11feat = torch.load(os.path.join(DATA_DIR, 'country_x_by_year.pt'),  weights_only=False)

with open(os.path.join(DATA_DIR, 'country_mapping.pkl'), 'rb') as f: c_map = pickle.load(f)
with open(os.path.join(DATA_DIR, 'product_mapping.pkl'), 'rb') as f: p_map = pickle.load(f)

c_feat_df = pd.read_csv(os.path.join(DATA_DIR, 'country_features.csv'))
BACI_COLS = ['log_export', 'n_products', 'avg_rca', 'max_rca']
c_x_4feat = {}
for yr in sorted(c_feat_df['year'].unique()):
    yd = c_feat_df[c_feat_df['year'] == yr].copy()
    yd['idx'] = yd['country'].map(c_map['to_idx'])
    yd = yd.dropna(subset=['idx']).sort_values('idx')
    c_x_4feat[int(yr)] = torch.tensor(yd[BACI_COLS].values, dtype=torch.float32)

print(f'4-feat shape: {c_x_4feat[TEST_YEAR].shape}  |  11-feat shape: {c_x_11feat[TEST_YEAR].shape}')

# ── Model definitions ─────────────────────────────────────────────────────────
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

# ── Data helpers ──────────────────────────────────────────────────────────────
def build_snap(year, c_x):
    d = HeteroData()
    d['country'].x = c_x[year]
    d['product'].x = p_x_by_yr[year]
    ei = edge_idx_by_yr[year].long()
    d['country', 'exports',     'product'].edge_index = ei
    d['product', 'rev_exports', 'country'].edge_index = ei.flip(0)
    return d

def build_sample(obs_yr, ldf, c_x):
    snaps = [build_snap(y, c_x) for y in range(obs_yr - 4, obs_yr + 1)]
    row   = ldf[ldf['year'] == obs_yr].copy().reset_index(drop=True)
    ci_s  = row['country'].map(c_map['to_idx'])
    pi_s  = row['product'].map(p_map['to_idx'])
    ok    = ci_s.notna() & pi_s.notna()
    ci, pi = ci_s[ok].astype(int).values, pi_s[ok].astype(int).values
    lv     = row.loc[ok, 'label'].values
    return {
        'snapshots': snaps,
        'labels': {'edge_label_index': torch.tensor([ci, pi], dtype=torch.long),
                   'edge_label':       torch.tensor(lv, dtype=torch.float32)},
        'year': int(obs_yr),
        'countries_raw': row.loc[ok, 'country'].values,
        'products_raw':  row.loc[ok, 'product'].values,
    }

def to_dev(samp, dev):
    for s in samp['snapshots']:
        s['country'].x = s['country'].x.to(dev)
        s['product'].x = s['product'].x.to(dev)
        for et in s.edge_types:
            s[et].edge_index = s[et].edge_index.to(device=dev, dtype=torch.long)
    samp['labels']['edge_label_index'] = samp['labels']['edge_label_index'].to(dev)
    samp['labels']['edge_label']       = samp['labels']['edge_label'].to(dev)

@torch.no_grad()
def get_scores(mdl, pred, samp, dev):
    mdl.eval(); pred.eval()
    to_dev(samp, dev)
    z = mdl(samp['snapshots'])
    return torch.sigmoid(pred(z['country'], z['product'],
                              samp['labels']['edge_label_index'])).cpu().numpy()

# ── Train + save ──────────────────────────────────────────────────────────────
def train_and_save(name, tr, va, c_in, save_path):
    meta = tr[0]['snapshots'][0].metadata()
    enc  = BipartiteEncoder(c_in, HIDDEN, meta).to(DEVICE)
    mdl  = TemporalGNN(enc, HIDDEN).to(DEVICE)
    pred = LinkPredictor(HIDDEN).to(DEVICE)

    all_lv = torch.cat([s['labels']['edge_label'] for s in tr])
    n_pos  = all_lv.sum().item()
    n_neg  = (all_lv == 0).sum().item()
    pw     = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    crit   = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt    = torch.optim.Adam(list(mdl.parameters()) + list(pred.parameters()),
                               lr=LR, weight_decay=WD)

    best_vpa, best_state, no_imp = -1.0, None, 0
    print(f'\nTraining {name}  |  c_in={c_in}  |  pos_weight={n_neg/n_pos:.1f}x')

    for ep in range(1, EPOCHS + 1):
        mdl.train(); pred.train(); ep_loss = 0.0
        for samp in tr:
            to_dev(samp, DEVICE)
            opt.zero_grad()
            z    = mdl(samp['snapshots'])
            loss = crit(pred(z['country'], z['product'], samp['labels']['edge_label_index']),
                        samp['labels']['edge_label'])
            loss.backward()
            nn.utils.clip_grad_norm_(list(mdl.parameters()) + list(pred.parameters()), 1.0)
            opt.step()
            ep_loss += loss.item()

        vscores = get_scores(mdl, pred, va, DEVICE)
        vlabels = va['labels']['edge_label'].cpu().numpy()
        vp, vr, _ = precision_recall_curve(vlabels, vscores)
        vpa = auc(vr, vp)

        if vpa > best_vpa:
            best_vpa   = vpa
            best_state = ({k: v.cpu().clone() for k, v in mdl.state_dict().items()},
                          {k: v.cpu().clone() for k, v in pred.state_dict().items()})
            no_imp = 0
        else:
            no_imp += 1

        if ep % 10 == 0:
            print(f'  Ep {ep:3d}  loss={ep_loss/len(tr):.4f}  val_PR-AUC={vpa:.4f}  best={best_vpa:.4f}')
        if no_imp >= PATIENCE:
            print(f'  Early stop at epoch {ep}  (best val PR-AUC={best_vpa:.4f})')
            break

    torch.save({
        'mdl_state':  best_state[0],
        'pred_state': best_state[1],
        'c_in':   c_in,
        'hidden': HIDDEN,
        'meta':   meta,
    }, save_path)
    print(f'  Checkpoint saved -> {save_path}')

# ── Run ───────────────────────────────────────────────────────────────────────
torch.manual_seed(42); np.random.seed(42)

ckpt_4f  = os.path.join(CKPT_DIR, 'gnn_4f.pt')
ckpt_11f = os.path.join(CKPT_DIR, 'gnn_11f.pt')

print('Building 4F datasets...')
tr4  = [build_sample(y, train_lbl, c_x_4feat) for y in sorted(train_lbl['year'].unique())]
va4  = build_sample(VAL_YEAR, val_lbl, c_x_4feat)
train_and_save('GNN-4F (BACI only)', tr4, va4, c_in=4, save_path=ckpt_4f)

torch.manual_seed(42); np.random.seed(42)
print('\nBuilding 11F datasets...')
tr11 = [build_sample(y, train_lbl, c_x_11feat) for y in sorted(train_lbl['year'].unique())]
va11 = build_sample(VAL_YEAR, val_lbl, c_x_11feat)
train_and_save('GNN-11F (BACI+WDI)', tr11, va11, c_in=11, save_path=ckpt_11f)

print('\nAll checkpoints saved:')
for p in [ckpt_4f, ckpt_11f]:
    size_mb = os.path.getsize(p) / 1e6
    print(f'  {p}  ({size_mb:.1f} MB)')
