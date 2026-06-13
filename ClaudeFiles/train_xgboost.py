"""
Train XGBoost on GPU and save checkpoint to data/models/xgboost/xgb_model.pkl.

Features per (country, product) pair (51 dims total):
  - 11 country features  : 4 BACI (log_export, n_products, avg_rca, max_rca)
                           + 7 WDI (gdp_pc, capital_formation, tertiary_enrollment,
                                    fdi_inflows, manufacturing_va, internet_users, population)
  -  3 product features  : log_world_export, ubiquity, avg_rca
  - 32 PCA-LLM features  : top-32 PCA components of 768-dim FinLang product embeddings (L2-normed)
  -  1 density score     : phi-weighted fraction of neighbours the country exports
  -  1 ECI score         : country economic complexity index
  -  3 RCA history       : raw RCA at t-2, t-1, t

Training data: train_labels.csv (years 2000-2012, ~1.7M pairs).
Strict TRAIN_CUTOFF=2012 — no val/test leakage.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

DATA_DIR     = 'data'
TRAIN_CUTOFF = 2012
PCA_DIM      = 32
XGB_DIR      = os.path.join(DATA_DIR, 'models', 'xgboost')
XGB_CKPT     = os.path.join(XGB_DIR, 'xgb_model.pkl')
os.makedirs(XGB_DIR, exist_ok=True)

BACI_COLS    = ['log_export', 'n_products', 'avg_rca', 'max_rca']
WDI_COLS     = ['gdp_pc', 'capital_formation', 'tertiary_enrollment',
                'fdi_inflows', 'manufacturing_va', 'internet_users', 'population']
COUNTRY_COLS = BACI_COLS + WDI_COLS   # 11 dims
PROD_COLS    = ['log_world_export', 'ubiquity', 'avg_rca']   # 3 dims

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
print(f'Feature dims: {len(COUNTRY_COLS)} country + {len(PROD_COLS)} product + '
      f'{PCA_DIM} PCA-LLM + 1 density + 1 ECI + 3 RCA-history = '
      f'{len(COUNTRY_COLS) + len(PROD_COLS) + PCA_DIM + 5} total')

# ── Load raw data ─────────────────────────────────────────────────────────────
print('\nLoading raw data...')
smooth    = pd.read_csv(os.path.join(DATA_DIR, 'M_cpt_smoothed.csv'))
rca_df    = pd.read_csv(os.path.join(DATA_DIR, 'rca_cpt.csv'))
train_lbl = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
c_feat_df = pd.read_csv(os.path.join(DATA_DIR, 'country_features_enriched.csv'))  # 11 cols
p_feat_df = pd.read_csv(os.path.join(DATA_DIR, 'product_features.csv'))

countries = sorted(smooth['country'].unique())
products  = sorted(smooth['product'].unique())
C, P = len(countries), len(products)
c_idx = {c: i for i, c in enumerate(countries)}
p_idx = {p: i for i, p in enumerate(products)}
print(f'Countries: {C}  Products: {P}')

# ── PCA-compressed LLM product embeddings ─────────────────────────────────────
# Fit PCA on the full 768-dim embeddings (using all products — no train/test split
# needed here because embeddings are derived from product *names*, not export data).
print(f'\nFitting PCA({PCA_DIM}) on product LLM embeddings...')
llm_emb = torch.load(os.path.join(DATA_DIR, 'product_llm_embeddings.pt'),
                      weights_only=False, map_location='cpu').float().numpy()  # [P, 768]
pca = PCA(n_components=PCA_DIM, random_state=42)
pca.fit(llm_emb)
llm_pca = pca.transform(llm_emb).astype(np.float32)   # [P, 32]
# L2-normalise rows so scale matches the other features
llm_pca /= np.linalg.norm(llm_pca, axis=1, keepdims=True).clip(min=1e-8)
explained = pca.explained_variance_ratio_.sum()
print(f'PCA explains {explained * 100:.1f}% of embedding variance.')

# Map product id → row index in llm_pca (products list is 0-indexed same as p_idx)
# llm_pca[p_idx[p]] gives the PCA vector for product p.

# ── Proximity matrix (train years only, GPU-accelerated) ──────────────────────
print('\nBuilding proximity matrix from training years (GPU)...')
co_exp_t  = torch.zeros((P, P), dtype=torch.float32, device=DEVICE)
any_exp_t = torch.zeros((P, P), dtype=torch.float32, device=DEVICE)

for yr in sorted(y for y in smooth['year'].unique() if y <= TRAIN_CUTOFF):
    rows = smooth[smooth['year'] == yr]
    ci_yr = rows['country'].map(c_idx).values
    pi_yr = rows['product'].map(p_idx).values

    # Build M as sparse COO indices for efficiency
    M_indices = torch.LongTensor([ci_yr, pi_yr]).to(DEVICE)
    M_values = torch.ones(len(ci_yr), dtype=torch.float32, device=DEVICE)
    M_sparse = torch.sparse_coo_tensor(M_indices, M_values, (C, P), device=DEVICE).float()
    M_dense = M_sparse.to_dense()

    # Compute co-export and any-export
    co = M_dense.T @ M_dense
    ex = M_dense.sum(dim=0)
    co_exp_t  += co
    any_exp_t += ex.unsqueeze(1) + ex.unsqueeze(0) - co

# Compute phi on GPU
phi_t = torch.where(any_exp_t > 0, co_exp_t / (any_exp_t + 1e-9), torch.zeros_like(co_exp_t))
phi_t.fill_diagonal_(0.0)
phi_row_sum_t = phi_t.sum(dim=1)

# Move to CPU for later use
phi = phi_t.cpu().numpy().astype(np.float32)
phi_row_sum = phi_row_sum_t.cpu().numpy().astype(np.float32)
print(f'Proximity matrix done (GPU: {DEVICE}).')


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_M(year):
    M = np.zeros((C, P), dtype=np.float32)
    yr = smooth[smooth['year'] == year]
    M[yr['country'].map(c_idx).values, yr['product'].map(p_idx).values] = 1.0
    return M


def build_features(df_pairs, t):
    """
    Build feature matrix for (country, product) pairs at observation year t.
    Returns float32 array of shape [N, 51].
    Vectorized with GPU acceleration where possible.
    """
    ci_vals = df_pairs['country'].map(c_idx).values.astype(np.int64)
    pi_vals = df_pairs['product'].map(p_idx).values.astype(np.int64)

    # ── Density and ECI from M_t (GPU) ──────────────────────────────────────
    rows_M = smooth[smooth['year'] == t]
    ci_M = rows_M['country'].map(c_idx).values
    pi_M = rows_M['product'].map(p_idx).values
    M_indices = torch.LongTensor([ci_M, pi_M]).to(DEVICE)
    M_values = torch.ones(len(ci_M), dtype=torch.float32, device=DEVICE)
    M_t_sparse = torch.sparse_coo_tensor(M_indices, M_values, (C, P), device=DEVICE).float()
    M_t_dense = M_t_sparse.to_dense()

    kc_t = M_t_dense.sum(dim=1); kp_t = M_t_dense.sum(dim=0)
    kcs_t = torch.where(kc_t > 0, kc_t, torch.ones_like(kc_t))
    kps_t = torch.where(kp_t > 0, kp_t, torch.ones_like(kp_t))
    kc_n, kp_n = kc_t.float(), kp_t.float()
    for _ in range(20):
        kc_n = (1.0 / kcs_t) * (M_t_dense @ kp_n)
        kp_n = (1.0 / kps_t) * (M_t_dense.T @ kc_n)
    eci_t = (kc_n - kc_n.mean()) / (kc_n.std() + 1e-9)

    phi_t = torch.from_numpy(phi).to(DEVICE).float()
    dens_t = M_t_dense @ phi_t.T / (torch.from_numpy(phi_row_sum).to(DEVICE).unsqueeze(0) + 1e-9)

    eci_t_np = eci_t.cpu().numpy().astype(np.float32)
    dens_t_np = dens_t.cpu().numpy().astype(np.float32)

    # ── Country features (vectorized) ────────────────────────────────────────
    avail_c = sorted(c_feat_df['year'].unique())
    c_yr = max(y for y in avail_c if y <= t)
    cf = c_feat_df[c_feat_df['year'] == c_yr].set_index('country')[COUNTRY_COLS]
    c_feats = np.zeros((C, len(COUNTRY_COLS)), dtype=np.float32)
    for i, c in enumerate(countries):
        if c in cf.index:
            c_feats[i] = cf.loc[c].values.astype(np.float32)
    c_feats_batch = c_feats[ci_vals]  # [N, 11]

    # ── Product features (vectorized) ────────────────────────────────────────
    avail_p = sorted(p_feat_df['year'].unique())
    p_yr = max(y for y in avail_p if y <= t)
    pf = p_feat_df[p_feat_df['year'] == p_yr].set_index('product')[PROD_COLS]
    p_feats = np.zeros((P, len(PROD_COLS)), dtype=np.float32)
    for i, p in enumerate(products):
        if p in pf.index:
            p_feats[i] = pf.loc[p].values.astype(np.float32)
    p_feats_batch = p_feats[pi_vals]  # [N, 3]

    # ── LLM PCA features (vectorized) ───────────────────────────────────────
    llm_feats_batch = llm_pca[pi_vals]  # [N, 32]

    # ── Density and ECI (indexed) ──────────────────────────────────────────
    dens_batch = dens_t_np[ci_vals, pi_vals].reshape(-1, 1)  # [N, 1]
    eci_batch = eci_t_np[ci_vals].reshape(-1, 1)  # [N, 1]

    # ── RCA history — build dense [C, P, 3] array, then index in one shot ──
    hist_yrs = [t - 2, t - 1, t]
    rca_sub = rca_df[rca_df['year'].isin(hist_yrs)].copy()
    rca_sub['ci'] = rca_sub['country'].map(c_idx)
    rca_sub['pi'] = rca_sub['product'].map(p_idx)
    rca_sub = rca_sub.dropna(subset=['ci', 'pi'])
    rca_sub['ci'] = rca_sub['ci'].astype(int)
    rca_sub['pi'] = rca_sub['pi'].astype(int)

    rca_arr = np.zeros((C, P, 3), dtype=np.float32)
    for k, yr in enumerate(hist_yrs):
        yr_rows = rca_sub[rca_sub['year'] == yr]
        rca_arr[yr_rows['ci'].values, yr_rows['pi'].values, k] = yr_rows['rca'].values.astype(np.float32)

    rca_batch = rca_arr[ci_vals, pi_vals]  # [N, 3] — single array index, no loop

    # ── Concatenate all features ────────────────────────────────────────────
    X = np.hstack([c_feats_batch, p_feats_batch, llm_feats_batch,
                   dens_batch, eci_batch, rca_batch])
    return X.astype(np.float32)


# ── Build training matrix year by year ───────────────────────────────────────
print('\nBuilding training feature matrix...')
train_years = sorted(train_lbl['year'].unique())
X_parts, y_parts = [], []

for t_yr in train_years:
    df_yr = train_lbl[train_lbl['year'] == t_yr].reset_index(drop=True)
    print(f'  year {t_yr}: {len(df_yr):,} pairs ...', end=' ', flush=True)
    X_yr = build_features(df_yr, t_yr)
    X_parts.append(X_yr)
    y_parts.append(df_yr['label'].values.astype(np.float32))
    print(f'done  shape={X_yr.shape}')

X_train = np.vstack(X_parts)
y_train = np.concatenate(y_parts)
pos_rate         = y_train.mean()
scale_pos_weight = (1.0 - pos_rate) / (pos_rate + 1e-9)
print(f'\nX_train: {X_train.shape}  dtype={X_train.dtype}')
print(f'pos_rate={pos_rate:.4f}  scale_pos_weight={scale_pos_weight:.2f}')


# ── Train on GPU ──────────────────────────────────────────────────────────────
print(f'\nTraining XGBoost on {DEVICE.upper()} (this may take a few minutes)...')
model = xgb.XGBClassifier(
    n_estimators       = 500,
    max_depth          = 6,
    learning_rate      = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 10,
    scale_pos_weight   = scale_pos_weight,
    eval_metric        = 'logloss',
    tree_method        = 'hist',   # GPU-accelerated histogram method
    device             = DEVICE,   # 'cuda' or 'cpu'
    random_state       = 42,
    n_jobs             = -1,
)

model.fit(X_train, y_train, verbose=50)


# ── Save ──────────────────────────────────────────────────────────────────────
# Save both the XGBoost model and the fitted PCA so inference can reuse it.
bundle = {'model': model, 'pca': pca}
with open(XGB_CKPT, 'wb') as f:
    pickle.dump(bundle, f)
print(f'\nModel + PCA saved to {XGB_CKPT}')

# Sanity check on a slice of the training set
sample_size = min(10_000, len(X_train))
preds = model.predict_proba(X_train[:sample_size])[:, 1]
auroc = roc_auc_score(y_train[:sample_size], preds)
print(f'Sanity AUROC on first {sample_size:,} train rows: {auroc:.4f}')
