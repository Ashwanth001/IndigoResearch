"""
Recompute baseline CWR values using percentile-ranked scores
so the threshold 0.5 means "top half of predictions" for all methods.
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

DATA_DIR   = 'data'
TEST_YEAR  = 2015
TRAIN_CUTOFF = 2012

# ── Load data ─────────────────────────────────────────────────────────────────
smooth   = pd.read_csv(os.path.join(DATA_DIR, 'M_cpt_smoothed.csv'))
rca      = pd.read_csv(os.path.join(DATA_DIR, 'rca_cpt.csv'))
test_lbl = pd.read_csv(os.path.join(DATA_DIR, 'test_labels.csv'))

countries = sorted(smooth['country'].unique())
products  = sorted(smooth['product'].unique())
C, P = len(countries), len(products)
c_idx = {c: i for i, c in enumerate(countries)}
p_idx = {p: i for i, p in enumerate(products)}

def build_M(year):
    M = np.zeros((C, P), dtype=np.float32)
    yr = smooth[smooth['year'] == year]
    ci = yr['country'].map(c_idx).values
    pi = yr['product'].map(p_idx).values
    M[ci, pi] = 1.0
    return M

# ── PCI proxy (matches step 12: -ubiquity/max_ubiquity) ──────────────────────
rca_ref  = rca[rca['year'] == 2010]
ubiq     = rca_ref.groupby('product')['rca'].apply(lambda x: (x >= 1).sum())
max_ubiq = ubiq.max()
pci_dict = {int(p): float(-u / max_ubiq) for p, u in ubiq.items()}

# ── PCI weights for test pairs ────────────────────────────────────────────────
test_lbl = test_lbl.copy()
test_lbl['pci'] = test_lbl['product'].map(pci_dict).fillna(0.0)
min_pci = test_lbl['pci'].min()
test_lbl['w'] = test_lbl['pci'] - min_pci   # shift to non-negative
tot_w = test_lbl.loc[test_lbl['label'] == 1, 'w'].sum()

def cwr_pct(scores_series, labels_series, w_series):
    """CWR with percentile-ranked scores. Threshold = top 50% of predictions."""
    pct = scores_series.rank(pct=True)
    hit_w = w_series[(labels_series == 1) & (pct >= 0.5)].sum()
    return float(hit_w / tot_w) if tot_w > 0 else 0.0

# ── Proximity matrix (training years only) ────────────────────────────────────
print('Building proximity matrix from training years...')
train_years = [y for y in smooth['year'].unique() if y <= TRAIN_CUTOFF]
co_export  = np.zeros((P, P), dtype=np.float32)
any_export = np.zeros((P, P), dtype=np.float32)
for yr in sorted(train_years):
    M = build_M(yr)
    co      = M.T @ M
    exp     = M.sum(axis=0)
    any_mat = exp[:, None] + exp[None, :] - co
    co_export  += co
    any_export += any_mat
phi = np.where(any_export > 0, co_export / (any_export + 1e-9), 0.0)
np.fill_diagonal(phi, 0.0)
phi_row_sum = phi.sum(axis=1)

# ── Compute scores for test pairs ─────────────────────────────────────────────
M_t = build_M(TEST_YEAR)
ci_arr = test_lbl['country'].map(c_idx).values
pi_arr = test_lbl['product'].map(p_idx).values

# ECI / PCI
kc = M_t.sum(axis=1); kp = M_t.sum(axis=0)
kc_safe = np.where(kc > 0, kc, 1.0); kp_safe = np.where(kp > 0, kp, 1.0)
kc_n, kp_n = kc.astype(float), kp.astype(float)
for _ in range(20):
    kc_n = (1.0 / kc_safe) * (M_t @ kp_n)
    kp_n = (1.0 / kp_safe) * (M_t.T @ kc_n)
eci = (kc_n - kc_n.mean()) / (kc_n.std() + 1e-9)
pci = (kp_n - kp_n.mean()) / (kp_n.std() + 1e-9)

# Density
dens_mat = (M_t @ phi) / (phi_row_sum[None, :] + 1e-9)

def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-9)

scores = {
    'RCA Persistence': None,   # computed separately below
    'Density':         dens_mat[ci_arr, pi_arr],
    'ECI':             eci[ci_arr],
    'ECI + Density':   minmax(eci[ci_arr]) + minmax(dens_mat[ci_arr, pi_arr]),
}

# RCA Persistence: mean(RCA>=1) over past 3 years
hist_yrs = [TEST_YEAR - 2, TEST_YEAR - 1, TEST_YEAR]
rca_hist = rca[rca['year'].isin(hist_yrs)][['country', 'product', 'year', 'rca']]
rca_hist = rca_hist.merge(test_lbl[['country', 'product']], on=['country', 'product'])
rca_wide = rca_hist.pivot_table(index=['country', 'product'], columns='year', values='rca', fill_value=0)
for yr in hist_yrs:
    if yr not in rca_wide.columns:
        rca_wide[yr] = 0
rca_wide['score'] = (rca_wide[hist_yrs] >= 1).mean(axis=1)
persist_merged = test_lbl.merge(rca_wide[['score']], on=['country', 'product'], how='left').fillna(0)
scores['RCA Persistence'] = persist_merged['score'].values

# ── Compute CWR with percentile ranking ───────────────────────────────────────
print('\nBaseline CWR (percentile-ranked scores, threshold = top 50%):')
print('-' * 45)
results = {}
for name, sc in scores.items():
    s = pd.Series(sc)
    cwr = cwr_pct(s, test_lbl['label'], test_lbl['w'])
    results[name] = round(cwr, 4)
    print(f'  {name:<20}: {cwr:.4f}')

print('\nPaste these into the baselines list in cell 27 (replacing the last column):')
for name, cwr in results.items():
    print(f'  {name}: CWR = {cwr}')
