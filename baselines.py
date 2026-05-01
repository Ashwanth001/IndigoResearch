"""
baselines.py  —  ECI / Product Space Baselines
================================================
Computes the three canonical baselines required for Table 1 of the paper.
These baselines answer the research question: does the GNN beat established
economic complexity measures?

Baselines implemented
---------------------
1. **Density (Product Space proximity)**
   For each (country c, candidate product p) pair at year t,
   density(c, p, t) = Σ_{p' ∈ M_t(c)} φ(p, p') / Σ_{p'} φ(p, p')
   where φ(p, p') is the product-pair co-export proximity.
   A high density means many of c's current products are close to p in
   the product space — the standard Hidalgo et al. (2009) predictor.

2. **ECI score (country fitness)**
   Economic Complexity Index computed via the Method of Reflections (MoR)
   with 20 iterations. Used as a country-level prior: wealthier, more
   complex countries are more likely to diversify into any new product.

3. **PCI score (product complexity)**
   Product Complexity Index from the same MoR iteration. Used as a
   product-level prior: less ubiquitous, more complex products are
   harder to enter.

Evaluation
----------
For each of the three test years (matching step8_split.py), the script
computes:
  • AUROC
  • Precision@50
  • Recall@50

using the same labels_h5.csv that the GNN trains on, so results are
directly comparable.

Usage
-----
    python3.14 baselines.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
SMOOTH_FILE = os.path.join(DATA_DIR, "M_cpt_smoothed.csv")   # step3 output
LABELS_FILE = os.path.join(DATA_DIR, "labels_h5.csv")        # step4 output
TEST_YEARS  = [2015]     # mirrors step8_split.py test window
VAL_YEARS   = [2013]     # mirrors step8_split.py val window

# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading data...")
if not os.path.exists(SMOOTH_FILE):
    raise FileNotFoundError(f"{SMOOTH_FILE} not found — run step3 first.")
if not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"{LABELS_FILE} not found — run step4 first.")

edges_df  = pd.read_csv(SMOOTH_FILE)   # columns: year, country, product, M
labels_df = pd.read_csv(LABELS_FILE)  # columns: year, country, product, label

years    = sorted(edges_df['year'].unique())
countries = sorted(edges_df['country'].unique())
products  = sorted(edges_df['product'].unique())

C, P = len(countries), len(products)
country_idx = {c: i for i, c in enumerate(countries)}
product_idx = {p: i for i, p in enumerate(products)}
print(f"Loaded {C} countries, {P} products, {len(years)} years.\n")

# ─── Helper: build binary M matrix for a given year ──────────────────────────
def build_M(year: int) -> np.ndarray:
    """Return binary [C×P] RCA matrix for 'year' from smoothed edges."""
    M = np.zeros((C, P), dtype=np.float32)
    yr = edges_df[edges_df['year'] == year]
    ci = yr['country'].map(country_idx).values
    pi = yr['product'].map(product_idx).values
    M[ci, pi] = 1.0
    return M


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 1 — Product Space Density
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Computing Product Space Proximity matrix φ...")

# φ(p, p') = |countries that export BOTH| / |countries that export EITHER|
# Computed over ALL training years combined (up to TRAIN_CUTOFF = max(years ≤ 2012))
TRAIN_CUTOFF = 2012
train_years  = [y for y in years if y <= TRAIN_CUTOFF]

# Accumulate co-occurrence counts over training years for robustness
co_export = np.zeros((P, P), dtype=np.float32)
any_export = np.zeros((P, P), dtype=np.float32)

print(f"  Building φ from {len(train_years)} training years...")
for yr in train_years:
    M = build_M(yr)                     # [C, P]
    # For each pair (p, p'): co-exports = countries with BOTH; any = with EITHER
    # Use matrix multiplication for efficiency
    co  = M.T @ M                       # [P, P]  : |{c : c exports p AND p'}|
    exp = M.sum(axis=0)                 # [P]     : |{c : c exports p}|
    # any(p, p') = |{c exports p}| + |{c exports p'}| - co(p, p')
    any_mat = exp[:, None] + exp[None, :] - co
    co_export  += co
    any_export += any_mat

# Proximity (Jaccard-like)
phi = np.where(any_export > 0, co_export / (any_export + 1e-9), 0.0)
np.fill_diagonal(phi, 0.0)         # self-similarity not useful
phi_row_sum = phi.sum(axis=1)      # normalisation denominator [P]
print(f"  φ computed: shape {phi.shape}, mean={phi.mean():.4f}\n")


def density_scores(M_t: np.ndarray) -> np.ndarray:
    """
    density[c, p] = (M_t[c,:] @ phi[:, p]) / (phi[:, p].sum() + 1e-9)
    Returns [C×P] matrix of density scores.
    """
    # numerator[c, p] = sum_{p' in basket_c} phi[p, p']
    numerator   = M_t @ phi                        # [C, P]
    denominator = phi_row_sum[None, :] + 1e-9      # [1, P]
    return numerator / denominator                 # [C, P]


# ══════════════════════════════════════════════════════════════════════════════
# BASELINE 2 & 3 — ECI and PCI via Method of Reflections (20 iterations)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Computing ECI / PCI via Method of Reflections (20 iters)...")


def compute_eci_pci(M: np.ndarray, n_iter: int = 20):
    """
    Method of Reflections (Hidalgo & Hausmann 2009).
    Returns ECI [C] and PCI [P] as numpy arrays.
    """
    kc = M.sum(axis=1)   # diversification [C]
    kp = M.sum(axis=0)   # ubiquity        [P]

    # Avoid division by zero for products/countries with zero exports
    kc_safe = np.where(kc > 0, kc, 1.0)
    kp_safe = np.where(kp > 0, kp, 1.0)

    kc_n = kc.copy().astype(float)
    kp_n = kp.copy().astype(float)

    for _ in range(n_iter):
        kc_new = (1.0 / kc_safe) * (M   @ kp_n)
        kp_new = (1.0 / kp_safe) * (M.T @ kc_n)
        kc_n, kp_n = kc_new, kp_new

    # Normalise to zero-mean unit-variance
    def normalise(x):
        std = x.std() + 1e-9
        return (x - x.mean()) / std

    return normalise(kc_n), normalise(kp_n)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════
def precision_at_k(y_true, y_score, k=50):
    top_k = np.argsort(y_score)[::-1][:k]
    return y_true[top_k].sum() / k


def recall_at_k(y_true, y_score, k=50):
    top_k = np.argsort(y_score)[::-1][:k]
    n_pos = y_true.sum()
    return y_true[top_k].sum() / (n_pos + 1e-9)


def evaluate_baseline(name: str, scores: np.ndarray,
                       y_true: np.ndarray) -> dict:
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float('nan')
    p50 = precision_at_k(y_true, scores, k=50)
    r50 = recall_at_k(y_true, scores, k=50)
    print(f"  [{name}] AUC={auc:.4f}  P@50={p50:.4f}  R@50={r50:.4f}")
    return {'baseline': name, 'auc': auc, 'precision@50': p50, 'recall@50': r50}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE ON EACH EVAL YEAR
# ══════════════════════════════════════════════════════════════════════════════
all_results = []

for split_name, eval_years in [("VAL", VAL_YEARS), ("TEST", TEST_YEARS)]:
    print("=" * 60)
    print(f"Split: {split_name}  |  Years: {eval_years}")
    print("=" * 60)

    for eval_year in eval_years:
        print(f"\n--- Eval year: {eval_year} ---")

        # Build M at t (observation year from labels_df, not eval_year itself)
        # labels_df['year'] = t (the observation year), the label is for t+5
        obs_year = eval_year   # labels are tagged with observation year t

        available_years = labels_df['year'].unique()   # plain numpy array, no ambiguity

        if obs_year not in available_years:
            # Fallback: use the closest year present in labels
            obs_year = min(available_years, key=lambda y: abs(y - eval_year))
            print(f"  (No exact match for {eval_year}; using obs_year={obs_year})")

        year_labels = labels_df[labels_df['year'] == obs_year]
        if year_labels.empty:
            print(f"  No labels for year {eval_year}, skipping.")
            continue

        # Build observation-year M matrix
        M_t = build_M(obs_year)

        # Map labels to flat index
        ci_labels = year_labels['country'].map(country_idx).values
        pi_labels = year_labels['product'].map(product_idx).values
        y_true    = year_labels['label'].values.astype(float)

        # --- Density baseline ---
        dens = density_scores(M_t)
        dens_scores = dens[ci_labels, pi_labels]
        r = evaluate_baseline(f"Density (year={eval_year})", dens_scores, y_true)
        r['year'] = eval_year; r['split'] = split_name
        all_results.append(r)

        # --- ECI baseline (country score only) ---
        eci, pci = compute_eci_pci(M_t)
        eci_scores = eci[ci_labels]   # higher ECI → more likely to diversify
        r = evaluate_baseline(f"ECI    (year={eval_year})", eci_scores, y_true)
        r['year'] = eval_year; r['split'] = split_name
        all_results.append(r)

        # --- PCI baseline (product score only, inverted: low PCI = easy entry) ---
        pci_scores = -pci[pi_labels]   # negate: low complexity → easier to enter
        r = evaluate_baseline(f"PCI    (year={eval_year})", pci_scores, y_true)
        r['year'] = eval_year; r['split'] = split_name
        all_results.append(r)

        # --- Combined: ECI + Density (simple additive ensemble) ---
        # Normalise both to [0,1] first
        def minmax(x):
            rng = x.max() - x.min() + 1e-9
            return (x - x.min()) / rng

        combined = minmax(eci_scores) + minmax(dens_scores)
        r = evaluate_baseline(f"ECI+D  (year={eval_year})", combined, y_true)
        r['year'] = eval_year; r['split'] = split_name
        all_results.append(r)

# ─── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — Baseline Results")
print("=" * 60)
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

out_file = os.path.join(DATA_DIR, "baseline_results.csv")
results_df.to_csv(out_file, index=False)
print(f"\nResults saved to {out_file}")
print("\nBaseline computation complete.")
