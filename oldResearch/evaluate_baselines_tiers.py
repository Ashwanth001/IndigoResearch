import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from evaluator import universal_evaluation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
SMOOTH_FILE = os.path.join(DATA_DIR, "M_cpt_smoothed.csv")
LABELS_FILE = os.path.join(DATA_DIR, "labels_h5.csv")
TEST_YEAR   = 2015

# ─── Load Data ────────────────────────────────────────────────────────────────
logger.info("Loading baseline data...")
edges_df  = pd.read_csv(SMOOTH_FILE)
labels_df = pd.read_csv(LABELS_FILE)

countries = sorted(edges_df['country'].unique())
products  = sorted(edges_df['product'].unique())
C, P = len(countries), len(products)
country_idx = {c: i for i, c in enumerate(countries)}
product_idx = {p: i for i, p in enumerate(products)}

def build_M(year: int) -> np.ndarray:
    M = np.zeros((C, P), dtype=np.float32)
    yr = edges_df[edges_df['year'] == year]
    ci = yr['country'].map(country_idx).values
    pi = yr['product'].map(product_idx).values
    M[ci, pi] = 1.0
    return M

def compute_eci_pci(M: np.ndarray, n_iter: int = 20):
    kc = M.sum(axis=1); kp = M.sum(axis=0)
    kc_safe = np.where(kc > 0, kc, 1.0); kp_safe = np.where(kp > 0, kp, 1.0)
    kc_n = kc.copy().astype(float); kp_n = kp.copy().astype(float)
    for _ in range(n_iter):
        kc_new = (1.0 / kc_safe) * (M @ kp_n)
        kp_new = (1.0 / kp_safe) * (M.T @ kc_n)
        kc_n, kp_n = kc_new, kp_new
    def normalise(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    return normalise(kc_n), normalise(kp_n)

# Compute Proximity Phi once (from training years <= 2012)
TRAIN_CUTOFF = 2012
train_years = sorted([y for y in edges_df['year'].unique() if y <= TRAIN_CUTOFF])
co_export = np.zeros((P, P), dtype=np.float32)
any_export = np.zeros((P, P), dtype=np.float32)

logger.info(f"Computing proximity matrix from {len(train_years)} years...")
for yr in train_years:
    M = build_M(yr)
    co = M.T @ M
    exp = M.sum(axis=0)
    any_mat = exp[:, None] + exp[None, :] - co
    co_export += co
    any_export += any_mat

phi = np.where(any_export > 0, co_export / (any_export + 1e-9), 0.0)
np.fill_diagonal(phi, 0.0)
phi_row_sum = phi.sum(axis=1)

def density_scores(M_t: np.ndarray) -> np.ndarray:
    numerator = M_t @ phi
    denominator = phi_row_sum[None, :] + 1e-9
    return numerator / denominator

def minmax(x):
    rng = x.max() - x.min() + 1e-9
    return (x - x.min()) / rng

# ─── Main Evaluation ──────────────────────────────────────────────────────────
def evaluate_baselines():
    logger.info(f"Starting 3-Tiered Evaluation for Baselines (Year {TEST_YEAR})...")
    
    # Prep Test Labels
    year_labels = labels_df[labels_df['year'] == TEST_YEAR]
    ci_labels = year_labels['country'].map(country_idx).values
    pi_labels = year_labels['product'].map(product_idx).values
    y_true = year_labels['label'].values
    
    ground_truth_df = pd.DataFrame({
        'country': year_labels['country'],
        'product': year_labels['product'],
        'label': y_true
    })

    # Prepare for results
    baseline_results = []
    plt.figure(figsize=(10, 8))
    
    M_t = build_M(TEST_YEAR)
    eci, pci = compute_eci_pci(M_t)
    product_pci_dict = {products[i]: pci[i] for i in range(P)}

    # Define Baselines
    dens = density_scores(M_t)
    
    baselines = {
        "Density": dens[ci_labels, pi_labels],
        "ECI": eci[ci_labels],
        "PCI": -pci[pi_labels], # Negative PCI because lower complexity is easier
        "ECI+Density": minmax(eci[ci_labels]) + minmax(dens[ci_labels, pi_labels])
    }

    colors = ['green', 'orange', 'purple', 'red']
    
    for (name, scores), color in zip(baselines.items(), colors):
        logger.info(f"Evaluating {name}...")
        
        predictions_df = pd.DataFrame({
            'country': year_labels['country'],
            'product': year_labels['product'],
            'score': scores
        })
        
        # 1. Tiered Metrics
        res = universal_evaluation(predictions_df, ground_truth_df, product_pci_dict, k=20)
        res['Baseline'] = name
        baseline_results.append(res)
        
        # 2. PR-AUC Plotting
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=color, lw=2, label=f'{name} (AUC={pr_auc:.4f})')

    # Add GNN comparison if plot exists (or just plot the baseline curve)
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('Baseline Comparison: Global PR Curve', fontsize=22)
    plt.legend(loc='best', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig("baselines_pr_auc_comparison.png", dpi=300, bbox_inches='tight')
    
    # Summary Table
    results_df = pd.DataFrame(baseline_results)
    # Reorder columns to put Baseline first
    cols = ['Baseline'] + [c for c in results_df.columns if c != 'Baseline']
    results_df = results_df[cols]
    
    print("\n" + "="*80)
    print("BASELINE 3-TIERED EVALUATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    results_df.to_csv("data/baseline_tiered_results.csv", index=False)
    logger.info("Results saved to data/baseline_tiered_results.csv")

if __name__ == "__main__":
    evaluate_baselines()
