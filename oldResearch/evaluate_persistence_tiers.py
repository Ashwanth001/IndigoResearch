import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from evaluator import universal_evaluation
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = "data"
RCA_FILE = os.path.join(DATA_DIR, "rca_cpt.csv")
TEST_LABELS_FILE = os.path.join(DATA_DIR, "GNNModelTraining", "test_labels.csv")
SMOOTH_FILE = os.path.join(DATA_DIR, "M_cpt_smoothed.csv")
TEST_YEAR = 2015

def evaluate_persistence():
    logger.info("Starting 3-Tiered Evaluation for Persistence Baseline...")
    
    # 1. Load test labels
    logger.info("Loading test labels...")
    test_labels = pd.read_csv(TEST_LABELS_FILE)
    relevant_pairs = test_labels[['country', 'product']].drop_duplicates()
    
    # 2. Extract RCA history (2015, 2014, 2013)
    history_years = [TEST_YEAR, TEST_YEAR-1, TEST_YEAR-2]
    logger.info(f"Extracting RCA history for years {history_years}...")
    
    # Check total size for progress bar (28M lines estimated from notebook)
    chunks = pd.read_csv(RCA_FILE, chunksize=1000000)
    filtered_list = []
    
    for chunk in tqdm(chunks, total=28, desc="Filtering RCA history"):
        # Filter for the relevant years first to save memory
        chunk_years = chunk[chunk['year'].isin(history_years)]
        # Filter for the relevant (country, product) pairs
        filtered_list.append(chunk_years.merge(relevant_pairs, on=['country', 'product']))
    
    rca_filtered = pd.concat(filtered_list)
    logger.info(f"Filtered RCA history size: {len(rca_filtered)} rows")
    
    # 3. Pivot to wide format to calculate score
    logger.info("Calculating Persistence Scores...")
    rca_wide = rca_filtered.pivot_table(
        index=['country', 'product'], 
        columns='year', 
        values='rca', 
        fill_value=0
    )
    
    # Ensure all history years are columns even if missing in RCA data
    for yr in history_years:
        if yr not in rca_wide.columns:
            rca_wide[yr] = 0
            
    # Score = Mean of [RCA >= 1] over history years
    rca_wide['score'] = (rca_wide[history_years] >= 1).mean(axis=1)
    
    # 4. Merge scores back into the test set
    final_bench = test_labels.merge(rca_wide[['score']], on=['country', 'product'], how='left').fillna(0)
    
    # 5. Get PCI weights for Tier 2
    # Import the calculation logic from evaluate_gnn_tiers.py
    from evaluate_gnn_tiers import compute_pci_for_year
    product_pci_dict = compute_pci_for_year(TEST_YEAR)
    
    # 6. Final Evaluation
    logger.info("Running universal evaluator...")
    results = universal_evaluation(
        final_bench[['country', 'product', 'score']], 
        test_labels[['country', 'product', 'label']], 
        product_pci_dict,
        k=20
    )
    
    # 7. Plot PR-AUC Curve
    logger.info("Plotting PR-AUC Curve...")
    y_true = test_labels['label'].values
    y_score = final_bench['score'].values
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='cyan', lw=3, label=f'Persistence (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('Persistence Baseline: Global PR Curve', fontsize=22)
    plt.legend(loc='best', fontsize=16)
    plt.grid(alpha=0.3)
    plt.savefig("persistence_pr_auc.png", dpi=300, bbox_inches='tight')
    
    # Summary Table
    print("\n" + "="*50)
    print("PERSISTENCE 3-TIERED EVALUATION RESULTS")
    print("="*50)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:30}: {value:.4f}")
        else:
            print(f"{metric:30}: {value}")
    print("="*50)
    
    # Save results
    res_df = pd.DataFrame([results])
    res_df['Baseline'] = 'Persistence'
    res_df.to_csv("data/persistence_tiered_results.csv", index=False)

if __name__ == "__main__":
    evaluate_persistence()
