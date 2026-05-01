import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, ndcg_score
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_pr_auc(y_true, y_scores):
    """
    Tier 3: Global Investor Perspective
    Calculates the Precision-Recall Area Under Curve.
    PR-AUC is more robust for highly imbalanced datasets where the negative class is dominant.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def calculate_ndcg_at_k(group, k=20):
    """
    Calculates NDCG@k for a specific country group.
    """
    if len(group) < 2 or group['label'].sum() == 0:
        return np.nan
    
    # sklearn's ndcg_score expects 2D arrays
    y_true = np.asarray([group['label']])
    y_score = np.asarray([group['score']])
    
    return ndcg_score(y_true, y_score, k=k)

def calculate_precision_at_k(group, k=20):
    """
    Calculates Precision@k for a specific country group.
    """
    if len(group) == 0:
        return np.nan
    
    top_k = group.sort_values('score', ascending=False).head(k)
    return top_k['label'].mean()

def calculate_weighted_recall(merged_df, product_pci_dict):
    """
    Tier 2: Economic Value Perspective (Complexity-Weighted)
    Formula: Σ(Correct Predictions * PCI) / Σ(All Actual New Products * PCI)
    
    We shift PCI values to be non-negative (min=0) to ensure they act as positive 'value' weights.
    """
    # Map PCI values to the dataframe
    # If product_pci_dict is a simple dict of {prod_id: pci}
    merged_df['pci'] = merged_df['product'].map(product_pci_dict)
    
    # Handle missing PCI values (fill with mean or 0)
    if merged_df['pci'].isnull().any():
        mean_pci = merged_df['pci'].mean()
        merged_df['pci'] = merged_df['pci'].fillna(mean_pci if not np.isnan(mean_pci) else 0)

    # Shift PCI to be non-negative if it contains negative values (it's often Z-scored)
    min_pci = merged_df['pci'].min()
    if min_pci < 0:
        merged_df['pci_weight'] = merged_df['pci'] - min_pci
    else:
        merged_df['pci_weight'] = merged_df['pci']

    # Filter for actual positives (ground truth = 1)
    actual_positives = merged_df[merged_df['label'] == 1]
    total_weighted_value = (actual_positives['pci_weight']).sum()
    
    if total_weighted_value == 0:
        return 0.0

    # For 'Correct Predictions', we usually need to define a threshold or look at top-k.
    # However, 'Weighted Recall' in this context usually refers to the entire probability-weighted sum
    # or the recall of the model's top predictions.
    # The user says: "Instead of every True Positive counting as '1', we weight successful predictions".
    # This implies we look at the model's hits. Let's assume Top-50 or a high-confidence threshold.
    # Standard recall uses a threshold. Let's use a standard threshold of 0.5 or 
    # let's assume 'Correct Predictions' refers to all samples where label=1, 
    # but the metric is often calculated at a specific cutoff.
    
    # A more common way to do 'Weighted Recall' in IR is to use the score as a weight 
    # but that's not what was asked.
    # If we treat it as a global metric:
    # Let's define 'Correct Predictions' as those with label=1 AND score > 0.5 (or top-K)
    # But usually, it's calculated as a curve or at a specific K.
    # Given Tier 1 uses K=20, let's use a global top-K or a threshold.
    
    # I'll implement it as: Recall of the top 5% of global predictions (common for sparse trade data)
    # or just use a threshold of 0.5.
    threshold = 0.5
    correct_predictions = merged_df[(merged_df['label'] == 1) & (merged_df['score'] >= threshold)]
    successful_weighted_value = correct_predictions['pci_weight'].sum()
    
    return successful_weighted_value / total_weighted_value

def universal_evaluation(predictions_df, ground_truth_df, product_pci_dict, k=20):
    """
    The Universal Evaluator for GNNs and Economic Baselines.
    
    predictions_df: [country, product, score] (and optionally year)
    ground_truth_df: [country, product, label] (and optionally year)
    product_pci_dict: {product_id: pci_value}
    """
    logger.info("Starting Universal Evaluation...")
    
    # 1. Merge predictions and ground truth
    # Use year in join if available
    join_cols = ['country', 'product']
    if 'year' in predictions_df.columns and 'year' in ground_truth_df.columns:
        join_cols.append('year')
        
    merged = pd.merge(predictions_df, ground_truth_df, on=join_cols)
    
    if merged.empty:
        logger.error("Merge resulted in an empty DataFrame. Check country/product IDs.")
        return {}

    # --- Tier 3: Global Investor Perspective ---
    global_pr_auc = calculate_pr_auc(merged['label'], merged['score'])
    logger.info(f"Tier 3: Global PR-AUC = {global_pr_auc:.4f}")

    # --- Tier 1: Minister of Economy Perspective ---
    # Group by country and calculate per-country metrics
    # We use a loop or include_groups=False to avoid future warnings
    country_ndcg_list = []
    country_precision_list = []
    
    for _, group in merged.groupby('country'):
        country_ndcg_list.append(calculate_ndcg_at_k(group, k=k))
        country_precision_list.append(calculate_precision_at_k(group, k=k))
    
    macro_ndcg = np.nanmean(country_ndcg_list)
    macro_precision = np.nanmean(country_precision_list)
    
    logger.info(f"Tier 1: Macro NDCG@{k} = {macro_ndcg:.4f}")
    logger.info(f"Tier 1: Macro Precision@{k} = {macro_precision:.4f}")

    # --- Tier 2: Economic Value Perspective ---
    weighted_recall = calculate_weighted_recall(merged, product_pci_dict)
    logger.info(f"Tier 2: Complexity-Weighted Recall = {weighted_recall:.4f}")

    return {
        "Global PR-AUC": global_pr_auc,
        f"Macro NDCG@{k}": macro_ndcg,
        f"Macro Precision@{k}": macro_precision,
        "Complexity-Weighted Recall": weighted_recall,
        "n_countries": len([x for x in country_ndcg_list if not np.isnan(x)]),
        "n_samples": len(merged)
    }

if __name__ == "__main__":
    # Example usage / Test block
    print("Running evaluator.py self-test...")
    
    # Mock data
    test_preds = pd.DataFrame({
        'country': ['A', 'A', 'B', 'B'],
        'product': [1, 2, 1, 2],
        'score': [0.9, 0.1, 0.4, 0.8]
    })
    test_truth = pd.DataFrame({
        'country': ['A', 'A', 'B', 'B'],
        'product': [1, 2, 1, 2],
        'label': [1, 0, 0, 1]
    })
    test_pci = {1: 2.5, 2: 0.5}
    
    results = universal_evaluation(test_preds, test_truth, test_pci, k=1)
    print("\nResults:", results)
