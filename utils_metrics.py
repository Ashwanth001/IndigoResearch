import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd

def binary_auc(y_true, y_score):
    """Compute ROC-AUC score."""
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5

def precision_at_k(y_true, y_score, k=50, edge_label_index=None):
    """
    Compute Precision@k.
    If edge_label_index is provided, it groups by country (row 0).
    """
    if edge_label_index is None:
        # Global Precision@k if no grouping
        idx = np.argsort(y_score)[::-1][:k]
        return np.mean(y_true[idx])
    
    # Grouped by country
    df = pd.DataFrame({
        'country': edge_label_index[0],
        'label': y_true,
        'score': y_score
    })
    
    precisions = []
    for _, group in df.groupby('country'):
        if group['label'].sum() == 0:
            continue
        
        # Sort by score
        group = group.sort_values('score', ascending=False)
        top_k = group.head(k)
        precisions.append(top_k['label'].mean())
        
    return np.mean(precisions) if precisions else 0.0

def recall_at_k(y_true, y_score, k=50, edge_label_index=None):
    """
    Compute Recall@k.
    If edge_label_index is provided, it groups by country (row 0).
    """
    if edge_label_index is None:
        # Global Recall@k if no grouping
        total_positives = np.sum(y_true)
        if total_positives == 0: return 0.0
        idx = np.argsort(y_score)[::-1][:k]
        return np.sum(y_true[idx]) / total_positives
    
    # Grouped by country
    df = pd.DataFrame({
        'country': edge_label_index[0],
        'label': y_true,
        'score': y_score
    })
    
    recalls = []
    for _, group in df.groupby('country'):
        total_pos = group['label'].sum()
        if total_pos == 0:
            continue
        
        # Sort by score
        group = group.sort_values('score', ascending=False)
        top_k = group.head(k)
        recalls.append(top_k['label'].sum() / total_pos)
        
    return np.mean(recalls) if recalls else 0.0
