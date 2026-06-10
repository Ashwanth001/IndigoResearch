import torch
import pandas as pd
import numpy as np
import pickle
import os
import torch_geometric.transforms as T
from models import BipartiteEncoder, TemporalBipartiteGNN, LinkPredictor
from evaluator import universal_evaluation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
CHECKPOINT_PATH = r"checkpoints\GNN\best_model.pt"
TEST_DATA_PATH  = r"data\GNNModelTraining\test_data.pt"
SMOOTH_FILE     = r"data\M_cpt_smoothed.csv"
COUNTRY_MAP     = r"data\country_mapping.pkl"
PRODUCT_MAP     = r"data\product_mapping.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_pci_for_year(year=2015):
    """Borrow logic from baselines.py to compute PCI for the test year."""
    logger.info(f"Computing PCI weights for year {year}...")
    df = pd.read_csv(SMOOTH_FILE)
    yr_df = df[df['year'] == year]
    
    countries = sorted(df['country'].unique())
    products = sorted(df['product'].unique())
    C, P = len(countries), len(products)
    
    country_idx = {c: i for i, c in enumerate(countries)}
    product_idx = {p: i for i, p in enumerate(products)}
    
    M = np.zeros((C, P), dtype=np.float32)
    ci = yr_df['country'].map(country_idx).values
    pi = yr_df['product'].map(product_idx).values
    M[ci, pi] = 1.0
    
    # Method of Reflections (20 iterations)
    kc = M.sum(axis=1)
    kp = M.sum(axis=0)
    kc_safe = np.where(kc > 0, kc, 1.0)
    kp_safe = np.where(kp > 0, kp, 1.0)
    kc_n = kc.copy().astype(float)
    kp_n = kp.copy().astype(float)
    
    for _ in range(20):
        kc_new = (1.0 / kc_safe) * (M @ kp_n)
        kp_new = (1.0 / kp_safe) * (M.T @ kc_n)
        kc_n, kp_n = kc_new, kp_new
    
    # Normalise
    pci = (kp_n - kp_n.mean()) / (kp_n.std() + 1e-9)
    
    # Return as dict: {product_name: pci_value}
    return {products[i]: pci[i] for i in range(P)}

def evaluate_gnn():
    # 1. Load Mappings
    with open(COUNTRY_MAP, "rb") as f:
        country_map = pickle.load(f)
    with open(PRODUCT_MAP, "rb") as f:
        product_map = pickle.load(f)
    
    # 2. Load Model
    logger.info(f"Loading GNN model from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    # Load test data to infer dims
    test_batch = torch.load(TEST_DATA_PATH, weights_only=False)
    if isinstance(test_batch, list):
        test_batch = test_batch[0]
        
    to_undirected = T.ToUndirected()
    sample_snap = to_undirected(test_batch['snapshots'][0])
    country_in_dim = sample_snap['country'].x.size(1)
    product_in_dim = sample_snap['product'].x.size(1)
    metadata = sample_snap.metadata()
    
    hidden_dim = 128
    temp_hidden_dim = 128
    
    encoder = BipartiteEncoder(country_in_dim, product_in_dim, hidden_dim=hidden_dim, metadata=metadata)
    model = TemporalBipartiteGNN(encoder, hidden_dim=hidden_dim, temporal_hidden_dim=temp_hidden_dim)
    predictor = LinkPredictor(in_dim=temp_hidden_dim * 2, hidden_dim=128)
    
    model.load_state_dict(checkpoint['model'])
    predictor.load_state_dict(checkpoint['predictor'])
    model.to(DEVICE).eval()
    predictor.to(DEVICE).eval()
    
    # 3. Perform Inference
    logger.info("Running model inference on test set...")
    to_undirected = T.ToUndirected()
    snapshots = [to_undirected(s).to(DEVICE) for s in test_batch['snapshots']]
    labels = test_batch['labels']
    edge_label_index = labels['edge_label_index'].to(DEVICE)
    y_true = labels['edge_label'].cpu().numpy()
    
    with torch.no_grad():
        z_dict = model(snapshots)
        logits = predictor(z_dict['country'], z_dict['product'], edge_label_index)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    # 4. Prepare DataFrames for Evaluator
    # We need to map indices back to names for Tier 1 evaluation
    c_indices = edge_label_index[0].cpu().numpy()
    p_indices = edge_label_index[1].cpu().numpy()
    
    c_names = [country_map['to_name'][idx] for idx in c_indices]
    p_names = [product_map['to_name'][idx] for idx in p_indices]
    
    predictions_df = pd.DataFrame({
        'country': c_names,
        'product': p_names,
        'score': probs,
        'year': test_batch['year']
    })
    
    ground_truth_df = pd.DataFrame({
        'country': c_names,
        'product': p_names,
        'label': y_true,
        'year': test_batch['year']
    })
    
    # 5. Get PCI weights
    product_pci_dict = compute_pci_for_year(test_batch['year'])
    
    # 6. Final Tiered Evaluation
    results = universal_evaluation(predictions_df, ground_truth_df, product_pci_dict, k=20)
    
    # 7. Plot Global PR-AUC Curve
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc
    
    logger.info("Plotting Global PR-AUC Curve...")
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=3, label=f'GNN (AUC = {pr_auc:.4f})')
    
    # Baseline: random guessing precision
    baseline_precision = y_true.sum() / len(y_true)
    plt.axhline(y=baseline_precision, color='red', linestyle='--', lw=2, label=f'Baseline ({baseline_precision:.4f})')
    
    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.title('Global Precision-Recall Curve (Tier 3)', fontsize=22)
    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.3)
    
    plot_path = "gnn_pr_auc_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"PR-AUC curve saved to {plot_path}")
    
    # Summary Print
    print("\n" + "="*50)
    print("GNN 3-TIERED EVALUATION RESULTS")
    print("="*50)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:30}: {value:.4f}")
        else:
            print(f"{metric:30}: {value}")
    print("="*50)

if __name__ == "__main__":
    evaluate_gnn()
