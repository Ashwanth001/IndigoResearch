import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from models import BipartiteEncoder, TemporalBipartiteGNN, LinkPredictor
from utils_metrics import binary_auc, precision_at_k, recall_at_k
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)

class TemporalDataset(Dataset):
    def __init__(self, data_list):
        # Handle case where val/test are single dicts instead of lists
        if isinstance(data_list, dict):
            self.data = [data_list]
        else:
            self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Since batch_size=1, just return the first element
    return batch[0]

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load data
    print("Loading data...")
    train_data = torch.load('data/train_data.pt', weights_only=False)
    val_data = torch.load('data/val_data.pt', weights_only=False)
    test_data = torch.load('data/test_data.pt', weights_only=False)

    # To fix to_hetero node update issues, we must make snapshots undirected
    import torch_geometric.transforms as T
    to_undirected = T.ToUndirected()

    def make_undirected(data_list):
        if isinstance(data_list, dict):
            # Val/Test
            data_list['snapshots'] = [to_undirected(s) for s in data_list['snapshots']]
        else:
            # Train
            for item in data_list:
                item['snapshots'] = [to_undirected(s) for s in item['snapshots']]

    print("Converting snapshots to undirected for training...")
    make_undirected(train_data)
    make_undirected(val_data)
    make_undirected(test_data)

    train_loader = DataLoader(TemporalDataset(train_data), batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(TemporalDataset(val_data), batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(TemporalDataset(test_data), batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Infer input dimensions
    sample_snapshot = train_data[0]['snapshots'][0]
    country_in_dim = sample_snapshot['country'].x.size(1)
    product_in_dim = sample_snapshot['product'].x.size(1)
    metadata = sample_snapshot.metadata()
    
    print(f"Metadata (Undirected): {metadata}")
    print(f"Input dims: Country={country_in_dim}, Product={product_in_dim}")

    # Step 2: Initialize model
    hidden_dim = 128
    temp_hidden_dim = 128
    
    encoder = BipartiteEncoder(country_in_dim, product_in_dim, hidden_dim=hidden_dim, metadata=metadata)
    model = TemporalBipartiteGNN(encoder, hidden_dim=hidden_dim, temporal_hidden_dim=temp_hidden_dim)
    predictor = LinkPredictor(in_dim=temp_hidden_dim * 2, hidden_dim=128)

    model = model.to(device)
    predictor = predictor.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), 
        lr=1e-3, weight_decay=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_auc = 0.0

    print("Starting training loop...")
    for epoch in range(1, 101):
        # TRAIN
        model.train()
        predictor.train()
        total_loss = 0
        
        for batch in train_loader:
            snapshots = [s.to(device) for s in batch['snapshots']]
            labels = batch['labels']
            edge_label_index = labels['edge_label_index'].to(device)
            edge_label = labels['edge_label'].to(device)

            optimizer.zero_grad()
            z_dict = model(snapshots)
            logits = predictor(z_dict['country'], z_dict['product'], edge_label_index)
            
            loss = criterion(logits, edge_label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # VALIDATE
        model.eval()
        predictor.eval()
        val_auc, val_p50, val_r50 = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                snapshots = [s.to(device) for s in batch['snapshots']]
                labels = batch['labels']
                edge_label_index = labels['edge_label_index'].to(device)
                edge_label = labels['edge_label']

                z_dict = model(snapshots)
                logits = predictor(z_dict['country'], z_dict['product'], edge_label_index)
                probs = torch.sigmoid(logits).cpu().numpy()
                y_true = edge_label.numpy()
                
                # Metrics
                val_auc = binary_auc(y_true, probs)
                val_p50 = precision_at_k(y_true, probs, k=50, edge_label_index=labels['edge_label_index'].numpy())
                val_r50 = recall_at_k(y_true, probs, k=50, edge_label_index=labels['edge_label_index'].numpy())

        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val P@50: {val_p50:.4f} | Val R@50: {val_r50:.4f}")

        # Save Best
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model': model.state_dict(),
                'predictor': predictor.state_dict(),
                'epoch': epoch,
                'val_auc': best_val_auc
            }, os.path.join(checkpoint_dir, 'best_model.pt'))

    # TEST
    print("\nTraining complete. Evaluating on test set...")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model'])
    predictor.load_state_dict(checkpoint['predictor'])
    
    model.eval()
    predictor.eval()
    test_auc, test_p50, test_r50 = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            snapshots = [s.to(device) for s in batch['snapshots']]
            labels = batch['labels']
            edge_label_index = labels['edge_label_index'].to(device)
            edge_label = labels['edge_label']

            z_dict = model(snapshots)
            logits = predictor(z_dict['country'], z_dict['product'], edge_label_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true = edge_label.numpy()
            
            test_auc = binary_auc(y_true, probs)
            test_p50 = precision_at_k(y_true, probs, k=50, edge_label_index=labels['edge_label_index'].numpy())
            test_r50 = recall_at_k(y_true, probs, k=50, edge_label_index=labels['edge_label_index'].numpy())

    print(f"BEST VAL AUC EPOCH: {checkpoint['epoch']}")
    print(f"TEST METRICS: AUC = {test_auc:.4f}, P@50 = {test_p50:.4f}, R@50 = {test_r50:.4f}")

if __name__ == "__main__":
    train()
