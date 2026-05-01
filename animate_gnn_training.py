import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import matplotlib.animation as animation
from models import BipartiteEncoder, TemporalBipartiteGNN, LinkPredictor
import torch.nn.functional as F

# Aestehtic Setup
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 18})

class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        if isinstance(data_list, dict): self.data = [data_list]
        else: self.data = data_list
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch): return batch[0]

def run_animation_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data for animation...")
    train_data = torch.load('data/train_data.pt', weights_only=False)
    val_data = torch.load('data/val_data.pt', weights_only=False)

    # Undirected snapshots for GNN stability
    import torch_geometric.transforms as T
    to_undirected = T.ToUndirected()
    for item in train_data: item['snapshots'] = [to_undirected(s) for s in item['snapshots']]
    val_data['snapshots'] = [to_undirected(s) for s in val_data['snapshots']]

    train_loader = DataLoader(TemporalDataset(train_data), batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Model Setup
    sample_snapshot = train_data[0]['snapshots'][0]
    metadata = sample_snapshot.metadata()
    hidden_dim, temp_hidden_dim = 128, 128
    
    encoder = BipartiteEncoder(sample_snapshot['country'].x.size(1), sample_snapshot['product'].x.size(1), hidden_dim=hidden_dim, metadata=metadata)
    model = TemporalBipartiteGNN(encoder, hidden_dim=hidden_dim, temporal_hidden_dim=temp_hidden_dim)
    predictor = LinkPredictor(in_dim=temp_hidden_dim * 2, hidden_dim=128)

    model.to(device)
    predictor.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Prep for animation
    val_snapshots = [s.to(device) for s in val_data['snapshots']]
    all_embeddings = []
    losses = []
    epochs = 30 # Enough to see movement

    print(f"Starting mini-training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        predictor.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            snapshots = [s.to(device) for s in batch['snapshots']]
            labels = batch['labels']
            z_dict = model(snapshots)
            logits = predictor(z_dict['country'], z_dict['product'], labels['edge_label_index'].to(device))
            loss = criterion(logits, labels['edge_label'].to(device).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        # Capture validation embeddings
        model.eval()
        with torch.no_grad():
            z_val = model(val_snapshots)
            # Combine country and product embeddings for holistic view
            combined = torch.cat([z_val['country'], z_val['product']], dim=0).cpu().numpy()
            all_embeddings.append(combined)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # Animation generation
    print("Generating animation frames...")
    pca = PCA(n_components=2)
    # Fit PCA on the last epoch to get stable axes
    pca.fit(all_embeddings[-1])
    
    num_countries = val_data['snapshots'][0]['country'].x.size(0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update(i):
        ax.clear()
        embed_2d = pca.transform(all_embeddings[i])
        
        c_embed = embed_2d[:num_countries]
        p_embed = embed_2d[num_countries:]
        
        ax.scatter(c_embed[:, 0], c_embed[:, 1], c='#00d2ff', s=100, label='Countries', alpha=0.7, edgecolors='white')
        ax.scatter(p_embed[:, 0], p_embed[:, 1], c='#ff006e', s=50, label='Products', alpha=0.5)
        
        ax.set_title(f"GNN Embedding Evolution (Epoch {i+1})", fontsize=24, fontweight='bold')
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.legend(loc='upper right')
        ax.grid(alpha=0.1)
        
        # Add loss info
        ax.text(0.05, 0.95, f"Loss: {losses[i]:.4f}", transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    ani = animation.FuncAnimation(fig, update, frames=len(all_embeddings), interval=200)
    
    # Save as GIF
    gif_path = "gnn_training_animation.gif"
    try:
        ani.save(gif_path, writer='pillow')
        print(f"Animation saved to {gif_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        # Fallback: Save first and last frame
        plt.figure(figsize=(24, 10))
        plt.subplot(1, 2, 1)
        update(0)
        plt.subplot(1, 2, 2)
        update(len(all_embeddings)-1)
        plt.savefig("model_evolution_static.png")
        print("Static comparison saved to model_evolution_static.png")

if __name__ == "__main__":
    run_animation_training()
