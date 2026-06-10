import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle
import os

# Set aesthetic style
plt.style.use('dark_background')
plt.rcParams.update({'font.size': 22}) # Using large font as requested

def visualize_data_grouping():
    print("Loading data for grouping visualization...")
    data_dir = "data"
    train_path = os.path.join(data_dir, "train_data.pt")
    country_map_path = os.path.join(data_dir, "country_mapping.pkl")
    product_map_path = os.path.join(data_dir, "product_mapping.pkl")

    # Load data
    train_data = torch.load(train_path, weights_only=False)
    with open(country_map_path, "rb") as f:
        country_map = pickle.load(f)
    with open(product_map_path, "rb") as f:
        product_map = pickle.load(f)

    # Use the first sample's last snapshot for features
    sample = train_data[0]
    last_snapshot = sample['snapshots'][-1]
    
    country_x = last_snapshot['country'].x.numpy()
    product_x = last_snapshot['product'].x.numpy()
    
    print(f"Country features shape: {country_x.shape}")
    print(f"Product features shape: {product_x.shape}")

    # t-SNE reduction
    print("Running t-SNE (this might take a moment)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    
    # Countries
    c_tsne = tsne.fit_transform(country_x)
    # Products (might be many, let's take a sample if too large, or just run it)
    if product_x.shape[0] > 1000:
        indices = np.random.choice(product_x.shape[0], 1000, replace=False)
        p_tsne = tsne.fit_transform(product_x[indices])
    else:
        p_tsne = tsne.fit_transform(product_x)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # 1. Countries
    axes[0].scatter(c_tsne[:, 0], c_tsne[:, 1], c='#00d2ff', s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
    axes[0].set_title("Country Feature Clusters (t-SNE)", fontsize=28, fontweight='bold', pad=25)
    
    # Annotate some countries
    idx_to_country = country_map['to_name']
    for i in np.random.choice(len(c_tsne), 15, replace=False):
        axes[0].annotate(idx_to_country[i], (c_tsne[i, 0], c_tsne[i, 1]), fontsize=20, fontweight='bold', alpha=0.9, color='white')

    # 2. Products
    axes[1].scatter(p_tsne[:, 0], p_tsne[:, 1], c='#ff006e', s=50, alpha=0.6, edgecolors='white', linewidth=0.3)
    axes[1].set_title("Product Feature Clusters (t-SNE)", fontsize=28, fontweight='bold', pad=25)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout()
    output_path = "data_grouping_clusters.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Cluster visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_data_grouping()
