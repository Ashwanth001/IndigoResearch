import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import os

# Define directories
output_dir = "data"
train_data_file = os.path.join(output_dir, "train_data.pt")
val_data_file = os.path.join(output_dir, "val_data.pt")
test_data_file = os.path.join(output_dir, "test_data.pt")

class TemporalBipartiteDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of dicts with 'snapshots' and 'labels'
        """
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

print(f"Starting Step 10: Initializing Data Loaders...")

if not os.path.exists(train_data_file):
    print(f"Error: {train_data_file} not found. Did you run Step 9?")
    exit(1)

# Load data
print(f"Loading data from {train_data_file}...")
train_samples = torch.load(train_data_file, weights_only=False)

# Create dataset
train_dataset = TemporalBipartiteDataset(train_samples)

# Create loader (batch_size=1, since each sample is a full year of graphs/labels)
print("Initializing DataLoader (batch_size=1)...")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Example: iterate through one batch
for batch in train_loader:
    snapshots = batch['snapshots']  # list of 5 graphs
    labels = batch['labels']  # dict with edge_label_index and edge_label
    year = batch['year']
    
    print(f"\nExample Batch Information:")
    print(f"Year: {year.item()}")
    print(f"Num snapshots in batch: {len(snapshots)}")
    print(f"Labeled edges (targets): {labels['edge_label'].shape[1] if len(labels['edge_label'].shape) > 1 else labels['edge_label'].shape[0]}")
    print(f"Example Edge Indices (first 5): {labels['edge_label_index'][:, :5]}")
    break

print("\nValidation Statistics for Step 10:")
print(f"DataLoader initialized for {len(train_dataset)} training years.")
print(f"Data objects are ready for model training.")
