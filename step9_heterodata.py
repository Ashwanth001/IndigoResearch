import torch
import pandas as pd
import pickle
import os
from torch_geometric.data import HeteroData

# Define directories
output_dir = "data"
edge_file = os.path.join(output_dir, "edge_index_by_year.pt")
country_x_file = os.path.join(output_dir, "country_x_by_year.pt")
product_x_file = os.path.join(output_dir, "product_x_by_year.pt")
country_map_file = os.path.join(output_dir, "country_mapping.pkl")
product_map_file = os.path.join(output_dir, "product_mapping.pkl")

train_labels_file = os.path.join(output_dir, "train_labels.csv")
val_labels_file = os.path.join(output_dir, "val_labels.csv")
test_labels_file = os.path.join(output_dir, "test_labels.csv")

train_data_output = os.path.join(output_dir, "train_data.pt")
val_data_output = os.path.join(output_dir, "val_data.pt")
test_data_output = os.path.join(output_dir, "test_data.pt")

print(f"Starting Step 9: Creating PyTorch Geometric HeteroData Objects...")

def create_hetero_snapshot(year, country_x_by_year, product_x_by_year, edge_index_by_year):
    """Create a single HeteroData object for one year."""
    data = HeteroData()
    # Node features
    data['country'].x = country_x_by_year[year]
    data['product'].x = product_x_by_year[year]
    # Edges (country → product)
    data['country', 'exports', 'product'].edge_index = edge_index_by_year[year]
    return data

def create_temporal_batch(year, labels_df, country_x_by_year, product_x_by_year, edge_index_by_year, country_map, product_map):
    """
    Create a single predictive sample: 5-year history + future labels at year+5.
    (Self-note: labels at t+5 were generated from year 'year')
    """
    # History: 5 consecutive years ending at 'year'
    snapshots = [create_hetero_snapshot(y, country_x_by_year, product_x_by_year, edge_index_by_year) 
                 for y in range(year-4, year+1)]
    
    # Labels for year+5 (predicting what happens with labels marked for this 'year')
    sample_labels = labels_df[labels_df['year'] == year].copy()
    
    # Map country/product names to indices (must match feature/edge indexing)
    country_idx = sample_labels['country'].map(country_map['to_idx']).values
    product_idx = sample_labels['product'].map(product_map['to_idx']).values
    label_values = sample_labels['label'].values
    
    # edge_label_index: [2, num_labeled_edges]
    edge_label_index = torch.tensor([country_idx, product_idx], dtype=torch.long)
    edge_label = torch.tensor(label_values, dtype=torch.float)
    
    labels_dict = {
        'edge_label_index': edge_label_index,
        'edge_label': edge_label
    }
    
    return {'snapshots': snapshots, 'labels': labels_dict, 'year': int(year)}

# Load all data
print("Loading all node features and edges...")
edge_index_by_year = torch.load(edge_file, weights_only=False)
country_x_by_year = torch.load(country_x_file, weights_only=False)
product_x_by_year = torch.load(product_x_file, weights_only=False)

with open(country_map_file, "rb") as f:
    country_map = pickle.load(f)
with open(product_map_file, "rb") as f:
    product_map = pickle.load(f)

# Load labels split
train_labels = pd.read_csv(train_labels_file)
val_labels = pd.read_csv(val_labels_file)
test_labels = pd.read_csv(test_labels_file)

# Create training samples
print("Processing training samples...")
train_samples = []
for year in sorted(train_labels['year'].unique()):
    print(f"Creating sample for year {year} (requires history {year-4}-{year})...")
    sample = create_temporal_batch(year, train_labels, country_x_by_year, product_x_by_year, edge_index_by_year, country_map, product_map)
    train_samples.append(sample)

# Create validation sample
print("Processing validation sample (2013)...")
val_sample = create_temporal_batch(2013, val_labels, country_x_by_year, product_x_by_year, edge_index_by_year, country_map, product_map)

# Create test sample
print("Processing test sample (2015)...")
test_sample = create_temporal_batch(2015, test_labels, country_x_by_year, product_x_by_year, edge_index_by_year, country_map, product_map)

# Save
print(f"Saving final .pt data objects to {train_data_output}, {val_data_output}, and {test_data_output}...")
torch.save(train_samples, train_data_output)
torch.save(val_sample, val_data_output)
torch.save(test_sample, test_data_output)

# Validation Statistics
print("\nValidation Statistics for Step 9:")
print(f"Total training years: {len(train_samples)}")
print(f"Validation target year: {val_sample['year']}")
print(f"Test target year: {test_sample['year']}")
sample_year = train_samples[0]['year']
print(f"Sample (year {sample_year}): {len(train_samples[0]['snapshots'])} graph snapshots.")
print(f"Labeled edges for year {sample_year}: {train_samples[0]['labels']['edge_label'].shape[0]}")
