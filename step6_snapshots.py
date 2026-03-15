import pandas as pd
import torch
import pickle
import os

# Define directories
output_dir = "data"
input_file = os.path.join(output_dir, "M_cpt_smoothed.csv")
edge_output = os.path.join(output_dir, "edge_index_by_year.pt")
country_map_output = os.path.join(output_dir, "country_mapping.pkl")
product_map_output = os.path.join(output_dir, "product_mapping.pkl")

print(f"Starting Step 6: Building Temporal Graph Snapshots...")

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Did you run Step 3?")
    exit(1)

# Load edges (M=1)
edges = pd.read_csv(input_file)

# Create integer mappings for countries and products
# (Crucial: These mappings must be consistent across all steps and years)
print("Creating node mappings...")
country_list = sorted(edges['country'].unique())
product_list = sorted(edges['product'].unique())

country_to_idx = {c: i for i, c in enumerate(country_list)}
product_to_idx = {p: i for i, p in enumerate(product_list)}

idx_to_country = {i: c for c, i in country_to_idx.items()}
idx_to_product = {i: p for p, i in product_to_idx.items()}

# Save mappings
print(f"Saving mappings to {country_map_output} and {product_map_output}...")
with open(country_map_output, "wb") as f:
    pickle.dump({'to_idx': country_to_idx, 'to_name': idx_to_country}, f)
with open(product_map_output, "wb") as f:
    pickle.dump({'to_idx': product_to_idx, 'to_name': idx_to_product}, f)

# Build edge_index per year
# edge_index: [2, num_edges] for source (country) and target (product)
print("Building edge indexes per year...")
edge_index_by_year = {}
years = sorted(edges['year'].unique())

for year in years:
    year_edges = edges[edges['year'] == year]
    
    # Convert to integer indices
    country_idx = year_edges['country'].map(country_to_idx).values
    product_idx = year_edges['product'].map(product_to_idx).values
    
    # Edge index: [2, num_edges]
    edge_index = torch.tensor([country_idx, product_idx], dtype=torch.long)
    edge_index_by_year[year] = edge_index

# Save as .pt file
print(f"Saving edge indexes to {edge_output}...")
torch.save(edge_index_by_year, edge_output)

# Validation Statistics
print("\nValidation Statistics for Step 6:")
print(f"Output: {edge_output}")
print(f"Total years processed: {len(edge_index_by_year)}")
print(f"Num countries: {len(country_to_idx)}")
print(f"Num products: {len(product_to_idx)}")
sample_year = years[0]
print(f"Snapshot edges (year {sample_year}): {edge_index_by_year[sample_year].shape[1]}")
