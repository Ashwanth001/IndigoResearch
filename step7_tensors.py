import pandas as pd
import torch
import pickle
import os

# Define directories
output_dir = "data"
country_feat_file = os.path.join(output_dir, "country_features.csv")
product_feat_file = os.path.join(output_dir, "product_features.csv")
country_map_file = os.path.join(output_dir, "country_mapping.pkl")
product_map_file = os.path.join(output_dir, "product_mapping.pkl")

country_x_output = os.path.join(output_dir, "country_x_by_year.pt")
product_x_output = os.path.join(output_dir, "product_x_by_year.pt")

print(f"Starting Step 7: Creating Node Feature Tensors...")

if not all(os.path.exists(f) for f in [country_feat_file, product_feat_file, country_map_file, product_map_file]):
    print("Error: Required input files not found. Did you run previous steps?")
    exit(1)

# Load features and mappings
country_feat = pd.read_csv(country_feat_file)
product_feat = pd.read_csv(product_feat_file)

with open(country_map_file, "rb") as f:
    country_map = pickle.load(f)
with open(product_map_file, "rb") as f:
    product_map = pickle.load(f)

# Sort based on indices to ensure tensors match edge index indices
country_x_by_year = {}
product_x_by_year = {}

years = sorted(country_feat['year'].unique())

for year in years:
    # Country features for this year
    # We must ensure that row 'i' of our tensor is for country with index 'i'
    year_country = country_feat[country_feat['year'] == year].copy()
    year_country['idx'] = year_country['country'].map(country_map['to_idx'] or -1)
    # Filter out any that weren't in the mapping (though they should all be there)
    year_country = year_country.sort_values('idx')
    
    # Feature columns: log_export, n_products, avg_rca, max_rca
    country_x = torch.tensor(
        year_country[['log_export', 'n_products', 'avg_rca', 'max_rca']].values,
        dtype=torch.float
    )
    country_x_by_year[year] = country_x
    
    # Product features for this year
    year_product = product_feat[product_feat['year'] == year].copy()
    year_product['idx'] = year_product['product'].map(product_map['to_idx'] or -1)
    year_product = year_product.sort_values('idx')
    
    # Feature columns: log_world_export, ubiquity, avg_rca
    product_x = torch.tensor(
        year_product[['log_world_export', 'ubiquity', 'avg_rca']].values,
        dtype=torch.float
    )
    product_x_by_year[year] = product_x

# Save
print(f"Saving feature tensors to {country_x_output} and {product_x_output}...")
torch.save(country_x_by_year, country_x_output)
torch.save(product_x_by_year, product_x_output)

# Validation Statistics
print("\nValidation Statistics for Step 7:")
print(f"Num years: {len(years)}")
sample_year = years[0]
print(f"Country feature shape (year {sample_year}): {country_x_by_year[sample_year].shape}")
print(f"Product feature shape (year {sample_year}): {product_x_by_year[sample_year].shape}")
