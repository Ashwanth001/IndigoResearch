import pandas as pd
import numpy as np
import os

# Define directories
output_dir = "data"
input_file = os.path.join(output_dir, "rca_cpt.csv")
country_output = os.path.join(output_dir, "country_features.csv")
product_output = os.path.join(output_dir, "product_features.csv")

print(f"Starting Step 5: Creating Node Features...")

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Did you run Step 2?")
    exit(1)

# Load RCA aggregated exports
exports = pd.read_csv(input_file)

# Country Features per year
print("Creating country features per year...")
country_features = []
countries = sorted(exports['country'].unique())
years = sorted(exports['year'].unique())

for year in years:
    year_data = exports[exports['year'] == year]
    # Total export value (log scale) per country
    c_totals = year_data.groupby('country')['value'].sum()
    # Number of products with RCA >= 1
    c_rca_count = year_data[year_data['rca'] >= 1].groupby('country').size()
    # Average RCA across all products
    c_avg_rca = year_data.groupby('country')['rca'].mean()
    # Max RCA
    c_max_rca = year_data.groupby('country')['rca'].max()

    for c in countries:
        features = {
            'year': year,
            'country': c,
            'log_export': np.log1p(c_totals.get(c, 0)),
            'n_products': c_rca_count.get(c, 0),
            'avg_rca': c_avg_rca.get(c, 0),
            'max_rca': c_max_rca.get(c, 0),
        }
        country_features.append(features)

country_feat_df = pd.DataFrame(country_features)

# Normalize country features (z-score per year)
print("Normalizing country features...")
for col in ['log_export', 'n_products', 'avg_rca', 'max_rca']:
    country_feat_df[col] = country_feat_df.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

country_feat_df.to_csv(country_output, index=False)

# Product Features per year
print("Creating product features per year...")
product_features = []
products = sorted(exports['product'].unique())

for year in years:
    year_data = exports[exports['year'] == year]
    # World export value for this product
    p_totals = year_data.groupby('product')['value'].sum()
    # Ubiquity: count countries with RCA >= 1
    p_ubiquity = year_data[year_data['rca'] >= 1].groupby('product').size()
    # Average RCA across countries for this product
    p_avg_rca = year_data.groupby('product')['rca'].mean()

    for p in products:
        features = {
            'year': year,
            'product': p,
            'log_world_export': np.log1p(p_totals.get(p, 0)),
            'ubiquity': p_ubiquity.get(p, 0),
            'avg_rca': p_avg_rca.get(p, 0),
        }
        product_features.append(features)

product_feat_df = pd.DataFrame(product_features)

# Normalize product features (z-score per year)
print("Normalizing product features...")
for col in ['log_world_export', 'ubiquity', 'avg_rca']:
    product_feat_df[col] = product_feat_df.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

product_feat_df.to_csv(product_output, index=False)

# Validation Statistics
print("\nValidation Statistics for Step 5:")
print(f"Country Features: {len(country_feat_df)} rows, saved to {country_output}")
print(f"Product Features: {len(product_feat_df)} rows, saved to {product_output}")
print(f"Unique features (country): {['log_export', 'n_products', 'avg_rca', 'max_rca']}")
print(f"Unique features (product): {['log_world_export', 'ubiquity', 'avg_rca']}")
