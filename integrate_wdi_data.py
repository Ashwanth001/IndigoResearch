import pandas as pd
import numpy as np
import os
import torch
import pickle
from torch_geometric.data import HeteroData

# Paths
wdi_csv_path = r"c:\Users\Ashwa\Ash_projects\Indigo Research\WDI_csv\WDICSV.csv"
baci_codes_path = r"c:\Users\Ashwa\Ash_projects\Indigo Research\BACIDataset1995\country_codes_V202601.csv"
data_dir = "data"

# Step A — Load and filter WDI data
print("Step A — Loading and filtering WDI data...")
indicators = {
    'NY.GDP.PCAP.KD': 'gdp_pc',
    'NE.GDI.TOTL.ZS': 'capital_formation',
    'SE.TER.ENRR': 'tertiary_enrollment',
    'BX.KLT.DINV.WD.GD.ZS': 'fdi_inflows',
    'NV.IND.MANF.ZS': 'manufacturing_va',
    'IT.NET.USER.ZS': 'internet_users',
    'SP.POP.TOTL': 'population'
}

# The WDICSV.csv file has years as column names form 1960 to 2025
# Low_memory=False to avoid DtypeWarning on large WDI files
df_wdi = pd.read_csv(wdi_csv_path, low_memory=False)

# Filter indicators
df_wdi = df_wdi[df_wdi['Indicator Code'].isin(indicators.keys())]

# Keep only years 1995-2022
year_cols = [str(y) for y in range(1995, 2023)]
id_cols = ['Country Code', 'Indicator Code']
df_wdi = df_wdi[id_cols + year_cols]

# Reshape from wide to long
df_long = df_wdi.melt(id_vars=id_cols, var_name='year', value_name='value')
df_long['year'] = df_long['year'].astype(int)
df_long.rename(columns={'Country Code': 'country_code_iso3', 'Indicator Code': 'indicator_code'}, inplace=True)
print("Step A — Finished.\n")

# Step B — Pivot to one row per country-year
print("Step B — Pivoting and transforming WDI data...")
# Clean values: WDI uses empty strings or strings for missing, ensure numeric
df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

df_pivot = df_long.pivot_table(index=['country_code_iso3', 'year'], 
                              columns='indicator_code', 
                              values='value').reset_index()

# Ensure all indicators exist as columns even if missing in raw data
for code in indicators.keys():
    if code not in df_pivot.columns:
        df_pivot[code] = np.nan

# Rename columns
df_pivot.rename(columns=indicators, inplace=True)

# Apply log transformation to gdp_pc and population
# Use log1p to handle any 0 values gracefully
df_pivot['gdp_pc'] = np.log1p(df_pivot['gdp_pc'])
df_pivot['population'] = np.log1p(df_pivot['population'])
print("Step B — Finished.\n")

# Step C — Match WDI country codes to BACI country codes
print("Step C — Matching WDI country codes to BACI country codes...")
df_baci_codes = pd.read_csv(baci_codes_path)
# BACI file has columns: country_code, country_name, country_iso2, country_iso3
mapping = df_baci_codes[['country_iso3', 'country_code']].dropna()
mapping.columns = ['country_code_iso3', 'baci_country_code']

# Join mapping
df_wdi_matched = df_pivot.merge(mapping, on='country_code_iso3', how='inner')
print(f"Matched {df_wdi_matched['country_code_iso3'].nunique()} countries.")
print("Step C — Finished.\n")

# Step D — Handle missing values
print("Step D — Handling missing values...")
indicators_cols = list(indicators.values())

# Missingness before filling
missing_before = df_wdi_matched[indicators_cols].isna().mean()
for col in indicators_cols:
    if missing_before[col] > 0.20:
        print(f"WARNING: Indicator '{col}' has {missing_before[col]:.1%} missing values.")

# Forward-fill then backward-fill per country
df_wdi_matched = df_wdi_matched.sort_values(['baci_country_code', 'year'])
df_wdi_matched[indicators_cols] = df_wdi_matched.groupby('baci_country_code')[indicators_cols].ffill()
df_wdi_matched[indicators_cols] = df_wdi_matched.groupby('baci_country_code')[indicators_cols].bfill()

# Year-level median for remaining gaps (e.g. countries with no data at all for an indicator)
for col in indicators_cols:
    if df_wdi_matched[col].isna().any():
        medians = df_wdi_matched.groupby('year')[col].transform('median')
        df_wdi_matched[col] = df_wdi_matched[col].fillna(medians)

# Final check
remaining_na = df_wdi_matched[indicators_cols].isna().sum().sum()
print(f"Remaining NaN values: {remaining_na}")
assert remaining_na == 0
print("Step D — Finished.\n")

# Step E — Normalize features
print("Step E — Normalizing features...")
# Z-score normalize each feature per year
for col in indicators_cols:
    df_wdi_matched[col] = df_wdi_matched.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

wdi_features_file = os.path.join(data_dir, "wdi_features.csv")
df_wdi_matched.to_csv(wdi_features_file, index=False)
print(f"Saved WDI features to {wdi_features_file}")
print("Step E — Finished.\n")

# Step F — Merge with existing country features
print("Step F — Merging with existing country features...")
country_feat_file = os.path.join(data_dir, "country_features.csv")
df_baci_feat = pd.read_csv(country_feat_file)

# Merging. 
df_wdi_join = df_wdi_matched.rename(columns={'baci_country_code': 'country'})

# Merge with left join to retain all BACI country-years
df_enriched = df_baci_feat.merge(df_wdi_join[['country', 'year'] + indicators_cols], 
                                 on=['country', 'year'], 
                                 how='left')

# Fill missing with 0 (for countries in BACI but missing in WDI)
df_enriched[indicators_cols] = df_enriched[indicators_cols].fillna(0)

# Save result
enriched_file = os.path.join(data_dir, "country_features_enriched.csv")
df_enriched.to_csv(enriched_file, index=False)
print(f"Saved enriched country features to {enriched_file}")
print("Step F — Finished.\n")

# Step G — Rebuild country feature tensors
print("Step G — Rebuilding country feature tensors...")
with open(os.path.join(data_dir, "country_mapping.pkl"), "rb") as f:
    country_map = pickle.load(f)

country_x_by_year = {}
years = sorted(df_enriched['year'].unique())
all_feat_cols = ['log_export', 'n_products', 'avg_rca', 'max_rca'] + indicators_cols

for year in years:
    year_data = df_enriched[df_enriched['year'] == year].copy()
    year_data['idx'] = year_data['country'].map(country_map['to_idx'])
    
    # Sort by mapped index to match edge_index indices
    year_data = year_data.dropna(subset=['idx']).sort_values('idx')
    
    country_x = torch.tensor(
        year_data[all_feat_cols].values,
        dtype=torch.float
    )
    country_x_by_year[int(year)] = country_x

torch.save(country_x_by_year, os.path.join(data_dir, "country_x_by_year.pt"))
print("Step G — Finished.\n")

# Step H — Rebuild train/val/test .pt files
print("Step H — Rebuilding train/val/test .pt files...")
# Graph structure files
edge_index_by_year = torch.load(os.path.join(data_dir, "edge_index_by_year.pt"), weights_only=False)
product_x_by_year = torch.load(os.path.join(data_dir, "product_x_by_year.pt"), weights_only=False)
with open(os.path.join(data_dir, "product_mapping.pkl"), "rb") as f:
    product_map = pickle.load(f)

def create_hetero_snapshot(year):
    data = HeteroData()
    data['country'].x = country_x_by_year[year]
    data['product'].x = product_x_by_year[year]
    data['country', 'exports', 'product'].edge_index = edge_index_by_year[year]
    return data

def create_temporal_sample(year, labels_df):
    # History: 5 years lookback [t-4, t]
    snapshots = [create_hetero_snapshot(y) for y in range(year-4, year+1)]
    
    # Target: labels for year t+5
    sample_labels = labels_df[labels_df['year'] == year].copy()
    country_indices = sample_labels['country'].map(country_map['to_idx']).values
    product_indices = sample_labels['product'].map(product_map['to_idx']).values
    label_values = sample_labels['label'].values
    
    edge_label_index = torch.tensor([country_indices, product_indices], dtype=torch.long)
    edge_label = torch.tensor(label_values, dtype=torch.float)
    
    return {
        'snapshots': snapshots,
        'labels': {'edge_label_index': edge_label_index, 'edge_label': edge_label},
        'year': int(year)
    }

# Load label splits
train_labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
val_labels = pd.read_csv(os.path.join(data_dir, "val_labels.csv"))
test_labels = pd.read_csv(os.path.join(data_dir, "test_labels.csv"))

# Build updated .pt files
train_samples = [create_temporal_sample(y, train_labels) for y in sorted(train_labels['year'].unique())]
val_sample = create_temporal_sample(2013, val_labels)
test_sample = create_temporal_sample(2015, test_labels)

torch.save(train_samples, os.path.join(data_dir, "train_data.pt"))
torch.save(val_sample, os.path.join(data_dir, "val_data.pt"))
torch.save(test_sample, os.path.join(data_dir, "test_data.pt"))
print("Step H — Finished.\n")

# Step I — Validation checks
print("Step I — Validation checks...")
# 1. Shape check
sample_year = 2010
print(f"Shape of country features tensor for year {sample_year}: {country_x_by_year[sample_year].shape}")

# 2. Match stats
baci_total = df_baci_feat['country'].nunique()
matched = df_wdi_matched['baci_country_code'].nunique()
print(f"Total countries in BACI: {baci_total}")
print(f"Matched countries with WDI data: {matched}")
print(f"BACI countries with no WDI (zero-filled): {baci_total - matched}")

# 3. Missingness report
print("\n% Missingness per indicator before filling:")
for col_name in indicators_cols:
    print(f"{col_name}: {missing_before[col_name]:.1%}")

# 4. File existence checks
for f in ["train_data.pt", "val_data.pt", "test_data.pt"]:
    exists = os.path.exists(os.path.join(data_dir, f))
    print(f"File '{f}' updated: {exists}")

print("\nWDI Integration Pipeline Complete.")
