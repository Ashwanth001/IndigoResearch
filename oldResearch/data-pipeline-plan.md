# Data Pipeline Plan: BACI to PyTorch Temporal Bipartite GNN

## Overview

This document provides a comprehensive, step-by-step data pipeline to transform raw BACI trade data (1995–2022) into PyTorch Geometric tensors ready for training a temporal bipartite GNN that predicts new RCA activations 5 years ahead.

**Starting Point:** Raw BACI files (latest revision, 1995+)  
**End Point:** PyTorch `.pt` files containing training/validation/test splits of temporal bipartite graphs

---

## Pipeline Architecture

```
Raw BACI CSV files
    ↓
Step 1: Load and aggregate to country-product-year totals
    ↓
Step 2: Compute RCA matrix per year
    ↓
Step 3: Apply 3-year rolling window smoothing to RCA
    ↓
Step 4: Binarize to M[c,p,t] = 1 if smoothed RCA ≥ 1
    ↓
Step 5: Generate positive and negative labels (new activations)
    ↓
Step 6: Create node features (country and product)
    ↓
Step 7: Build temporal graph snapshots (edge_index per year)
    ↓
Step 8: Split into train/val/test windows
    ↓
Step 9: Convert to PyTorch Geometric HeteroData format
    ↓
Step 10: Save as .pt files
```

---

## Step 1: Load and Aggregate BACI Data

### Input Files
- `BACI_HS17_Y*.csv` (yearly files, 1995–2022)
- Columns: `t` (year), `i` (exporter country code), `j` (importer country code), `k` (HS6 product code), `v` (trade value in thousands USD), `q` (quantity)

### Task
Aggregate bilateral flows to total exports per exporter-product-year (sum over all importers):

```python
import pandas as pd
import glob

# Load all yearly BACI files
baci_files = sorted(glob.glob("BACI_HS17_Y*.csv"))
df_list = []

for file in baci_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Concatenate all years
baci = pd.concat(df_list, ignore_index=True)

# Aggregate: X_cpt = sum over j (all importers)
exports = baci.groupby(['t', 'i', 'k'])['v'].sum().reset_index()
exports.columns = ['year', 'country', 'product', 'value']

# Save intermediate result
exports.to_csv("exports_cpt.csv", index=False)
```

### Output
- `exports_cpt.csv` with columns: `year`, `country`, `product`, `value`
- Approximately 28 years × 200 countries × 5000 products = ~28 million rows (sparse, many zero entries will be implicit)

---

## Step 2: Compute RCA Matrix Per Year

### Formula
For each year t:

\[
\text{RCA}_{cpt} = \frac{X_{cpt} / \sum_p X_{cpt}}{\sum_c X_{cpt} / \sum_{c,p} X_{cpt}}
\]

Where:
- \(X_{cpt}\) = exports of country c, product p, year t
- \(\sum_p X_{cpt}\) = total exports of country c in year t
- \(\sum_c X_{cpt}\) = world exports of product p in year t
- \(\sum_{c,p} X_{cpt}\) = total world exports in year t

### Implementation

```python
import numpy as np

# Load aggregated exports
exports = pd.read_csv("exports_cpt.csv")

# Compute totals per year
exports['country_total'] = exports.groupby(['year', 'country'])['value'].transform('sum')
exports['product_total'] = exports.groupby(['year', 'product'])['value'].transform('sum')
exports['world_total'] = exports.groupby('year')['value'].transform('sum')

# Compute RCA
exports['rca'] = (exports['value'] / exports['country_total']) / \
                 (exports['product_total'] / exports['world_total'])

# Drop intermediate columns
exports = exports[['year', 'country', 'product', 'value', 'rca']]

# Save RCA matrix
exports.to_csv("rca_cpt.csv", index=False)
```

### Output
- `rca_cpt.csv` with columns: `year`, `country`, `product`, `value`, `rca`
- Only contains non-zero trade flows; zero flows implicitly have RCA = 0

---

## Step 3: Apply 3-Year Rolling Window Smoothing

### Purpose
Remove one-year noise from crisis years (2009, 2020). A product counts as RCA=1 only if it has RCA ≥ 1 in at least **2 out of 3 consecutive years**.

### Implementation

```python
# Pivot to wide format for rolling window
rca_wide = exports.pivot_table(
    index=['country', 'product'],
    columns='year',
    values='rca',
    fill_value=0
)

# Apply 3-year rolling window: count years with RCA ≥ 1
rca_binary = (rca_wide >= 1).astype(int)
rca_smoothed = rca_binary.rolling(window=3, axis=1, min_periods=2).sum() >= 2

# Convert back to binary (True/False → 1/0)
rca_smoothed = rca_smoothed.astype(int)

# Melt back to long format
rca_smoothed = rca_smoothed.reset_index().melt(
    id_vars=['country', 'product'],
    var_name='year',
    value_name='M'
)

# Filter to only edges where M=1 (to keep data sparse)
edges = rca_smoothed[rca_smoothed['M'] == 1].copy()

# Save binary edge matrix
edges.to_csv("M_cpt_smoothed.csv", index=False)
```

### Output
- `M_cpt_smoothed.csv` with columns: `year`, `country`, `product`, `M` (all M=1)
- This is your **binary country-product matrix** per year (edges only)

---

## Step 4: Generate Positive and Negative Labels

### Definitions
- **Positive (new activation):** \(M_{cp,t} = 0\) AND \(M_{cp,t+5} = 1\) AND \(M_{cp,t+6} = 1\)
  - (Country c does NOT export product p with RCA≥1 at year t, but DOES at t+5 and t+6)
  - The t+6 condition ensures it's a sustained activation, not a one-year fluke
  
- **Negative (no activation):** \(M_{cp,t} = 0\) AND \(M_{cp,t+5} = 0\)
  - Country c does not export product p at t and still doesn't at t+5

### Implementation

```python
# Load smoothed edges
edges = pd.read_csv("M_cpt_smoothed.csv")

# Create a set of (country, product, year) tuples for fast lookup
edge_set = set(zip(edges['country'], edges['product'], edges['year']))

# Get all unique countries and products
countries = sorted(edges['country'].unique())
products = sorted(edges['product'].unique())

# Generate labels for each valid year window
labels = []
h = 5  # prediction horizon

for year in range(1995, 2017):  # up to 2017 so that t+6 ≤ 2022
    if year + h + 1 > 2022:
        continue
    
    for c in countries:
        for p in products:
            has_at_t = (c, p, year) in edge_set
            has_at_t5 = (c, p, year + h) in edge_set
            has_at_t6 = (c, p, year + h + 1) in edge_set
            
            # Positive: new sustained activation
            if not has_at_t and has_at_t5 and has_at_t6:
                labels.append({
                    'year': year,
                    'country': c,
                    'product': p,
                    'label': 1
                })
            
            # Negative: stays non-exporter
            # Sample negatives (don't store all, only sample 2x positives)
            elif not has_at_t and not has_at_t5:
                if np.random.rand() < 0.05:  # 5% sampling rate for negatives
                    labels.append({
                        'year': year,
                        'country': c,
                        'product': p,
                        'label': 0
                    })

labels_df = pd.DataFrame(labels)

# Balance classes: sample negatives to match 2x positives
n_pos = (labels_df['label'] == 1).sum()
neg_samples = labels_df[labels_df['label'] == 0].sample(n=2*n_pos, random_state=42)
pos_samples = labels_df[labels_df['label'] == 1]
labels_balanced = pd.concat([pos_samples, neg_samples], ignore_index=True)

# Save labels
labels_balanced.to_csv("labels_h5.csv", index=False)
```

### Output
- `labels_h5.csv` with columns: `year`, `country`, `product`, `label`
- Contains balanced positive/negative examples for link prediction

---

## Step 5: Create Node Features

### Country Features

Create simple baseline features per country per year:

```python
# Load RCA data
exports = pd.read_csv("rca_cpt.csv")

# Country features per year
country_features = []

for year in range(1995, 2023):
    year_data = exports[exports['year'] == year]
    
    for c in countries:
        c_data = year_data[year_data['country'] == c]
        
        features = {
            'year': year,
            'country': c,
            # Total export value (log scale)
            'log_export': np.log1p(c_data['value'].sum()),
            # Number of products with RCA ≥ 1
            'n_products': (c_data['rca'] >= 1).sum(),
            # Average RCA across all products
            'avg_rca': c_data['rca'].mean(),
            # Max RCA
            'max_rca': c_data['rca'].max(),
        }
        country_features.append(features)

country_feat_df = pd.DataFrame(country_features)

# Normalize features (z-score per year)
for col in ['log_export', 'n_products', 'avg_rca', 'max_rca']:
    country_feat_df[col] = country_feat_df.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

country_feat_df.to_csv("country_features.csv", index=False)
```

### Product Features

```python
# Product features per year
product_features = []

for year in range(1995, 2023):
    year_data = exports[exports['year'] == year]
    
    for p in products:
        p_data = year_data[year_data['product'] == p]
        
        features = {
            'year': year,
            'product': p,
            # World export value for this product
            'log_world_export': np.log1p(p_data['value'].sum()),
            # Number of countries with RCA ≥ 1 (ubiquity)
            'ubiquity': (p_data['rca'] >= 1).sum(),
            # Average RCA across countries
            'avg_rca': p_data['rca'].mean(),
        }
        product_features.append(features)

product_feat_df = pd.DataFrame(product_features)

# Normalize
for col in ['log_world_export', 'ubiquity', 'avg_rca']:
    product_feat_df[col] = product_feat_df.groupby('year')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

product_feat_df.to_csv("product_features.csv", index=False)
```

### Output
- `country_features.csv`: 28 years × 200 countries = 5,600 rows, 6 columns
- `product_features.csv`: 28 years × 5000 products = 140,000 rows, 5 columns

---

## Step 6: Build Temporal Graph Snapshots

### Task
For each year, create an edge list (edge_index) representing all country-product pairs with M=1.

```python
import torch

# Load edges
edges = pd.read_csv("M_cpt_smoothed.csv")

# Create integer mappings for countries and products
country_to_idx = {c: i for i, c in enumerate(sorted(edges['country'].unique()))}
product_to_idx = {p: i for i, p in enumerate(sorted(edges['product'].unique()))}

idx_to_country = {i: c for c, i in country_to_idx.items()}
idx_to_product = {i: p for p, i in product_to_idx.items()}

# Save mappings
import pickle
with open("country_mapping.pkl", "wb") as f:
    pickle.dump({'to_idx': country_to_idx, 'to_name': idx_to_country}, f)
with open("product_mapping.pkl", "wb") as f:
    pickle.dump({'to_idx': product_to_idx, 'to_name': idx_to_product}, f)

# Build edge_index per year
edge_index_by_year = {}

for year in range(1995, 2023):
    year_edges = edges[edges['year'] == year]
    
    # Convert to integer indices
    country_idx = year_edges['country'].map(country_to_idx).values
    product_idx = year_edges['product'].map(product_to_idx).values
    
    # Edge index: [2, num_edges] with [source, target] rows
    edge_index = torch.tensor([country_idx, product_idx], dtype=torch.long)
    
    edge_index_by_year[year] = edge_index

# Save
torch.save(edge_index_by_year, "edge_index_by_year.pt")
```

### Output
- `edge_index_by_year.pt`: dict mapping year → edge_index tensor
- `country_mapping.pkl`, `product_mapping.pkl`: bidirectional mappings

---

## Step 7: Create Node Feature Tensors

```python
# Load features
country_feat = pd.read_csv("country_features.csv")
product_feat = pd.read_csv("product_features.csv")

# Load mappings
with open("country_mapping.pkl", "rb") as f:
    country_map = pickle.load(f)
with open("product_mapping.pkl", "rb") as f:
    product_map = pickle.load(f)

# Create feature tensors per year
country_x_by_year = {}
product_x_by_year = {}

for year in range(1995, 2023):
    # Country features for this year
    year_country = country_feat[country_feat['year'] == year].sort_values('country')
    year_country['idx'] = year_country['country'].map(country_map['to_idx'])
    year_country = year_country.sort_values('idx')
    
    country_x = torch.tensor(
        year_country[['log_export', 'n_products', 'avg_rca', 'max_rca']].values,
        dtype=torch.float
    )
    country_x_by_year[year] = country_x
    
    # Product features for this year
    year_product = product_feat[product_feat['year'] == year].sort_values('product')
    year_product['idx'] = year_product['product'].map(product_map['to_idx'])
    year_product = year_product.sort_values('idx')
    
    product_x = torch.tensor(
        year_product[['log_world_export', 'ubiquity', 'avg_rca']].values,
        dtype=torch.float
    )
    product_x_by_year[year] = product_x

# Save
torch.save(country_x_by_year, "country_x_by_year.pt")
torch.save(product_x_by_year, "product_x_by_year.pt")
```

### Output
- `country_x_by_year.pt`: dict of year → tensor [num_countries, 4]
- `product_x_by_year.pt`: dict of year → tensor [num_products, 3]

---

## Step 8: Split Into Train/Validation/Test Windows

### Split Definition

```
Training:   2000 → 2005, 2001 → 2006, ..., 2012 → 2017  (13 windows)
Validation: 2013 → 2018  (1 window)
Test:       2015 → 2020  (1 window)
```

Each window uses a **5-year lookback** of graph snapshots as input.

### Implementation

```python
# Load labels
labels = pd.read_csv("labels_h5.csv")

# Split labels by year
train_labels = labels[labels['year'].between(2000, 2012)]
val_labels = labels[labels['year'] == 2013]
test_labels = labels[labels['year'] == 2015]

# Save split labels
train_labels.to_csv("train_labels.csv", index=False)
val_labels.to_csv("val_labels.csv", index=False)
test_labels.to_csv("test_labels.csv", index=False)
```

### Output
- `train_labels.csv`: ~13 years of prediction targets
- `val_labels.csv`: 2013 targets
- `test_labels.csv`: 2015 targets

---

## Step 9: Create PyTorch Geometric HeteroData Objects

### Format
Each training example is a **temporal sequence** of 5 yearly graph snapshots, plus labels for year t+5.

```python
from torch_geometric.data import HeteroData
import pickle

# Load all data
edge_index_by_year = torch.load("edge_index_by_year.pt")
country_x_by_year = torch.load("country_x_by_year.pt")
product_x_by_year = torch.load("product_x_by_year.pt")

with open("country_mapping.pkl", "rb") as f:
    country_map = pickle.load(f)
with open("product_mapping.pkl", "rb") as f:
    product_map = pickle.load(f)

def create_hetero_snapshot(year):
    """Create a single HeteroData object for one year."""
    data = HeteroData()
    
    # Node features
    data['country'].x = country_x_by_year[year]
    data['product'].x = product_x_by_year[year]
    
    # Edges (country → product)
    data['country', 'exports', 'product'].edge_index = edge_index_by_year[year]
    
    return data

def create_temporal_sample(year, labels_df):
    """
    Create a training sample: 5-year history + labels at year+5.
    
    Returns:
        snapshots: list of 5 HeteroData objects (year-4 to year)
        labels: dict with 'edge_label_index' and 'edge_label'
    """
    # Input: 5 consecutive years ending at 'year'
    snapshots = [create_hetero_snapshot(y) for y in range(year-4, year+1)]
    
    # Labels: predictions for year+5
    sample_labels = labels_df[labels_df['year'] == year].copy()
    
    country_idx = sample_labels['country'].map(country_map['to_idx']).values
    product_idx = sample_labels['product'].map(product_map['to_idx']).values
    label_values = sample_labels['label'].values
    
    # edge_label_index: [2, num_samples]
    edge_label_index = torch.tensor([country_idx, product_idx], dtype=torch.long)
    edge_label = torch.tensor(label_values, dtype=torch.float)
    
    labels_dict = {
        'edge_label_index': edge_label_index,
        'edge_label': edge_label
    }
    
    return snapshots, labels_dict

# Create training samples
train_labels = pd.read_csv("train_labels.csv")
train_samples = []

for year in range(2000, 2013):  # 2000 to 2012
    snapshots, labels = create_temporal_sample(year, train_labels)
    train_samples.append({
        'snapshots': snapshots,
        'labels': labels,
        'year': year
    })

# Create validation sample
val_labels = pd.read_csv("val_labels.csv")
val_snapshots, val_label_dict = create_temporal_sample(2013, val_labels)
val_sample = {
    'snapshots': val_snapshots,
    'labels': val_label_dict,
    'year': 2013
}

# Create test sample
test_labels = pd.read_csv("test_labels.csv")
test_snapshots, test_label_dict = create_temporal_sample(2015, test_labels)
test_sample = {
    'snapshots': test_snapshots,
    'labels': test_label_dict,
    'year': 2015
}

# Save
torch.save(train_samples, "train_data.pt")
torch.save(val_sample, "val_data.pt")
torch.save(test_sample, "test_data.pt")

print(f"Created {len(train_samples)} training samples")
print(f"Created 1 validation sample")
print(f"Created 1 test sample")
```

### Output
- `train_data.pt`: list of 13 training samples
- `val_data.pt`: single validation sample
- `test_data.pt`: single test sample

Each sample contains:
- `snapshots`: list of 5 HeteroData graphs (5-year history)
- `labels`: dict with `edge_label_index` [2, N] and `edge_label` [N]

---

## Step 10: Data Loader for Training

### Create a Custom Dataset

```python
from torch.utils.data import Dataset, DataLoader

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

# Load data
train_samples = torch.load("train_data.pt")

# Create dataset
train_dataset = TemporalBipartiteDataset(train_samples)

# Create loader (batch_size=1 for now, since each sample is a full year)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Example: iterate through one batch
for batch in train_loader:
    snapshots = batch['snapshots']  # list of 5 graphs
    labels = batch['labels']  # dict with edge_label_index and edge_label
    year = batch['year']
    
    print(f"Year: {year.item()}")
    print(f"Num snapshots: {len(snapshots)}")
    print(f"Num labeled edges: {labels['edge_label'].shape[0]}")
    break
```

---

## Final File Structure

After running this pipeline, you will have:

```
data/
├── exports_cpt.csv                 # Step 1 output
├── rca_cpt.csv                     # Step 2 output
├── M_cpt_smoothed.csv              # Step 3 output
├── labels_h5.csv                   # Step 4 output
├── country_features.csv            # Step 5 output
├── product_features.csv            # Step 5 output
├── edge_index_by_year.pt           # Step 6 output
├── country_mapping.pkl             # Step 6 output
├── product_mapping.pkl             # Step 6 output
├── country_x_by_year.pt            # Step 7 output
├── product_x_by_year.pt            # Step 7 output
├── train_labels.csv                # Step 8 output
├── val_labels.csv                  # Step 8 output
├── test_labels.csv                 # Step 8 output
├── train_data.pt                   # Step 9 output (READY FOR MODEL)
├── val_data.pt                     # Step 9 output (READY FOR MODEL)
└── test_data.pt                    # Step 9 output (READY FOR MODEL)
```

---

## Validation Checks

Before moving to model training, verify:

1. **No data leakage:** Test year (2015) is never seen during training
2. **Balanced labels:** ~2:1 negative:positive ratio in labels
3. **Consistent mappings:** country_to_idx and product_to_idx are the same across all years
4. **Edge counts:** edge_index_by_year should show ~10,000–50,000 edges per year (sparse)
5. **Feature shapes:**
   - country_x: [~200, 4]
   - product_x: [~5000, 3]
   - edge_label_index: [2, N] where N is number of labeled pairs

Run these checks:

```python
# Check train/val/test year ranges
print("Train years:", sorted(train_labels['year'].unique()))
print("Val year:", val_labels['year'].unique())
print("Test year:", test_labels['year'].unique())

# Check label balance
print("\nLabel distribution:")
print(train_labels['label'].value_counts())

# Check feature dimensions
sample_year = 2010
print(f"\nFeature shapes for year {sample_year}:")
print(f"Country features: {country_x_by_year[sample_year].shape}")
print(f"Product features: {product_x_by_year[sample_year].shape}")
print(f"Edges: {edge_index_by_year[sample_year].shape}")
```

---

## Next Steps After Pipeline

Once you have `train_data.pt`, `val_data.pt`, and `test_data.pt`:

1. **Implement the GNN model** (separate guide)
2. **Define training loop** with edge-level BCE loss
3. **Evaluate on val set** to tune hyperparameters
4. **Final evaluation on test set** (2015 → 2020)
5. **Compare against baselines** (RCA persistence, product space, Random Forest)

---

## Notes for Your Coding Agent

- All intermediate CSV files should be saved to allow debugging
- Use `pandas` for data wrangling, `numpy` for computations, `torch` for final tensors
- The 3-year smoothing window requires at least 2 years with RCA≥1 (not 3 out of 3)
- Negative sampling rate of 5% in Step 4 is a heuristic — adjust if you get too few/many negatives
- All features are z-score normalized per year to prevent temporal bias
- The 5-year lookback means the first trainable year is 2000 (needs 1996–2000 history)

---

## Troubleshooting Common Issues

| Issue | Solution |
|---|---|
| Out of memory when loading all years | Process years in chunks, save intermediate .pt files per year |
| Too many negative samples | Reduce sampling rate in Step 4, or use stratified sampling |
| Features are all NaN | Check for division by zero in RCA computation; add epsilon (1e-8) |
| Edge index has wrong shape | Should be [2, num_edges], not [num_edges, 2]; transpose if needed |
| Labels don't match any edges | Verify country/product mappings are consistent across all files |

---

This pipeline is production-ready. Each step is modular and can be run independently. Save intermediate files to allow rerunning specific steps without recomputing everything.
