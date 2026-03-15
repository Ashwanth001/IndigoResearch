import pandas as pd
import torch
import os
import random

# Define directories
output_dir = "data"
input_file = os.path.join(output_dir, "M_cpt_smoothed.csv")
output_file = os.path.join(output_dir, "labels_h5.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting Step 4 (CUDA Accelerated): Generating Prediction Labels (h=5)...")

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Did you run Step 3?")
    exit(1)

# Load smoothed edges (only where M=1)
df = pd.read_csv(input_file)

# We'll re-convert to a 3D Tensor [Years, Countries, Products] on GPU
years = sorted(df['year'].unique())
countries = sorted(df['country'].unique())
products = sorted(df['product'].unique())

year_map = {y: i for i, y in enumerate(years)}
country_map = {c: i for i, c in enumerate(countries)}
product_map = {p: i for i, p in enumerate(products)}

print(f"Allocating 3D Tensor on {device}...")
M = torch.zeros((len(years), len(countries), len(products)), dtype=torch.uint8, device=device)

# Fill indices
idx_y = torch.tensor(df['year'].map(year_map).values, device=device)
idx_c = torch.tensor(df['country'].map(country_map).values, device=device)
idx_p = torch.tensor(df['product'].map(product_map).values, device=device)
M[idx_y, idx_c, idx_p] = 1

# Generate labels (h=5, horizon target t+5, sustained at t+6)
h = 5
total_years = len(years)
# We can only generate labels for years t where t+h+1 < total_years
limit = total_years - h - 1

print(f"Generating labels for {limit} years on CUDA...")

# 1. Slice Tensors for t, t+5, t+6
# Masks [T_valid, C, P]
M_t = M[:limit]
M_t5 = M[h:limit+h]
M_t6 = M[h+1:limit+h+1]

# 2. Extract Positive Indices (new sustained activation)
positives_mask = (M_t == 0) & (M_t5 == 1) & (M_t6 == 1)
pos_indices = torch.nonzero(positives_mask)
print(f"Found {len(pos_indices)} positive instances.")

# 3. Extract Negative Indices (stay 0)
negatives_mask = (M_t == 0) & (M_t5 == 0)
# Instead of keeping all (millions), we'll sample negative indices
# For efficiency, we'll sub-sample directly on GPU if possible
# Or just get all and sample in pandas
neg_indices = torch.nonzero(negatives_mask)
print(f"Total possible negative pool: {len(neg_indices)} instances.")

# --- Sampling ---
# We want roughly 2x negatives relative to positives
target_neg = min(len(neg_indices), 2 * len(pos_indices))
print(f"Sampling {target_neg} negatives to balance classes...")

# GPU sampling indices
sample_idx = torch.randperm(len(neg_indices), device=device)[:target_neg]
neg_sampled = neg_indices[sample_idx]

# --- Combine and Convert back ---
print("Converting back to long format and saving...")
all_labels = []

# Positives
pos_df = pd.DataFrame({
    'year': [years[i] for i in pos_indices[:, 0].cpu()],
    'country': [countries[i] for i in pos_indices[:, 1].cpu()],
    'product': [products[i] for i in pos_indices[:, 2].cpu()],
    'label': 1
})

# Negatives
neg_df = pd.DataFrame({
    'year': [years[i] for i in neg_sampled[:, 0].cpu()],
    'country': [countries[i] for i in neg_sampled[:, 1].cpu()],
    'product': [products[i] for i in neg_sampled[:, 2].cpu()],
    'label': 0
})

final_df = pd.concat([pos_df, neg_df], ignore_index=True)
final_df.to_csv(output_file, index=False)

# Validation Statistics
print("\nValidation Statistics for Step 4:")
print(f"Output: {output_file}")
print(f"Positive samples: {len(pos_df)}")
print(f"Negative samples: {len(neg_df)}")
print("Label generation (CUDA) Finished.")
