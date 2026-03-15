import pandas as pd
import torch
import os

# Define directories
output_dir = "data"
input_file = os.path.join(output_dir, "exports_cpt.csv")
output_file = os.path.join(output_dir, "rca_cpt.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting Step 2 (CUDA Accelerated): Computing RCA Matrix using {device}...")

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Did you run Step 1?")
    exit(1)

# Load aggregated exports
df = pd.read_csv(input_file)

# Move data to GPU using PyTorch for massive speedup in math
# We'll compute the totals using grouping logic on the CPU briefly 
# because it's easier with pandas, then compute the division on GPU.

print("Calculating totals on CPU/GPU...")
df['country_total'] = df.groupby(['year', 'country'])['value'].transform('sum')
df['product_total'] = df.groupby(['year', 'product'])['value'].transform('sum')
df['world_total'] = df.groupby('year')['value'].transform('sum')

# Convert to tensors for the division (accelerated on GPU)
val_t = torch.tensor(df['value'].values, dtype=torch.float32, device=device)
c_total_t = torch.tensor(df['country_total'].values, dtype=torch.float32, device=device)
p_total_t = torch.tensor(df['product_total'].values, dtype=torch.float32, device=device)
w_total_t = torch.tensor(df['world_total'].values, dtype=torch.float32, device=device)

print("Running RCA division on CUDA...")
# RCA = (X_cpt / CountryTotal) / (ProductTotal / WorldTotal)
rca_t = (val_t / (c_total_t + 1e-8)) / ((p_total_t + 1e-8) / w_total_t)

# Move back to CPU and save
df['rca'] = rca_t.cpu().numpy()

# Drop intermediate columns
df = df[['year', 'country', 'product', 'value', 'rca']]

# Save RCA matrix
df.to_csv(output_file, index=False)

# Validation Statistics
print("\nValidation Statistics for Step 2:")
print(f"Output file: {output_file}")
print(f"Max RCA: {df['rca'].max():.4f}")
print(f"RCA >= 1 entries: {(df['rca'] >= 1).sum()}")
print("Step 2 Finished.")
