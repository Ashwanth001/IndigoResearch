import pandas as pd
import torch
import os

# Define directories
output_dir = "data"
input_file = os.path.join(output_dir, "rca_cpt.csv")
output_file = os.path.join(output_dir, "M_cpt_smoothed.csv")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Starting Step 3 (CUDA Accelerated): Smoothing and Binarizing...")

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Did you run Step 2?")
    exit(1)

# Load RCA data
df = pd.read_csv(input_file)

# 1. Map IDs to fixed indices for Tensor operations
years = sorted(df['year'].unique())
countries = sorted(df['country'].unique())
products = sorted(df['product'].unique())

year_map = {y: i for i, y in enumerate(years)}
country_map = {c: i for i, c in enumerate(countries)}
product_map = {p: i for i, p in enumerate(products)}

# 2. Create a sparse representation using a 3D Binary Tensor: [Years, Countries, Products]
print(f"Allocating 3D Tensor: [{len(years)}, {len(countries)}, {len(products)}] on {device}")
# M_binary: initialized to zero
M_binary = torch.zeros((len(years), len(countries), len(products)), dtype=torch.float32, device=device)

# Fill indices where RCA >= 1
mask = df['rca'] >= 1.0
active_df = df[mask]

# Extract indices for mapping
idx_y = torch.tensor(active_df['year'].map(year_map).values, device=device)
idx_c = torch.tensor(active_df['country'].map(country_map).values, device=device)
idx_p = torch.tensor(active_df['product'].map(product_map).values, device=device)

# Set RCA=1 markers in tensor
M_binary[idx_y, idx_c, idx_p] = 1.0

# 3. Apply 3-Year Rolling Window Sum (CUDA Accelerated)
print("Applying 3-year rolling window sum using 1D Convolution...")
# Reshape to [C*P, 1, Y] for 1D convolution across the Year axis
# (Treating each country-product pair as a 1D time series)
M_reshaped = M_binary.permute(1, 2, 0).reshape(-1, 1, len(years))

# Convolution kernel of [1, 1, 3] ones (sum of previous years)
# (Padding=1 to keep same length, though we should ignore first few years)
kernel = torch.ones((1, 1, 3), device=device)
M_sum = torch.nn.functional.conv1d(M_reshaped, kernel, padding=1)

# M=1 if sum >= 2 (RCA >= 1 in at least 2 out of 3 years)
M_smoothed = (M_sum >= 2.0).float().reshape(len(countries), len(products), len(years)).permute(2, 0, 1)

# 4. Filter back to sparse long-format and save
# Find indices where M_smoothed == 1
nonzero_indices = torch.nonzero(M_smoothed)
print(f"Found {len(nonzero_indices)} edges with RCA >= 1 (smoothed).")

# Convert back to DataFrame
edge_list = {
    'year': [years[i] for i in nonzero_indices[:, 0].cpu()],
    'country': [countries[i] for i in nonzero_indices[:, 1].cpu()],
    'product': [products[i] for i in nonzero_indices[:, 2].cpu()],
    'M': [1] * len(nonzero_indices)
}

edges_df = pd.DataFrame(edge_list)
edges_df.to_csv(output_file, index=False)

# Validation Statistics
print("\nValidation Statistics for Step 3:")
print(f"Output: {output_file}")
print(f"Total smoothed edges: {len(edges_df)}")
print("Smoothing (CUDA) Finished.")
