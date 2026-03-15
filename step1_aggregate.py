import pandas as pd
import glob
import os
import sys

# Define directories
data_dir = r"c:\Users\Ashwa\Ash_projects\Indigo Research\BACIDataset1995"
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Load and Aggregate BACI Data
# File pattern for the user's dataset
baci_files = sorted(glob.glob(os.path.join(data_dir, "BACI_HS92_Y*_V202601.csv")))
print(f"Found {len(baci_files)} BACI files.")

if not baci_files:
    print("No BACI files found! Please check the dataset path.")
    sys.exit(1)

# Only process years 1995-2022 as per plan logic?
# The plan mentions 1995–2022. Let's filter if needed, 
# although having more years (up to 2024) might be better. 
# Let's use all available files to be comprehensive.

df_list = []
for file in baci_files:
    print(f"Loading {os.path.basename(file)}...")
    df = pd.read_csv(file)
    # Aggregate: X_cpt = sum over j (all importers)
    # Columns in BACI are t, i, j, k, v, q
    # We want to group by year (t), exporter (i), product (k)
    exports_year = df.groupby(['t', 'i', 'k'])['v'].sum().reset_index()
    exports_year.columns = ['year', 'country', 'product', 'value']
    df_list.append(exports_year)

# Concatenate all years
print("Concatenating aggregated data...")
exports = pd.concat(df_list, ignore_index=True)

# Save intermediate result
output_file = os.path.join(output_dir, "exports_cpt.csv")
exports.to_csv(output_file, index=False)

# Validation Statistics
print("\nValidation Statistics for Step 1:")
print(f"Output file: {output_file}")
print(f"File created: {os.path.exists(output_file)}")
print(f"Total rows: {len(exports)}")
print(f"Year range: {exports['year'].min()} to {exports['year'].max()}")
print(f"Unique countries: {exports['country'].nunique()}")
print(f"Unique products: {exports['product'].nunique()}")
print(f"Total trade value: {exports['value'].sum():,.0f}")
