"""
build_tabular_dataset.py
========================
Builds tabular train and test CSVs for baseline models (CatBoost, KNN).
The dataset mirrors the exact train/test split used by the GNN.

Key Logic:
- Labels come from train_labels.csv and test_labels.csv (ignoring validation).
- The 'year' column in labels is the anchor year `t` (the last year of the
  5-year lookback window). The target (RCA in 5 years) is the label column.
- Features are derived from:
    - rca_cpt.csv          -> rca_t, rca_t-1, ..., rca_t-4 (trajectory)
    - country_features_enriched.csv -> country trade + WDI features at t and t-4
    - product_features.csv -> product features at t and t-4
    - product_density is calculated from the edge structure at time t.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "TabularDataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. LOAD CORE DATA ──────────────────────────────────────────────────────────
print("Step 1: Loading source data files...")

train_labels = pd.read_csv(os.path.join(DATA_DIR, "GNNModelTraining", "train_labels.csv"))
test_labels  = pd.read_csv(os.path.join(DATA_DIR, "GNNModelTraining", "test_labels.csv"))

# Country features: year, country, log_export, n_products, avg_rca, max_rca,
#                   gdp_pc, capital_formation, tertiary_enrollment, fdi_inflows,
#                   manufacturing_va, internet_users, population
country_feat = pd.read_csv(os.path.join(DATA_DIR, "country_features_enriched.csv"))

# Product features: year, product, log_world_export, ubiquity, avg_rca
product_feat = pd.read_csv(os.path.join(DATA_DIR, "product_features.csv"))

print("Step 2: Building fast RCA lookup (chunked filtered read)...")

def load_filtered_rca(label_df):
    """
    Load only the RCA rows that are relevant to the given label dataframe.
    We only need years [t-4 .. t] for every (country, product) pair in labels.
    This avoids loading the full 14M-row file into memory at once.
    """
    # Determine which years we actually need
    anchor_years = label_df['year'].unique()
    needed_years = set()
    for y in anchor_years:
        for delta in range(5):           # t-4, t-3, t-2, t-1, t
            needed_years.add(y - delta)

    relevant_pairs = label_df[['country', 'product']].drop_duplicates()

    chunks = []
    rca_iter = pd.read_csv(
        os.path.join(DATA_DIR, "rca_cpt.csv"),
        chunksize=1_000_000,
        usecols=['year', 'country', 'product', 'rca']
    )
    for chunk in tqdm(rca_iter, desc="  Filtering RCA chunks"):
        chunk = chunk[chunk['year'].isin(needed_years)]
        chunk = chunk.merge(relevant_pairs, on=['country', 'product'], how='inner')
        if len(chunk) > 0:
            chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


# ── 2. GRAND FEATURE BUILDER ───────────────────────────────────────────────────

def build_features(label_df: pd.DataFrame, rca_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Given a label dataframe and the pre-filtered RCA dataframe, construct the
    full tabular feature matrix for CatBoost / KNN.
    """
    print(f"\n[{split_name}] Building feature matrix for {len(label_df):,} samples...")

    # Pivot RCA to wide: index=(country, product), columns=year
    print(f"  Pivoting RCA data...")
    rca_wide = rca_df.pivot_table(
        index=['country', 'product'],
        columns='year',
        values='rca',
        fill_value=0.0
    )

    records = []
    anchor_years = sorted(label_df['year'].unique())

    for t in tqdm(anchor_years, desc=f"  Processing anchor years"):
        year_labels = label_df[label_df['year'] == t].copy()

        # ── Country features at t and t-4 ──────────────────────────────────
        cf_t    = country_feat[country_feat['year'] == t].set_index('country')
        cf_tm4  = country_feat[country_feat['year'] == t - 4].set_index('country')

        # ── Product features at t and t-4 ──────────────────────────────────
        pf_t    = product_feat[product_feat['year'] == t].set_index('product')
        pf_tm4  = product_feat[product_feat['year'] == t - 4].set_index('product')

        # ── Product Density (graph topology proxy) ──────────────────────────
        # product_density_t = fraction of products active for this country at t
        # that are related (by ubiquity proximity) to the target product.
        # Simple tractable proxy: for each (country, product) pair,
        #   density = (# products country exports at t with RCA>=1) / (total products at t)
        # This captures the GNN's neighbourhood signal without the full graph.
        products_at_t = rca_df[(rca_df['year'] == t) & (rca_df['rca'] >= 1)]\
            .groupby('country')['product'].count().rename('active_count')
        total_products_at_t = rca_df[rca_df['year'] == t]\
            .groupby('country')['product'].count().rename('total_count')
        density_df = pd.concat([products_at_t, total_products_at_t], axis=1).fillna(0)
        density_df['product_density_t'] = density_df['active_count'] / (density_df['total_count'] + 1e-8)

        for _, row in year_labels.iterrows():
            c = row['country']
            p = row['product']
            label = row['label']

            # -----------------------------------------------------------
            # Section 2: Edge Features – RCA Trajectory
            # -----------------------------------------------------------
            rca_vals = {}
            if (c, p) in rca_wide.index:
                cp_row = rca_wide.loc[(c, p)]
                for delta, col_name in enumerate(['rca_t_minus_4', 'rca_t_minus_3',
                                                  'rca_t_minus_2', 'rca_t_minus_1',
                                                  'rca_t']):
                    yr = t - (4 - delta)
                    rca_vals[col_name] = cp_row.get(yr, 0.0)
            else:
                for col_name in ['rca_t_minus_4', 'rca_t_minus_3',
                                  'rca_t_minus_2', 'rca_t_minus_1', 'rca_t']:
                    rca_vals[col_name] = 0.0

            rca_vals['rca_momentum'] = rca_vals['rca_t'] - rca_vals['rca_t_minus_4']

            # product_density_t
            rca_vals['product_density_t'] = density_df.at[c, 'product_density_t'] \
                if c in density_df.index else 0.0

            # -----------------------------------------------------------
            # Section 3: Country Trade Features
            # -----------------------------------------------------------
            ctf = {}
            if c in cf_t.index:
                row_t = cf_t.loc[c]
                ctf['country_log_export_t']  = row_t.get('log_export',  0.0)
                ctf['country_n_products_t']  = row_t.get('n_products',  0.0)
                ctf['country_avg_rca_t']     = row_t.get('avg_rca',     0.0)
                ctf['country_max_rca_t']     = row_t.get('max_rca',     0.0)
            else:
                ctf = {k: 0.0 for k in ['country_log_export_t', 'country_n_products_t',
                                          'country_avg_rca_t', 'country_max_rca_t']}

            # n_products momentum
            n_products_tm4 = cf_tm4.loc[c, 'n_products'] if c in cf_tm4.index else 0.0
            ctf['country_n_products_momentum'] = ctf['country_n_products_t'] - n_products_tm4

            # -----------------------------------------------------------
            # Section 4: Country WDI Features
            # -----------------------------------------------------------
            wdi = {}
            wdi_cols = ['gdp_pc', 'capital_formation', 'tertiary_enrollment',
                        'fdi_inflows', 'manufacturing_va', 'internet_users', 'population']
            if c in cf_t.index:
                row_t = cf_t.loc[c]
                for col in wdi_cols:
                    wdi[f'{col}_t'] = row_t.get(col, 0.0)
            else:
                for col in wdi_cols:
                    wdi[f'{col}_t'] = 0.0

            # GDP momentum (% change over 5-year window)
            gdp_t   = wdi['gdp_pc_t']
            gdp_tm4 = cf_tm4.loc[c, 'gdp_pc'] if c in cf_tm4.index else 0.0
            wdi['gdp_pc_momentum'] = (gdp_t - gdp_tm4) / (abs(gdp_tm4) + 1e-8)

            # -----------------------------------------------------------
            # Section 5: Product Features
            # -----------------------------------------------------------
            pf = {}
            if p in pf_t.index:
                row_t_p = pf_t.loc[p]
                pf['product_log_world_export_t'] = row_t_p.get('log_world_export', 0.0)
                pf['product_ubiquity_t']         = row_t_p.get('ubiquity',         0.0)
                pf['product_avg_rca_t']          = row_t_p.get('avg_rca',          0.0)
            else:
                pf = {k: 0.0 for k in ['product_log_world_export_t',
                                         'product_ubiquity_t', 'product_avg_rca_t']}

            # Ubiquity momentum
            ubiquity_tm4 = pf_tm4.loc[p, 'ubiquity'] if p in pf_tm4.index else 0.0
            pf['product_ubiquity_momentum'] = pf['product_ubiquity_t'] - ubiquity_tm4

            # -----------------------------------------------------------
            # Assemble record
            # -----------------------------------------------------------
            record = {
                # Identifiers & Target
                'country_id':         c,
                'product_id':         p,
                'anchor_year':        t,
                'target_future_rca':  label,
                # Edge/Trajectory features
                **rca_vals,
                # Country trade
                **ctf,
                # WDI
                **wdi,
                # Product
                **pf,
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


# ── 3. BUILD TRAIN AND TEST DATASETS ──────────────────────────────────────────
print("\n=== Processing TRAIN set ===")
rca_train = load_filtered_rca(train_labels)
train_df  = build_features(train_labels, rca_train, "TRAIN")

print("\n=== Processing TEST set ===")
rca_test  = load_filtered_rca(test_labels)
test_df   = build_features(test_labels, rca_test, "TEST")

# ── 4. SAVE ────────────────────────────────────────────────────────────────────
train_path = os.path.join(OUTPUT_DIR, "train_tabular.csv")
test_path  = os.path.join(OUTPUT_DIR, "test_tabular.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n" + "="*60)
print(f"  Train CSV: {train_path}")
print(f"  Train rows: {len(train_df):,}  |  Columns: {len(train_df.columns)}")
print(f"  Test  CSV: {test_path}")
print(f"  Test  rows: {len(test_df):,}  |  Columns: {len(test_df.columns)}")
print(f"  Feature columns: {list(train_df.columns)}")
print("="*60)
print("Done. Files saved to data/TabularDataset/")
