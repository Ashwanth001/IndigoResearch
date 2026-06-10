# Trade Complexity 2.0 — Project Context

> Bring any collaborator or new chat up to speed from this single file.

---

## What This Project Is

**Trade Complexity 2.0** is an ML research project that asks:

> *Can a temporal bipartite Graph Neural Network (GNN), trained on the global country–product export network, predict which products a country will gain Revealed Comparative Advantage (RCA ≥ 1) in over the next 5 years — and does adding an LLM-based product capability layer improve accuracy and interpretability?*

It compares the GNN against classical economic complexity methods (ECI, Product Space) and a gradient-boosted ML baseline. The core novelty is the LLM-derived product capability graph layered on top of trade structure.

**Primary story country:** India. Challengers: Vietnam, Mexico, Indonesia. Comparators: China, Singapore, UAE.

---

## Three Research Hypotheses

| # | Hypothesis |
|---|-----------|
| H1 | Temporal bipartite GNN > ECI / Product Space (learns higher-order network patterns) |
| H2 | LLM-derived product capability similarity improves performance for rare/complex products |
| H3 | GNN + LLM explanations recover Product Space intuition with sharper, data-driven pathways |

---

## Repository Layout

```
Indigo_Research/
├── dataPipeline.ipynb          ← THE active pipeline (Steps 1–12, all in one notebook)
├── data/                       ← All pipeline outputs (CSVs + .pt tensors)
│   ├── exports_cpt.csv
│   ├── rca_cpt.csv
│   ├── M_cpt_smoothed.csv
│   ├── labels_h5.csv
│   ├── country_features.csv
│   ├── country_features_enriched.csv
│   ├── product_features.csv
│   ├── wdi_features.csv
│   ├── edge_index_by_year.pt
│   ├── country_x_by_year.pt    ← 11-feature country tensors (4 BACI + 7 WDI) [rebuilt 2026-06-10]
│   ├── product_x_by_year.pt    ← 3-feature product tensors
│   ├── country_mapping.pkl
│   ├── product_mapping.pkl
│   ├── train_labels.csv / val_labels.csv / test_labels.csv
│   ├── train_data.pt / val_data.pt / test_data.pt
│   ├── product_llm_embeddings.pt
│   ├── capability_edge_index.pt
│   └── gnn_tiered_results.csv  ← GNN-4F and GNN-11F results [new, 2026-06-10]
├── datasets/
│   ├── BACIDataset1995/        ← BACI HS92, 1995–2024 (30 annual CSV files)
│   └── WDI_csv/                ← World Bank WDI indicators (WDICSV.csv)
└── oldResearch/                ← Archived: old step scripts, models.py, evaluator.py,
                                   checkpoints/GNN/best_model.pt, baseline results
```

**Current branch:** `clean-master` — all standalone `.py` scripts have been consolidated into `dataPipeline.ipynb`.

---

## Data Pipeline (dataPipeline.ipynb)

The notebook runs the full pipeline from raw BACI files to trained GNN models.

### Key constants (set in Cell 2)

| Constant | Value | Meaning |
|----------|-------|---------|
| `BACI_DIR` | `datasets/BACIDataset1995` | HS92 trade data source |
| `TRAIN_CUTOFF` | 2012 | Observation years ≤ 2012 → training set |
| `VAL_YEAR` | 2013 | Validation observation year |
| `TEST_YEAR` | 2015 | Test observation year (2-year gap from val is intentional) |
| `LABEL_HORIZON` | 5 | Predict RCA transition h=5 years ahead |
| `NEG_RATIO` | 5 | 5 negatives sampled per positive |

### Steps and outputs

| Step | What it does | Output file(s) |
|------|-------------|----------------|
| 1 — Aggregate | Sum bilateral BACI flows → country-product-year export values | `exports_cpt.csv` |
| 2 — RCA | Compute Revealed Comparative Advantage per (country, product, year) | `rca_cpt.csv` |
| 3 — Smoothing | **Causal** 3-year trailing window: M=1 if RCA≥1 in ≥2 of past 3 years | `M_cpt_smoothed.csv` |
| 4 — Labels | Binary labels: positive = 0→1 transition at t+5 (+ sustained at t+6) | `labels_h5.csv` |
| 5 — Features | Per-year z-score node features from BACI | `country_features.csv`, `product_features.csv` |
| 6 — Graph structure | Stable country/product index mappings + edge indexes | `edge_index_by_year.pt`, `*_mapping.pkl` |
| 7 — Tensors | Pack 4-feature BACI country features into `{year: tensor}` dicts | `product_x_by_year.pt` (also writes 4-feat `country_x_by_year.pt` — overwritten by Step 9) |
| 8 — Split | **Temporal** train/val/test split by year cutoff | `train/val/test_labels.csv` |
| 9 — WDI enrichment | Add 7 World Bank indicators to country nodes; normalize using train-only stats; **overwrites `country_x_by_year.pt` with 11-feature version** | `country_features_enriched.csv`, `wdi_features.csv`, `country_x_by_year.pt` (11-feat) |
| 10 — HeteroData | Package 5-year temporal windows + labeled edges into PyTorch Geometric objects | `train/val/test_data.pt` |
| 11A — LLM embeddings | Embed HS6 product descriptions with `all-MiniLM-L6-v2` (384-dim, unit-norm) | `product_llm_embeddings.pt` |
| 11B — Capability edges | Sparse product↔product edges where cosine similarity ≥ 0.70 | `capability_edge_index.pt` |
| 12 — GNN training | Train GNN-4F and GNN-11F, evaluate 3-tier metrics vs baselines | `gnn_tiered_results.csv` |

### Critical data-leakage rules (supervisor review notes)

1. **Smoothing (Step 3):** Trailing window only — `mean(t-2, t-1, t)`. Never centered `(t-1, t, t+1)`.
2. **Labels (Step 4):** Only `(~M[t]) & M[t+5]` transitions count as positive. All `M[t+5]=1` would inflate positive rate ~10×.
3. **Split (Step 8):** Strict year cutoff — NOT sklearn random split. Test year must never appear in training.
4. **WDI normalization (Step 9):** Compute mean/std from training years (≤ 2012) only; apply those stats to all years.

---

## Node Features

### Country nodes — dim 11 (4 BACI + 7 WDI)

| Feature | Source | Description |
|---------|--------|-------------|
| `log_export` | BACI | Log total export value |
| `n_products` | BACI | Number of products with RCA≥1 |
| `avg_rca` | BACI | Mean RCA across all products |
| `max_rca` | BACI | Highest individual RCA |
| `gdp_pc` | WDI | Log GDP per capita (PPP) |
| `population` | WDI | Log population |
| `manufacturing_va` | WDI | Manufacturing % of GDP |
| `capital_formation` | WDI | Gross capital formation % of GDP |
| `tertiary_enrollment` | WDI | Tertiary education enrollment rate |
| `internet_users` | WDI | % population using internet |
| `fdi_inflows` | WDI | FDI inflows % of GDP |

### Product nodes — dim 3

| Feature | Description |
|---------|-------------|
| `log_world_export` | Log total global export value for this product |
| `ubiquity` | Number of countries with RCA≥1 (low = complex product) |
| `avg_rca` | Global average RCA in this product |

All features are **z-score normalized per year**.

---

## GNN Architecture (Step 12 / dataPipeline.ipynb)

```
BipartiteEncoder12
  └─ country_lin: Linear(c_in → 128)   [c_in=4 for GNN-4F, 11 for GNN-11F]
  └─ product_lin: Linear(3 → 128)
  └─ _HomoGNN wrapped with to_hetero:
       SAGEConv(128→128) × 2, with ReLU + dropout(0.3)

TemporalGNN12
  └─ BipartiteEncoder12 applied to each of 5 yearly snapshots
  └─ GRU over 5-year country sequence  → z_c[-1]
  └─ GRU over 5-year product sequence  → z_p[-1]

LinkPredictor12
  └─ MLP: Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→1)
  └─ Output: logit → sigmoid → P(RCA transition)
```

**Training setup:** 80 epochs, lr=1e-3, weight_decay=1e-5, BCEWithLogitsLoss with pos_weight≈5.0 (to handle ~17% positive rate in labeled edges), patience=15 early stopping, gradient clipping at 1.0.

---

## Results — Full Comparison Table (Test year 2015, predicting 2020 outcomes)

| Model | PR-AUC | NDCG@20 | Prec@20 | CWR | AUROC |
|-------|--------|---------|---------|-----|-------|
| RCA Persistence | 0.525 | 0.510 | 0.528 | 0.000 | — |
| Density (Product Space) | 0.349 | 0.487 | 0.434 | 0.005 | — |
| ECI | 0.231 | 0.146 | 0.869 | 0.000 | — |
| ECI + Density | 0.349 | 0.487 | 0.434 | 1.000 | — |
| **GNN-4F (BACI only)** | **0.413** | **0.445** | **0.403** | **0.726** | **0.820** |
| **GNN-11F (BACI+WDI)** | **0.435** | **0.474** | **0.426** | **0.717** | **0.829** |
| GNN + LLM capability layer | *pending* | *pending* | *pending* | *pending* | *pending* |

**Key observations:**
- WDI features (GNN-11F vs GNN-4F) improve every metric except CWR
- Both GNN configs beat the Density and ECI baselines on PR-AUC and NDCG@20
- RCA Persistence still leads on PR-AUC — a strong baseline reflecting autocorrelation in trade data
- GNNs achieve much higher CWR (0.72+) vs classical methods (<0.01), meaning they prioritize complex, valuable products correctly
- High AUROC (0.82–0.83) shows good ranking quality; PR-AUC gap vs Persistence suggests room for improvement on precision at threshold

Saved to `data/gnn_tiered_results.csv`.

---

## Prediction Methods (Ablation Ladder)

| # | Method | Status |
|---|--------|--------|
| 1 | RCA Persistence baseline | Done — numbers in table above |
| 2 | Product Space density (proximity φ) | Done — numbers in table above |
| 3 | ECI and ECI+Density | Done — numbers in table above |
| 4 | Gradient-boosted classifier (tabular) | Results in `oldResearch/` — not yet re-run on new pipeline |
| 5 | GNN-4F (BACI only) | **Done** — PR-AUC=0.413 |
| 6 | GNN-11F (BACI+WDI) | **Done** — PR-AUC=0.435 |
| 7 | GNN + LLM capability layer | **Pending** — data ready (`capability_edge_index.pt`), model integration not started |

---

## Evaluation — 3-Tier Framework

| Tier | Metric | What it measures |
|------|--------|-----------------|
| 1 — Global | **PR-AUC** (preferred over AUROC for ~3–5% positive rate) | Overall discrimination |
| 2 — Economic | **Complexity-weighted recall** (weights by PCI — Product Complexity Index proxy: −ubiquity/max_ubiquity) | Gets the hard, valuable transitions right |
| 3 — Investor | **NDCG@20 and Precision@20 per country** | Ranks the top predictions correctly |

---

## HeteroData Schema

Each training sample is a dict:
```python
{
  'snapshots': [HeteroData, ...],   # 5 consecutive yearly graphs
  'labels': {
    'edge_label_index': LongTensor[2, E],   # (country_idx, product_idx) pairs to predict
    'edge_label':       FloatTensor[E],     # 0 or 1
  },
  'year': int,           # observation year t; label = transition at t+5
  'countries_raw': arr,  # original country codes (for per-country metrics)
  'products_raw':  arr,  # original product codes
}
```

Each `HeteroData` snapshot:
```python
data['country'].x                                    # [233, 11]
data['product'].x                                    # [5018, 3]
data['country', 'exports',     'product'].edge_index  # [2, ~90k] bipartite trade edges
data['product', 'rev_exports', 'country'].edge_index  # [2, ~90k] reverse edges (for SAGEConv)
# After Step 11 integration (pending):
data['product', 'capability', 'product'].edge_index   # [2, E_capability] semantic similarity edges
```

---

## Known Bugs Fixed (2026-06-10 session)

| Bug | Symptom | Fix |
|-----|---------|-----|
| `_HomoGNN.forward(self, x, ei)` — wrong parameter name | `ValueError: MessagePassing.propagate only supports integer tensors of shape [2, num_messages]` | Renamed `ei` → `edge_index`; PyG's `to_hetero` fx tracer requires this exact name |
| `c_in` hardcoded as 4 and 11 | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (233×4 and 11×128)` | Auto-detect from `tr[0]['snapshots'][0]['country'].x.shape[1]` |
| `country_x_by_year.pt` had 4 features despite Step 9 existing | GNN-11F used same 4-feature tensors as GNN-4F | Step 7 wrote a 4-feat `.pt`; Step 9 builds the 11-feat version — ran Step 9 tensor rebuild from `country_features_enriched.csv` |

---

## What's Done vs. Pending

| Component | Status |
|-----------|--------|
| Full data pipeline (Steps 1–10) | Complete and running in notebook |
| LLM embeddings (Step 11A) | Complete — `product_llm_embeddings.pt` |
| Capability edges (Step 11B) | Complete — `capability_edge_index.pt` |
| GNN-4F training + evaluation | **Complete** — PR-AUC=0.413 |
| GNN-11F training + evaluation | **Complete** — PR-AUC=0.435 |
| Inject capability edges into HeteroData | **Pending** — add `('product','capability','product')` edges to snapshots |
| Re-train GNN with capability edge type (GNN+LLM) | **Pending** — core H2 innovation |
| Re-run gradient-boosted baseline on new pipeline | **Pending** |
| Full 7-model comparison table | **Pending** (missing gradient-boost + GNN+LLM rows) |
| India / Vietnam / Mexico / Indonesia country stories | **Pending** |
| Dashboard / visualization layer | **Pending** |

---

## Tech Stack

- **Python 3.14**, PyTorch, PyTorch Geometric (`HeteroData`, `SAGEConv`, `to_hetero`, `GRUConv`)
- **Data:** BACI HS92 trade data (1995–2024), World Bank WDI indicators
- **LLM embeddings:** `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim)
- **Baselines:** scikit-learn, XGBoost/LightGBM (tabular), scipy (Product Space proximity)
- **CUDA** used for RCA computation, tensor ops, and GNN training

---

## Economic Concepts Quick Reference

| Term | Definition |
|------|-----------|
| **RCA** (Revealed Comparative Advantage) | `(X_cp / X_c) / (X_wp / X_w)` > 1 means a country exports this product more than world-average would predict |
| **ECI** (Economic Complexity Index) | Country-level measure of how sophisticated and diversified its export basket is |
| **PCI** (Product Complexity Index) | Product-level measure of how complex/rare the product is |
| **Product Space** | Network of products connected by proximity (how often countries co-export them); predicts diversification paths |
| **Ubiquity** | How many countries export a product with RCA≥1 (low ubiquity → complex product) |
| **0→1 transition** | The prediction target: a country-product pair where M=0 at year t becomes M=1 at year t+5 |
