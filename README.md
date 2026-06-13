# Trade Complexity 2.0

Predicts which products a country will gain Revealed Comparative Advantage (RCA ≥ 1) in over the next **5 years**, using a temporal bipartite GNN trained on global country–product export networks (BACI HS92, 1995–2024), with an optional LLM-derived product capability graph layer.

---

## What this project does

Given the global trade network at year *t*, the model scores every (country, product) pair where the country does **not** currently export the product, and predicts which ones will reach RCA ≥ 1 by year *t+5*. This is framed as a link-prediction problem on a temporal bipartite graph.

---

## Methods compared

| Family | Methods |
|--------|---------|
| Classical baselines | RCA Persistence, Density, ECI, ECI + Density |
| Embedding baseline | KNN (PCA-32 LLM embeddings) |
| Tabular ML | XGBoost (51-dim feature vector, GPU-trained) |
| GNN | GNN-4F, GNN-11F (BACI+WDI), GNN-11F+LLM, GNN-LLM PCA variants |

---

## Evaluation framework (3 tiers)

| Tier | Metric | What it captures |
|------|--------|-----------------|
| 1 — Global | **PR-AUC** | Overall discrimination (~17% positive rate — prefer over AUROC) |
| 2 — Economic | **CWR** (Complexity-Weighted Recall) | Gets the hard, high-complexity transitions right |
| 3 — Investor | **NDCG@20 / Prec@20** | Correctly ranks the top predictions per country |

Evaluated on two held-out test years (t=2015, t=2016) to check cross-year stability.

---

## Running the pipeline

### 1. Build data artifacts
Run `dataPipeline.ipynb` top-to-bottom (Steps 1–12). This produces all `.pt` tensors and CSVs in `data/`.

### 2. Train XGBoost (GPU)
```powershell
python3.14 ClaudeFiles/train_xgboost.py
```
Saves checkpoint to `data/models/xgboost/xgb_model.pkl` (~5 min on GPU).

### 3. Train GNNs
Run `new_gnn_training_fixed.ipynb`, or use the standalone fallback:
```powershell
python3.14 train_and_save_gnns.py
```
Checkpoints saved to `data/models/gnn/checkpoints/`.

### 4. Evaluate — internal benchmarking (sampled test set)
Run `internal_benchmarking.ipynb` top-to-bottom. Produces tables and plots for all methods on the stratified-sampled test labels.

### 5. Evaluate — full universe
Run `full_universe_eval.ipynb` top-to-bottom. Scores every (country, product) pair in the full universe (no sampling). Results saved to `full_universe_eval/`.

---

## GNN architecture

```
BipartiteEncoder
  country_lin : Linear(c_in → 128)    # c_in = 4 (GNN-4F) or 11 (GNN-11F)
  product_lin : Linear(3 → 128)
  gnn         : to_hetero(SAGEConv(128→128) × 2, ReLU + dropout(0.3))

TemporalGNN
  BipartiteEncoder applied to each of 5 yearly snapshots
  GRU over 5-year country sequence → z_c[-1]
  GRU over 5-year product sequence → z_p[-1]

LinkPredictor
  MLP: Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→1) → sigmoid
```

The GNN+LLM variant adds `('product','capability','product')` edges derived from top-20 nearest neighbours in FinLang (768-dim) embedding space, PCA-compressed to 32 dims.

---

## XGBoost feature vector (51 dims)

| Group | Dims | Features |
|-------|------|---------|
| Country (BACI) | 4 | log_export, n_products, avg_rca, max_rca |
| Country (WDI) | 7 | gdp_pc, capital_formation, tertiary_enrollment, fdi_inflows, manufacturing_va, internet_users, population |
| Product | 3 | log_world_export, ubiquity, avg_rca |
| PCA-LLM | 32 | Top-32 PCA components of FinLang product embeddings (L2-normalised) |
| Network | 2 | Density score, ECI |
| RCA history | 3 | Raw RCA at t-2, t-1, t |

---

## File layout

```
dataPipeline.ipynb              Active pipeline — Steps 1–12
evaluation.ipynb                Early 8-method comparison notebook
internal_benchmarking.ipynb     All methods on stratified-sampled test set (t=2015, 2016)
full_universe_eval.ipynb        All methods on full (country, product) universe
new_gnn_training_fixed.ipynb    GNN training with PCA-LLM capability edges
gnn_training.ipynb              GNN training exploration

ClaudeFiles/
  train_xgboost.py              GPU XGBoost training script
  plot_ib_results.py            Plots for internal_benchmarking (5 figures)
  plot_results.py               Plots for full_universe_eval (4 figures)
  fix_encoding.py               Mojibake fix utility
  check_and_fix.py              Encoding check + fix utility
  patch_notebooks.py            Notebook patching utility

data/                           Pipeline outputs (.pt tensors, CSVs, model checkpoints)
datasets/BACIDataset1995/       BACI HS92 1995–2024 (30 annual CSVs)
datasets/WDI_csv/               World Bank WDI (WDICSV.csv)
internal_benchmarking/          Result CSVs + plots (internal eval)
full_universe_eval/             Result CSVs + plots (full universe eval)
oldResearch/                    Archived scripts and baselines
```

---

## Key constants

| Constant | Value | Notes |
|----------|-------|-------|
| `TRAIN_CUTOFF` | 2012 | Years ≤ 2012 used for training |
| `VAL_YEAR` | 2013 | Validation observation year |
| `TEST_YEAR` | 2015 | Primary test year (2-year gap from val is intentional) |
| `LABEL_HORIZON` | 5 | Predict RCA gain *h=5* years ahead |
| `NEG_RATIO` | 5 | Negatives sampled per positive (~17% positive rate) |
| `HIDDEN` | 128 | GNN hidden dimension |
| `EPOCHS` | 80 | Max training epochs (patience=15 early stopping) |

---

## Data leakage rules

These must not be violated or the experiment is invalid:

1. **Smoothing:** Trailing window only — `mean(t-2, t-1, t)`. Never centered.
2. **Labels:** Positive = `(~M[t]) & M[t+5]`. Never `M[t+5] == 1` directly.
3. **Split:** Strict year cutoff — `year ≤ TRAIN_CUTOFF` for training. No random splits.
4. **WDI normalisation:** Compute mean/std from training years (≤ 2012) only; apply those stats to all years.

---

## Tech stack

- Python 3.14, PyTorch, PyTorch Geometric (`HeteroData`, `SAGEConv`, `to_hetero`)
- `sentence-transformers` — `FinLang/finance-embeddings-investopedia` (768-dim)
- scikit-learn, scipy, XGBoost — baselines and tabular model
- CUDA throughout; falls back to CPU automatically
