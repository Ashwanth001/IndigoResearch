# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

**Trade Complexity 2.0** — predicts which products a country will gain Revealed Comparative Advantage (RCA ≥ 1) in over the next 5 years, using a temporal bipartite GNN trained on global country–product export networks (BACI HS92, 1995–2024), with an optional LLM-derived product capability graph layer.

See `CONTEXT.md` for the full research context, hypotheses, data schema, results table, and pending work.

---

## Running the Pipeline

The canonical entry point is `dataPipeline.ipynb` — all 12 steps run top-to-bottom in a single notebook.

**Standalone fallbacks** (use when the notebook kernel crashes or for isolated reruns):

```powershell
# Step 11A — LLM embeddings (~20 min, downloads FinLang model on first run)
python run_step11_embeddings.py

# Train GNN-4F and GNN-11F and save checkpoints to data/checkpoints/
python train_and_save_gnns.py

# Recompute baseline CWR values using percentile-ranked scores
python compute_baseline_cwr.py
```

**Full evaluation** (all 8 methods, produces comparison plots and `data/full_evaluation_results.csv`):
Run `evaluation.ipynb` top-to-bottom. GNN cells load from `data/models/gnn/checkpoints/` if present, otherwise train from scratch (~5 min per GNN).

GNN training exploration notebook: `gnn_training.ipynb`.

---

## Architecture

Three-module stack defined inline in `dataPipeline.ipynb` (Step 12) and mirrored in `train_and_save_gnns.py` and `evaluation.ipynb`:

```
BipartiteEncoder
  country_lin: Linear(c_in → 128)    # c_in = 4 (GNN-4F) or 11 (GNN-11F)
  product_lin: Linear(3 → 128)
  gnn: to_hetero(_HomoGNN)            # SAGEConv(128→128) × 2, ReLU + dropout(0.3)

TemporalGNN
  BipartiteEncoder applied to each of 5 yearly snapshots
  GRU over 5-year country sequence → z_c[-1]
  GRU over 5-year product sequence → z_p[-1]

LinkPredictor
  MLP: Linear(256→128) → ReLU → Dropout(0.2) → Linear(128→1) → sigmoid
```

`_HomoGNN.forward` **must** use the parameter name `edge_index` (not `ei`) — PyG's `to_hetero` fx tracer requires it exactly.

**GNN+LLM variant** adds `('product','capability','product')` edges to each snapshot. These come from `data/capability_edge_index.pt` (top-20 nearest neighbours in FinLang embedding space). The checkpoint is `data/models/gnn/checkpoints/gnn_11f_llm.pt`.

---

## Data Pipeline — Critical Leakage Rules

Violating any of these would invalidate the experiment:

1. **Smoothing (Step 3):** Trailing window only — `mean(t-2, t-1, t)`. Never centered `(t-1, t, t+1)`.
2. **Labels (Step 4):** Positive = `(~M[t]) & M[t+5]`. Never `M[t+5] == 1` directly (would inflate positive rate ~10×).
3. **Split (Step 8):** Strict year cutoff — `year ≤ TRAIN_CUTOFF` for training. Never sklearn random split.
4. **WDI normalization (Step 9):** Compute mean/std from training years (≤ 2012) only; apply those stats to all years.

---

## Key Constants (Cell 2 of dataPipeline.ipynb)

| Constant | Value | Notes |
|----------|-------|-------|
| `TRAIN_CUTOFF` | 2012 | Years ≤ 2012 → training set |
| `VAL_YEAR` | 2013 | Validation observation year |
| `TEST_YEAR` | 2015 | Test observation year (2-year gap from val is intentional) |
| `LABEL_HORIZON` | 5 | Predict transition h=5 years ahead |
| `NEG_RATIO` | 5 | Negatives sampled per positive (~17% positive rate) |
| `HIDDEN` | 128 | GNN hidden dimension |
| `EPOCHS` | 80 | Max training epochs (patience=15 early stopping) |

---

## HeteroData Schema

Each sample is a dict `{'snapshots': [HeteroData×5], 'labels': {...}, 'year': int, 'countries_raw': arr, 'products_raw': arr}`.

Each snapshot:
- `data['country'].x` — `[233, 4]` (GNN-4F) or `[233, 11]` (GNN-11F)
- `data['product'].x` — `[5018, 3]`
- `data['country','exports','product'].edge_index` — `[2, ~90k]`
- `data['product','rev_exports','country'].edge_index` — `[2, ~90k]` (required for SAGEConv)
- `data['product','capability','product'].edge_index` — `[2, 144192]` (GNN+LLM only)

Edge indexes must be cast to `torch.long` at load time and in `build_snap`/`to_dev` — `edge_index_by_year.pt` was originally saved as float.

---

## Evaluation — 3-Tier Framework

| Tier | Metric | What it measures |
|------|--------|-----------------|
| 1 — Global | **PR-AUC** | Overall discrimination (~3–5% positive rate; prefer over AUROC) |
| 2 — Economic | **CWR** (Complexity-Weighted Recall) | Gets the hard, valuable (low-ubiquity) transitions right |
| 3 — Investor | **NDCG@20 / Prec@20** (macro, per-country) | Ranks the top predictions correctly |

**CWR** uses percentile-ranked scores so the 0.5 threshold means "top half of predictions" — makes it method-agnostic (works for RCA values, densities, and sigmoid probabilities alike).

---

## Tech Stack

- Python 3.14, PyTorch, PyTorch Geometric (`HeteroData`, `SAGEConv`, `to_hetero`)
- `sentence-transformers` — `FinLang/finance-embeddings-investopedia` (768-dim) for product embeddings
- scikit-learn, scipy (baselines); XGBoost/LightGBM (tabular baseline in `oldResearch/`)
- CUDA used throughout; falls back to CPU automatically

## File Layout

```
dataPipeline.ipynb          ← Active pipeline — Steps 1–12, all in one notebook
evaluation.ipynb            ← Full 8-method comparison (run after pipeline)
gnn_training.ipynb          ← GNN training exploration
run_step11_embeddings.py    ← Standalone Step 11A fallback
train_and_save_gnns.py      ← Standalone GNN training fallback
compute_baseline_cwr.py     ← Recomputes CWR for classical baselines
data/                       ← All pipeline outputs (CSVs + .pt tensors)
datasets/BACIDataset1995/   ← BACI HS92 1995–2024 (30 annual CSVs)
datasets/WDI_csv/           ← World Bank WDI (WDICSV.csv)
oldResearch/                ← Archived old step scripts, baselines, checkpoints
```
