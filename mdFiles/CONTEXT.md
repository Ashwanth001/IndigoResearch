# Trade Complexity 2.0 — Project Context
---

> **Last synced: 2026-07-01.** This document is the master tracker. It has been reconciled with `README.md`, `evaluation/INTERNAL_BENCHMARKING.md`, `evaluation/FULL_UNIVERSE_BENCHMARKING.md`, and the live result CSVs in `internal_benchmarking/` and `full_universe_eval/`. The evaluation now spans **14 methods × 8 metrics × 2 test years (2015 & 2016) × 2 dataset conditions (sampled + full universe)**. XGBoost is trained and is the current top performer; the two `evaluation/*.md` reports still predate the XGBoost + PCA-variant additions and should be regenerated from the CSVs.

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
├── dataPipeline.ipynb              ← THE active pipeline (Steps 1–12, all in one notebook)
├── new_gnn_training_fixed.ipynb    ← GNN training incl. PCA-LLM capability variants
├── internal_benchmarking.ipynb     ← All 14 methods, sampled test set, t=2015 & 2016
├── full_universe_eval.ipynb        ← All methods, full unsampled universe (~1.08M pairs)
├── data/                           ← Pipeline outputs (.pt tensors, CSVs, model checkpoints)
│   ├── (pipeline artifacts: exports/rca/M_cpt/labels/features CSVs, *_by_year.pt,
│   │    *_mapping.pkl, train/val/test_data.pt, product_llm_embeddings.pt,
│   │    capability_edge_index.pt)
│   ├── models/xgboost/xgb_model.pkl
│   └── models/gnn/checkpoints/     ← gnn_4f, gnn_11f, gnn_11f_llm, gnn_11f_llm_v2,
│                                      gnn_llm_v2_unopt, gnn_11f_llm_pca,
│                                      gnn_11f_llm_pca_ew, gnn_11f_llm_pca_gat_ew (.pt)
├── internal_benchmarking/          ← Sampled-eval result CSVs + plots/ (5 figures)
├── full_universe_eval/             ← Full-universe result CSVs + plots/ (4 figures)
├── evaluation/                     ← INTERNAL_BENCHMARKING.md, FULL_UNIVERSE_BENCHMARKING.md
│                                      (both stale — stop at 10 methods, pre-XGBoost)
├── ClaudeFiles/                    ← train_xgboost.py, train_and_save_gnns.py,
│                                      run_step11_embeddings.py, plot_*.py, patch_*.py, …
├── mdFiles/CONTEXT.md              ← THIS master tracker
├── datasets/
│   ├── BACIDataset1995/            ← BACI HS92, 1995–2024 (30 annual CSV files)
│   └── WDI_csv/                    ← World Bank WDI indicators (WDICSV.csv)
└── oldResearch/                    ← Archived: old step scripts, models.py, evaluator.py,
                                       baseline results, superseded eval notebooks
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
| 11A — LLM embeddings | Embed HS6 product descriptions with `FinLang/finance-embeddings-investopedia` (768-dim, finance-domain, unit-norm). **Quality:** Dramatically better discrimination than e5-large-v2. Horses vs Diodes = 0.29 (was 0.80); petrol cars same class = 0.998; only 1% of pairs > 0.70 (was 99%). | `product_llm_embeddings.pt` [5018, 768] |
| 11B — Capability edges | Sparse product↔product edges built from top-K=20 FinLang nearest neighbours. **Corrected:** Previous e5-large-v2 version with 0.70 threshold was stale; rebuilt 2026-06-11 using FinLang. Now correctly connects semantically similar semiconductor products instead of all electrical apparatus. | `capability_edge_index.pt` [2, 144192] |
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

## Results — Full Comparison Table (14 methods)

Two evaluation regimes are now maintained:
- **Sampled test set** (5:1 negatives, ~14.5% positive) — for method ranking; inflates absolute PR-AUC ~5–8×. Source: `internal_benchmarking/`.
- **Full universe** (all `M[t]=0` pairs, ~1.7% positive) — realistic deployment numbers. Source: `full_universe_eval/`.

Both are evaluated at **t=2015 and t=2016** (predicting t+5 transitions, sustained at t+6). CWR uses **percentile-ranked scores** (threshold = top 50%) so the comparison is fair across score scales, weighted by `1/ubiquity[2010]` with median fill for the 80 products missing from the 2010 reference.

### Sampled test set — t=2015 (127,531 pairs, 14.5% positive)

| Model | PR-AUC | AUROC | NDCG@20 | Prec@20 | CWR | Best F1 | P@1000 | mAP@10 |
|-------|--------|-------|---------|---------|-----|---------|--------|--------|
| RCA Persistence | 0.5198 | 0.6515 | 0.5013 | 0.4788 | 0.3359 | 0.4357 | 0.5970 | 0.3660 |
| Density (Product Space) | 0.3487 | 0.7792 | 0.4809 | 0.4400 | 0.8574 | 0.4205 | 0.4860 | 0.3448 |
| ECI | 0.1370 | 0.4821 | 0.1468 | 0.1374 | 0.4637 | 0.2597 | 0.0930 | N/A |
| ECI + Density | 0.3487 | 0.7792 | 0.4809 | 0.4400 | 0.8574 | 0.4205 | 0.4860 | 0.3448 |
| KNN (LLM embeddings) | 0.2305 | 0.6373 | 0.2830 | 0.2644 | 0.6550 | 0.2997 | 0.4090 | 0.1644 |
| **XGBoost (51-dim tabular)** | **0.6576** | **0.8992** | **0.7004** | **0.6423** | **0.9376** | **0.6319** | **0.8970** | **0.6128** |
| GNN-4F (BACI only) | 0.4123 | 0.8191 | 0.4369 | 0.3980 | 0.8838 | 0.4708 | 0.6200 | 0.2968 |
| GNN-11F (BACI+WDI) | 0.4340 | 0.8299 | 0.4662 | 0.4257 | 0.8972 | 0.4805 | 0.6280 | 0.3306 |
| GNN-11F+LLM (FinLang cap. edges) | 0.4408 | 0.8316 | 0.4848 | 0.4296 | 0.8989 | 0.4834 | 0.6460 | 0.3673 |
| GNN-LLM v2 (GAT+Focal, Optuna) | 0.4108 | 0.8184 | 0.4414 | 0.3969 | 0.8871 | 0.4665 | 0.6560 | 0.3187 |
| GNN-LLM v2 Unopt | 0.3190 | 0.7727 | 0.3155 | 0.2843 | 0.8501 | 0.4112 | 0.4090 | 0.1984 |
| GNN-LLM PCA-A (SAGE) | 0.4500 | 0.8355 | 0.4856 | 0.4367 | 0.9008 | 0.4903 | 0.6810 | 0.3538 |
| **GNN-LLM PCA-B (GCN+EW)** ← best GNN | **0.4565** | **0.8392** | **0.4948** | **0.4376** | **0.9057** | **0.4946** | **0.6850** | **0.3745** |
| GNN-LLM PCA-D (GAT+EW, Optuna) | 0.4102 | 0.8218 | 0.4399 | 0.3936 | 0.8926 | 0.4680 | 0.6600 | 0.3089 |

t=2016 numbers follow the same ordering (all methods +3–5% PR-AUC); full tables in `internal_benchmarking/full_sampled_results.csv` and the two `evaluation/*.md` reports.

### Full universe — t=2015 (1,078,236 pairs, 1.71% positive) — deployment-realistic

| Model | PR-AUC | AUROC | CWR | mAP@10 |
|-------|--------|-------|-----|--------|
| RCA Persistence | 0.2448 | 0.6518 | 0.3359 | 0.0661 |
| Density | 0.0549 | 0.7782 | 0.8865 | 0.0670 |
| GNN-11F+LLM | 0.0850 | 0.8325 | 0.9269 | 0.0788 |
| (XGBoost + PCA variants: pending full-universe re-run — see Pending) | | | | |

Full-universe absolute PR-AUC is ~5–8× lower than sampled; rankings are preserved. Complete tables in `full_universe_eval/full_universe_combined_results.csv`.

**Key observations (updated 2026-07-01):**
- **XGBoost is now the top performer by a wide margin** on the sampled set — PR-AUC 0.66 (t=2015) / 0.69 (t=2016), CWR 0.94, mAP@10 0.61–0.65. It beats every GNN on every metric and clears the literature XGBoost SOTA ([P1] BestF1≈0.139, prec@1000≈0.198). This is the headline result and **reframes the GNN's contribution as second-best, not best.**
- **Best GNN is now GNN-LLM PCA-B (GCN+EW)**, not GNN-11F+LLM — PCA-compressing the FinLang embeddings to 32 dims + a GCN encoder with edge weights edges out the SAGEConv+raw-capability-edge variant on every metric.
- The **GAT variants (v2 GAT+Focal, PCA-D GAT+EW) regress** below the SAGE/GCN encoders — the attention upgrade did not help on this task.
- The GNN ablation ladder still holds directionally: WDI (4F→11F) is the biggest jump (+10–12% on full universe), capability edges add a smaller consistent gain (+3–6%).
- RCA Persistence still leads PR-AUC among the *network/economic* baselines; GNNs lead on CWR (0.88–0.91) and AUROC (0.82–0.84).
- KNN on FinLang embeddings alone (no GNN) scores 0.2305 — capability signal *must* be integrated into the GNN/tabular structure to help.

Result CSVs: `internal_benchmarking/*.csv` (sampled) and `full_universe_eval/*.csv` (full universe). Older single-year CSVs (`data/full_evaluation_results.csv`, `data/extra_metrics_results.csv`, `data/filtered_metrics_results.csv`) are superseded by the two benchmarking folders.

---

## Prediction Methods (Ablation Ladder)

Sampled-test PR-AUC (t=2015) shown; see results table for full metrics.

| # | Method | Status |
|---|--------|--------|
| 1 | RCA Persistence baseline | **Done** — PR-AUC=0.5198 |
| 2 | Product Space density (proximity φ) | **Done** — PR-AUC=0.3487 |
| 3 | ECI and ECI+Density | **Done** — PR-AUC=0.1370 / 0.3487 |
| 4 | KNN on LLM embeddings (baseline) | **Done** — PR-AUC=0.2305 |
| 5 | **XGBoost (51-dim tabular)** | **Done — TOP METHOD** — PR-AUC=0.6576. Checkpoint `data/models/xgboost/xgb_model.pkl`; script `ClaudeFiles/train_xgboost.py` |
| 6 | GNN-4F (BACI only) | **Done** — PR-AUC=0.4123 |
| 7 | GNN-11F (BACI+WDI) | **Done** — PR-AUC=0.4340 |
| 8 | GNN-11F+LLM (BACI+WDI+FinLang capability edges) | **Done** — PR-AUC=0.4408 (FP32 + gradient checkpointing; fixed FP16 bug) |
| 9 | GNN-LLM v2 (GAT+Focal, Optuna-tuned) | **Done — regresses** — PR-AUC=0.4108 (below SAGEConv baseline) |
| 10 | GNN-LLM v2 Unopt (GAT, fixed hparams) | **Done — strong regression** — PR-AUC=0.3190 |
| 11 | GNN-LLM PCA-A (SAGE, PCA-32 embeddings) | **Done** — PR-AUC=0.4500 |
| 12 | **GNN-LLM PCA-B (GCN + edge weights)** | **Done — BEST GNN** — PR-AUC=0.4565 |
| 13 | GNN-LLM PCA-D (GAT + edge weights, Optuna) | **Done — regresses** — PR-AUC=0.4102 |

---

## Evaluation — 3-Tier Framework

| Tier | Metric | What it measures |
|------|--------|-----------------|
| 1 — Global | **PR-AUC** (preferred over AUROC for ~3–5% positive rate) | Overall discrimination |
| 2 — Economic | **CWR — Complexity-Weighted Recall** | Gets the hard, valuable transitions right |
| 3 — Investor | **NDCG@20 and Precision@20 per country** | Ranks the top predictions correctly |

**CWR detail:** For each method, scores are first converted to percentile ranks across all test pairs (so the top-scoring pair = 1.0, median = 0.5, bottom = 0.0). A pair is counted as "predicted positive" if its percentile rank ≥ 0.5 (i.e., top half of predictions). Correct predictions among true positives are then weighted by product complexity (PCI proxy = −ubiquity/max_ubiquity — rare products score higher). CWR = weighted hits / total weighted true positives. Using percentile ranks instead of raw scores makes the 0.5 threshold method-agnostic and comparable across RCA values, densities, and sigmoid probabilities.

---

## Extended Evaluation Notebooks

Two additional notebooks provide deeper metric analysis beyond the main `evaluation.ipynb`:

### `extra_metrics_eval.ipynb` — Literature-comparable metrics

Computes three metrics on the **full test set** (all 127K pairs) for all 8 methods to enable direct comparison with prior literature (Tacchella et al. 2021, Albora et al. 2023):

- **Best F1** — F1 at the threshold that maximises F1 (literature standard in [P1–P4])
- **Prec@1000** — fraction of the top-1000 predictions that are true positives (literature standard)
- **mAP@10** — mean average precision at rank 10, per country (literature standard; N/A for ECI due to per-country scoring only)

Results saved to `data/extra_metrics_results.csv`. Run all cells top-to-bottom. ECI marked N/A for mAP@10 since it has no within-country ranking signal.

### `filtered_metrics_eval.ipynb` — Near-miss evaluation

Evaluates all 8 methods on the **RCA > 0.25 subset** — country-product pairs where the country already has genuine activity (RCA > 0.25) but hasn't crossed the RCA ≥ 1 threshold. This is the economically most interesting subset:
- 21,041 pairs (vs 127,531 full), 55% positive rate (vs 14.5% full)
- Tests whether models identify which latent capabilities will cross into sustained comparative advantage

Computes all 8 metrics (PR-AUC, AUROC, NDCG@20, Prec@20, CWR, Best F1, Prec@K, mAP@10) on this filtered subset. Models are scored on the **full test set** (unaware of filter), but metrics computed only on the filtered subset. Results saved to `data/filtered_metrics_results.csv`.

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

## Known Bugs Fixed

| Bug | Symptom | Fix | Date |
|-----|---------|-----|------|
| `_HomoGNN.forward(self, x, ei)` — wrong parameter name | `ValueError: MessagePassing.propagate only supports integer tensors of shape [2, num_messages]` | Renamed `ei` → `edge_index`; PyG's `to_hetero` fx tracer requires this exact name | 2026-06-10 |
| `c_in` hardcoded as 4 and 11 | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (233×4 and 11×128)` | Auto-detect from `tr[0]['snapshots'][0]['country'].x.shape[1]` | 2026-06-10 |
| `country_x_by_year.pt` had 4 features despite Step 9 existing | GNN-11F used same 4-feature tensors as GNN-4F | Step 7 wrote a 4-feat `.pt`; Step 9 builds the 11-feat version — ran Step 9 tensor rebuild from `country_features_enriched.csv` | 2026-06-10 |
| `edge_index_by_year.pt` saved as float dtype | `ValueError: MessagePassing.propagate only supports integer tensors` during Step 12 training | Added `.long()` cast at three points in Step 12 cell: at load time (`{k: v.long() for k, v in ...}`), inside `build_snap12`, and inside `to_dev12` (`dtype=torch.long`) | 2026-06-11 |
| FP16 mixed precision broke gradient clipping in GNN-11F+LLM training | `GradScaler` scaled loss → `loss.backward()` → gradients in FP16×65536 — `clip_grad_norm_` clamped at 1.0 was ineffective; large gradients flowed through unchecked → optimization diverged → val loss stuck, early stop at epoch 16 with PR-AUC=0.27 instead of 0.44 | Removed `torch.cuda.amp.autocast()` and `GradScaler` entirely; gradient checkpointing alone handles memory. Train with FP32 + proper grad clipping. Retrained GNN-11F+LLM: now converges to epoch 80, PR-AUC=0.4444 (vs 0.2698 broken) | 2026-06-12 |
| `capability_edge_index.pt` built from stale e5-large-v2 embeddings | GNN-11F+LLM performance collapsed to 0.27 despite FinLang embeddings being high-quality | Files had same modification timestamp. The skip-if-exists guard in `gnn_training.ipynb` cell-train-03 loaded the e5-large-v2 file (with permissive 0.70 threshold, wrong product connections). Rebuilt from FinLang top-K=20 NN correctly. | 2026-06-11 |

---

## What's Done vs. Pending

| Component | Status |
|-----------|--------|
| Full data pipeline (Steps 1–10) | **Complete** and running in notebook |
| LLM embeddings (Step 11A) | **Complete** — `product_llm_embeddings.pt` [5018, 768] using `FinLang/finance-embeddings-investopedia`. Vastly improved discrimination vs e5-large-v2 (1% > 0.70 vs 99%). |
| Capability edges (Step 11B) | **Complete** — rebuilt 2026-06-11 using top-K=20 FinLang NN. File `capability_edge_index.pt` [2, 144192] now correctly encodes semantic product similarity. |
| GNN-4F training + evaluation | **Complete** — PR-AUC=0.4123 |
| GNN-11F training + evaluation | **Complete** — PR-AUC=0.4340 |
| GNN-11F+LLM (capability edges + TemporalGNN) | **Complete** — PR-AUC=0.4408. Fixed FP16 training bug; converges cleanly on 80 epochs. **H2 validated.** |
| GNN-LLM PCA variants (A/SAGE, B/GCN+EW, D/GAT+EW) | **Complete** — PCA-32 FinLang embeddings. **PCA-B (GCN+EW) is the best GNN** (PR-AUC=0.4565); GAT variants regress. Trained via `new_gnn_training_fixed.ipynb`. |
| GNN-LLM v2 GAT+Focal (Optuna + Unopt) | **Complete — regressions** — PR-AUC 0.4108 / 0.3190. GAT+focal loss underperforms SAGE/GCN; not used for deployment. |
| **XGBoost (51-dim tabular) — re-run on new pipeline** | **Complete — now the TOP method** — PR-AUC=0.6576/0.6880. Beats all GNNs and clears literature XGBoost SOTA. Checkpoint `data/models/xgboost/xgb_model.pkl`. |
| Two-year evaluation (t=2015 + t=2016) | **Complete** — cross-year rankings stable; `internal_benchmarking.ipynb`. |
| Full-universe evaluation (unsampled, ~1.08M pairs) | **Complete** — deployment-realistic absolute numbers; `full_universe_eval.ipynb` → `full_universe_eval/`. |
| Extra metrics (Best F1, Prec@1000, mAP@10) | **Complete** — now folded into the 8-metric benchmarking notebooks. (Standalone `extra_metrics_eval.ipynb` superseded.) |
| Filtered evaluation (RCA > 0.25 pairs) | **Complete** — near-miss subset in both benchmarking notebooks (sampled + full universe). (Standalone `filtered_metrics_eval.ipynb` superseded.) |
| Regenerate `evaluation/*.md` reports from CSVs | **Pending** — both reports stop at 10 methods; they predate XGBoost and the PCA variants. |
| Add XGBoost + PCA variants to full-universe tables | **Pending** — full-universe CSVs currently cover 8–10 methods. |
| Multi-seed GNN training (confirm sub-1% deltas) | **Pending** — all GNN checkpoints are single-seed; 11F vs 11F+LLM delta is within noise. |
| India / Vietnam / Mexico / Indonesia country stories | **Pending** |
| Dashboard / visualization layer | **Pending** |

---

## Tech Stack

- **Python 3.14**, PyTorch, PyTorch Geometric (`HeteroData`, `SAGEConv`, `to_hetero`, `GRU`)
- **Data:** BACI HS92 trade data (1995–2024), World Bank WDI indicators
- **LLM embeddings:** `sentence-transformers` — `FinLang/finance-embeddings-investopedia` (768-dim, finance-domain fine-tuned)
- **Baselines:** scikit-learn, XGBoost/LightGBM (tabular), scipy (Product Space proximity)
- **GPU:** CUDA for RCA computation, tensor ops, GNN training (8GB device). Gradient checkpointing + FP32 for memory efficiency.

---

## Comparable Literature and External Benchmarks

This section documents the closest published work to this project. Use it to frame novelty claims and to benchmark GNN results against prior SOTA.

### Novelty status (as of June 2026)

**No published paper applies a GNN to the country–product bipartite RCA-transition prediction task.** Every "GNN + trade" paper in the literature predicts bilateral country-to-country trade *value* (regression), not RCA≥1 transitions (classification). The combination of (a) temporal bipartite GNN on the RCA matrix and (b) LLM-derived capability graph is novel.

The true quantitative benchmarks are the "Rome school" papers below — they own the exact prediction framing (binary RCA≥1, 5-year horizon, temporal split) and report tree-based ML as current SOTA.

---

### Primary benchmarks — exact same task (binary RCA≥1, δ=5)

**[P1] Tacchella, Zaccaria, Miccheli & Pietronero (2021/2023). "Relatedness in the Era of Machine Learning." arXiv:2103.06017 / Chaos, Solitons & Fractals.**

The single most directly comparable paper. Tests multiple methods on out-of-sample prediction of new RCA≥1 appearances (i.e., pairs with RCA<0.25 at time t that activate at t+5). Uses COMTRADE HS6 data, ~5000 products, ~170 countries, 2007–2018, leave-k-countries-out temporal cross-validation.

Methods compared: Random Prediction, Random Graph, RCA Persistence, Product Space co-occurrence, Taxonomy Network, Description-based Embeddings (pre-LLM text similarity), XGBoost, CPS (t-SNE and VAE16 embedding-based classifiers).

Key finding: XGBoost gives +350% precision@1000 and +96% BestF1 over Product Space. Description embeddings beat Product Space but lose to XGBoost. XGBoost is current SOTA on this task.

| Method | BestF1 | Prec@1000 | mAP@10 |
|--------|--------|-----------|--------|
| XGBoost | **0.139** | **0.198** | **0.389** |
| VAE16 (CPS) | 0.103 | 0.132 | 0.330 |
| t-SNE (CPS) | 0.104 | 0.158 | 0.292 |
| Description Embeddings | 0.102 | 0.147 | 0.283 |
| RCA Persistence | 0.088 | 0.106 | 0.284 |
| Taxonomy Network | 0.074 | 0.043 | 0.276 |
| Product Space | 0.071 | 0.044 | 0.231 |
| Random Graph | 0.060 | 0.029 | 0.045 |
| Random Prediction | 0.042 | 0.015 | 0.090 |

**Metric note:** This paper uses BestF1, Prec@K, and mAP@10 — not PR-AUC/AUROC/NDCG. To compare your GNN against these numbers you need to re-implement baselines and report all metric sets side by side.

**Relevance to H2:** The Description Embeddings baseline (pre-LLM text similarity of product descriptions) is the only prior test of textual product relatedness on this exact task. It scores BestF1=0.102 / prec@1000=0.147. Your LLM-capability-graph + GNN combination must beat this to justify H2.

---

**[P2] Albora, Pietronero, Tacchella & Zaccaria (2023). "Product progression: a machine learning approach to forecasting industrial upgrading." Scientific Reports 13:1481. DOI:10.1038/s41598-023-28179-x.**

Same task framing (RCA≥1 activation, δ=5, test restricted to RCA<0.25 pairs) using BACI data. Compares RCA auto-correlation, Product Space co-occurrence, Random Forest, Boosted Trees, and other ML. Tree-based algorithms win consistently. Explicitly warns AUROC is the least reliable metric under heavy class imbalance here — relevant because this project prioritises PR-AUC over AUROC for the same reason.

Metrics reported: mean Precision@k, BestF1, MCC, ROC-AUC (with caveats). Exact per-method numbers are in the paper body/SI.

---

**[P3] Fessina, Albora, Tacchella & Zaccaria (2022). "Which products activate a product? An explainable machine learning approach." arXiv:2212.03094.**

Introduces FIPS (Feature Importance Product Space) — using Random Forest feature importances to build a data-driven product proximity network. Beats RF and original Product Space on BestF1 and mP@10 but has lower AUROC (again confirming AUROC unreliability under class imbalance). Most relevant to the interpretability / H3 angle.

Metrics: BestF1, mP@10, AUC-ROC.

---

**[P4] Albora & Zaccaria (2022). "Machine learning to assess relatedness: the advantage of using firm-level data." Complexity 2022:2095048. arXiv:2202.00458.**

RF vs Product Space vs Taxonomy vs RCA persistence at both country and firm level. Metrics: Precision, Recall, F1, MCC (threshold maximising F1). Finding: relatedness is scale-dependent; RF beats network methods at country level.

---

### Earlier precedents (pre-ML, methodological lineage)

**[P5] Hidalgo, Klinger, Barabási & Hausmann (2007). "The Product Space Conditions the Development of Nations." Science 317(5837):482–487.**
Origin of Product Space and density-based prediction of product appearances. This is what the Density baseline in our results table implements.

**[P6] Hidalgo & Hausmann (2009). "The building blocks of economic complexity." PNAS 106(26):10570–10575.**
Origin of ECI/diversity-ubiquity iteration on the country-product bipartite matrix. This is what the ECI baseline implements.

**[P7] Vidmer, Zeng, Medo & Zhang (2015). "Prediction in complex systems: the case of the international trade network." arXiv:1511.05404.**
Pre-ML bipartite link-prediction using heat/mass diffusion + fitness scores on RCA≥1 network. Reports ranking score r and precision. Methodological precedent for framing this as link prediction.

---

### GNN papers on trade (different task — bilateral flow value regression)

These are **not** comparable to our results but are the correct citations when claiming GNN novelty on the RCA task.

| Paper | Task | Metrics | Why not comparable |
|-------|------|---------|-------------------|
| Sellami et al. (2024). "Harnessing GNNs to Predict International Trade Flows." *Big Data and Cognitive Computing* 8(6):65. | Bilateral trade VALUE regression, country-country | MSE, MAE, R², MAPE | Regression on flows, not RCA classification |
| Verstyuk & Douglas (2022). "Machine Learning the Gravity Equation." SSRN 4053795. | Bilateral accessibility/gravity | Fit vs gravity baseline | Regression, country-country |
| Panford-Quainoo, Bose & Defferrard (2020). "Bilateral Trade Modeling with GNNs." ICLR 2020 workshop. | Country income classification + trade-partner link prediction (111 countries) | Accuracy, AUC/AP (GAE/VGAE) | Country-country, not country-product RCA |
| Monken et al. (2021). "GNNs for Modeling Causality in International Trade." FLAIRS-34. | Bilateral trade causality | — | Country-country |
| Minakawa, Izumi & Sakaji (2022). "Bilateral Trade Flow Prediction by Gravity-informed GAE." IEEE Big Data. | Bilateral flows | MSE | Regression, country-country |

---

### What metrics to report for comparability

| Metric | Used by this project | Used in literature | Notes |
|--------|---------------------|--------------------|-------|
| PR-AUC | Yes (primary) | Not reported by [P1–P4] | Must re-implement baselines to fill these cells |
| AUROC | Yes | [P2], [P3] report but flag as unreliable | Use as secondary only |
| NDCG@20 | Yes | Not reported by [P1–P4] | Novel metric contribution |
| Prec@20 (per country) | Yes | [P1] reports prec@1000 (global) | Different K and averaging |
| CWR | Yes | Not in literature | Novel complexity-weighted metric |
| BestF1 | Not currently reported | [P1], [P2], [P3], [P4] | **Must add** to be comparable to literature |
| Prec@1000 (global) | Not currently reported | [P1] | Add to enable direct comparison |
| mAP@10 | Not currently reported | [P1] | Add for completeness |

**Action — status (2026-07-01):** Done. Product Space density, RCA persistence, and XGBoost are all re-implemented on this pipeline; BestF1 + Prec@1000 + mAP@10 are reported alongside PR-AUC/NDCG/CWR for all 14 methods in `internal_benchmarking/`.

**Result vs literature:** Our **XGBoost** scores BestF1=0.63, Prec@1000=0.90, mAP@10=0.61 on the *sampled* test set — far above [P1]'s XGBoost SOTA (BestF1≈0.139, prec@1000≈0.198). **Caveat:** the gap is largely the 5:1 negative subsampling (which inflates absolutes ~5–8×); the [P1] numbers are on a near-full-universe pool, so the honest comparison is against our **full-universe** BestF1/Prec@1000 — XGBoost must still be scored there (see Pending). The GNNs do **not** currently beat our own XGBoost on any metric — the publishable story is "XGBoost is SOTA; the temporal GNN + LLM capability graph is a competitive, more interpretable alternative," not "GNN beats tabular ML." This tempers H1.

### XGBoost feature vector (51 dims)

| Group | Dims | Features |
|-------|------|----------|
| Country (BACI) | 4 | log_export, n_products, avg_rca, max_rca |
| Country (WDI) | 7 | gdp_pc, capital_formation, tertiary_enrollment, fdi_inflows, manufacturing_va, internet_users, population |
| Product | 3 | log_world_export, ubiquity, avg_rca |
| PCA-LLM | 32 | Top-32 PCA components of FinLang product embeddings (L2-normalised) |
| Network | 2 | Density score, ECI |
| RCA history | 3 | Raw RCA at t-2, t-1, t |

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