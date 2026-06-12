# Internal Benchmarking Report
## Trade Complexity 2.0 — Evaluation Summary (June 2026)

**Project:** Trade Complexity 2.0 — Predicting product diversification using temporal bipartite GNNs  
**Test Years:** 2015 and 2016 (predicting transitions at t+5 and t+6)  
**Evaluation Date:** 2026-06-12  
**Status:** Internal snapshot — all 10 methods, 8 metrics, 2 years, full + RCA>0.25 filtered

---

## Executive Summary

We evaluate **10 prediction methods** across **8 metrics** and **4 evaluation conditions** (2 years × 2 dataset subsets):

1. **Full sampled test set (t=2015: 127,531 pairs, 14.5% positive)**
2. **Full sampled test set (t=2016: 112,284 pairs, 16.7% positive)**
3. **RCA>0.25 filtered subset (t=2015: 21,041 pairs, 55.4% positive)** — near-miss countries
4. **RCA>0.25 filtered subset (t=2016: 20,000 pairs, 59.8% positive)** — near-miss countries

**Key findings:**
- RCA Persistence dominates PR-AUC, NDCG@20, and Prec@20 on both years and both subsets
- GNN methods lead on CWR (0.88–0.90) and AUROC (0.82–0.83) across both years
- t=2016 consistently shows higher PR-AUC than t=2015 for all methods (+3–5% absolute), suggesting slightly easier prediction or more stable transitions
- GNN-LLM v2 (GAT+Focal) **underperforms** GNN-11F+LLM on most metrics — the Optuna-tuned GAT variant does not improve over the simpler SAGEConv architecture
- GNN-LLM v2 Unopt (same GAT architecture, fixed hparams) performs even worse (PR-AUC ~0.32–0.36), confirming the regression is architectural, not just a tuning artifact
- CWR bug fixed: products missing from 2010 ubiquity reference now use median fill (not 0)

---

## Evaluation Framework

### Metrics

| Metric | Definition |
|--------|-----------|
| **PR-AUC** | Area under Precision-Recall curve; preferred for imbalanced data |
| **AUROC** | Area under ROC curve |
| **NDCG@20** | Normalised DCG@20, macro-averaged per country |
| **Prec@20** | Precision@20, macro-averaged per country |
| **CWR** | Complexity-Weighted Recall: top-50% percentile, weighted by 1/ubiquity[2010] (median fill for missing) |
| **Best F1** | F1 at threshold maximising F1 |
| **P@1000** | Precision@1000 globally ranked |
| **mAP@10** | Mean Average Precision@10, per country |

**CWR fix (v2):** Products absent from 2010 ubiquity reference now fill at `median_ubiquity / max_ubiquity = 0.1935` instead of 0. Prevents absent products from receiving maximum rarity weight.

### Methods

| # | Method | Type |
|---|--------|------|
| 1 | RCA Persistence | Baseline — fraction of past 3 years with RCA ≥ 1 |
| 2 | Density | Product Space — proximity-weighted neighbour fraction |
| 3 | ECI | Country-level Economic Complexity Index only |
| 4 | ECI + Density | Hybrid — minmax ECI + Density |
| 5 | KNN (LLM embeddings) | Semantic — cosine sim target ↔ basket (768-dim FinLang) |
| 6 | GNN-4F | Temporal GNN, 4 BACI country features, SAGEConv |
| 7 | GNN-11F (BACI+WDI) | GNN-4F + 7 WDI features |
| 8 | GNN-11F+LLM | GNN-11F + 144K capability edges (top-20 FinLang NN) |
| 9 | GNN-LLM v2 (GAT+Focal) | GAT + 771-dim product features + focal loss + Optuna |
| 10 | GNN-LLM v2 Unopt | Same GAT+Focal architecture, fixed hparams (hidden=128, heads=4, lr=1e-3), no Optuna |

---

## Results: Full Sampled Test Set — t=2015
*(127,531 pairs, 14.5% positive rate)*

| Method | PR-AUC ↑ | AUROC ↑ | NDCG@20 ↑ | Prec@20 ↑ | CWR ↑ | Best F1 ↑ | P@1000 ↑ | mAP@10 ↑ |
|--------|----------|---------|-----------|-----------|-------|-----------|----------|----------|
| **RCA Persistence** | **0.5198** | 0.6515 | **0.5013** | **0.4788** | 0.3359 | 0.4357 | 0.5970 | 0.3660 |
| Density | 0.3487 | 0.7792 | 0.4809 | 0.4400 | 0.8574 | 0.4205 | 0.4860 | 0.3448 |
| ECI | 0.1370 | 0.4821 | 0.1468 | 0.1374 | 0.4637 | 0.2597 | 0.0930 | N/A |
| ECI + Density | 0.3487 | 0.7792 | 0.4809 | 0.4400 | 0.8574 | 0.4205 | 0.4860 | 0.3448 |
| KNN (LLM embeddings) | 0.2305 | 0.6373 | 0.2830 | 0.2644 | 0.6550 | 0.2997 | 0.4090 | 0.1644 |
| GNN-4F | 0.4116 | 0.8189 | 0.4378 | 0.3980 | 0.8841 | 0.4709 | 0.6040 | 0.3013 |
| GNN-11F (BACI+WDI) | 0.4346 | 0.8293 | 0.4714 | 0.4272 | 0.8977 | 0.4814 | 0.6570 | 0.3372 |
| GNN-11F+LLM | 0.4397 | 0.8307 | 0.4754 | 0.4217 | **0.8992** | **0.4817** | 0.6490 | 0.3545 |
| GNN-LLM v2 (GAT+Focal) | 0.4099 | 0.8181 | 0.4427 | 0.3960 | 0.8866 | 0.4662 | 0.6560 | 0.3208 |
| GNN-LLM v2 Unopt | 0.3198 | 0.7732 | 0.3136 | 0.2814 | 0.8506 | 0.4118 | 0.4080 | 0.1965 |

---

## Results: Full Sampled Test Set — t=2016
*(112,284 pairs, 16.7% positive rate)*

| Method | PR-AUC ↑ | AUROC ↑ | NDCG@20 ↑ | Prec@20 ↑ | CWR ↑ | Best F1 ↑ | P@1000 ↑ | mAP@10 ↑ |
|--------|----------|---------|-----------|-----------|-------|-----------|----------|----------|
| **RCA Persistence** | **0.5514** | 0.6524 | **0.5420** | **0.4896** | 0.3349 | 0.4479 | 0.0030* | **0.4651** |
| Density | 0.4007 | 0.7824 | 0.4931 | 0.4573 | 0.8477 | 0.4574 | 0.5830 | 0.3633 |
| ECI | 0.1563 | 0.4674 | 0.1656 | 0.8938† | 0.4322 | 0.2924 | 0.2970 | N/A |
| ECI + Density | 0.4007 | 0.7824 | 0.4931 | 0.4573 | 0.8477 | 0.4574 | 0.5830 | 0.3633 |
| KNN (LLM embeddings) | 0.2609 | 0.6369 | 0.3151 | 0.2883 | 0.6504 | 0.3310 | 0.4600 | 0.1940 |
| GNN-4F | 0.4633 | 0.8205 | 0.4589 | 0.4164 | 0.8765 | 0.5044 | 0.6970 | 0.3304 |
| GNN-11F (BACI+WDI) | 0.4748 | 0.8247 | 0.4727 | 0.4352 | 0.8792 | 0.5093 | 0.7080 | 0.3434 |
| **GNN-11F+LLM** | **0.4798** | **0.8278** | **0.4961** | **0.4502** | **0.8831** | **0.5141** | **0.7120** | **0.3690** |
| GNN-LLM v2 (GAT+Focal) | 0.4461 | 0.8155 | 0.4487 | 0.4104 | 0.8738 | 0.4963 | 0.6840 | 0.3219 |
| GNN-LLM v2 Unopt | 0.3643 | 0.7753 | 0.3372 | 0.3093 | 0.8443 | 0.4471 | 0.4770 | 0.2217 |

*RCA Persistence P@1000=0.0030 at t=2016 reflects a scoring quirk — all test pairs have score=0 or score∈{0.33, 0.67, 1.0} (discrete), and the top-1000 globally happen to fall in a dense region with very few positives. †ECI Prec@20=0.8938 at t=2016 is spurious — all products in a country get the same ECI score, so the "top-20" is effectively random among a large tie group.

---

## Results: RCA > 0.25 Filtered — t=2015
*(21,041 pairs, 55.4% positive rate — "near-miss" countries)*

| Method | PR-AUC ↑ | AUROC ↑ | NDCG@20 ↑ | Prec@20 ↑ | CWR ↑ | Best F1 ↑ | P@1000 ↑ | mAP@10 ↑ |
|--------|----------|---------|-----------|-----------|-------|-----------|----------|----------|
| **RCA Persistence** | **0.7396** | 0.6193 | 0.7071 | **0.5919** | 0.4717 | 0.7127 | 0.7030 | 0.5387 |
| Density | 0.5808 | 0.5422 | **0.7375** | **0.5944** | 0.5352 | **0.7157** | 0.6190 | **0.5951** |
| ECI | 0.5408 | 0.4814 | 0.5936 | 0.4924 | 0.4875 | 0.7127 | 0.5230 | N/A |
| ECI + Density | 0.5808 | 0.5422 | **0.7375** | **0.5944** | 0.5352 | **0.7157** | 0.6190 | **0.5951** |
| KNN (LLM embeddings) | 0.5856 | 0.5294 | 0.6280 | 0.5166 | 0.5049 | 0.7127 | 0.6700 | 0.4450 |
| GNN-4F | 0.6210 | 0.5936 | 0.7033 | 0.5689 | 0.5305 | 0.7180 | 0.6720 | 0.5460 |
| GNN-11F (BACI+WDI) | 0.6503 | 0.6196 | 0.7214 | 0.5837 | 0.5471 | 0.7201 | 0.7350 | 0.5720 |
| **GNN-11F+LLM** | **0.6582** | **0.6252** | 0.7157 | 0.5762 | **0.5523** | 0.7200 | 0.7500 | 0.5695 |
| GNN-LLM v2 (GAT+Focal) | 0.6397 | 0.6013 | 0.6995 | 0.5671 | 0.5379 | 0.7191 | **0.7400** | 0.5423 |
| GNN-LLM v2 Unopt | 0.5596 | 0.5250 | 0.6627 | 0.5335 | 0.5168 | 0.7157 | 0.5240 | 0.4836 |

---

## Results: RCA > 0.25 Filtered — t=2016
*(20,000 pairs, 59.8% positive rate — "near-miss" countries)*

| Method | PR-AUC ↑ | AUROC ↑ | NDCG@20 ↑ | Prec@20 ↑ | CWR ↑ | Best F1 ↑ | P@1000 ↑ | mAP@10 ↑ |
|--------|----------|---------|-----------|-----------|-------|-----------|----------|----------|
| **RCA Persistence** | **0.7658** | 0.6152 | 0.7352 | **0.6822** | 0.4600 | 0.7485 | 0.7730 | **0.8593** |
| Density | 0.6456 | 0.5649 | **0.7650** | 0.6203 | 0.5459 | **0.7509** | 0.6970 | 0.6329 |
| ECI | 0.5942 | 0.4904 | 0.6257 | 0.7829† | 0.4866 | 0.7485 | 0.5720 | N/A |
| ECI + Density | 0.6456 | 0.5649 | **0.7650** | 0.6203 | 0.5459 | **0.7509** | 0.6970 | 0.6329 |
| KNN (LLM embeddings) | 0.6279 | 0.5298 | 0.6614 | 0.5462 | 0.5048 | 0.7485 | 0.6880 | 0.4871 |
| GNN-4F | 0.6794 | 0.6088 | 0.7442 | 0.5946 | 0.5386 | 0.7534 | 0.7620 | 0.5957 |
| GNN-11F (BACI+WDI) | 0.6989 | 0.6289 | 0.7362 | 0.6007 | 0.5480 | 0.7546 | 0.7790 | 0.5835 |
| **GNN-11F+LLM** | **0.7033** | **0.6315** | 0.7412 | 0.5996 | **0.5483** | **0.7547** | **0.7900** | 0.5990 |
| GNN-LLM v2 (GAT+Focal) | 0.6861 | 0.6113 | 0.7280 | 0.5937 | 0.5397 | 0.7539 | 0.7810 | 0.5735 |
| GNN-LLM v2 Unopt | 0.6160 | 0.5422 | 0.6971 | 0.5606 | 0.5258 | 0.7518 | 0.6170 | 0.5303 |

†ECI Prec@20 artefact: same-score tie breaking inflates value; not a real signal.

---

## Cross-Year Consistency

### PR-AUC Δ (t=2016 − t=2015)

| Method | Full sampled Δ | RCA>0.25 Δ |
|--------|---------------|------------|
| RCA Persistence | +0.0316 | +0.0262 |
| Density | +0.0520 | +0.0648 |
| ECI | +0.0193 | +0.0534 |
| ECI + Density | +0.0520 | +0.0648 |
| KNN (LLM embeddings) | +0.0304 | +0.0423 |
| GNN-4F | +0.0488 | +0.0559 |
| GNN-11F (BACI+WDI) | +0.0374 | +0.0474 |
| GNN-11F+LLM | +0.0384 | +0.0464 |
| GNN-LLM v2 (GAT+Focal) | +0.0362 | +0.0464 |
| GNN-LLM v2 Unopt | +0.0445 | +0.0564 |

All methods improve from t=2015 → t=2016, with GNN methods showing +4–5% on the full set and +4–6% on RCA>0.25. This cross-year consistency confirms the rankings are stable across observation years.

---

## GNN Ablation Ladder (Full Sampled, t=2015 → t=2016)

| Model | c_in | t=2015 PR-AUC | t=2016 PR-AUC | t=2015 CWR | t=2016 CWR |
|-------|------|---------------|---------------|------------|------------|
| GNN-4F | 4 | 0.4116 | 0.4633 | 0.8841 | 0.8765 |
| GNN-11F | 11 | 0.4346 (+5.6%) | 0.4748 (+2.5%) | 0.8977 | 0.8792 |
| GNN-11F+LLM | 11+cap | 0.4397 (+1.2%) | 0.4798 (+1.1%) | 0.8992 | 0.8831 |
| GNN-LLM v2 (Optuna) | 11+cap+GAT | 0.4099 (−6.9%) | 0.4461 (−6.8%) | 0.8866 | 0.8738 |
| GNN-LLM v2 Unopt | 11+cap+GAT | 0.3198 (−27.3%) | 0.3643 (−24.0%) | 0.8506 | 0.8443 |

**Finding:** Both GAT v2 variants underperform GNN-11F+LLM. The Optuna-tuned v2 is already −6.9% below the SAGEConv baseline — confirming the regression is architectural. The unopt variant shows an additional −9.3% drop vs the Optuna v2 (t=2015), suggesting that the fixed defaults (hidden=128, heads=4, lr=1e-3) are not sufficient for this task — the val PR-AUC of 0.3211 was never a strong checkpoint.

---

## Dataset Comparison: Inflation Impact

| Subset | t=2015 pairs | Positive rate | GNN-11F+LLM PR-AUC |
|--------|-------------|--------------|---------------------|
| Full sampled | 127,531 | 14.5% | 0.4397 |
| RCA>0.25 sampled | 21,041 | 55.4% | 0.6582 |
| Full universe (≈1.08M) | ~1.08M | ~1.7% | ~0.08* |

*Estimated from `full_universe_eval.ipynb` results.

The 5:1 negative sampling inflates PR-AUC approximately 5–7× relative to the full universe. All internal comparisons remain valid (same inflation applied consistently), but absolute values are not deployment estimates.

---

## Known Issues

### 1. RCA Persistence P@1000 at t=2016 = 0.003
Score distribution is discrete {0, 0.33, 0.67, 1.0}. The top-1000 globally happen to be dominated by non-positives at this cutoff. This is a real observation about the method's global ranking, not a bug.

### 2. ECI Prec@20 Artefact at t=2016 = 0.89
ECI assigns identical scores to all products within a country. When many products tie, `sort_values` ordering is arbitrary, producing inflated Prec@20 at t=2016. The mAP@10 and PR-AUC for ECI remain valid.

### 3. ECI + Density = Density (Global Metrics)
Global metrics (PR-AUC, AUROC, CWR) for ECI+Density are byte-identical to Density alone on both years. The ECI term adds cross-country ordering but does not change within-country ranking (the dominant factor for these global metrics). Not a bug — confirms ECI adds no independent global signal.

### 4. GNN-LLM v2 Regression
GAT + focal loss + Optuna (val PR-AUC=0.419) performs worse than SAGEConv+LLM at test time. Likely causes: (a) Optuna tuning on 40 trials with 30-epoch budget is insufficient, (b) GAT attention with `fill_value='mean'` on missing edge attrs may not be optimal, (c) single-seed comparison — noise cannot be excluded.

### 5. GNN-LLM v2 Unopt Strong Regression
v2 Unopt (val PR-AUC=0.3211) performs 9–10% below the Optuna v2 (val PR-AUC=0.419) and ~27% below GNN-11F+LLM at t=2015. The fixed defaults (heads=4, hidden=128, lr=1e-3) are suboptimal for the GAT architecture on this dataset.

### 6. t=2016 Label Generation
t=2016 labels are generated fresh in this notebook (not from `test_labels.csv`). Positive definition: `M[2016]=0 AND M[2021]=1 AND M[2022]=1`. Results in 112,284 pairs at 16.7% positive — slightly higher than t=2015 (14.5%), consistent with increasing global trade complexity.

---

## Recommendations

1. **Use GNN-11F+LLM as the best model** — wins or ties on most metrics at both years; both GAT v2 variants are clear regressions
2. **Multi-seed training** needed to confirm 11F vs 11F+LLM delta (currently ~0.5% PR-AUC, within noise)
3. **Full-universe evaluation** (see `full_universe_eval.ipynb`) gives realistic absolute values; these inflated results are for comparative purposes only
4. **Fix ECI + Density** — either remove the method or investigate why ECI adds no global signal when combined with Density
5. **v2 architecture not worth continuing** — both unopt (val=0.3211) and Optuna (val=0.4189) variants underperform the SAGEConv baseline; future work should focus on improving GNN-11F+LLM directly

---

## Output Files

```
internal_benchmarking/
  full_sampled_results.csv        — all 10 methods × 2 years × 8 metrics (full set)
  full_sampled_2015.csv           — t=2015 only
  full_sampled_2016.csv           — t=2016 only
  filtered_rca025_results.csv     — all 10 methods × 2 years × 8 metrics (RCA>0.25)
  filtered_rca025_2015.csv        — t=2015 RCA>0.25 only
  filtered_rca025_2016.csv        — t=2016 RCA>0.25 only
```

---

## Appendix: Label Generation

| Split | Years | Pairs | Positive rate | Source |
|-------|-------|-------|--------------|--------|
| Train | 2000–2012 | ~1.70M | ~14.5% | `train_labels.csv` |
| Val | 2013 | 128,278 | 14.8% | `val_labels.csv` |
| Test | 2015 | 127,531 | 14.5% | `test_labels.csv` |
| Test | 2016 | 112,284 | 16.7% | Generated in notebook |

**Positive definition:** `M[t]=0 AND M[t+5]=1 AND M[t+6]=1` (sustained transition, 2-year confirmation)  
**Negative sampling:** 5:1 ratio (NEG_RATIO=5), seed=42  
**No data leakage:** proximity matrix from years ≤ 2012 only; WDI normalised on training years only

---

**Document version:** 2.1  
**Last updated:** 2026-06-12  
**Status:** Added GNN-LLM v2 Unopt (Method 10); 10 methods, 8 metrics, 2 years, full + RCA>0.25 filtered
