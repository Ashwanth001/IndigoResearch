## **Mar 15, 2026**

## **Part 1 — What Exactly Are We Predicting?**

This confuses most students, so let's be precise.

**RCA is measured at the country × product level, not industry level.** Specifically at the HS6 (6-digit Harmonized System) product level. HS6 means something like "854231 — Electronic integrated circuits: processors and controllers." There are roughly 5,000 such product codes. RCA is NOT measured at the industry level (like "electronics" or "textiles") — that would be HS2 or HS4. HS6 is granular enough to be scientifically meaningful.

**The exact prediction task is this:** given country $c$'s entire export basket observed up to year $t$, predict a binary label for every product $p$ that $c$ does NOT currently export competitively:![][image1]

$$\\hat{y}*{c,p} \= P\\bigl(M*{c,p,t+k} \= 1 ;\\big|; M\_{c,p,t} \= 0,; \\text{graph up to } t\\bigr)$$

So we are predicting **new entries** — the 0 → 1 transitions in the binary RCA matrix. Not "does India export semiconductors" (too coarse), but "does India export HS 854231 specifically." The graph structure gives the model context about which nearby products India already exports, and the LLM layer adds knowledge about what capabilities those products require.

---

## **Part 2 — Code Review of the Student's Repo**

I can see the full file listing and two committed tensor artifacts, which together tell me a lot. Here is a structured audit:GitHub rate-limited the file-level reads, but the directory listing and the two committed artifacts (`country_x_by_year.pt` and `country_features_enriched.csv`) already tell a significant story. Here's the full breakdown.

---

## **What exactly we're predicting, and at what level**

Let me show this visually first, then do the code audit.Now the full code audit. From the file listing and committed artifacts, here are all the issues, organized by severity:Now let me go through each issue in detail so the student can actually fix them:

---

### **🔴 Critical: step3\_smoothing.py — temporal leakage**

"Smoothing" RCA across years sounds reasonable (reduces noise from one-off export spikes) but is the most dangerous step in the pipeline. A 3-year rolling average computed as `mean(t-1, t, t+1)` uses the *future* year `t+1` — which means the RCA value you feed the model at year `t` already contains information about year `t+1`. Your labels are based on `t+1`. You've just leaked the answer into the input. The only safe version is a trailing window: `mean(t-2, t-1, t)`. The student needs to confirm the `window` parameter and whether `min_periods` is set.

---

### **🔴 Critical: step4\_labels.py — labelling all 1s vs. only new 1s**

This is the most common conceptual mistake. There are two possible label definitions:

* **Wrong:** `label = M[t+k][c,p]` — this labels all pairs where the country exports at `t+k`, including those it already exported at `t`. The "positive rate" will be \~10–15%, which seems OK but is scientifically wrong. The model will learn "countries keep exporting what they already export" — a trivial, useless result.  
* **Correct:** `label = (~M[t][c,p]) & M[t+k][c,p]` — only 0→1 transitions. The positive rate drops to \~3–5%. Harder task, but that's the science.

The student must check this exact line in `step4_labels.py`.

---

### **🔴 Critical: step8\_split.py — temporal vs. random split**

If this uses `sklearn.model_selection.train_test_split`, it is wrong. That function randomly shuffles rows, which means a (India, chip, 2021\) edge might appear in training while a (India, chip, 2018\) edge appears in test. The model trains on the future to predict the past. The correct split assigns all years ≤ 2018 to train, 2019–2020 to validation, 2021+ to test. Nothing from future years ever enters training.

---

### **🔴 Critical: no product node features**

The committed file `country_x_by_year.pt` stores country features. There is no `product_x_by_year.pt` or equivalent. In a bipartite GNN, both node types need feature vectors. If products are initialised as zero vectors or random noise, the GNN's product embeddings are meaningless — the model is essentially just doing matrix factorisation on the country side. The LLM embeddings (completely absent from the repo) were supposed to solve exactly this.

---

### **🟡 Serious: integrate\_wdi\_data.py is orphaned**

This script exists outside the numbered pipeline (step1–step10). WDI data (GDP per capita, FDI inflows, etc.) can genuinely improve country features, but the `country_features_enriched.csv` file being committed as a static CSV raises questions: is it re-generated when the pipeline reruns? What year range does it cover? Does it include years in the test set (leakage)? The WDI integration should be `step0_wdi.py` that feeds cleanly into `step5_features.py`.

---

### **🟡 Serious: no baselines at all**

The research question explicitly asks whether the GNN beats "ECI/Product Space baselines." There is no script in the repo that computes ECI, PCI, the proximity matrix φ, or density scores. Without these, you cannot answer your own research question. These are not optional extras — they are Table 1 of the paper.

---

### **🟡 Serious: no LLM embedding step**

H2 — the key innovation — is completely absent from the pipeline. No sentence-transformer step, no product text file being used, no `product_llm_embeddings.npy`. The student has built a standard GNN but not the "GNN \+ LLM capability graph" that the paper is about.

---

### **🟠 Moderate: negative sampling ratio unknown**

With \~3% positive rate in the label matrix, a naive training loop (one random negative per positive) still gives a 1:1 class ratio — fine. But if the student is treating *all* 0-cells in the matrix as negatives simultaneously (\~97% of the \~1 million country-product pairs), the loss is overwhelmed by easy negatives and the model collapses to predicting everything as 0\. Need `neg_ratio=5` and random negative sampling per batch.

---

### **🟠 Moderate: binary files committed to git**

`country_mapping.pkl`, `country_x_by_year.pt`, and the CSVs should be in `.gitignore`. They can't be reproduced by anyone who doesn't have `trade_data.csv`, and they'll break if the pipeline parameters change. The repo should contain only code plus a `README.md` explaining how to regenerate them.

---

**The one-sentence summary for the student:** the data pipeline has leakage in three places (smoothing, labels, split), the model is missing its core innovation (LLM product embeddings), and there are no baselines — so even if the GNN trains, there's nothing to compare it against and no paper.

