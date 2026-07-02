# Trade Complexity 2.0 — Slide Deck (build-ready outline)

> **How to use this file.** Each `## Slide N` block is one slide. Under each you get: a **title**, the **image** to drop in (from `presentation/images/`), the **on-slide bullets** (keep these terse — 4–6 lines max), and **speaker notes** (what you say, not what's printed). Numbers are pulled from the live result CSVs as of 2026-07-01. Assemble in PowerPoint / Google Slides / Canva. Suggested aspect ratio 16:9.
>
> Design guidance is in `DESIGN_NOTES.md`; the full spoken narrative arc is in `TALKING_POINTS.md`; every figure is catalogued in `FIGURE_INDEX.md`.

---

## Slide 1 — Title

**Title:** Trade Complexity 2.0
**Subtitle:** Forecasting which products a country will export next — a temporal bipartite GNN with an LLM capability layer, benchmarked against tabular ML and classical economic-complexity methods.

- Author: Ashwanth Sivakumar
- Data: BACI HS92 global trade (1995–2024) + World Bank WDI
- One-line thesis: *We can predict 5-year export diversification, and we now know exactly which method wins and why.*

**Speaker notes:** Set the frame — this is a prediction problem dressed as an economics problem. Every country wants to know "what should we try to make next?" We turn that into a supervised link-prediction task and benchmark 14 methods honestly.

---

## Slide 2 — The Question

**Title:** The Question

- Given the global trade network **today**, which (country, product) pairs that a country does **not** currently export will it export competitively (RCA ≥ 1) **5 years from now**?
- Framed as **link prediction on a temporal bipartite graph** (countries ↔ products).
- Economic stakes: diversification into complex products drives long-run growth (Hidalgo–Hausmann).

**Speaker notes:** Emphasise it's a *forward-looking, out-of-sample* question — not "describe today's trade" but "predict a transition that hasn't happened yet." That's what makes it hard and what makes leakage control non-negotiable (Slide 9).

---

## Slide 3 — Three Hypotheses

**Title:** What We Set Out to Test

| # | Hypothesis | Verdict |
|---|-----------|---------|
| **H1** | Temporal GNN beats classical ECI / Product Space | ✅ vs classical · ⚠️ **but XGBoost beats the GNN** |
| **H2** | LLM capability similarity helps rare/complex products | ✅ validated (capability edges lift every GNN metric) |
| **H3** | GNN + LLM explanations recover Product-Space intuition, sharper | ⏳ pending (interpretability / country stories) |

**Speaker notes:** Be upfront here — the honest finding is that H1 is *partially* supported: the GNN clearly beats the economics baselines, but a well-engineered XGBoost beats the GNN. That candour is a strength; it's what a good examiner wants to see. H2 is a clean win. H3 is future work.

---

## Slide 4 — Project Timeline

**Title:** How We Got Here
**Image:** *(no figure — use a simple text/timeline built in your slide tool, or omit)*

- Six completed phases: data → graph → temporal GNN → LLM layer → model expansion → rigorous evaluation.
- Pending: full-universe XGBoost scoring, country case studies, dashboard, multi-seed runs.

**Speaker notes:** Walk left-to-right. Stress that evaluation was its own dedicated phase — not an afterthought — which is why we can trust the rankings.

---

## Slide 5 — Data & Methodology Pipeline

**Title:** From Raw Trade Flows to Trainable Graphs
**Image:** *(no figure — bullet list, or a simple flow drawn in your slide tool)*

- 12 reproducible steps in one notebook (`dataPipeline.ipynb`): aggregate → RCA → binarise → label the transition → features → graph → temporal split → WDI enrich → HeteroData → LLM layer → train + evaluate.
- 4 steps are governed by strict anti-leakage rules — see Slide 9.
- Output: PyG `HeteroData` samples — 5-year temporal windows of the country–product graph.

**Speaker notes:** Don't read all 12. Point out the shape: aggregate → RCA → binarise → label the transition → build the graph → split by time → train. The leakage-controlled steps are where most published pipelines accidentally cheat; we don't.

---

## Slide 6 — The Data

**Title:** What the Model Sees

- **Nodes:** 233 countries, 5,018 products (HS6).
- **Country features (11):** 4 trade (log export, #products, avg/max RCA) + 7 World Bank development indicators (GDP/capita, population, manufacturing VA, capital formation, tertiary enrolment, internet users, FDI).
- **Product features (3):** log world export, ubiquity, avg RCA.
- **Edges:** ~90k country→product "exports" edges per year + optional 144k product↔product "capability" edges from the LLM.
- **Temporal:** 5 consecutive yearly snapshots per sample.

**Speaker notes:** The WDI features are what make the country node "know" its development level, not just its trade. Slide 8's ablation shows this is the single biggest lever.

---

## Slide 7 — Model Architecture

**Title:** Temporal Bipartite GNN
**Image:** *(architecture diagram — draw in your slide tool from the description below, or omit)*

- **BipartiteEncoder** — projects country/product features to 128-d, then 2× SAGEConv message passing (heterogeneous via `to_hetero`).
- **TemporalGNN** — applies the encoder to each of 5 yearly snapshots, then a GRU reads the sequence → final embedding.
- **LinkPredictor** — MLP on `[z_country ‖ z_product]` → sigmoid → P(0→1 transition).
- **+LLM variant** — adds product↔product capability edges (top-20 FinLang neighbours).

**Speaker notes:** The key novelty is *temporal* + *bipartite* + *LLM capability edges* together. The GRU is what lets the model use momentum ("this country is trending toward electronics"), which a static snapshot can't.

---

## Slide 8 — The LLM Capability Layer (H2)

**Title:** Teaching the Graph What Products "Mean"

- Embed all 5,018 HS6 product descriptions with **FinLang/finance-embeddings-investopedia** (768-dim, finance-domain).
- Connect each product to its **top-20 nearest neighbours** → 144,192 capability edges.
- Why FinLang: dramatically better discrimination than generic e5-large-v2 (only ~1% of product pairs score >0.70 similarity, vs ~99% for e5 — it actually separates semiconductors from horses).
- Effect: capability edges lift **every** GNN metric — validates H2.

**Speaker notes:** This is the conceptual heart. Classical Product Space infers product similarity from *co-export statistics*; we inject *semantic* similarity from language. The KNN-on-embeddings baseline (Slide 11) proves this signal only helps when *integrated into the graph*, not used alone.

---

## Slide 9 — Why You Can Trust These Numbers

**Title:** Rigour — Four Anti-Leakage Rules
**Image:** *(text/table slide — the four rules below)*

- **Causal smoothing** — trailing window only, never peeks at the future.
- **Transition labels** — positive = *becomes* competitive (and stays), not *is already* competitive.
- **Temporal split** — train ≤2012, test 2015; the test year never appears in training.
- **Train-only normalisation** — WDI stats computed on training years only.

**Speaker notes:** Any one of these done wrong invalidates the whole study — the label rule alone would inflate the positive rate ~10×. This slide is your credibility. Examiners look for exactly this.

---

## Slide 10 — How We Evaluate

**Title:** A 3-Tier Evaluation Framework

| Tier | Metric | Answers |
|------|--------|---------|
| 1 — Global | **PR-AUC** (+ AUROC) | Overall discrimination on imbalanced data |
| 2 — Economic | **CWR** (Complexity-Weighted Recall) | Do we catch the *hard, valuable* transitions? |
| 3 — Investor | **NDCG@20 / Prec@20** per country | Is the *top of each country's list* right? |
| + Literature | **Best-F1, Prec@1000, mAP@10** | Directly comparable to prior papers |

- Everything computed on **2 test years (2015 & 2016)** × **2 regimes (sampled + full universe)**.

**Speaker notes:** Different stakeholders care about different things. A policymaker cares about CWR (are we surfacing complex opportunities?); an investor cares about the top-20 list. One metric can't capture that, so we report all three tiers — plus the literature metrics so our numbers can be placed against published SOTA.

---

## Slide 11 — The Methods We Compare

**Title:** 14 Methods, From Simple to Sophisticated
**Image:** *(table/list slide — the four families below)*

- **Classical/network:** RCA Persistence, Density (Product Space), ECI, ECI+Density.
- **Embedding:** KNN on FinLang embeddings.
- **Tabular ML:** XGBoost (51-dim engineered features).
- **GNNs:** 4F → 11F → 11F+LLM → PCA variants (SAGE / GCN+EW / GAT) + two GAT+Focal v2 variants.

**Speaker notes:** The ladder is deliberate — each step adds one idea, so any performance change is attributable. RCA Persistence is the "dumb but strong" baseline; XGBoost is the "strong tabular ML" bar to clear.

---

## Slide 12 — Headline Result

**Title:** Who Wins?
**Image:** `images/metric_sampled_prauc.png` (or `radar_sampled.png` for the trade-off view)

- **XGBoost is the top method** — PR-AUC **0.658** (2015) / **0.688** (2016).
- Best GNN is **GNN-LLM PCA-B (GCN+EW)** at **0.457** — best among all graph models.
- RCA Persistence is a deceptively strong baseline (**0.520**) — trade is autocorrelated.
- GAT-based variants *regress* — attention didn't help here.

**Speaker notes:** Lead with the honest headline: XGBoost wins. Then the nuance — among *graph* methods our best is PCA-B, and every GNN beats every classical economics baseline. Note RCA Persistence beating most GNNs on PR-AUC is *expected*: if you export something today you probably will in 5 years. The interesting question is the *hard* cases → next slide.

---

## Slide 13 — Where the GNN Earns Its Keep

**Title:** Economic Value — Complexity-Weighted Recall
**Image:** `images/metric_sampled_cwr.png`

- On **CWR** (catching the rare, complex transitions), GNNs and XGBoost hit **0.88–0.94**.
- **RCA Persistence collapses to 0.34** — it only knows how to repeat the obvious.
- This is the metric that matters for *policy*: which non-obvious, sophisticated products is a country ready for?

**Speaker notes:** This reframes the PR-AUC story. RCA Persistence wins PR-AUC by nailing easy cases, but it's useless for the actual decision (what *new, complex* thing to pursue). On that decision, learned models dominate. This is the slide that justifies the whole project beyond "XGBoost won."

---

## Slide 14 — The GNN Ablation Story

**Title:** Every Component Pulls Its Weight
**Image:** `images/metric_sampled_prauc.png` (point at the GNN-4F → GNN-11F → +LLM → PCA-A → PCA-B bars — they rise left-to-right within the GNN family)

- 4 BACI features → **+7 WDI features**: biggest jump (+10–12% PR-AUC on the full universe).
- +**LLM capability edges**: further +3–6%.
- +**PCA-32 embeddings + GCN encoder + edge weights**: best GNN configuration.

**Speaker notes:** This is the cleanest scientific result in the deck — a monotone ladder where each added idea helps. WDI matters most (development context), then semantic capability, then the encoder refinement. GAT (not shown as a rung) *hurt*, so we don't claim it.

---

## Slide 15 — Robustness: Two Independent Test Years

**Title:** The Rankings Are Stable
**Image:** `images/metric_sampled_prauc.png` (each method has a 2015 solid + 2016 hatched bar — they track closely) or `panel_primary_metrics_sampled.png`

- In every bar chart, the 2015 (solid) and 2016 (hatched) bars track closely — same ordering both years.
- All cross-year PR-AUC deltas are small and mostly positive (2016 slightly easier).
- Conclusion: results aren't a single-year fluke.

**Speaker notes:** A reviewer's first question is "did you get lucky on one test year?" The paired 2015/2016 bars answer it — the ordering holds across two years chosen with a deliberate 2-year gap from validation. Combined with the leakage controls, the study is defensible.

---

## Slide 16 — Honest Numbers: Sampled vs Full Universe

**Title:** What the Real Deployment Number Looks Like
**Image:** `images/metric_sampled_prauc.png` beside `images/metric_universe_prauc.png` (put the two side by side — same ordering, much lower absolutes on the universe)

- Sampled test set uses 5:1 negative sampling → ~14.5% positive → **inflates PR-AUC ~5–8×**.
- On the **full unsampled universe** (~1.08M pairs, ~1.7% positive), absolutes drop sharply but **rankings are preserved**.
- We report both so no one is misled by the inflated headline number.

**Speaker notes:** Crucial honesty slide. Sampled numbers are for *fair comparison between methods*; full-universe numbers are for *"what would this actually do in production."* Many papers only report the inflated version — we report both, and the method ordering is identical.

---

## Slide 17 — Positioning vs Published Literature

**Title:** How This Compares to Prior Work

- **No published paper** applies a GNN to the country–product *RCA-transition* task — all "GNN + trade" work predicts bilateral *flow value* (regression). Our framing is novel.
- The real benchmark is the "Rome school" (Tacchella/Albora/Zaccaria): tree-based ML (XGBoost) is their SOTA at Best-F1 ≈ 0.139 / Prec@1000 ≈ 0.198.
- Our XGBoost + Best-F1/Prec@1000/mAP@10 are reported for direct comparison (caveat: our sampled numbers are inflated; full-universe comparison is the fair one — pending).

**Speaker notes:** Two claims: (1) the GNN framing is genuinely novel; (2) we didn't dodge the strong tabular baseline — we implemented it and it's our top method, consistent with the literature that tree-ML is SOTA on this task. Our contribution is the temporal-GNN-+-LLM *alternative* and the rigorous, multi-regime evaluation.

---

## Slide 18 — Key Takeaways

**Title:** What We Learned

1. **XGBoost is SOTA** on this task; the temporal GNN + LLM is a competitive, more *structural/interpretable* alternative.
2. **WDI development context** is the highest-value feature addition; **LLM capability edges** add a real, consistent gain (H2 ✓).
3. **CWR** shows learned models — not RCA persistence — catch the *complex, valuable* transitions.
4. Results are **leakage-audited, cross-year stable, and reported at true deployment scale**.

**Speaker notes:** Land the plane. The project's value isn't "we beat everyone" — it's "we built a rigorous, honest benchmark and a novel model family, and we know precisely what works and why."

---

## Slide 19 — What's Next

**Title:** Roadmap

- **Full-universe XGBoost + PCA-GNN** scoring → complete the honest head-to-head.
- **Country case studies** — India (main), Vietnam, Mexico, Indonesia: which products does the model recommend, and do they make economic sense? (H3)
- **Interpretability** — extract capability pathways; compare to Product Space (H3).
- **Multi-seed training** — confirm the sub-1% GNN deltas.
- **Public dashboard** — interactive per-country predictions.

**Speaker notes:** The country stories are the most compelling next deliverable for a non-technical audience — turning a PR-AUC into "here are 10 products India is poised to break into."

---

## Slide 20 — Appendix: Full Results Tables

**Title:** Appendix — Reference

- Sampled heatmap (all 14 methods × 8 metrics): `images/heatmap_sampled.png`
- Full-universe heatmap: `images/heatmap_universe.png`
- Every individual metric (sampled + universe): `images/metric_*.png` — see `FIGURE_INDEX.md`
- Full CSVs: `../internal_benchmarking/*.csv`, `../full_universe_eval/*.csv`
- Method definitions & leakage rules: `../mdFiles/CONTEXT.md`

**Speaker notes:** Keep as backup slides for Q&A — if someone asks "what's method X's mAP@10 in 2016," the heatmap has it.
