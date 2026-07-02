# Talking Points — the spoken narrative arc

This is the *story* to tell around the slides — the through-line that makes a listener understand, not just see numbers. Read this once before presenting; it turns 20 slides into one argument.

---

## The one-sentence version

> "We turned 'what should a country make next?' into an honest, leakage-controlled prediction benchmark of 14 methods — found that XGBoost wins overall but a temporal GNN with an LLM capability layer is the best *structural* model and uniquely good at the hard, economically valuable cases."

---

## The arc, in five beats

**1. The hook (Slides 1–3).**
Every developing economy asks the same question: which new industries are we actually ready for? Classical economics answers this with the "Product Space" — a static map of which products co-occur. We ask whether modern ML — a temporal graph network that watches the trade network *move* over time, plus a language model that understands what products *are* — can answer it better. We had three hypotheses, and we'll tell you honestly which held.

**2. The setup (Slides 4–8).**
Getting this right is 80% data discipline. We aggregate 30 years of global trade, compute comparative advantage, and — critically — define the target as a *transition* (a country going from not-competitive to competitive-and-staying-there). Then we build a temporal bipartite graph and layer on an LLM that embeds every product's description, so the graph "knows" that microprocessors and diodes are related even if no single country exports both.

**3. The credibility firewall (Slides 9–10).**
Here's why you should believe us. Prediction studies on trade are *littered* with subtle leakage — smoothing that peeks at the future, labels that reward the already-obvious, random splits that let the test year bleed into training. We enumerate four rules we never break, and we evaluate on three tiers of metrics because "accuracy" means different things to a policymaker vs an investor.

**4. The payoff (Slides 11–17).**
Now the results, and we lead with the uncomfortable truth: **XGBoost beats our GNN.** We're not hiding it — a well-engineered tabular model is genuinely the strongest, which matches the published literature. *But* two things rescue the GNN story. First, on **complexity-weighted recall** — catching the rare, sophisticated transitions that actually matter for development — learned models crush the naive baselines that win on raw PR-AUC. Second, our **ablation ladder** is a clean scientific result: development indicators help most, LLM capability edges help consistently, and we can attribute every gain. Then we prove it's not luck (two test years) and we're honest about scale (the real deployment number is much lower than the headline).

**5. The landing (Slides 18–20).**
The contribution isn't a leaderboard win. It's (a) a *novel* framing — nobody has applied a GNN to RCA-transition prediction before; (b) a *rigorous, multi-regime* benchmark others can build on; and (c) a clear map of what works and why. Next: turn the predictions into concrete country stories.

---

## Anticipated tough questions (and honest answers)

**Q: If XGBoost wins, why bother with the GNN?**
A: Three reasons. (1) On the economically meaningful metric — catching complex, non-obvious transitions (CWR) — the GNN matches XGBoost and both crush the classical baselines. (2) The GNN is *structural*: it produces embeddings and capability pathways we can interpret (H3, ongoing), whereas XGBoost's feature vector is flatter. (3) XGBoost actually *consumes* GNN-adjacent signals (PCA-LLM embeddings, density, ECI) in its 51-dim vector — the representations we built feed the winner.

**Q: Isn't RCA Persistence beating most GNNs embarrassing?**
A: No — it's expected and informative. Trade is highly autocorrelated; if you export something today you likely will in 5 years. Persistence wins PR-AUC by nailing the *easy* cases. But it scores 0.34 on CWR — it's blind to the hard, valuable transitions, which is exactly what we care about.

**Q: Your headline PR-AUC (0.66) — is that real?**
A: That's the *sampled* number, inflated ~5–8× by negative subsampling for fair method comparison. The true full-universe number is much lower (~0.08 for the GNN, ~0.24 for persistence) at ~1.7% positive rate. We report both; rankings are identical. We never present only the flattering number.

**Q: Why did GAT (attention) hurt?**
A: On this task, with these features and single-seed training, the GAT variants regressed vs SAGE/GCN — both the Optuna-tuned and fixed-hparam versions. We report it as a negative result rather than tuning until it looks good. Multi-seed runs are pending to rule out noise.

**Q: How is this novel — there are GNN+trade papers?**
A: Every published GNN-on-trade paper predicts bilateral *flow value* (a regression between two countries). Ours predicts a *country–product RCA≥1 transition* (classification on a bipartite graph) with a temporal component and an LLM capability layer. That specific combination is, to our knowledge, unpublished.

**Q: Where's the LLM actually helping?**
A: Two places. In the GNN, capability edges lift every metric (the ablation rung). In XGBoost, 32 PCA components of the FinLang embeddings are part of the 51-dim vector. The KNN-on-embeddings baseline (0.23 PR-AUC) proves the semantic signal is real but *only useful when integrated* — used raw it's weak.

---

## Numbers worth memorising

- XGBoost PR-AUC: **0.658** (2015) / **0.688** (2016) — top method.
- Best GNN (PCA-B GCN+EW): **0.457** (2015).
- RCA Persistence: **0.520** PR-AUC but only **0.34** CWR.
- GNN/XGBoost CWR: **0.88–0.94**.
- WDI ablation jump: **+10–12%** PR-AUC (full universe). LLM edges: **+3–6%**.
- Scale: 233 countries, 5,018 products, ~1.08M full-universe pairs (~1.7% positive).
- Sampling inflation: **~5–8×**.
