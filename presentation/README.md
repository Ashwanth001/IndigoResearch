# Presentation Materials — Trade Complexity 2.0

Everything needed to build a slide deck about this research. Assemble the slides yourself in PowerPoint / Google Slides / Canva using the outline and images here.

## What's in this folder

| File | Purpose |
|------|---------|
| **`SLIDE_DECK.md`** | The main deliverable — 20 slides, each with title, image, on-slide bullets, and speaker notes. Build from this. |
| **`TALKING_POINTS.md`** | The spoken narrative arc + anticipated Q&A with honest answers + numbers to memorise. Read before presenting. |
| **`FIGURE_INDEX.md`** | Catalogue of every figure (in this folder and the two eval folders), what it shows, which slide it belongs to. |
| **`DESIGN_NOTES.md`** | Colour system, slide-count options (full / short / lightning), typography, how to frame the "XGBoost wins" story. |
| **`build_presentation_figures.py`** | Generates the evaluation-results figures into `images/`. |
| **`images/`** | 22 evaluation-results PNGs — one bar chart per metric, a 2×2 primary panel, a radar/pentagon chart, and a heatmap; each for the sampled and full-universe regimes. |

These are pure **results charts** (bar / radar / heatmap). Denser analytical versions of the same data live in `../internal_benchmarking/plots/` and `../full_universe_eval/plots/` (see `FIGURE_INDEX.md`).

## Quick start

1. Read `TALKING_POINTS.md` for the story.
2. Open `SLIDE_DECK.md` and build slides top to bottom, dropping in each named image.
3. Use `DESIGN_NOTES.md` for colours/fonts and to pick full vs short vs lightning length.

## Regenerating figures

Safe to re-run any time (they read the live result CSVs):

```powershell
python3.14 presentation/build_presentation_figures.py     # 22 evaluation-results figures
python3.14 ClaudeFiles/plot_ib_results.py                 # sampled analytical plots
python3.14 ClaudeFiles/plot_results.py                    # full-universe analytical plots
```

## Source of truth

All numbers in the deck trace to `../mdFiles/CONTEXT.md` (the master project tracker) and the CSVs in `../internal_benchmarking/` and `../full_universe_eval/`. If you re-run the benchmarking notebooks and results change, update `CONTEXT.md`, re-run the plot scripts, and adjust the numbers quoted in `SLIDE_DECK.md`.

## Status snapshot (2026-07-01)

- 14 methods evaluated; **XGBoost is the top method**, best GNN is **GNN-LLM PCA-B (GCN+EW)**.
- Two test years (2015, 2016), sampled + full-universe regimes, leakage-audited.
- Pending (flagged in Slide 19): full-universe XGBoost scoring, country case studies, interpretability, multi-seed, dashboard.
