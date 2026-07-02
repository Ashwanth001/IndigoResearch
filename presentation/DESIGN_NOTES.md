# Design Notes — assembling the deck

Practical guidance so the slides look coherent and the audience follows the argument.

## Colour system (used consistently in every generated figure)
Match your slide accents to these so figures don't clash:

| Meaning | Hex | Where |
|---------|-----|-------|
| Classical / network baselines | `#6baed6` (blue) | RCA-Persist, Density, ECI |
| Embedding baseline (KNN-LLM) | `#74c476` (green) | KNN |
| **XGBoost (top method)** | `#fd8d3c` (orange) | headline emphasis |
| GNN family | `#9e9ac8` (purple) | all GNN variants |
| **Best GNN (PCA-B)** | `#54278f` (dark purple) | highlight |
| Accent / warnings / leakage | `#e6550d` (strong orange) | leakage rules, callouts |
| Ink / text | `#222222` | body text |

Suggested slide theme: white or very light background, dark charcoal text, one orange accent. Avoid a busy template — the figures carry the visual weight.

## Slide-count options
- **Full deck (~20 slides):** defence / thesis committee / detailed review.
- **Short talk (~10 min, 8 slides):** 1 (title), 2 (question), 7 (architecture), 9 (leakage), 10 (evaluation), 12 (headline), 13 (CWR), 18 (takeaways).
- **Lightning (~5 min, 4 slides):** 2 (question), 7 (architecture), 12 (headline), 13 (CWR).

## Typography
- Titles: 28–34 pt bold. Body bullets: 18–22 pt. Never smaller than 16 pt on a projected slide.
- Max ~6 bullets/slide; each bullet ≤ 2 lines. The `SLIDE_DECK.md` bullets are already trimmed to this.

## Handling the "XGBoost wins" narrative
This is the deck's rhetorical challenge. Structure it as a *reveal*, not a *concession*:
1. Slide 12 states it plainly (credibility through honesty).
2. Slide 13 immediately reframes with CWR (learned models win where it matters).
3. Slide 14 shows the GNN is scientifically clean (monotone ablation).
4. Slide 18 lands the real contribution (novel framing + rigorous benchmark).
Never apologise for it — a candid negative result reads as competence.

## Figure embedding tips
- All figures in `images/` are evaluation-results charts (bar / radar / heatmap) and already have titles + subtitles baked in — you can hide the slide title or keep both (the baked title is smaller).
- Individual-metric bar charts (`metric_*.png`) are wide — give them near-full-slide width. They're sorted best→worst and colour-coded by family, so they're readable at a glance.
- The radar/pentagon (`radar_*.png`) is square — center it with the legend to the right; great as a single "who's best overall" slide.
- Heatmaps (`heatmap_*.png`) are dense — embed at full size on a dedicated appendix slide; they're meant to be read, not glanced at.
- Slides 4, 5, 7, 9, 11 (timeline / pipeline / architecture / leakage rules / method taxonomy) have **no figure** — build them as simple text, tables, or a flow drawn in your slide tool. The content for each is in `SLIDE_DECK.md`.

## If you later want a real .pptx
`python-pptx` is the standard route: one slide per `## Slide N`, `add_picture()` for the image, bullets as a text frame. Say the word and this can be scripted from `SLIDE_DECK.md` deterministically.
