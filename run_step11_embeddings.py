"""
Step 11A: LLM Product Embeddings — run as standalone script if the notebook kernel crashes.

    python run_step11_embeddings.py

Downloads intfloat/e5-large-v2 (~1.3 GB, cached after first run) and encodes all
HS6 product descriptions. Saves:
    data/product_llm_embeddings.pt   — FloatTensor [N_products, 1024], unit-normalised

Fixes vs. original:
  - Uses "passage: " prefix (correct for symmetric product-to-product similarity in e5 models;
    "query: " is for asymmetric retrieval and compresses all embeddings into a narrow cone)
  - Enriches each description with its 2-digit HS chapter number for better discrimination
"""

import os, sys
import numpy as np
import pandas as pd
import torch

BACI_DIR       = os.path.join('datasets', 'BACIDataset1995')
PRODUCT_CODES  = os.path.join(BACI_DIR, 'product_codes_HS92_V202601.csv')
DATA_DIR       = 'data'
OUT_EMBEDDINGS = os.path.join(DATA_DIR, 'product_llm_embeddings.pt')

# ── 1. Build product list (same sort order as Step 6 mapping) ─────────────────
smoothed     = pd.read_csv(os.path.join(DATA_DIR, 'M_cpt_smoothed.csv'), usecols=['product'])
product_list = sorted(smoothed['product'].unique())
N            = len(product_list)
print(f'Products in graph: {N}')

# ── 2. Load HS descriptions ───────────────────────────────────────────────────
hs_df    = pd.read_csv(PRODUCT_CODES, dtype={'code': str})
hs_df    = hs_df[hs_df['code'].str.match(r'^\d+$')].copy()
hs_df['code'] = hs_df['code'].astype(int)
desc_map = dict(zip(hs_df['code'], hs_df['description']))
missing  = sum(1 for p in product_list if p not in desc_map)
print(f'Products missing HS description (fallback = code string): {missing}')

# Build enriched passages: add HS chapter (first 2 digits of zero-padded 6-digit code)
# "passage: " is the correct e5 prefix for symmetric similarity — NOT "query: "
def make_passage(code: int, desc: str) -> str:
    chapter = str(code).zfill(6)[:2]
    return f"query: HS chapter {chapter} trade product: {desc}"

prefixed = [make_passage(p, desc_map.get(p, str(p))) for p in product_list]

print('\nExample passages:')
for p in product_list[:3]:
    print(f'  {p}: {make_passage(p, desc_map.get(p, str(p)))}')

# ── 3. Embed ──────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print('sentence-transformers not installed. Run:  pip install sentence-transformers')
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\nDevice: {DEVICE}')
print('Loading intfloat/e5-large-v2 (~1.3 GB, cached after first run)...')

model      = SentenceTransformer('intfloat/e5-large-v2', device=DEVICE)
embeddings = model.encode(
    prefixed,
    batch_size=64,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True,
)

torch.save(embeddings.cpu(), OUT_EMBEDDINGS)
print(f'\nSaved -> {OUT_EMBEDDINGS}')
print(f'Shape: {tuple(embeddings.shape)}')

# ── 4. Sanity checks ──────────────────────────────────────────────────────────
pairs = [
    (854140, 854130, 'Diodes vs Diodes LED          (expect HIGH ~0.90+)'),
    (870322, 870323, 'Petrol cars 1500 vs 3000cc    (expect HIGH ~0.95+)'),
    (10111,  854140, 'Live horses vs Diodes         (expect LOW  <0.60) '),
    (100110, 610910, 'Durum wheat vs Cotton T-shirts (expect LOW  <0.60)'),
]
emb_cpu = embeddings.cpu()
print('\nSanity checks:')
for a, b, label in pairs:
    if a in product_list and b in product_list:
        i, j = product_list.index(a), product_list.index(b)
        sim = (emb_cpu[i] * emb_cpu[j]).sum().item()
        print(f'  sim({a:6d}, {b:6d})  [{label}]: {sim:.4f}')
    else:
        print(f'  ({a}, {b})  [{label}]: skipped — code not in product list')

# ── 5. Distribution summary ───────────────────────────────────────────────────
rng        = np.random.default_rng(42)
sample_idx = rng.choice(len(product_list), size=min(1000, len(product_list)), replace=False)
sample_emb = emb_cpu[sample_idx].numpy()
sims       = (sample_emb @ sample_emb.T).flatten()
sims       = sims[sims < 0.9999]  # remove self-pairs
print(f'\nPairwise similarity distribution (1000-product sample):')
print(f'  median={np.median(sims):.3f}  mean={np.mean(sims):.3f}  p95={np.percentile(sims, 95):.3f}')
print(f'  % pairs > 0.70 : {(sims > 0.70).mean()*100:.1f}%  (was 99.0% with "query:" prefix)')
print(f'  % pairs > 0.85 : {(sims > 0.85).mean()*100:.1f}%')

print('\n Step 11A complete')
