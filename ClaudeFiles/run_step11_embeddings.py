"""
Step 11A: LLM Product Embeddings — run as standalone script if the notebook kernel crashes.

    python run_step11_embeddings.py

Model: FinLang/investopedia_embedding (768-dim, finance-domain fine-tuned on top of BAAI/bge-base-en-v1.5)
No special prefix required — model encodes sentences directly.

Saves:
    data/product_llm_embeddings.pt   — FloatTensor [N_products, 768], unit-normalised
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

# No prefix needed for FinLang/investopedia_embedding
descriptions = [desc_map.get(p, str(p)) for p in product_list]

print(f'\nExample inputs:')
for p in product_list[:3]:
    print(f'  {p}: {desc_map.get(p, str(p))}')

# ── 3. Embed ──────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print('sentence-transformers not installed. Run:  pip install sentence-transformers')
    sys.exit(1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'\nDevice: {DEVICE}')
print('Loading FinLang/finance-embeddings-investopedia (finance-domain, 768-dim)...')

model      = SentenceTransformer('FinLang/finance-embeddings-investopedia', device=DEVICE)
embeddings = model.encode(
    descriptions,
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
print(f'  % pairs > 0.70 : {(sims > 0.70).mean()*100:.1f}%')
print(f'  % pairs > 0.85 : {(sims > 0.85).mean()*100:.1f}%')

print('\nStep 11A complete (FinLang/finance-embeddings-investopedia)')
