import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

EMB_PATH = os.path.join(DATA_DIR, 'product_llm_embeddings.pt')
assert os.path.exists(EMB_PATH), 'Run: python run_step11_embeddings.py  then re-run this cell'

# ── Load embeddings and product metadata ─────────────────────────────────────
emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').numpy()  # [N, 1024]

smoothed     = pd.read_csv(os.path.join(DATA_DIR, 'M_cpt_smoothed.csv'), usecols=['product'])
product_list = sorted(smoothed['product'].unique())

hs_df    = pd.read_csv(PRODUCT_CODES, dtype={'code': str})
hs_df    = hs_df[hs_df['code'].str.match(r'^\d+$')].copy()
hs_df['code'] = hs_df['code'].astype(int)
desc_map = dict(zip(hs_df['code'], hs_df['description']))

# HS chapter = first 2 digits of the zero-padded 6-digit code
chapter_arr = np.array([int(str(p).zfill(6)[:2]) for p in product_list])
N           = len(product_list)
print(f'Loaded {N} product embeddings  |  {len(set(chapter_arr.tolist()))} HS chapters')

# ── Panel 1: Pairwise cosine similarity distribution ─────────────────────────
np.random.seed(42)
sample_idx = np.random.choice(N, size=min(3000, N), replace=False)
emb_s      = emb[sample_idx]
sim_matrix = emb_s @ emb_s.T
triu_idx   = np.triu_indices(len(sample_idx), k=1)
pairwise   = sim_matrix[triu_idx]

# ── Panel 3: t-SNE (PCA 1024->50, then t-SNE 50->2) ─────────────────────────
print('Running PCA (1024 -> 50 dims) ...')
pca     = PCA(n_components=50, random_state=42)
emb_pca = pca.fit_transform(emb)
print(f'PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%')

tsne_n   = min(1500, N)
tsne_idx = np.random.choice(N, size=tsne_n, replace=False)
print(f'Running t-SNE on {tsne_n} products ...')
tsne      = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42, verbose=0)
emb_2d    = tsne.fit_transform(emb_pca[tsne_idx])
tsne_chap = chapter_arr[tsne_idx]

# ── Panel 4: Within vs between chapter similarity ─────────────────────────────
within_sims, between_sims = [], []
for ch in sorted(set(chapter_arr.tolist())):
    idx_in  = np.where(chapter_arr == ch)[0]
    idx_out = np.where(chapter_arr != ch)[0]
    if len(idx_in) < 2: continue
    s_in  = idx_in[:30]
    s_out = np.random.choice(idx_out, size=min(30, len(idx_out)), replace=False)
    within_sims.append((emb[s_in] @ emb[s_in].T)[np.triu_indices(len(s_in), k=1)].mean())
    between_sims.append((emb[s_in] @ emb[s_out].T).flatten().mean())

# ── Nearest-neighbour helper ──────────────────────────────────────────────────
def top_neighbours(code, k=4):
    if code not in product_list: return []
    i    = product_list.index(code)
    sims = emb @ emb[i]
    sims[i] = -1
    top = np.argsort(sims)[::-1][:k]
    return [(product_list[j], float(sims[j]), desc_map.get(product_list[j], '?')[:55]) for j in top]

spot = [(854140, 'Diodes'), (870322, 'Petrol cars <1500cc'),
        (610910, 'Cotton T-shirts'), (100110, 'Durum wheat'), (300490, 'Medicaments')]
spot = [(c, d) for c, d in spot if c in product_list]

# ── FIGURE ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)

# Panel 1
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(pairwise, bins=120, color='steelblue', edgecolor='none', alpha=0.85)
ax1.axvline(np.median(pairwise), color='red', linestyle='--',
            label=f'median = {np.median(pairwise):.3f}')
pct_hi = (pairwise > 0.70).mean() * 100
ax1.axvline(0.70, color='orange', linestyle=':', label=f'0.70 threshold ({pct_hi:.1f}% of pairs)')
ax1.set_title('Pairwise Cosine Similarity\n(3 000-product sample)', fontsize=11)
ax1.set_xlabel('Cosine similarity'); ax1.set_ylabel('Count')
ax1.legend(fontsize=9)

# Panel 2: nearest neighbours
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
y = 0.99
for code, label in spot:
    nbrs = top_neighbours(code)
    ax2.text(0, y, f'{label}  ({code})', fontsize=9, fontweight='bold',
             transform=ax2.transAxes, va='top')
    y -= 0.045
    for nc, nsim, ndesc in nbrs:
        ax2.text(0.02, y, f'{nsim:.3f}  {nc}  {ndesc}',
                 fontsize=7.5, transform=ax2.transAxes, va='top', color='#333')
        y -= 0.038
    y -= 0.015
ax2.set_title('Top-4 Nearest Neighbours', fontsize=11, pad=10)

# Panel 3: t-SNE
ax3 = fig.add_subplot(gs[1, 0])
uniq_ch  = sorted(set(tsne_chap.tolist()))
palette  = plt.cm.tab20(np.linspace(0, 1, min(len(uniq_ch), 20)))
ch_color = {ch: palette[i % 20] for i, ch in enumerate(uniq_ch)}
colors   = [ch_color[ch] for ch in tsne_chap]
ax3.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=5, alpha=0.55)
ax3.set_title(f't-SNE of Product Embeddings  (n={tsne_n})\nColoured by HS chapter (2-digit)', fontsize=11)
ax3.set_xticks([]); ax3.set_yticks([])

# Panel 4: within vs between
ax4 = fig.add_subplot(gs[1, 1])
bp = ax4.boxplot([within_sims, between_sims],
                 labels=['Within chapter', 'Between chapters'],
                 patch_artist=True,
                 boxprops=dict(facecolor='steelblue', alpha=0.65),
                 medianprops=dict(color='red', linewidth=2.5))
ax4.set_title('Cosine Similarity: Within vs Between HS Chapters', fontsize=11)
ax4.set_ylabel('Mean cosine similarity')
w_med = np.median(within_sims)
b_med = np.median(between_sims)
ax4.text(0.5, 0.04, f'Within={w_med:.3f}  Between={b_med:.3f}  delta={w_med-b_med:+.3f}',
         transform=ax4.transAxes, ha='center', fontsize=9, color='#333')

fig.suptitle('e5-large-v2 Product Embedding Quality  (HS92, 5 018 products)', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# ── Text summary ──────────────────────────────────────────────────────────────
print(f'Pairwise similarity summary (3000-product sample):')
print(f'  median = {np.median(pairwise):.3f}  |  mean = {pairwise.mean():.3f}  |  p95 = {np.percentile(pairwise, 95):.3f}')
print(f'  % pairs > 0.70  = {pct_hi:.1f}%  (these become capability graph edges)')
print(f'Within-chapter median  : {w_med:.3f}')
print(f'Between-chapter median : {b_med:.3f}')
delta = w_med - b_med
print(f'Delta (within-between) : {delta:+.3f}')
if delta > 0.02:
    print('-> Embeddings clearly discriminate between product categories')
elif delta > 0:
    print('-> Modest discrimination — some signal present')
else:
    print('-> No clear discrimination')
