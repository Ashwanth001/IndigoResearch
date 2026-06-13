"""
Patch evaluation notebooks:
1. KNN: swap 768-dim llm_emb -> 32-dim PCA embeddings in both notebooks
2. XGBoost: fix inplace_predict -> predict_proba in full_universe_eval
"""
import json, os

def get_cell(nb, cid):
    for cell in nb['cells']:
        if cell.get('id') == cid:
            return cell
    raise KeyError(cid)

def get_source(cell):
    return ''.join(cell['source'])

def set_source(cell, s):
    cell['source'] = s


# ══════════════════════════════════════════════════════════════════════════════
# 1. internal_benchmarking.ipynb — KNN in ib-2015-classical and ib-2016-classical
# ══════════════════════════════════════════════════════════════════════════════
IB_PATH = 'internal_benchmarking.ipynb'
with open(IB_PATH, encoding='utf-8') as f:
    ib = json.load(f)

OLD_KNN_BLOCK = (
    "# ── Method 5: KNN (LLM embeddings) ───────────────────────────────────────────\n"
    "country_basket = {}\n"
    "for c in df_t['country'].unique():\n"
    "    ci = c_idx.get(c, -1)\n"
    "    if ci < 0: continue\n"
    "    exported = np.where(M_t[ci] == 1)[0]\n"
    "    basket = llm_emb[exported].mean(axis=0) if len(exported) else np.zeros(llm_emb.shape[1])\n"
    "    norm = np.linalg.norm(basket)\n"
    "    country_basket[c] = basket / (norm + 1e-9)\n"
    "knn_sc = np.array([\n"
    "    float(llm_emb[p_idx[p]] @ country_basket[c]) if c in country_basket and p in p_idx else 0.0\n"
    "    for c, p in zip(df_t['country'].values, df_t['product'].values)\n"
    "], dtype=np.float32)\n"
    "evaluate(T, 'KNN (LLM embeddings)', knn_sc)"
)

NEW_KNN_BLOCK = (
    "# ── Method 5: KNN (PCA-LLM embeddings, 32-dim) ───────────────────────────────\n"
    "# Uses llm_pca_np [P, 32] — already fitted & L2-normalised in setup cell.\n"
    "country_basket = {}\n"
    "for c in df_t['country'].unique():\n"
    "    ci = c_idx.get(c, -1)\n"
    "    if ci < 0: continue\n"
    "    exported = np.where(M_t[ci] == 1)[0]\n"
    "    basket = llm_pca_np[exported].mean(axis=0) if len(exported) else np.zeros(PCA_DIM)\n"
    "    norm = np.linalg.norm(basket)\n"
    "    country_basket[c] = basket / (norm + 1e-9)\n"
    "knn_sc = np.array([\n"
    "    float(llm_pca_np[p_idx[p]] @ country_basket[c]) if c in country_basket and p in p_idx else 0.0\n"
    "    for c, p in zip(df_t['country'].values, df_t['product'].values)\n"
    "], dtype=np.float32)\n"
    "evaluate(T, 'KNN (LLM embeddings)', knn_sc)"
)

fixed_ib = 0
for cid in ('ib-2015-classical', 'ib-2016-classical'):
    cell = get_cell(ib, cid)
    src = get_source(cell)
    if OLD_KNN_BLOCK in src:
        set_source(cell, src.replace(OLD_KNN_BLOCK, NEW_KNN_BLOCK))
        fixed_ib += 1
        print(f'  IB: fixed KNN in {cid}')
    else:
        print(f'  IB: WARNING — KNN block not found in {cid}')
        # Show what we're looking for vs what's there around Method 5
        idx = src.find('Method 5')
        if idx >= 0:
            print('  Found near Method 5:', repr(src[idx:idx+200]))

with open(IB_PATH, 'w', encoding='utf-8') as f:
    json.dump(ib, f, indent=1, ensure_ascii=False)
print(f'internal_benchmarking.ipynb: {fixed_ib}/2 KNN cells updated')


# ══════════════════════════════════════════════════════════════════════════════
# 2. full_universe_eval.ipynb
# ══════════════════════════════════════════════════════════════════════════════
FU_PATH = 'full_universe_eval.ipynb'
with open(FU_PATH, encoding='utf-8') as f:
    fu = json.load(f)

# ── 2a. KNN cell (fu-m5) ─────────────────────────────────────────────────────
cell_knn = get_cell(fu, 'fu-m5')
src_knn = get_source(cell_knn)

NEW_KNN_FU = (
    "# Load 768-dim embeddings and derive 32-dim PCA version.\n"
    "EMB_PATH = os.path.join(DATA_DIR, 'product_llm_embeddings.pt')\n"
    "emb = torch.load(EMB_PATH, weights_only=False, map_location='cpu').numpy()  # [P, 768]\n"
    "print(f'Embeddings: {emb.shape}')\n"
    "\n"
    "from sklearn.decomposition import PCA as _PCA\n"
    "_pca_knn = _PCA(n_components=32, random_state=42)\n"
    "_pca_knn.fit(emb)\n"
    "emb_pca = _pca_knn.transform(emb).astype(np.float32)  # [P, 32]\n"
    "emb_pca /= np.linalg.norm(emb_pca, axis=1, keepdims=True).clip(min=1e-8)\n"
    "print(f'PCA-LLM: {emb_pca.shape}  '\n"
    "      f'(explains {_pca_knn.explained_variance_ratio_.sum()*100:.1f}% variance)')\n"
    "\n"
    "def knn_scores(M_t, df_yr):\n"
    "    # Vectorised: mean basket in 32-dim PCA space, cosine dot product.\n"
    "    basket = np.zeros((C, 32), dtype=np.float32)\n"
    "    for ci_val in range(C):\n"
    "        exported = np.where(M_t[ci_val] == 1)[0]\n"
    "        if len(exported):\n"
    "            b = emb_pca[exported].mean(axis=0)\n"
    "            norm = np.linalg.norm(b)\n"
    "            basket[ci_val] = b / (norm + 1e-9)\n"
    "    ci_arr = df_yr['ci'].values\n"
    "    pi_arr = df_yr['pi'].values\n"
    "    return (emb_pca[pi_arr] * basket[ci_arr]).sum(axis=1).astype(np.float32)\n"
    "\n"
    "for yr, M_t, df_yr in [(2015, M_2015, df_2015), (2016, M_2016, df_2016)]:\n"
    "    evaluate(yr, 'KNN (LLM embeddings)', knn_scores(M_t, df_yr))"
)

set_source(cell_knn, NEW_KNN_FU)
print('full_universe_eval.ipynb: KNN cell (fu-m5) replaced with PCA-LLM version')

# ── 2b. Fix XGBoost predict in fu-xgb-code ───────────────────────────────────
cell_xgb = get_cell(fu, 'fu-xgb-code')
src_xgb = get_source(cell_xgb)

OLD_XGB_FN = (
    "def xgb_scores_universe(t, df_yr):\n"
    "    \"\"\"GPU-accelerated XGBoost prediction over the full universe.\"\"\"\n"
    "    X = build_xgb_features_fu(df_yr, t).astype(np.float32)\n"
    "    \n"
    "    # FIX: Get the underlying Booster object which natively supports inplace_predict\n"
    "    booster = xgb_model_fu.get_booster()\n"
    "    \n"
    "    if _USE_CUPY_FU:\n"
    "        X_gpu = cp.asarray(X)\n"
    "        return np.asarray(booster.inplace_predict(X_gpu)).astype(np.float64)\n"
    "    return booster.inplace_predict(X).astype(np.float64)"
)

NEW_XGB_FN = (
    "def xgb_scores_universe(t, df_yr):\n"
    "    \"\"\"XGBoost probability scores over the full universe.\"\"\"\n"
    "    X = build_xgb_features_fu(df_yr, t).astype(np.float32)\n"
    "    return xgb_model_fu.predict_proba(X)[:, 1].astype(np.float64)"
)

if OLD_XGB_FN in src_xgb:
    set_source(cell_xgb, src_xgb.replace(OLD_XGB_FN, NEW_XGB_FN))
    print('full_universe_eval.ipynb: XGBoost predict_proba fixed')
else:
    # Try a looser match on the key bad lines
    if 'inplace_predict' in src_xgb:
        import re
        new_src = re.sub(
            r'def xgb_scores_universe\(t, df_yr\):.*?return.*?inplace_predict.*?\n',
            NEW_XGB_FN + '\n',
            src_xgb,
            flags=re.DOTALL
        )
        set_source(cell_xgb, new_src)
        print('full_universe_eval.ipynb: XGBoost fixed via regex')
    else:
        print('full_universe_eval.ipynb: XGBoost already clean (no inplace_predict found)')

with open(FU_PATH, 'w', encoding='utf-8') as f:
    json.dump(fu, f, indent=1, ensure_ascii=False)

# ── Verify ────────────────────────────────────────────────────────────────────
print('\n=== Verification ===')
with open(IB_PATH, encoding='utf-8') as f:
    ib2 = json.load(f)
for cid in ('ib-2015-classical', 'ib-2016-classical'):
    src = get_source(get_cell(ib2, cid))
    uses_pca = 'llm_pca_np' in src
    uses_full = 'llm_emb[exported]' in src
    print(f'IB {cid}: uses_pca={uses_pca}  uses_full_emb={uses_full}')

with open(FU_PATH, encoding='utf-8') as f:
    fu2 = json.load(f)
src_knn2 = get_source(get_cell(fu2, 'fu-m5'))
src_xgb2 = get_source(get_cell(fu2, 'fu-xgb-code'))
print(f'FU knn: uses emb_pca={("emb_pca" in src_knn2)}  uses full emb={("emb[exported]" in src_knn2)}')
print(f'FU xgb: uses predict_proba={("predict_proba" in src_xgb2)}  uses inplace_predict={("inplace_predict" in src_xgb2)}')
