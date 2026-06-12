# How LLM Embeddings Are Used in GNN-11F+LLM

## Summary
The LLM embeddings are used **only to construct a sparse product-product graph** — they are **NOT used as product features**. There is **NO edge weighting** on the capability edges.

---

## Current Implementation (GNN-11F+LLM)

### 1. How embeddings are generated
- **Model:** FinLang/finance-embeddings-investopedia (768-dim, finance-domain)
- **Input:** HS6 product description strings
- **Output:** `data/product_llm_embeddings.pt` — unit-normalized 768-dim vectors for each product

### 2. How they're used: Edge construction (no features)
```python
# Step 1: Compute pairwise cosine similarity
sim = embeddings @ embeddings.T        # [5018, 5018]

# Step 2: Threshold-based edge selection (cosine > 0.70)
mask = sim >= 0.70
src, dst = mask.nonzero(as_tuple=True)
edge_index = torch.stack([src, dst], dim=0)

# Step 3: Add to graph as product-product edges
d['product', 'capability', 'product'].edge_index = edge_index

# Result: 144,192 directed edges connecting semantically similar products
```

**Key point:** The embeddings themselves are **discarded after edge construction**. Only the edge_index is saved and used.

### 3. GNN architecture receiving these edges
```python
class BipartiteEncoder(nn.Module):
    def forward(self, x_dict, ei_dict):
        # x_dict['product'] = [5018, 3] — ONLY the 3 hand-crafted features
        # ei_dict includes:
        #   - ('country', 'exports', 'product') — trade edges
        #   - ('product', 'rev_exports', 'country')
        #   - ('product', 'capability', 'product') — LLM-derived edges
        
        return self.gnn({'country': self.country_lin(x_dict['country']),
                         'product': self.product_lin(x_dict['product'])},  # 3→128 dim
                         ei_dict)
```

The GNN applies **SAGEConv** to all edge types:
```python
class _HomoGNN(nn.Module):
    def forward(self, x, edge_index):
        # edge_index comes from the 3 edge types in HeteroData
        # SAGEConv signature: forward(x, edge_index, size=None)
        # NO edge_weight parameter is passed
        return self.c2(self.c1(x, edge_index).relu())
```

---

## What This Means

| Aspect | Status | Details |
|--------|--------|---------|
| **Embeddings as features?** | ❌ NO | Only 3 hand-crafted product features (ubiquity, log export, avg RCA) are used |
| **Edge weights?** | ❌ NO | All capability edges have equal weight in the graph; SAGEConv treats them as unweighted |
| **Embedding dimensions?** | ❌ DISCARDED | 768-dim vectors are used only for similarity calculation, then thrown away |
| **What's actually used?** | ✅ Topology | The **connectivity pattern** (144K product-product edges where cosine > 0.70) |
| **Where similarity matters?** | Indirectly | Cosine similarity decides the **binary presence/absence** of edges; all present edges have equal importance |

---

## Could You Improve This?

### Option A: Weight edges by cosine similarity (currently not done)
```python
# After computing similarity:
weights = sim[src, dst]  # actual cosine values, not binary
d['product', 'capability', 'product'].edge_weight = weights

# Then modify SAGEConv to use weights (requires custom implementation)
# because torch_geometric.SAGEConv.forward() does NOT accept edge_weight
```

**Why it's not done:** PyTorch Geometric's SAGEConv doesn't support edge weights natively. You'd need to either:
1. Use a different GNN layer (GCNConv supports it)
2. Implement a custom SAGEConv wrapper
3. Pre-weight the features instead

### Option B: Use embeddings as features (never attempted)
```python
# Concatenate LLM embeddings with hand-crafted features:
d['product'].x = torch.cat([
    hand_crafted_features,      # [5018, 3]
    llm_embeddings              # [5018, 768]
], dim=1)                       # -> [5018, 771]

# Then train with extra feature dimensions
product_lin = nn.Linear(771, hidden)  # was Linear(3, hidden)
```

**Why it's not done:** Untested; could help, could hurt. The 768-dim embeddings are domain-agnostic finance embeddings, not product-graph-specific learned features.

### Option C: Weighted combination (best middle ground)
```python
# Use cosine similarity to weight the contribution of neighboring products:
# Instead of "product A connects to product B", use weighted aggregation:
# "add B's features to A with weight = cosine_sim(A, B)"

# This requires GNN layers that natively support edge weights.
```

---

## Why GNN-11F+LLM Might Not Be Helping

Given that:
1. Edge weights are **all uniform** (no cosine weighting)
2. Embeddings are **completely discarded** (not used as features)
3. GNN validation PR-AUC early-stopped at **0.27** (very low)

The capability edges may be adding **noise rather than signal**:
- 144K new edges is a lot of connectivity to learn from sparse training data
- Without weighting, far and near neighbors are treated equally
- Without feature information, the GNN has no way to know **why** two products are connected

---

## Recommendation

Before claiming the LLM layer is useful, consider:

1. **Run multi-seed training** — is the improvement (4.2% PR-AUC delta: 0.638→0.659 on RCA>0.25 filtered set) consistent across seeds, or within noise?

2. **Try edge weighting** — Replace SAGEConv with GCNConv (which supports edge_weight) and weight edges by cosine similarity. This would make semantically closer products matter more.

3. **Try embeddings as features** — Concatenate 768-dim LLM embeddings with hand-crafted features; see if domain-specific semantic information helps.



---

**Status:** Current implementation is **topology-only** (unweighted, no feature integration). It is **not making full use** of the LLM embeddings' information.
