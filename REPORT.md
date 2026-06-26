# LLMAREC: LLM-Augmented Recommendation System
## Implementation Report

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [System Architecture](#3-system-architecture)
4. [Pipeline Stages](#4-pipeline-stages)
   - [Stage 1 – Profile Generation](#stage-1--profile-generation)
   - [Stage 2 – Embedding Generation](#stage-2--embedding-generation)
   - [Stage 3 – Hybrid Model Training](#stage-3--hybrid-model-training)
5. [Models Used](#5-models-used)
6. [Configuration & Hyperparameters](#6-configuration--hyperparameters)
7. [Profile Examples](#7-profile-examples)
8. [Code Walkthrough](#8-code-walkthrough)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Results](#10-results)
11. [Discussion](#11-discussion)

---

## 1. Project Overview

**LLMAREC** is an implementation of the **RLMRec** framework (*Representation Learning with Large Language Models for Recommendation*). It addresses a core limitation of traditional collaborative filtering systems: they rely exclusively on user-item interaction IDs, ignoring the rich semantic content that describes what users like and what items are about.

### Core Idea

The system bridges two worlds:

- **Collaborative Filtering (LightGCN)** — learns latent user/item representations from the interaction graph
- **LLM Semantic Understanding** — generates natural language profiles for each user and item, then encodes them into dense embedding vectors

These two streams are fused inside a hybrid neural model trained end-to-end with BPR (Bayesian Personalized Ranking) loss.

### Theoretical Motivation

From the RLMRec paper, the alignment objective is to **maximize mutual information** between collaborative embeddings `e` and semantic embeddings `s`:

```
maximize  I(e; s)
```

Two alignment strategies are described in the paper:

| Strategy | Objective | Analogy |
|---|---|---|
| **Contrastive** | `f(s, e) = exp(sim(σ(s), e))` | Pull positive (user, item) pairs together; push negatives apart |
| **Generative** | `f(s, e) = exp(sim(s, σ(e)))` | Masked reconstruction (MAE-inspired); predict masked CF embeddings from text |

This implementation uses **feature-level fusion**: text embeddings are injected into the LightGCN input layer via a learnable weight, and a downstream MLP predictor learns to combine ID-based and text-based interaction signals.

---

## 2. Dataset

The project uses the **MovieLens 1M** dataset, a standard benchmark for recommendation systems.

### Statistics

| Property | Value |
|---|---|
| Total users | 6,040 |
| Total movies | 3,883 |
| Total interactions | 1,000,210 |
| Training interactions | 800,193 (80%) |
| Test interactions | 200,017 (20%) |
| Rating scale | 1–5 (integer) |
| Relevance threshold | ≥ 3.5 |

### Raw Data Format

**`data/raw/movies.dat`** — pipe-delimited with `::` separator:
```
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
```

**`data/processed/train_ratings.csv`**:
```
user_id,movie_id,rating
0,1104,5
0,639,3
0,853,3
0,2162,5
1,1259,5
1,2853,4
```

---

## 3. System Architecture

The full system consists of three sequential stages that produce artifacts consumed by the next stage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LLMAREC PIPELINE                                   │
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  Raw Data   │    │  LLM Profiles   │    │    Gemini Embeddings        │ │
│  │             │    │                 │    │                             │ │
│  │  movies.dat │───▶│  users.json     │───▶│  users.npz  (6040 × 768)   │ │
│  │  train.csv  │    │  movies.json    │    │  movies.npz (3883 × 768)   │ │
│  └─────────────┘    └─────────────────┘    └──────────────┬──────────────┘ │
│                                                           │                │
│                                                           ▼                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        HybridModel (PyTorch)                          │ │
│  │                                                                       │ │
│  │   User ID ──▶ Embedding(64) ─┐                                        │ │
│  │                              ├──▶ LightGCN (3 layers) ──▶ u_l (64)   │ │
│  │   Item ID ──▶ Embedding(64) ─┘         ▲                  i_l (64)   │ │
│  │                                        │ text_weight ×               │ │
│  │   User Profile ──▶ Gemini(768) ──▶ Proj(64) ──▶ u_t (64)            │ │
│  │   Item Profile ──▶ Gemini(768) ──▶ Proj(64) ──▶ i_t (64)            │ │
│  │                                                                       │ │
│  │   score = dot(u_l, i_l) + MLP(LayerNorm(u_l·i_l) ∥ LayerNorm(u_t·i_t)) │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | File | Purpose |
|---|---|---|
| Profile Generator (users) | `user_profile.py` | Generate natural language user taste profiles via LLM |
| Profile Generator (movies) | `movie_profiles.py` | Generate natural language movie descriptions via LLM |
| Embedding Generator | `embedding.py` | Convert text profiles → 768-dim dense vectors |
| LightGCN Baseline | `lightgcn.py` | Pure GNN-based collaborative filtering |
| HybridModel | `hybrid_rec/model.py` | LightGCN + text fusion model |
| Data Loader | `hybrid_rec/data_loader.py` | Load ratings, build adjacency matrix, align embeddings |
| Training Loop | `hybrid_rec/train.py` | BPR training, evaluation, checkpointing |
| Config | `hybrid_rec/config.py` | Central hyperparameter dictionary |
| Benchmark | `benchmark.py` | Run both models and log comparison |

---

## 4. Pipeline Stages

### Stage 1 – Profile Generation

Two separate scripts generate text profiles using an LLM with an OpenAI-compatible API.

#### User Profile Generation (`user_profile.py`)

For each user, the script:
1. Collects all their rated movies from `train_ratings.csv`
2. Maps movie IDs to titles using `movies.dat`
3. Builds a structured prompt sorted by rating (highest first)
4. Calls the LLM API asynchronously (up to 10 concurrent requests)
5. Saves the result to `data/profiles/users.json` with atomic writes

**Prompt template:**

```
You are a movie taste analyst. Based on the following movie ratings from user {user_id},
write a concise user profile (3-5 sentences) describing their movie preferences,
favorite genres, and viewing patterns.

Ratings:
  - Schindler's List (1993): 5/5
  - The Godfather (1972): 5/5
  - Saving Private Ryan (1998): 4/5
  - American Pie (1999): 2/5
  ...

Respond with only the profile text, no headers or extra formatting.
```

**API parameters:** `temperature=0.7`, `max_tokens=300`

#### Movie Profile Generation (`movie_profiles.py`)

For each movie, the script:
1. Reads title and genre string from `movies.dat`
2. Builds a film critic-style prompt
3. Calls the LLM asynchronously and saves to `data/profiles/movies.json`

**Prompt template:**

```
You are a knowledgeable film expert. Based solely on your knowledge of cinema,
write a concise (3–5 sentences) profile for the movie titled "{title}",
which falls under the genres: {genre_list}. Include key themes, style,
notable aspects, and what kind of audience might enjoy it.

Respond with only the profile text—no titles, headers, or extra formatting.
```

**API parameters:** `temperature=0.7`, `max_tokens=300`

#### Checkpointing

Both scripts use a `ProfileCheckpointer` class that performs atomic writes — on each save, data is first written to a `.tmp` file then `os.replace`d to prevent corruption on interruption:

```python
class ProfileCheckpointer:
    async def save(self, entity_id: int, profile: str):
        async with self._lock:
            self._profiles[str(entity_id)] = profile
            self._flush()

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._profiles, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)
```

---

### Stage 2 – Embedding Generation

**`embedding.py`** converts the text profiles into dense vectors using the Gemini embedding model.

**Key details:**
- Model: `gemini-embedding-001` — produces 768-dimensional float32 vectors
- Concurrency: 20 simultaneous API requests
- Output format: `.npz` files with two arrays — `ids` (int32) and `embeddings` (float32)

```python
async def embed_one(sem, checkpointer, entity_id, text):
    async with sem:
        resp = await client.embeddings.create(
            model="gemini-embedding-001",
            input=text,
        )
    vector = resp.data[0].embedding
    await checkpointer.save(entity_id, vector)
```

**Output files:**

| File | Shape | Size |
|---|---|---|
| `data/embeddings/users.npz` | (6040, 768) | ~71 MB |
| `data/embeddings/movies.npz` | (3883, 768) | ~46 MB |

---

### Stage 3 – Hybrid Model Training

#### Graph Construction

A normalized bipartite adjacency matrix is built over the user-item interaction graph:

```python
def build_adj_matrix(train_df, user2idx, item2idx, n_users, n_items):
    # Build COO sparse matrix with user-item and item-user edges
    N = n_users + n_items
    A = sparse.coo_matrix((adj_data, (adj_row, adj_col)), shape=(N, N))

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    deg_inv_sqrt = np.power(deg, -0.5)
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
```

#### LightGCN Graph Convolution

Each layer propagates embeddings along edges; final embeddings are the mean across all layers (including layer 0):

```
e_u^(l+1) = Ã · e_u^(l)

e_u_final = (1 / (L+1)) · Σ_{l=0}^{L} e_u^(l)
```

In code:
```python
def forward(self, adj, u_text_proj=None, i_text_proj=None):
    u_e0 = self.user_emb.weight
    i_e0 = self.item_emb.weight

    # Inject text signal into initial embedding (with learnable weight)
    if u_text_proj is not None:
        u_e0 = u_e0 + self.text_weight * u_text_proj
        i_e0 = i_e0 + self.text_weight * i_text_proj

    e0 = torch.cat([u_e0, i_e0], dim=0)
    layer_embs = [e0]
    x = e0
    for _ in range(self.n_layers):
        x = torch.sparse.mm(adj, x)
        layer_embs.append(x)

    out = torch.stack(layer_embs, dim=0).mean(dim=0)
    return out[:n_users], out[n_users:]
```

#### Text Projection

Text embeddings (768-dim) are projected down to the LightGCN embedding space (64-dim):

```python
self.user_text_proj = nn.Sequential(
    nn.Linear(768, 64),
    nn.ReLU(),
    nn.LayerNorm(64),
    nn.Dropout(0.1)
)
```

#### Score Computation

The final score combines a LightGCN dot product with an MLP-derived residual:

```python
def forward(self, adj, user_indices, item_indices):
    u_l, i_l, u_t, i_t = self.get_all_embeddings(adj)

    u_batch = u_l[user_indices]
    i_batch = i_l[item_indices]

    # Base collaborative signal
    dot_score = (u_batch * i_batch).sum(dim=1, keepdim=True)

    # Feature interaction vectors
    interaction      = self.norm_inter(u_batch * i_batch)        # ID-based
    text_interaction = self.norm_text(u_t[user_indices] * i_t[item_indices])  # Semantic

    combined = torch.cat([interaction, text_interaction], dim=1)  # (B, 128)

    # MLP learns a residual correction
    residual = self.predictor(combined)

    return (dot_score + residual).squeeze(-1)
```

#### BPR Training Loss

The model is trained with Bayesian Personalized Ranking — for each user, a positive item (interacted with) and negative item (not interacted with) are sampled, and the model is trained to rank the positive higher:

```
L = -log σ(score(u, i+) − score(u, i−)) + λ · ||θ||²
```

```python
loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

l2_reg = sum(torch.norm(p)**2 for p in model.parameters() if p.requires_grad)
loss += weight_decay * 0.5 * l2_reg
```

---

## 5. Models Used

| Role | Model | Provider | Dimensions | Concurrency |
|---|---|---|---|---|
| User profile generation | `gemini-2.5-flash-lite` | GapGPT (Gemini-compatible) | N/A | 10 |
| Movie profile generation | `gemini-2.5-flash-lite` | GapGPT (Gemini-compatible) | N/A | 10 |
| Text embedding | `gemini-embedding-001` | GapGPT (Gemini-compatible) | 768 | 20 |
| Collaborative backbone | LightGCN (custom PyTorch) | Local | 64 | — |

**API endpoint:** `https://api.gapgpt.app/v1` (OpenAI-compatible interface to Gemini models)

**Client library:** `openai` (AsyncOpenAI for async/concurrent calls)

---

## 6. Configuration & Hyperparameters

**`hybrid_rec/config.py`:**

```python
CONFIG = {
    # Data paths
    "train_path":     "data/processed/train_ratings.csv",
    "test_path":      "data/processed/test_ratings.csv",
    "user_npz_path":  "data/embeddings/users.npz",
    "item_npz_path":  "data/embeddings/movies.npz",

    # Feature flags
    "use_text_embeddings": True,

    # LightGCN architecture
    "lgcn_emb_dim":   64,    # Collaborative embedding dimension
    "lgcn_layers":    3,     # Number of graph convolution layers

    # MLP predictor architecture
    "predictor_hidden_layers": [128, 64],  # Input(128) → 128 → 64 → 1

    # Training
    "epochs":         30,
    "batch_size":     1024,
    "lr":             1e-3,
    "weight_decay":   1e-4,
    "loss_type":      "bpr",   # "mse" or "bpr"
    "device":         "cuda" if torch.cuda.is_available() else "cpu",

    # Evaluation
    "K":                    10,    # Top-K recommendations
    "relevance_threshold":  3.5,   # Rating >= 3.5 is relevant
}
```

**LightGCN standalone hyperparameters (from `lightgcn.py`):**

| Parameter | Value |
|---|---|
| Embedding dimension | 64 |
| GCN layers | 3 |
| Epochs | 30 |
| Batch size | 2048 |
| Learning rate | 1e-3 |
| L2 regularization | 1e-4 |
| Optimizer | Adam |
| Embedding init | Xavier uniform |

**Profile generation parameters:**

| Parameter | Value |
|---|---|
| LLM model | `gemini-2.5-flash-lite` |
| Temperature | 0.7 |
| Max tokens | 300 |
| Concurrency | 10 |

**Embedding generation parameters:**

| Parameter | Value |
|---|---|
| Embedding model | `gemini-embedding-001` |
| Output dimension | 768 |
| Concurrency | 20 |

---

## 7. Profile Examples

### User Profiles

**User 5213** *(character-driven drama fan)*:
> User 5213 demonstrates a strong appreciation for character-driven dramas and independent films, particularly those from the 1990s, with a preference for critically acclaimed or lesser-known gems. They strongly dislike mainstream horror and thrillers, as well as certain classic noirs, suggesting a taste for subtlety and realism over genre conventions...

**User 5215** *(fantasy/comedy enthusiast)*:
> User 5215 exhibits a strong preference for fantasy and comedy, as evidenced by their perfect scores for *Billy Madison* and *The NeverEnding Story II*. They also appreciate classic horror and character-driven dramas, with high ratings for *Frankenstein Meets the Wolf Man* and *Red Firecracker, Green Firecracker*...

**User 5214** *(arthouse/prestige cinema)*:
> User 5214 displays a strong appreciation for critically acclaimed and often thought-provoking cinema, favoring dramas and character-driven narratives. They show a particular fondness for films that explore complex themes, evidenced by their high ratings for classics like *The Godfather* and *Eyes Wide Shut*...

---

### Movie Profiles

**Movie 1921 — *Pi* (1998)** *(Thriller)*:
> Darren Aronofsky's debut, *Pi*, is a stark, black-and-white descent into the mind of a brilliant but paranoid mathematician obsessed with finding patterns in the stock market and the universe. This psychological thriller masterfully blends paranoia, numerology, and the pursuit of ultimate knowledge, creating an unsettling atmosphere...

**Movie 3841 — *Air America* (1990)** *(Action|Comedy)*:
> "Air America" is a boisterous action-comedy that satirizes the Vietnam War era through the lens of a CIA-backed airline operating in Laos. It blends daring aerial sequences with irreverent humor, exploring themes of corruption, disillusionment, and the absurdity of covert operations...

**Movie 3360 — *Hoosiers* (1986)** *(Drama)*:
> "Hoosiers" is an inspiring 1986 drama that masterfully captures the underdog spirit of small-town Indiana basketball. Focusing on themes of redemption, community, and the power of belief, the film is renowned for its gritty realism and emotional resonance, anchored by Gene Hackman's compelling performance...

---

## 8. Code Walkthrough

### Full HybridModel Class

```python
class HybridModel(nn.Module):
    def __init__(self, n_users, n_items, lgcn_emb_dim, text_emb_dim,
                 lgcn_layers, use_text_embeddings=True,
                 predictor_hidden_layers=[128, 64]):
        super().__init__()
        self.use_text_embeddings = use_text_embeddings

        self.lightgcn = LightGCNModule(n_users, n_items, lgcn_emb_dim, lgcn_layers)

        if self.use_text_embeddings:
            # Frozen text embedding lookup tables (loaded from .npz)
            self.user_text_emb = nn.Embedding(n_users, text_emb_dim).requires_grad_(False)
            self.item_text_emb = nn.Embedding(n_items, text_emb_dim).requires_grad_(False)

            # Projection: 768 → 64
            self.user_text_proj = nn.Sequential(
                nn.Linear(text_emb_dim, lgcn_emb_dim),
                nn.ReLU(), nn.LayerNorm(lgcn_emb_dim), nn.Dropout(0.1)
            )
            self.item_text_proj = nn.Sequential(
                nn.Linear(text_emb_dim, lgcn_emb_dim),
                nn.ReLU(), nn.LayerNorm(lgcn_emb_dim), nn.Dropout(0.1)
            )
            input_dim = 2 * lgcn_emb_dim  # 128: ID interaction + text interaction
        else:
            input_dim = lgcn_emb_dim       # 64: ID interaction only

        self.norm_inter = nn.LayerNorm(lgcn_emb_dim)
        self.norm_text  = nn.LayerNorm(lgcn_emb_dim)
        self.predictor  = MLP([input_dim] + predictor_hidden_layers + [1])
```

### MLP Predictor

```python
class MLP(nn.Module):
    def __init__(self, layers_hidden, dropout=0.1):
        super().__init__()
        layers = []
        for in_f, out_f in zip(layers_hidden, layers_hidden[1:]):
            layers += [nn.Linear(in_f, out_f), nn.ReLU(), nn.Dropout(dropout)]
        self.model = nn.Sequential(*layers[:-2])  # remove last ReLU and Dropout

    def forward(self, x):
        return self.model(x)
```

For the hybrid model, MLP takes input `[128]` and produces `[128 → 64 → 1]`.

### Embedding Alignment in Data Loader

Text embeddings loaded from `.npz` are aligned to the internal index mapping:

```python
def load_text_embeddings(user_npz_path, item_npz_path, user2idx, item2idx):
    user_data = np.load(user_npz_path)
    u_ids, u_embs = user_data["ids"], user_data["embeddings"]
    u_id_to_idx = {uid: i for i, uid in enumerate(u_ids)}

    text_emb_dim = u_embs.shape[1]  # 768
    user_text_tensor = np.zeros((len(user2idx), text_emb_dim), dtype=np.float32)

    for uid, idx in user2idx.items():
        if uid in u_id_to_idx:
            user_text_tensor[idx] = u_embs[u_id_to_idx[uid]]

    # Same pattern for items...
    return torch.from_numpy(user_text_tensor), torch.from_numpy(item_text_tensor)
```

---

## 9. Evaluation Metrics

All metrics are computed at `K=10` with a relevance threshold of `3.5` (ratings ≥ 3.5 are treated as positive).

| Metric | Formula | Interpretation |
|---|---|---|
| **RMSE** | √(mean((ŷ − y)²)) | Rating prediction accuracy; lower is better |
| **MAE** | mean(\|ŷ − y\|) | Mean absolute rating error; lower is better |
| **Precision@K** | \|hits\| / K | Fraction of top-K recommendations that are relevant |
| **Recall@K** | \|hits\| / \|relevant\| | Fraction of all relevant items appearing in top-K |
| **F1@K** | 2·P·R / (P + R) | Harmonic mean of Precision and Recall |
| **HitRate@K** | Fraction of users with ≥1 hit in top-K | User-level coverage |
| **NDCG@K** | DCG / IDCG | Ranking quality; rewards relevant items ranked higher |
| **MAP@K** | Mean Average Precision | Area under the precision-recall curve |
| **MRR** | mean(1 / rank of first hit) | How early the first relevant item appears |
| **Coverage@K** | \|unique recommended\| / \|all items\| | Recommendation diversity |

---

## 10. Results

Results logged to `benchmark_results.log` (run date: 2026-05-30):

### LightGCN Baseline

Evaluated on all 6,040 test users.

| Metric | Value |
|---|---|
| **RMSE** | 1.8602 |
| **MAE** | 1.4848 |
| **Precision@10** | 0.2208 |
| **Recall@10** | 0.1457 |
| **F1@10** | 0.1435 |
| **HitRate@10** | 0.7642 |
| **NDCG@10** | 0.2708 |
| **MAP@10** | 0.1611 |
| **MRR** | 0.4893 |
| **Coverage@10** | 0.1816 (673 / 3883 items) |

### Hybrid Model (LightGCN + Text)

Evaluated on a random sample of 1,000 test users (for speed).

| Metric | Value |
|---|---|
| **RMSE** | 3.7432 |
| **MAE** | 3.3058 |
| **Precision@10** | 0.2273 |
| **Recall@10** | 0.1499 |
| **F1@10** | 0.1486 |
| **HitRate@10** | 0.7760 |
| **NDCG@10** | 0.2776 |
| **MAP@10** | 0.1654 |
| **MRR** | 0.4929 |
| **Coverage@10** | 0.1570 (582 / 3883 items) |

### Comparison Table

| Metric | LightGCN | Hybrid | Δ | Winner |
|---|---|---|---|---|
| RMSE ↓ | **1.8602** | 3.7432 | +1.8830 | LightGCN |
| MAE ↓ | **1.4848** | 3.3058 | +1.8210 | LightGCN |
| Precision@10 ↑ | 0.2208 | **0.2273** | +0.0065 | Hybrid |
| Recall@10 ↑ | 0.1457 | **0.1499** | +0.0042 | Hybrid |
| HitRate@10 ↑ | 0.7642 | **0.7760** | +0.0118 | Hybrid |
| NDCG@10 ↑ | 0.2708 | **0.2776** | +0.0068 | Hybrid |
| MAP@10 ↑ | 0.1611 | **0.1654** | +0.0043 | Hybrid |
| MRR ↑ | 0.4893 | **0.4929** | +0.0036 | Hybrid |
| Coverage@10 ↑ | **0.1816** | 0.1570 | −0.0246 | LightGCN |

---

## 11. Discussion

### What Worked

- **All ranking metrics improved** with the hybrid model: Precision@10, Recall@10, HitRate@10, NDCG@10, MAP@10, and MRR all show gains over the pure LightGCN baseline.
- The text injection mechanism (small initial `text_weight = 0.1`) successfully prevents the semantic signal from overwhelming collaborative signals during early training.
- The async profile and embedding generation pipeline with atomic checkpointing is robust to interruption across 10,000+ API calls.

### Areas for Improvement

- **RMSE is worse** in the hybrid model (3.74 vs 1.86). This suggests the model ranks items correctly but struggles to predict absolute rating values. The BPR loss optimizes for ranking, not calibrated score prediction — the dot product + MLP residual is not constrained to the 1–5 rating range.
- **Coverage is lower** for the hybrid (582 vs 673 unique items). The semantic signal may concentrate recommendations around well-described popular movies.
- The hybrid was evaluated on only 1,000 users (sampled for speed) vs 6,040 for LightGCN, so the comparison is not perfectly controlled.
- Further tuning of `text_weight` initialization, projection layer depth, and regularization strength could close the RMSE gap.

### File Structure Summary

```
LLMAREC/
├── user_profile.py          # Stage 1: generate user profiles via LLM
├── movie_profiles.py        # Stage 1: generate movie profiles via LLM
├── embedding.py             # Stage 2: embed profiles with Gemini
├── lightgcn.py              # Baseline model
├── hybrid_rec/
│   ├── config.py            # Hyperparameters
│   ├── model.py             # HybridModel architecture
│   ├── data_loader.py       # Data utilities
│   └── train.py             # Training and evaluation loop
├── benchmark.py             # Compare both models
├── benchmark_results.log    # Logged results
└── data/
    ├── raw/movies.dat
    ├── processed/{train,test}_ratings.csv
    ├── profiles/{users,movies}.json
    └── embeddings/{users,movies}.npz
```
