# LLMAREC

A from-scratch PyTorch reimplementation of **RLMRec** (Representation Learning with Large Language Models for Recommendation, WSDM 2024), benchmarked against a plain **LightGCN** baseline on MovieLens-1M.

The pipeline generates natural-language "profiles" for every user and movie with an LLM, embeds those profiles, and injects the resulting semantic vectors into a LightGCN collaborative-filtering backbone via a contrastive (InfoNCE) alignment loss — the core idea behind RLMRec's `con` variant.

## How it works

```
data/raw (MovieLens-1M)
      │
      ▼
   split.py ───────────────► data/processed/{train,test}_ratings.csv
      │
      ├──► user_profile.py ──► data/profiles/users.json   (LLM-written user taste summaries)
      ├──► movie_profiles.py ─► data/profiles/movies.json  (LLM-written movie synopses)
      │
      ▼
   embedding.py ────────────► data/embeddings/{users,movies}.npz  (text embeddings)
      │
      ▼
   run_hybrid.py ──► hybrid_rec/train.py ──► RLMRec-style LightGCN + KD/InfoNCE alignment
   lightgcn.py     ────────────────────────► plain LightGCN baseline
      │
      ▼
   benchmark.py ─── runs both models and logs comparable metrics
```

1. **`split.py`** also doubles as a simple item-based CF baseline (cosine similarity, RMSE/MAE) over the train/test split.
2. **`user_profile.py`** / **`movie_profiles.py`** call an LLM to turn each user's rating history (and each movie's title/genres) into a short natural-language profile, with checkpointing so runs can be resumed.
3. **`embedding.py`** embeds every profile with a text-embedding model and stores the vectors alongside their IDs in `.npz` files.
4. **`hybrid_rec/`** contains the actual model:
   - `model.py` — LightGCN backbone + BPR loss + RLMRec's contrastive/InfoNCE alignment loss (`con`), masked-reconstruction alignment (`gen`), or no alignment (`none`).
   - `data_loader.py` — builds the normalized user–item adjacency matrix and aligns text embeddings to the model's internal ID space.
   - `train.py` — training loop, ranking evaluation (Precision/Recall/NDCG/HitRate/MAP/MRR/Coverage @K), and early stopping on NDCG.
   - `config.py` — all hyperparameters (embedding dim, layers, alignment weight/temperature, etc.), mirroring the official RLMRec `lightgcn_plus` config where noted.
5. **`lightgcn.py`** is a dependency-free (no DGL/PyG) reimplementation of vanilla LightGCN used as the baseline.
6. **`benchmark.py`** runs LightGCN and RLMRec back-to-back and appends results to a log file for direct comparison.

## Dataset

Uses [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) (`ratings.dat`, `movies.dat`, `users.dat` in `data/raw/`). Place the raw MovieLens-1M files there before running `split.py`.

## Setup

```bash
git clone https://github.com/KiyaOfAUT/LLMAREC.git
cd LLMAREC
pip install -r requirements.txt
```

Requires Python 3.10+ (uses `dict[str, ...]` / `X | None` type hints), PyTorch 2.3, and an OpenAI-compatible LLM/embedding API endpoint.

Set your LLM credentials as environment variables rather than relying on any values checked into the scripts:

```bash
export LLM_BASE_URL="https://api.openai.com/v1"   # or any OpenAI-compatible endpoint
export LLM_API_KEY="your-api-key"
export LLM_MODEL="gpt-4o-mini"                     # chat model for profile generation
export CONCURRENCY=10                              # parallel in-flight requests
```

## Usage

Run each stage in order:

```bash
# 1. Split raw ratings into train/test
python split.py

# 2. Generate LLM profiles for users and movies (resumable)
python user_profile.py
python movie_profiles.py
python fix_empty_movie_profiles.py   # retry any profiles that came back empty

# 3. Embed the profiles
python embedding.py

# 4. Train models
python lightgcn.py        # baseline
python run_hybrid.py      # RLMRec-style hybrid model

# 5. Or benchmark both in one go
python benchmark.py --epochs 30 --log benchmark_results.log
```

## Results

Latest logged benchmark (30 epochs, MovieLens-1M, K=10):

| Metric | LightGCN | RLMRec (hybrid) |
|---|---|---|
| Precision@10 | 0.2207 | **0.2504** |
| Recall@10 | 0.1458 | **0.1769** |
| NDCG@10 | 0.2703 | **0.3094** |
| HitRate@10 | 0.7654 | **0.8228** |
| MAP@10 | 0.1606 | **0.1885** |
| MRR | 0.4869 | **0.5341** |
| Coverage@10 | 0.1802 | **0.2690** |

The RLMRec-aligned model outperforms plain LightGCN across all ranking metrics, consistent with the findings in the original RLMRec paper. (Full logs in `benchmark_results.log`.)

## Configuration

Key hyperparameters live in `hybrid_rec/config.py`:

- `alignment_mode`: `"con"` (contrastive/InfoNCE), `"gen"` (masked reconstruction), or `"none"`
- `info_weight`, `info_temperature`: alignment loss weight and temperature
- `lgcn_emb_dim`, `lgcn_layers`: LightGCN embedding size and propagation depth
- `epochs`, `batch_size`, `lr`, `weight_decay`, `early_stop_patience`
- `K`, `relevance_threshold`: ranking evaluation cutoff and relevance threshold for computing Precision/Recall/NDCG

## Reference

Ren, X. et al. *Representation Learning with Large Language Models for Recommendation.* WSDM 2024. [HKUDS/RLMRec](https://github.com/HKUDS/RLMRec)

He, X. et al. *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* SIGIR 2020.
