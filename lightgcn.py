# lightgcn.py
"""
LightGCN  –  Light Graph Convolution Network for Collaborative Filtering
He et al., SIGIR 2020

Pure PyTorch implementation.  No DGL / PyG dependency.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from tqdm import trange, tqdm


# ──────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────

def load_data(
    train_path: str = "data/processed/train_ratings.csv",
    test_path:  str = "data/processed/test_ratings.csv",
):
    def _clean(df):
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9]", "_", regex=True)
        )
        rename = {}
        if "userid" in df.columns:
            rename["userid"] = "user_id"
        if "movieid" in df.columns:
            rename["movieid"] = "movie_id"
        if rename:
            df.rename(columns=rename, inplace=True)
        if "timestamp" in df.columns:
            df.drop(columns=["timestamp"], inplace=True)
        return df[["user_id", "movie_id", "rating"]]

    return _clean(pd.read_csv(train_path)), _clean(pd.read_csv(test_path))


def build_mappings(train_df, test_df):
    """Create contiguous id maps that cover both train and test."""
    all_users = np.union1d(train_df["user_id"].unique(), test_df["user_id"].unique())
    all_items = np.union1d(train_df["movie_id"].unique(), test_df["movie_id"].unique())
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {m: i for i, m in enumerate(all_items)}
    idx2item = {i: m for m, i in item2idx.items()}
    return user2idx, item2idx, idx2item


def build_adj_matrix(train_df, user2idx, item2idx, n_users, n_items):
    """
    Build the normalised adjacency matrix of the user-item bipartite graph.
    Returns a coalesced sparse torch tensor.
    """
    rows = train_df["user_id"].map(user2idx).values
    cols = train_df["movie_id"].map(item2idx).values

    ur_row = rows
    ur_col = cols + n_users

    ll_row = cols + n_users
    ll_col = rows

    adj_row = np.concatenate([ur_row, ll_row])
    adj_col = np.concatenate([ur_col, ll_col])
    adj_data = np.ones(len(adj_row), dtype=np.float32)

    N = n_users + n_items
    A = sparse.coo_matrix((adj_data, (adj_row, adj_col)), shape=(N, N))

    # D^{-1/2}
    deg = np.array(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.zeros_like(deg)
    nonzero = deg > 0
    deg_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
    D_inv_sqrt = sparse.diags(deg_inv_sqrt)

    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    A_norm = A_norm.tocoo()

    indices = torch.LongTensor(np.vstack([A_norm.row, A_norm.col]))
    values  = torch.FloatTensor(A_norm.data)
    A_torch = torch.sparse_coo_tensor(indices, values, torch.Size([N, N])).coalesce()

    return A_torch

# ──────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────

class LightGCN(nn.Module):
    """
    LightGCN model.

    Parameters
    ----------
    n_users, n_items : int
    emb_dim : int           – latent dimension $d$
    n_layers : int          – number of graph convolution layers $L$
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users  = n_users
        self.n_items  = n_items
        self.emb_dim  = emb_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, adj: torch.Tensor):
        """
        Run L layers of graph convolution and return final user / item embeddings.

        $$\\mathbf{e}_u = \\frac{1}{L+1}\\sum_{l=0}^{L} \\mathbf{e}_u^{(l)}$$
        """
        e0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # (N+M, d)
        layer_embs = [e0]

        x = e0
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            layer_embs.append(x)

        # Mean pooling across layers
        out = torch.stack(layer_embs, dim=0).mean(dim=0)  # (N+M, d)
        user_final = out[: self.n_users]
        item_final = out[self.n_users :]
        return user_final, item_final

    def bpr_loss(self, user_final, item_final, users, pos_items, neg_items, reg_weight):
        """
        BPR pairwise loss + L2 regularisation on the *initial* embeddings only
        (as in the original paper).

        $$\\mathcal{L} = -\\sum \\ln\\sigma(\\hat{y}_{ui} - \\hat{y}_{uj})
                         + \\lambda(\\|\\mathbf{e}_u^{(0)}\\|^2
                                   + \\|\\mathbf{e}_i^{(0)}\\|^2
                                   + \\|\\mathbf{e}_j^{(0)}\\|^2)$$
        """
        u_emb   = user_final[users]
        pos_emb = item_final[pos_items]
        neg_emb = item_final[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 on initial embeddings
        u0   = self.user_emb(users)
        pos0 = self.item_emb(pos_items)
        neg0 = self.item_emb(neg_items)
        reg  = (u0.norm(2).pow(2) + pos0.norm(2).pow(2) + neg0.norm(2).pow(2)) / len(users)

        return bpr + reg_weight * reg


# ──────────────────────────────────────────────────────────────────────
# BPR sampler
# ──────────────────────────────────────────────────────────────────────

class BPRSampler:
    """Uniform negative sampling for BPR training."""

    def __init__(self, train_df, user2idx, item2idx, n_items):
        self.n_items = n_items
        self.user_pos = {}
        for uid, mid in zip(
            train_df["user_id"].map(user2idx).values,
            train_df["movie_id"].map(item2idx).values,
        ):
            self.user_pos.setdefault(uid, set()).add(mid)

        self.users = []
        self.pos_items = []
        for u, items in self.user_pos.items():
            for i in items:
                self.users.append(u)
                self.pos_items.append(i)
        self.users = np.array(self.users, dtype=np.int64)
        self.pos_items = np.array(self.pos_items, dtype=np.int64)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.users), size=batch_size)
        users = self.users[idx]
        pos   = self.pos_items[idx]
        neg   = np.empty(batch_size, dtype=np.int64)
        for k in range(batch_size):
            while True:
                j = rng.integers(0, self.n_items)
                if j not in self.user_pos[users[k]]:
                    neg[k] = j
                    break
        return users, pos, neg


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────

def train_lightgcn(
    model: LightGCN,
    adj: torch.Tensor,
    sampler: BPRSampler,
    *,
    n_epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 1e-3,
    reg: float = 1e-4,
    device: str = "cpu",
    val_df: pd.DataFrame = None,
    user2idx: dict = None,
    item2idx: dict = None,
):
    model = model.to(device)
    adj   = adj.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(42)

    n_batches = max(1, len(sampler.users) // batch_size)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for _ in range(n_batches):
            user_final, item_final = model(adj)

            users, pos, neg = sampler.sample(batch_size, rng)
            users_t = torch.LongTensor(users).to(device)
            pos_t   = torch.LongTensor(pos).to(device)
            neg_t   = torch.LongTensor(neg).to(device)

            loss = model.bpr_loss(
                user_final, item_final,
                users_t, pos_t, neg_t,
                reg_weight=reg,
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        msg = f"  Epoch {epoch:3d}/{n_epochs}  BPR loss={avg_loss:.4f}"

        if val_df is not None and user2idx is not None and item2idx is not None:
            if epoch % 5 == 0 or epoch == 1:
                val_metrics = _quick_val(model, adj, val_df, user2idx, item2idx, device)
                msg += f"  val RMSE={val_metrics['RMSE']:.4f}"
                print(msg)
        elif epoch % 5 == 0 or epoch == 1:
            print(msg)

    print("Training complete.")


@torch.no_grad()
def _quick_val(model, adj, val_df, user2idx, item2idx, device):
    model.eval()
    user_final, item_final = model(adj)

    u_idx = np.array([user2idx.get(u, -1) for u in val_df["user_id"].values])
    m_idx = np.array([item2idx.get(m, -1) for m in val_df["movie_id"].values])
    mask  = (u_idx >= 0) & (m_idx >= 0)

    preds = np.zeros(len(val_df), dtype=np.float64)
    if mask.any():
        ut = torch.LongTensor(u_idx[mask]).to(device)
        mt = torch.LongTensor(m_idx[mask]).to(device)
        scores = (user_final[ut] * item_final[mt]).sum(dim=1).cpu().numpy()
        preds[mask] = scores

    true_r = val_df["rating"].values.astype(np.float64)
    rmse = float(np.sqrt(np.mean((preds - true_r) ** 2)))
    return {"RMSE": rmse}


# ──────────────────────────────────────────────────────────────────────
# Prediction & recommendation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, adj, user_ids, movie_ids, user2idx, item2idx, device="cpu"):
    model.eval()
    user_final, item_final = model(adj)

    u_idx = np.array([user2idx.get(u, -1) for u in user_ids])
    m_idx = np.array([item2idx.get(m, -1) for m in movie_ids])

    preds = np.zeros(len(user_ids), dtype=np.float64)
    mask = (u_idx >= 0) & (m_idx >= 0)
    if mask.any():
        ut = torch.LongTensor(u_idx[mask]).to(device)
        mt = torch.LongTensor(m_idx[mask]).to(device)
        preds[mask] = (user_final[ut] * item_final[mt]).sum(dim=1).cpu().numpy()

    return preds


@torch.no_grad()
def recommend_topn(
    model, adj, user_ids, user2idx, item2idx, idx2item,
    n_items, n=10, train_df=None, device="cpu",
):
    model.eval()
    user_final, item_final = model(adj)

    # Build train mask
    train_mask = {}
    if train_df is not None:
        for uid, mid in zip(train_df["user_id"].values, train_df["movie_id"].values):
            uidx = user2idx.get(uid, -1)
            midx = item2idx.get(mid, -1)
            if uidx >= 0 and midx >= 0:
                train_mask.setdefault(uidx, set()).add(midx)

    item_emb_all = item_final.cpu().numpy()  # (n_items, d)
    results = {}

    for uid in tqdm(user_ids, desc="Generating top-N recs", unit="user"):
        uidx = user2idx.get(uid, -1)
        if uidx < 0:
            results[uid] = []
            continue

        u_vec = user_final[uidx].cpu().numpy()  # (d,)
        scores = item_emb_all @ u_vec            # (n_items,)

        rated = train_mask.get(uidx, set())
        if rated:
            scores[list(rated)] = -np.inf

        if n < len(scores):
            top_idx = np.argpartition(scores, -n)[-n:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
        else:
            top_idx = np.argsort(-scores)[:n]

        results[uid] = [idx2item[i] for i in top_idx if scores[i] > -np.inf]

    return results


# ──────────────────────────────────────────────────────────────────────
# Evaluation  (same metrics as cf.py / mf.py)
# ──────────────────────────────────────────────────────────────────────

def rating_metrics(true_ratings, pred_ratings):
    errors = pred_ratings - true_ratings
    return {
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE":  float(np.mean(np.abs(errors))),
    }


def ranking_metrics(recommendations, test_df, k=10, relevance_threshold=3.5):
    relevant_items = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    all_test_items = (
        test_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    )

    precisions, recalls, f1s = [], [], []
    ndcgs, aps, rrs, hits = [], [], [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))

    for uid, rec_list in recommendations.items():
        rel = relevant_items.get(uid, set())
        if uid not in all_test_items:
            continue
        rec_k = rec_list[:k]
        n_rec = len(rec_k)
        if n_rec == 0:
            precisions.append(0.0); recalls.append(0.0); f1s.append(0.0)
            ndcgs.append(0.0); aps.append(0.0); rrs.append(0.0); hits.append(0.0)
            continue

        rel_vec = np.array([1.0 if item in rel else 0.0 for item in rec_k])
        n_hit = rel_vec.sum()

        prec = n_hit / k
        rec_ = n_hit / len(rel) if len(rel) > 0 else 0.0
        f1   = (2 * prec * rec_ / (prec + rec_)) if (prec + rec_) > 0 else 0.0
        precisions.append(prec); recalls.append(rec_); f1s.append(f1)
        hits.append(1.0 if n_hit > 0 else 0.0)

        dcg  = np.sum(rel_vec[:k] * discounts[:n_rec])
        ideal_len = min(int(len(rel)), k)
        idcg = np.sum(discounts[:ideal_len]) if ideal_len > 0 else 0.0
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        cum_hits = np.cumsum(rel_vec)
        precision_at_i = cum_hits / np.arange(1, n_rec + 1, dtype=np.float64)
        ap = np.sum(precision_at_i * rel_vec) / min(len(rel), k) if len(rel) > 0 else 0.0
        aps.append(ap)

        rel_positions = np.where(rel_vec == 1.0)[0]
        rr = 1.0 / (rel_positions[0] + 1) if len(rel_positions) > 0 else 0.0
        rrs.append(rr)

    return {
        f"Precision_{k}": float(np.mean(precisions)),
        f"Recall_{k}":    float(np.mean(recalls)),
        f"F1_{k}":        float(np.mean(f1s)),
        f"HitRate_{k}":   float(np.mean(hits)),
        f"NDCG_{k}":      float(np.mean(ndcgs)),
        f"MAP_{k}":       float(np.mean(aps)),
        "MRR":            float(np.mean(rrs)),
        "n_eval_users":   len(precisions),
    }


def coverage_metric(recommendations, n_total_items, k=10):
    rec_items = set()
    for rec_list in recommendations.values():
        rec_items.update(rec_list[:k])
    cov = len(rec_items) / n_total_items if n_total_items > 0 else 0.0
    return {f"Coverage_{k}": float(cov), "unique_recommended": len(rec_items)}


def evaluate(model, adj, train_df, test_df, user2idx, item2idx, idx2item,
             n_items, k=10, relevance_threshold=3.5, device="cpu"):
    # Rating metrics
    print("Generating rating predictions...")
    pred_r = predict(
        model, adj,
        test_df["user_id"].values, test_df["movie_id"].values,
        user2idx, item2idx, device,
    )
    true_r = test_df["rating"].values.astype(np.float64)
    r_metrics = rating_metrics(true_r, pred_r)

    # Ranking metrics
    eval_users = test_df["user_id"].unique()
    print(f"Generating top-{k} recommendations for {len(eval_users)} users...")
    recs = recommend_topn(
        model, adj, eval_users, user2idx, item2idx, idx2item,
        n_items, n=k, train_df=train_df, device=device,
    )
    rank_m = ranking_metrics(recs, test_df, k=k, relevance_threshold=relevance_threshold)
    cov_m  = coverage_metric(recs, n_items, k=k)

    return {**r_metrics, **rank_m, **cov_m}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def run_experiment():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    # Hyperparameters
    EMB_DIM   = 64
    N_LAYERS  = 3
    N_EPOCHS  = 30
    BATCH     = 2048
    LR        = 1e-3
    REG       = 1e-4
    K         = 10
    REL_THR   = 3.5

    train_df, test_df = load_data()
    user2idx, item2idx, idx2item = build_mappings(train_df, test_df)
    n_users = len(user2idx)
    n_items = len(item2idx)

    print(f"Users: {n_users}  Items: {n_items}  Train: {len(train_df)}  Test: {len(test_df)}")

    adj = build_adj_matrix(train_df, user2idx, item2idx, n_users, n_items)

    model = LightGCN(n_users, n_items, emb_dim=EMB_DIM, n_layers=N_LAYERS)

    train_lightgcn(
        model, adj, BPRSampler(train_df, user2idx, item2idx, n_items),
        n_epochs=N_EPOCHS, batch_size=BATCH, lr=LR, reg=REG,
        device=DEVICE, val_df=test_df, user2idx=user2idx, item2idx=item2idx,
    )

    metrics = evaluate(
        model, adj.to(DEVICE), train_df, test_df,
        user2idx, item2idx, idx2item, n_items,
        k=K, relevance_threshold=REL_THR, device=DEVICE,
    )

    print(f"\n===== LightGCN Evaluation Results =====")
    print(f"  Rating prediction:")
    print(f"    RMSE            = {metrics['RMSE']:.4f}")
    print(f"    MAE             = {metrics['MAE']:.4f}")
    print(f"  Ranking (k={K}, relevance >= {REL_THR}):")
    print(f"    Precision_{K}    = {metrics[f'Precision_{K}']:.4f}")
    print(f"    Recall_{K}       = {metrics[f'Recall_{K}']:.4f}")
    print(f"    F1_{K}           = {metrics[f'F1_{K}']:.4f}")
    print(f"    HitRate_{K}      = {metrics[f'HitRate_{K}']:.4f}")
    print(f"    NDCG_{K}         = {metrics[f'NDCG_{K}']:.4f}")
    print(f"    MAP_{K}          = {metrics[f'MAP_{K}']:.4f}")
    print(f"    MRR             = {metrics['MRR']:.4f}")
    print(f"  Coverage:")
    print(f"    Coverage_{K}     = {metrics[f'Coverage_{K}']:.4f}")
    print(f"    Unique items    = {metrics['unique_recommended']}")
    print(f"  Evaluated users   = {metrics['n_eval_users']}")
    
    return metrics

if __name__ == "__main__":
    run_experiment()
