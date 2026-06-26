import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from .model import HybridModel
from .data_loader import load_ratings, build_mappings, load_text_embeddings, build_adj_matrix
from .config import CONFIG

class RatingDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        self.users = torch.LongTensor(df["user_id"].map(user2idx).values)
        self.items = torch.LongTensor(df["movie_id"].map(item2idx).values)
        self.ratings = torch.FloatTensor(df["rating"].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class BPRSampler:
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

def train_one_epoch(model, loader, adj, optimizer, criterion, device, loss_type="mse"):
    model.train()
    total_loss = 0
    
    if loss_type == "bpr":
        # loader is BPRSampler
        rng = np.random.default_rng()
        n_batches = max(1, len(loader.users) // CONFIG["batch_size"])
        
        for _ in tqdm(range(n_batches), desc="Training (BPR)", leave=False):
            u, i, j = loader.sample(CONFIG["batch_size"], rng)
            u_t = torch.LongTensor(u).to(device)
            i_t = torch.LongTensor(i).to(device)
            j_t = torch.LongTensor(j).to(device)
            
            optimizer.zero_grad()
            
            # Efficiently get all embeddings once
            u_l, i_l, u_t_emb, i_t_emb = model.get_all_embeddings(adj)
            
            pos_scores = model.score_pairs(u_l, i_l, u_t_emb, i_t_emb, u_t, i_t)
            neg_scores = model.score_pairs(u_l, i_l, u_t_emb, i_t_emb, u_t, j_t)
            
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
            
            # Add small L2 regularization on weights (predictor and embeddings)
            # (Similar to LightGCN but on all trainable params)
            l2_reg = 0
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg += torch.norm(param)**2
            loss += CONFIG["weight_decay"] * 0.5 * l2_reg
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / n_batches
    else:
        # loader is DataLoader with RatingDataset
        for u_idx, i_idx, ratings in tqdm(loader, desc="Training (MSE)", leave=False):
            u_idx, i_idx, ratings = u_idx.to(device), i_idx.to(device), ratings.to(device)
            optimizer.zero_grad()
            outputs = model(adj, u_idx, i_idx)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * u_idx.size(0)
        return total_loss / len(loader.dataset)

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
        f"Precision_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall_{k}":    float(np.mean(recalls)) if recalls else 0.0,
        f"F1_{k}":        float(np.mean(f1s)) if f1s else 0.0,
        f"HitRate_{k}":   float(np.mean(hits)) if hits else 0.0,
        f"NDCG_{k}":      float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"MAP_{k}":       float(np.mean(aps)) if aps else 0.0,
        "MRR":            float(np.mean(rrs)) if rrs else 0.0,
        "n_eval_users":   len(precisions),
    }

def coverage_metric(recommendations, n_total_items, k=10):
    rec_items = set()
    for rec_list in recommendations.values():
        rec_items.update(rec_list[:k])
    cov = len(rec_items) / n_total_items if n_total_items > 0 else 0.0
    return {f"Coverage_{k}": float(cov), "unique_recommended": len(rec_items)}

@torch.no_grad()
def evaluate_model(model, val_loader, adj, train_df, test_df, user2idx, item2idx, idx2item, n_items, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    all_preds = []
    all_targets = []
    for u_idx, i_idx, ratings in tqdm(val_loader, desc="Rating Eval", leave=False):
        u_idx, i_idx, ratings = u_idx.to(device), i_idx.to(device), ratings.to(device)
        outputs = model(adj, u_idx, i_idx)
        loss = criterion(outputs, ratings)
        total_loss += loss.item() * u_idx.size(0)
        all_preds.append(outputs.cpu().numpy())
        all_targets.append(ratings.cpu().numpy())
    
    mse = float(total_loss / len(val_loader.dataset))
    rmse = float(np.sqrt(mse))
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mae = float(np.mean(np.abs(all_preds - all_targets)))
    
    k = CONFIG["K"]
    rel_thr = CONFIG["relevance_threshold"]
    
    eval_users = test_df["user_id"].unique()
    if len(eval_users) > 1000:
        np.random.seed(42)
        eval_users = np.random.choice(eval_users, 1000, replace=False)
        
    print(f"Generating top-{k} recommendations for {len(eval_users)} users...")
    recs = model.recommend_topn(
        adj, eval_users, user2idx, item2idx, idx2item,
        n_items, n=k, train_df=train_df, device=device
    )
    
    rank_m = ranking_metrics(recs, test_df, k=k, relevance_threshold=rel_thr)
    cov_m  = coverage_metric(recs, n_items, k=k)
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, **rank_m, **cov_m}

def main():
    device = CONFIG["device"]
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_df, test_df = load_ratings(CONFIG["train_path"], CONFIG["test_path"])
    user2idx, item2idx = build_mappings(train_df, test_df)
    idx2item = {idx: item for item, idx in item2idx.items()}
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(train_df)}")
    
    # 2. Build Adjacency Matrix for LightGCN
    adj = build_adj_matrix(train_df, user2idx, item2idx, n_users, n_items).to(device)
    
    # 3. Load Text Embeddings
    if CONFIG["use_text_embeddings"]:
        user_text, item_text = load_text_embeddings(
            CONFIG["user_npz_path"], CONFIG["item_npz_path"], user2idx, item2idx
        )
        text_emb_dim = user_text.shape[1]
        print(f"Text embedding dimension: {text_emb_dim}")
    else:
        user_text, item_text = None, None
        text_emb_dim = 0
    
    # 4. Initialize Model
    model = HybridModel(
        n_users=n_users,
        n_items=n_items,
        lgcn_emb_dim=CONFIG["lgcn_emb_dim"],
        text_emb_dim=text_emb_dim,
        lgcn_layers=CONFIG["lgcn_layers"],
        use_text_embeddings=CONFIG["use_text_embeddings"],
        predictor_hidden_layers=CONFIG["predictor_hidden_layers"],
    ).to(device)
    
    if CONFIG["use_text_embeddings"]:
        model.user_text_emb.weight.data.copy_(user_text)
        model.item_text_emb.weight.data.copy_(item_text)
    
    # 5. Data Loaders / Samplers
    loss_type = CONFIG.get("loss_type", "mse")
    if loss_type == "bpr":
        train_loader = BPRSampler(train_df, user2idx, item2idx, n_items)
    else:
        train_dataset = RatingDataset(train_df, user2idx, item2idx)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=(device=="cuda"))
    
    test_dataset = RatingDataset(test_df, user2idx, item2idx)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=(device=="cuda"))
    
    # 6. Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.MSELoss()
    
    # 7. Training Loop
    best_rmse = float('inf')
    K = CONFIG["K"]
    REL_THR = CONFIG["relevance_threshold"]

    for epoch in range(1, CONFIG["epochs"] + 1):
        loss = train_one_epoch(model, train_loader, adj, optimizer, criterion, device, loss_type=loss_type)
        
        # Evaluate every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == 1 or epoch == CONFIG["epochs"]:
            metrics = evaluate_model(model, test_loader, adj, train_df, test_df, user2idx, item2idx, idx2item, n_items, device)
            
            print(f"\n===== Epoch {epoch} Evaluation Results =====")
            print(f"  Loss ({loss_type.upper()}): {loss:.4f}")
            print(f"  Rating prediction:")
            print(f"    RMSE            = {metrics['RMSE']:.4f}")
            print(f"    MAE             = {metrics['MAE']:.4f}")
            print(f"  Ranking (k={K}, relevance >= {REL_THR}):")
            print(f"    Precision_{K}    = {metrics[f'Precision_{K}']:.4f}")
            print(f"    Recall_{K}       = {metrics[f'Recall_{K}']:.4f}")
            print(f"    NDCG_{K}         = {metrics[f'NDCG_{K}']:.4f}")
            print(f"    HitRate_{K}      = {metrics[f'HitRate_{K}']:.4f}")
            print(f"  Coverage:")
            print(f"    Coverage_{K}     = {metrics[f'Coverage_{K}']:.4f}")
            
            if metrics['RMSE'] < best_rmse:
                best_rmse = metrics['RMSE']
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), f"models/hybrid_mlp_{loss_type}_model.pt")
        else:
            print(f"Epoch {epoch}/{CONFIG['epochs']} - Loss: {loss:.4f}")

    print(f"Training finished. Best RMSE: {best_rmse:.4f}")
    
    # Final evaluation to get full metrics
    final_metrics = evaluate_model(model, test_loader, adj, train_df, test_df, user2idx, item2idx, idx2item, n_items, device)
    return final_metrics

if __name__ == "__main__":
    main()
