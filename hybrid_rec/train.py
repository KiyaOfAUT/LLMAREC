import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from .model import RLMRecModel
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
        pos = self.pos_items[idx]
        neg = np.empty(batch_size, dtype=np.int64)
        for k in range(batch_size):
            while True:
                j = rng.integers(0, self.n_items)
                if j not in self.user_pos[users[k]]:
                    neg[k] = j
                    break
        return users, pos, neg


def train_one_epoch(model, sampler, adj, optimizer, device, config, rng):
    model.train()
    model.clear_inference_cache()
    n_batches = max(1, len(sampler.users) // config["batch_size"])
    total_loss = 0.0
    bpr_sum, kd_sum = 0.0, 0.0

    for _ in tqdm(range(n_batches), desc="Training (BPR+RLMRec)", leave=False):
        u, i, j = sampler.sample(config["batch_size"], rng)
        users_t = torch.LongTensor(u).to(device)
        pos_t = torch.LongTensor(i).to(device)
        neg_t = torch.LongTensor(j).to(device)

        optimizer.zero_grad()
        loss, parts = model.cal_training_loss(
            adj,
            users_t,
            pos_t,
            neg_t,
            keep_rate=config.get("keep_rate", 1.0),
            info_weight=config.get("info_weight", 1e-2),
            temperature=config.get("info_temperature", 0.2),
            reg_weight=config["weight_decay"],
            mask_ratio=config.get("mask_ratio", 0.1),
            rng=rng,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bpr_sum += parts["bpr"]
        kd_sum += parts["kd"]

    return {
        "total": total_loss / n_batches,
        "bpr": bpr_sum / n_batches,
        "kd": kd_sum / n_batches,
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
        f1 = (2 * prec * rec_) / (prec + rec_) if (prec + rec_) > 0 else 0.0
        precisions.append(prec); recalls.append(rec_); f1s.append(f1)
        hits.append(1.0 if n_hit > 0 else 0.0)

        dcg = np.sum(rel_vec[:k] * discounts[:n_rec])
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
        f"Recall_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"F1_{k}": float(np.mean(f1s)) if f1s else 0.0,
        f"HitRate_{k}": float(np.mean(hits)) if hits else 0.0,
        f"NDCG_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"MAP_{k}": float(np.mean(aps)) if aps else 0.0,
        "MRR": float(np.mean(rrs)) if rrs else 0.0,
        "n_eval_users": len(precisions),
    }


def coverage_metric(recommendations, n_total_items, k=10):
    rec_items = set()
    for rec_list in recommendations.values():
        rec_items.update(rec_list[:k])
    cov = len(rec_items) / n_total_items if n_total_items > 0 else 0.0
    return {f"Coverage_{k}": float(cov), "unique_recommended": len(rec_items)}


@torch.no_grad()
def evaluate_model(model, val_loader, adj, train_df, test_df, user2idx, item2idx, idx2item, n_items, device, config):
    model.eval()
    model.clear_inference_cache()
    user_final, item_final = model.forward(adj, training=False)

    all_preds = []
    all_targets = []
    for u_idx, i_idx, ratings in tqdm(val_loader, desc="Rating Eval", leave=False):
        u_idx = u_idx.to(device)
        i_idx = i_idx.to(device)
        scores = model.score_pairs(user_final, item_final, u_idx, i_idx)
        all_preds.append(scores.cpu().numpy())
        all_targets.append(ratings.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    errors = all_preds - all_targets
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    k = config["K"]
    rel_thr = config["relevance_threshold"]

    eval_users = test_df["user_id"].unique()
    max_users = config.get("max_eval_users")
    if max_users is not None and len(eval_users) > max_users:
        np.random.seed(42)
        eval_users = np.random.choice(eval_users, max_users, replace=False)

    print(f"Generating top-{k} recommendations for {len(eval_users)} users...")
    recs = model.recommend_topn(
        adj, eval_users, user2idx, item2idx, idx2item,
        n_items, n=k, train_df=train_df, device=device,
    )

    rank_m = ranking_metrics(recs, test_df, k=k, relevance_threshold=rel_thr)
    cov_m = coverage_metric(recs, n_items, k=k)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, **rank_m, **cov_m}


def run_experiment(n_epochs: int | None = None):
    config = dict(CONFIG)
    if n_epochs is not None:
        config["epochs"] = n_epochs
        if n_epochs < config.get("early_stop_patience", 10):
            config["early_stop_patience"] = n_epochs + 1

    device = config["device"]
    alignment_mode = config.get("alignment_mode", "con")
    print(f"Using device: {device}")
    print(f"RLMRec alignment mode: {alignment_mode}")
    print(f"Training RLMRec for {config['epochs']} epochs")

    train_df, test_df = load_ratings(config["train_path"], config["test_path"])
    user2idx, item2idx = build_mappings(train_df, test_df)
    idx2item = {idx: item for item, idx in item2idx.items()}
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"Users: {n_users}, Items: {n_items}, Interactions: {len(train_df)}")

    adj = build_adj_matrix(train_df, user2idx, item2idx, n_users, n_items).to(device)

    if config["use_text_embeddings"]:
        user_text, item_text = load_text_embeddings(
            config["user_npz_path"], config["item_npz_path"], user2idx, item2idx
        )
        text_emb_dim = user_text.shape[1]
        print(f"Text embedding dimension: {text_emb_dim}")
    else:
        user_text, item_text = None, None
        text_emb_dim = 0

    model = RLMRecModel(
        n_users=n_users,
        n_items=n_items,
        lgcn_emb_dim=config["lgcn_emb_dim"],
        text_emb_dim=text_emb_dim,
        lgcn_layers=config["lgcn_layers"],
        use_text_embeddings=config["use_text_embeddings"],
        alignment_mode=alignment_mode,
    ).to(device)

    if config["use_text_embeddings"]:
        model.user_text_emb.weight.data.copy_(user_text)
        model.item_text_emb.weight.data.copy_(item_text)
        user_valid = (user_text.abs().sum(dim=1) > 0)
        item_valid = (item_text.abs().sum(dim=1) > 0)
        model.set_text_valid_masks(user_valid.to(device), item_valid.to(device))
        print(
            f"Valid text profiles: users={user_valid.sum().item()}/{n_users}, "
            f"items={item_valid.sum().item()}/{n_items}"
        )

    train_sampler = BPRSampler(train_df, user2idx, item2idx, n_items)

    test_dataset = RatingDataset(test_df, user2idx, item2idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=0.0)

    K = config["K"]
    REL_THR = config["relevance_threshold"]
    patience = config.get("early_stop_patience", 10)
    eval_every = config.get("eval_every", 3)

    best_ndcg = -1.0
    best_epoch = 0
    stale_epochs = 0
    rng = np.random.default_rng(config.get("seed", 2023))

    for epoch in range(1, config["epochs"] + 1):
        losses = train_one_epoch(model, train_sampler, adj, optimizer, device, config, rng)

        if epoch % eval_every == 0 or epoch == 1 or epoch == config["epochs"]:
            metrics = evaluate_model(
                model, test_loader, adj, train_df, test_df,
                user2idx, item2idx, idx2item, n_items, device, config,
            )

            print(f"\n===== Epoch {epoch} Evaluation Results =====")
            print(
                f"  Loss: total={losses['total']:.4f}  "
                f"bpr={losses['bpr']:.4f}  kd={losses['kd']:.4f}"
            )
            print(f"  Rating prediction (dot-product scores, uncalibrated):")
            print(f"    RMSE            = {metrics['RMSE']:.4f}")
            print(f"    MAE             = {metrics['MAE']:.4f}")
            print(f"  Ranking (k={K}, relevance >= {REL_THR}):")
            print(f"    Precision_{K}    = {metrics[f'Precision_{K}']:.4f}")
            print(f"    Recall_{K}       = {metrics[f'Recall_{K}']:.4f}")
            print(f"    NDCG_{K}         = {metrics[f'NDCG_{K}']:.4f}")
            print(f"    HitRate_{K}      = {metrics[f'HitRate_{K}']:.4f}")
            print(f"  Coverage:")
            print(f"    Coverage_{K}     = {metrics[f'Coverage_{K}']:.4f}")

            ndcg = metrics[f"NDCG_{K}"]
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_epoch = epoch
                stale_epochs = 0
                os.makedirs("models", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    f"models/rlmrec_{alignment_mode}_model.pt",
                )
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(
                        f"\nEarly stopping at epoch {epoch} "
                        f"(best NDCG@{K}={best_ndcg:.4f} at epoch {best_epoch})"
                    )
                    break
        else:
            print(
                f"Epoch {epoch}/{config['epochs']} - "
                f"loss={losses['total']:.4f} (bpr={losses['bpr']:.4f}, kd={losses['kd']:.4f})"
            )

    print(f"Training finished. Best NDCG@{K}: {best_ndcg:.4f} (epoch {best_epoch})")

    if os.path.exists(f"models/rlmrec_{alignment_mode}_model.pt"):
        model.load_state_dict(
            torch.load(f"models/rlmrec_{alignment_mode}_model.pt", map_location=device)
        )

    final_metrics = evaluate_model(
        model, test_loader, adj, train_df, test_df,
        user2idx, item2idx, idx2item, n_items, device, config,
    )
    return final_metrics


def main():
    return run_experiment()


if __name__ == "__main__":
    main()
