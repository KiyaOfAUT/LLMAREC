import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

init = nn.init.xavier_uniform_


def edge_dropout(adj: torch.Tensor, keep_rate: float) -> torch.Tensor:
    if keep_rate >= 1.0:
        return adj
    values = adj.values()
    indices = adj.indices()
    mask = (torch.rand(values.size(0), device=values.device) + keep_rate).floor().bool()
    return torch.sparse_coo_tensor(
        indices[:, mask], values[mask], adj.shape
    ).coalesce()


def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return F.softplus(neg_preds - pos_preds).sum()


def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp: float):
    """Official RLMRec InfoNCE (HKUDS/RLMRec loss_utils)."""
    normed1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    normed_all = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    pos_term = -(normed1 * normed2 / temp).sum(-1)
    deno_term = torch.log(
        torch.sum(torch.exp(normed1 @ normed_all.T / temp), dim=-1)
    )
    return (pos_term + deno_term).sum()


def reg_all_params(model: nn.Module):
    reg = 0.0
    for p in model.parameters():
        if p.requires_grad:
            reg = reg + p.norm(2).square()
    return reg


class LightGCNBackbone(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, n_layers: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        init(self.user_emb.weight)
        init(self.item_emb.weight)

    def propagate(
        self,
        adj: torch.Tensor,
        user_e0: torch.Tensor | None = None,
        item_e0: torch.Tensor | None = None,
        sum_layers: bool = True,
    ):
        if user_e0 is None:
            user_e0 = self.user_emb.weight
        if item_e0 is None:
            item_e0 = self.item_emb.weight

        e0 = torch.cat([user_e0, item_e0], dim=0)
        layer_embs = [e0]
        x = e0
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            layer_embs.append(x)

        if sum_layers:
            out = sum(layer_embs)
        else:
            out = torch.stack(layer_embs, dim=0).mean(dim=0)

        return out[:self.n_users], out[self.n_users:]


class RLMRecModel(nn.Module):
    """
    RLMRec contrastive / generative alignment on a LightGCN backbone.
    Training follows HKUDS/RLMRec; inference uses collaborative dot products only.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        lgcn_emb_dim: int,
        text_emb_dim: int,
        lgcn_layers: int,
        use_text_embeddings: bool = True,
        alignment_mode: str = "con",
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.lgcn_emb_dim = lgcn_emb_dim
        self.text_emb_dim = text_emb_dim
        self.use_text_embeddings = use_text_embeddings
        self.alignment_mode = alignment_mode if use_text_embeddings else "none"

        self.backbone = LightGCNBackbone(n_users, n_items, lgcn_emb_dim, lgcn_layers)
        self.final_user_emb = None
        self.final_item_emb = None

        if self.use_text_embeddings:
            self.user_text_emb = nn.Embedding(n_users, text_emb_dim).requires_grad_(False)
            self.item_text_emb = nn.Embedding(n_items, text_emb_dim).requires_grad_(False)
            self.register_buffer(
                "user_text_valid",
                torch.ones(n_users, dtype=torch.bool),
                persistent=False,
            )
            self.register_buffer(
                "item_text_valid",
                torch.ones(n_items, dtype=torch.bool),
                persistent=False,
            )

            hidden = (text_emb_dim + lgcn_emb_dim) // 2

            if self.alignment_mode == "con":
                self.profile_mlp = nn.Sequential(
                    nn.Linear(text_emb_dim, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, lgcn_emb_dim),
                )
                for m in self.profile_mlp:
                    if isinstance(m, nn.Linear):
                        init(m.weight)

            elif self.alignment_mode == "gen":
                self.mask_token = nn.Parameter(torch.empty(lgcn_emb_dim))
                init(self.mask_token.unsqueeze(0))
                self.sigma_up = nn.Sequential(
                    nn.Linear(lgcn_emb_dim, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, text_emb_dim),
                )
                for m in self.sigma_up:
                    if isinstance(m, nn.Linear):
                        init(m.weight)

    def set_text_valid_masks(self, user_valid: torch.Tensor, item_valid: torch.Tensor):
        self.user_text_valid.copy_(user_valid)
        self.item_text_valid.copy_(item_valid)

    def project_all_profiles(self):
        if self.alignment_mode != "con":
            return None, None
        usr = self.profile_mlp(self.user_text_emb.weight)
        itm = self.profile_mlp(self.item_text_emb.weight)
        usr = usr[self.user_text_valid]
        itm = itm[self.item_text_valid]
        return usr, itm

    def forward(
        self,
        adj: torch.Tensor,
        keep_rate: float = 1.0,
        masked_user_idx: torch.Tensor | None = None,
        masked_item_idx: torch.Tensor | None = None,
        training: bool = False,
    ):
        if not training and self.final_user_emb is not None:
            return self.final_user_emb, self.final_item_emb

        if training and keep_rate < 1.0:
            adj = edge_dropout(adj, keep_rate)

        user_e0 = self.backbone.user_emb.weight
        item_e0 = self.backbone.item_emb.weight

        if (
            self.alignment_mode == "gen"
            and masked_user_idx is not None
            and masked_user_idx.numel() > 0
        ):
            user_e0 = user_e0.clone()
            user_e0[masked_user_idx] = self.mask_token

        if (
            self.alignment_mode == "gen"
            and masked_item_idx is not None
            and masked_item_idx.numel() > 0
        ):
            item_e0 = item_e0.clone()
            item_e0[masked_item_idx] = self.mask_token

        user_final, item_final = self.backbone.propagate(adj, user_e0, item_e0)

        if not training:
            self.final_user_emb = user_final
            self.final_item_emb = item_final

        return user_final, item_final

    def clear_inference_cache(self):
        self.final_user_emb = None
        self.final_item_emb = None

    def score_pairs(
        self,
        user_final: torch.Tensor,
        item_final: torch.Tensor,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
    ):
        return (user_final[user_indices] * item_final[item_indices]).sum(dim=1)

    def contrastive_alignment_loss(
        self,
        user_final: torch.Tensor,
        item_final: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        temperature: float,
    ):
        usr_proj_all = self.profile_mlp(self.user_text_emb.weight)
        itm_proj_all = self.profile_mlp(self.item_text_emb.weight)

        anc_cf = user_final[users]
        pos_cf = item_final[pos_items]
        neg_cf = item_final[neg_items]

        anc_prf = usr_proj_all[users]
        pos_prf = itm_proj_all[pos_items]
        neg_prf = itm_proj_all[neg_items]

        usr_pool = usr_proj_all[self.user_text_valid]
        pos_pool = pos_prf

        kd = cal_infonce_loss(anc_cf, anc_prf, usr_pool, temperature)
        kd = kd + cal_infonce_loss(pos_cf, pos_prf, pos_pool, temperature)
        kd = kd + cal_infonce_loss(neg_cf, neg_prf, pos_pool, temperature)

        return kd / users.size(0)

    def generative_alignment_loss(
        self,
        user_final: torch.Tensor,
        item_final: torch.Tensor,
        masked_user_idx: torch.Tensor,
        masked_item_idx: torch.Tensor,
        temperature: float,
    ):
        losses = []
        count = 0

        if masked_user_idx.numel() > 0:
            u_e = user_final[masked_user_idx]
            u_s = self.user_text_emb(masked_user_idx)
            u_proj = self.sigma_up(u_e)
            losses.append(cal_infonce_loss(u_proj, u_s, u_s, temperature))
            count += 1

        if masked_item_idx.numel() > 0:
            i_e = item_final[masked_item_idx]
            i_s = self.item_text_emb(masked_item_idx)
            i_proj = self.sigma_up(i_e)
            losses.append(cal_infonce_loss(i_proj, i_s, i_s, temperature))
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=user_final.device)

        return sum(losses) / count

    def cal_training_loss(
        self,
        adj: torch.Tensor,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        keep_rate: float,
        info_weight: float,
        temperature: float,
        reg_weight: float,
        mask_ratio: float,
        rng: np.random.Generator,
    ):
        masked_users_t = None
        masked_items_t = None

        if self.alignment_mode == "gen":
            unique_users = torch.unique(users).cpu().numpy()
            unique_items = torch.unique(
                torch.cat([pos_items, neg_items]).cpu()
            ).numpy()
            masked_users = _sample_mask_np(unique_users, mask_ratio, rng)
            masked_items = _sample_mask_np(unique_items, mask_ratio, rng)
            if len(masked_users):
                masked_users_t = torch.LongTensor(masked_users).to(users.device)
            if len(masked_items):
                masked_items_t = torch.LongTensor(masked_items).to(users.device)

        user_final, item_final = self.forward(
            adj,
            keep_rate=keep_rate,
            masked_user_idx=masked_users_t,
            masked_item_idx=masked_items_t,
            training=True,
        )

        bpr = cal_bpr_loss(
            user_final[users],
            item_final[pos_items],
            item_final[neg_items],
        ) / users.size(0)

        reg = reg_weight * reg_all_params(self)

        kd = torch.tensor(0.0, device=users.device)
        if self.alignment_mode == "con":
            kd = self.contrastive_alignment_loss(
                user_final, item_final, users, pos_items, neg_items, temperature
            )
        elif self.alignment_mode == "gen" and (
            masked_users_t is not None or masked_items_t is not None
        ):
            kd = self.generative_alignment_loss(
                user_final,
                item_final,
                masked_users_t if masked_users_t is not None else torch.tensor([], device=users.device, dtype=torch.long),
                masked_items_t if masked_items_t is not None else torch.tensor([], device=users.device, dtype=torch.long),
                temperature,
            )

        total = bpr + reg + info_weight * kd
        return total, {"bpr": bpr.item(), "reg": reg.item(), "kd": kd.item()}

    @torch.no_grad()
    def recommend_topn(
        self,
        adj,
        user_ids,
        user2idx,
        item2idx,
        idx2item,
        n_items,
        n=10,
        train_df=None,
        device="cpu",
    ):
        self.eval()
        self.clear_inference_cache()
        user_final, item_final = self.forward(adj, training=False)

        train_mask = {}
        if train_df is not None:
            for uid, mid in zip(train_df["user_id"].values, train_df["movie_id"].values):
                uidx = user2idx.get(uid, -1)
                midx = item2idx.get(mid, -1)
                if uidx >= 0 and midx >= 0:
                    train_mask.setdefault(uidx, set()).add(midx)

        item_emb_all = item_final.cpu().numpy()
        results = {}

        from tqdm import tqdm
        for uid in tqdm(user_ids, desc="Generating top-N recs", unit="user"):
            uidx = user2idx.get(uid, -1)
            if uidx < 0:
                results[uid] = []
                continue

            u_vec = user_final[uidx].cpu().numpy()
            scores = item_emb_all @ u_vec

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


def _sample_mask_np(indices: np.ndarray, mask_ratio: float, rng: np.random.Generator):
    if len(indices) == 0 or mask_ratio <= 0:
        return np.array([], dtype=np.int64)
    n_mask = max(1, int(len(indices) * mask_ratio))
    if n_mask >= len(indices):
        return indices.copy()
    return rng.choice(indices, size=n_mask, replace=False)


HybridModel = RLMRecModel
