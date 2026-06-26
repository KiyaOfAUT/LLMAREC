import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, layers_hidden, dropout=0.1):
        super(MLP, self).__init__()
        layers = []
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # Remove last ReLU and Dropout if we want a single output
        layers = layers[:-2]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class LightGCNModule(nn.Module):
    def __init__(self, n_users, n_items, emb_dim, n_layers):
        super(LightGCNModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        
        # Learnable weight for text injection (starts small to not drown IDs)
        self.text_weight = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, adj, u_text_proj=None, i_text_proj=None):
        u_e0 = self.user_emb.weight
        i_e0 = self.item_emb.weight
        
        if u_text_proj is not None and i_text_proj is not None:
            # Use learnable scaling for text injection
            u_e0 = u_e0 + self.text_weight * u_text_proj
            i_e0 = i_e0 + self.text_weight * i_text_proj
            
        e0 = torch.cat([u_e0, i_e0], dim=0)
        layer_embs = [e0]
        x = e0
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            layer_embs.append(x)
        out = torch.stack(layer_embs, dim=0).mean(dim=0)
        return out[:self.n_users], out[self.n_users:]

class HybridModel(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        lgcn_emb_dim,
        text_emb_dim,
        lgcn_layers,
        use_text_embeddings=True,
        predictor_hidden_layers=[128, 64],
    ):
        super(HybridModel, self).__init__()
        self.use_text_embeddings = use_text_embeddings
        
        self.lightgcn = LightGCNModule(n_users, n_items, lgcn_emb_dim, lgcn_layers)
        
        if self.use_text_embeddings:
            # Text embeddings
            self.user_text_emb = nn.Embedding(n_users, text_emb_dim).requires_grad_(False)
            self.item_text_emb = nn.Embedding(n_items, text_emb_dim).requires_grad_(False)
            
            self.user_text_proj = nn.Sequential(
                nn.Linear(text_emb_dim, lgcn_emb_dim),
                nn.ReLU(),
                nn.LayerNorm(lgcn_emb_dim),
                nn.Dropout(0.1) # Add dropout for robustness
            )
            self.item_text_proj = nn.Sequential(
                nn.Linear(text_emb_dim, lgcn_emb_dim),
                nn.ReLU(),
                nn.LayerNorm(lgcn_emb_dim),
                nn.Dropout(0.1)
            )
            
            # input_dim is now 2 * lgcn_emb_dim [u_id*i_id, u_text*i_text]
            input_dim = 2 * lgcn_emb_dim
            self.norm_inter = nn.LayerNorm(lgcn_emb_dim)
            self.norm_text = nn.LayerNorm(lgcn_emb_dim)
        else:
            input_dim = lgcn_emb_dim
            self.norm_inter = nn.LayerNorm(lgcn_emb_dim)
        
        self.predictor = MLP([input_dim] + predictor_hidden_layers + [1])

    def get_all_embeddings(self, adj):
        if self.use_text_embeddings:
            u_text_proj = self.user_text_proj(self.user_text_emb.weight)
            i_text_proj = self.item_text_proj(self.item_text_emb.weight)
            u_lgcn_all, i_lgcn_all = self.lightgcn(adj, u_text_proj, i_text_proj)
            return u_lgcn_all, i_lgcn_all, u_text_proj, i_text_proj
            
        u_lgcn_all, i_lgcn_all = self.lightgcn(adj)
        return u_lgcn_all, i_lgcn_all, None, None

    def forward(self, adj, user_indices, item_indices):
        u_l, i_l, u_t, i_t = self.get_all_embeddings(adj)
        
        u_batch = u_l[user_indices]
        i_batch = i_l[item_indices]
        
        # Base LightGCN dot product
        dot_score = (u_batch * i_batch).sum(dim=1, keepdim=True)
        
        # ID interaction vector
        interaction = self.norm_inter(u_batch * i_batch)
        
        if self.use_text_embeddings:
            # TEXT interaction vector (high bandwidth)
            text_interaction = self.norm_text(u_t[user_indices] * i_t[item_indices])
            combined = torch.cat([interaction, text_interaction], dim=1)
        else:
            combined = interaction
            
        # Final score = ID Dot Product + MLP learned hybrid score
        residual = self.predictor(combined)
        return (dot_score + residual).squeeze(-1)

    def score_pairs(self, u_l, i_l, u_t, i_t, user_indices, item_indices):
        u_batch = u_l[user_indices]
        i_batch = i_l[item_indices]
        
        dot_score = (u_batch * i_batch).sum(dim=1, keepdim=True)
        interaction = self.norm_inter(u_batch * i_batch)
        
        if self.use_text_embeddings:
            text_interaction = self.norm_text(u_t[user_indices] * i_t[item_indices])
            combined = torch.cat([interaction, text_interaction], dim=1)
        else:
            combined = interaction
            
        residual = self.predictor(combined)
        return (dot_score + residual).squeeze(-1)

    @torch.no_grad()
    def recommend_topn(
        self, adj, user_ids, user2idx, item2idx, idx2item,
        n_items, n=10, train_df=None, device="cpu", batch_size=512
    ):
        self.eval()
        u_l_all, i_l_all, u_t_all, i_t_all = self.get_all_embeddings(adj)
        
        train_mask = {}
        if train_df is not None:
            for uid, mid in zip(train_df["user_id"].values, train_df["movie_id"].values):
                uidx = user2idx.get(uid, -1)
                midx = item2idx.get(mid, -1)
                if uidx >= 0 and midx >= 0:
                    train_mask.setdefault(uidx, set()).add(midx)

        results = {}
        item_rep_all = i_l_all

        from tqdm import tqdm
        for uid in tqdm(user_ids, desc="Generating top-N recs", unit="user"):
            uidx = user2idx.get(uid, -1)
            if uidx < 0:
                results[uid] = []
                continue

            u_batch = u_l_all[uidx].unsqueeze(0)
            if self.use_text_embeddings:
                u_text_batch = u_t_all[uidx].unsqueeze(0)
            
            scores = []
            for i in range(0, n_items, batch_size):
                end_i = min(i + batch_size, n_items)
                curr_items = item_rep_all[i:end_i]
                u_rep_repeated = u_batch.expand(curr_items.size(0), -1)
                
                # Base signal
                dot_score = (u_rep_repeated * curr_items).sum(dim=1, keepdim=True)
                interaction = self.norm_inter(u_rep_repeated * curr_items)
                
                if self.use_text_embeddings:
                    curr_i_text = i_t_all[i:end_i]
                    u_text_repeated = u_text_batch.expand(curr_i_text.size(0), -1)
                    text_interaction = self.norm_text(u_text_repeated * curr_i_text)
                    inp = torch.cat([interaction, text_interaction], dim=1)
                else:
                    inp = interaction
                
                residual = self.predictor(inp)
                s = (dot_score + residual).squeeze(-1)
                scores.append(s)
            
            scores = torch.cat(scores).cpu().numpy()
            
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
