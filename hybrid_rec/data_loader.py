import numpy as np
import pandas as pd
import torch
from scipy import sparse

def load_ratings(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Simple cleaning if needed
    train_df.columns = [c.lower().replace("userid", "user_id").replace("movieid", "movie_id") for c in train_df.columns]
    test_df.columns = [c.lower().replace("userid", "user_id").replace("movieid", "movie_id") for c in test_df.columns]
    
    return train_df, test_df

def build_mappings(train_df, test_df):
    all_users = np.unique(np.concatenate([train_df["user_id"].values, test_df["user_id"].values]))
    all_items = np.unique(np.concatenate([train_df["movie_id"].values, test_df["movie_id"].values]))
    
    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {m: i for i, m in enumerate(all_items)}
    
    return user2idx, item2idx

def load_text_embeddings(user_npz_path, item_npz_path, user2idx, item2idx):
    # User embeddings
    user_data = np.load(user_npz_path)
    u_ids = user_data["ids"]
    u_embs = user_data["embeddings"]
    u_id_to_idx = {uid: i for i, uid in enumerate(u_ids)}
    
    n_users = len(user2idx)
    text_emb_dim = u_embs.shape[1]
    
    user_text_tensor = np.zeros((n_users, text_emb_dim), dtype=np.float32)
    for uid, idx in user2idx.items():
        if uid in u_id_to_idx:
            user_text_tensor[idx] = u_embs[u_id_to_idx[uid]]
        # else: already zero
        
    # Item embeddings
    item_data = np.load(item_npz_path)
    i_ids = item_data["ids"]
    i_embs = item_data["embeddings"]
    i_id_to_idx = {mid: i for i, mid in enumerate(i_ids)}
    
    n_items = len(item2idx)
    item_text_tensor = np.zeros((n_items, text_emb_dim), dtype=np.float32)
    for mid, idx in item2idx.items():
        if mid in i_id_to_idx:
            item_text_tensor[idx] = i_embs[i_id_to_idx[mid]]
        # else: already zero
        
    return torch.from_numpy(user_text_tensor), torch.from_numpy(item_text_tensor)

def build_adj_matrix(train_df, user2idx, item2idx, n_users, n_items):
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

    # Normalize D^{-1/2} A D^{-1/2}
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
