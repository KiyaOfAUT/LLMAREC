# cf.py

import pandas as pd
import numpy as np
import math
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp

def load_data(
    train_path: str = 'data/processed/train_ratings.csv',
    test_path:  str = 'data/processed/test_ratings.csv'
):
    """
    Load train/test CSVs, normalize column names to
    ['user_id','movie_id','rating'], drop any timestamp column.
    """
    def _clean(df, name):
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(r'[^a-z0-9]', '_', regex=True)
        )
        rename_map = {}
        if 'userid' in df.columns:
            rename_map['userid'] = 'user_id'
        if 'movieid' in df.columns:
            rename_map['movieid'] = 'movie_id'
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)
        needed = {'user_id','movie_id','rating'}
        missing = needed - set(df.columns)
        if missing:
            raise KeyError(
                f"In {name} file missing columns {missing}. "
                f"Found: {df.columns.tolist()}"
            )
        return df[['user_id','movie_id','rating']]

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    train = _clean(train, 'train')
    test  = _clean(test, 'test')
    return train, test


def build_index(train_df: pd.DataFrame):
    """
    Build:
      item_users: movie_id -> { user_id: rating, ... }
      user_items: user_id  -> { movie_id: rating, ... }
    """
    item_users = defaultdict(dict)
    user_items = defaultdict(dict)
    for _, row in train_df.iterrows():
        u = int(row['user_id'])
        m = int(row['movie_id'])
        r = row['rating']
        item_users[m][u]  = r
        user_items[u][m]  = r
    return item_users, user_items


def compute_norms(item_users):
    """
    Precompute ||v_m|| for each item m.
    """
    norms = {}
    for m, ur in item_users.items():
        norms[m] = math.sqrt(sum(r*r for r in ur.values()))
    return norms


def cosine_sim(i, j, item_users, norms):
    """
    Cosine similarity between item i and j.
    """
    ui = item_users.get(i, {})
    uj = item_users.get(j, {})
    common = set(ui) & set(uj)
    if not common or norms.get(i, 0)==0 or norms.get(j, 0)==0:
        return 0.0
    num = sum(ui[u] * uj[u] for u in common)
    den = norms[i] * norms[j]
    return num/den if den > 0 else 0.0


def predict_item_based(
    user_id: int,
    movie_id: int,
    user_items,
    item_users,
    norms,
    k: int = 10,
    global_mean: float = 3.0
) -> float:
    """
    Predict rating by item-based CF (top-k similar items).
    Falls back to user-mean or global_mean.
    """
    rated = user_items.get(user_id, {})
    if not rated or movie_id not in item_users:
        return global_mean

    sims = []
    for m2, r_u_m2 in rated.items():
        s = cosine_sim(movie_id, m2, item_users, norms)
        if s > 0:
            sims.append((s, r_u_m2))

    if not sims:
        return np.mean(list(rated.values()))

    sims.sort(key=lambda x: x[0], reverse=True)
    topk = sims[:k]
    num  = sum(s * r for s, r in topk)
    den  = sum(abs(s)   for s, _ in topk)
    return num/den if den > 0 else np.mean(list(rated.values()))


def init_worker(shared_user_items, shared_item_users, shared_norms):
    """
    Initialize global variables in the worker processes.
    """
    global user_items, item_users, norms
    user_items = shared_user_items
    item_users = shared_item_users
    norms = shared_norms


def _eval_one(args):
    """
    Worker for one test record.
    args = (user_id, movie_id, true_rating, k, global_mean)
    Uses the module-level user_items, item_users, norms.
    Returns (squared_error, abs_error).
    """
    u, m, true_r, k, global_mean = args
    pred_r = predict_item_based(
        u, m,
        user_items,    # populated by init_worker
        item_users,
        norms,
        k=k,
        global_mean=global_mean
    )
    se = (pred_r - true_r)**2
    ae = abs(pred_r - true_r)
    return se, ae

def evaluate(
    test_df: pd.DataFrame,
    user_items,
    item_users,
    norms,
    k: int = 10,
    global_mean: float = 3.0
):
    """
    Compute
      RMSE = $$ \sqrt{\frac{1}{n}\sum (pred - true)^2} $$
      MAE  = $$ \frac{1}{n}\sum |pred - true| $$
    in parallel, with a tqdm progress bar.
    """
    # Number of test cases
    n = len(test_df)

    # Build a simple list of argument-tuples for each row
    args_list = [
        (
            int(row['user_id']),
            int(row['movie_id']),
            row['rating'],
            k,
            global_mean
        )
        for _, row in test_df.iterrows()
    ]

    se_total = 0.0
    ae_total = 0.0

    # Initialize the Pool with the dictionaries to make them global in each worker
    with mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(user_items, item_users, norms)) as pool:
        # imap_unordered yields results as they complete
        for se, ae in tqdm(
            pool.imap_unordered(_eval_one, args_list, chunksize=1000),
            total=n,
            desc="Evaluating",
            unit="it"
        ):
            se_total += se
            ae_total += ae

    rmse = math.sqrt(se_total / n)
    mae  = ae_total / n
    return rmse, mae


if __name__ == '__main__':
    # 1) Load splits
    train_df, test_df = load_data()

    # 2) Build CF indices (these become globals for _eval_one via initargs)
    item_users, user_items = build_index(train_df)

    # 3) Precompute norms
    norms = compute_norms(item_users)

    # 4) Global mean rating
    global_mean = train_df['rating'].mean()
    print(f"Global mean rating = {global_mean:.4f}")

    # 5) Evaluate (now parallel + progress bar)
    rmse, mae = evaluate(
        test_df,
        user_items,
        item_users,
        norms,
        k=10,
        global_mean=global_mean
    )
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
