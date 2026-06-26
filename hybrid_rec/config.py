import torch

CONFIG = {
    "train_path": "data/processed/train_ratings.csv",
    "test_path": "data/processed/test_ratings.csv",
    "user_npz_path": "data/embeddings/users.npz",
    "item_npz_path": "data/embeddings/movies.npz",

    "use_text_embeddings": True,

    # RLMRec alignment: "con" (contrastive), "gen" (masked reconstruction), "none"
    "alignment_mode": "con",
    # Official RLMRec uses kd_weight=1e-2 in lightgcn_plus.yml
    "info_weight": 1e-2,
    "info_temperature": 0.2,
    "mask_ratio": 0.1,
    "keep_rate": 0.8,

    # Official RLMRec LightGCN uses embedding_size=32
    "lgcn_emb_dim": 32,
    "lgcn_layers": 3,

    "epochs": 30,
    "batch_size": 4096,
    "lr": 1e-3,
    # Official reg_weight=1e-7 for amazon/yelp
    "weight_decay": 1e-7,
    "early_stop_patience": 10,
    "eval_every": 3,
    "seed": 2023,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "K": 10,
    "relevance_threshold": 3.5,
    # None = evaluate all test users (paper-style all-rank protocol)
    "max_eval_users": None,
}
