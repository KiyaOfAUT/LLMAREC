import torch

CONFIG = {
    "train_path": "data/processed/train_ratings.csv",
    "test_path": "data/processed/test_ratings.csv",
    "user_npz_path": "data/embeddings/users.npz",
    "item_npz_path": "data/embeddings/movies.npz",
    
    "use_text_embeddings": True,
    
    "lgcn_emb_dim": 64,
    "lgcn_layers": 3,
    
    "predictor_hidden_layers": [128, 64],
    
    "epochs": 30,
    "batch_size": 1024,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "loss_type": "bpr", # "mse" or "bpr"
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "K": 10,
    "relevance_threshold": 3.5,
}
