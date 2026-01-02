"""
Training Script for Sparse Autoencoders

This script trains a single SAE model on a dataset.
Use this for individual model training without the full pipeline.

Usage:
    python train.py --model TopK --language_model 1B --model_path /path/to/llm \
        --data_path /path/to/train_data --layer 12 --hidden_size 2048 \
        --latent_size 16384 --k 64 --batch_size 64 --max_length 512 \
        --num_epochs 1 --lr 0.0005 --betas 0.9 0.999 --seed 42 --steps 10 \
        --device cuda:0 --use_wandb 1 --wandb_project my_project

Supported Models:
    - Vanilla: Standard SAE with L1 regularization
    - Gated: SAE with gating mechanism
    - TopK: SAE with top-k activation
    - JumpReLU: SAE with learned thresholds
    - RouteSAE: Multi-layer routing SAE
    - Crosscoder: Multi-layer processing SAE

Author: RouteSAE Contributors
License: MIT
"""

from utils import parse_args, set_seed, Trainer


if __name__ == '__main__':
    cfg = parse_args()
    set_seed(cfg.seed)
    trainer = Trainer(cfg)
    trainer.run()