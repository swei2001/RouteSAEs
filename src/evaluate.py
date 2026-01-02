"""
Evaluation Script for Sparse Autoencoders

This script evaluates a trained SAE model on a test dataset using various metrics.

Usage:
    python evaluate.py --model TopK --language_model 1B --model_path /path/to/llm \
        --data_path /path/to/eval_data --SAE_path ../SAE_models/model.pt \
        --layer 12 --hidden_size 2048 --latent_size 16384 --k 64 \
        --metric NormMSE --batch_size 64 --max_length 512 \
        --device cuda:0 --use_wandb 1 --wandb_project eval_project

Available Metrics:
    - NormMSE: Normalized mean squared error between original and reconstructed activations
    - KLDiv: KL divergence between original and reconstructed output distributions
    - DeltaCE: Difference in cross-entropy loss (impact on next-token prediction)
    - Recovered: Percentage recovery from zero ablation

Author: RouteSAE Contributors
License: MIT
"""

from utils import parse_args, Evaluater


if __name__ == '__main__':
    cfg = parse_args()
    evaluater = Evaluater(cfg)
    evaluater.run()