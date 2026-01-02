"""
Main Pipeline Script for RouteSAE

This script runs the complete pipeline including:
1. Training the SAE model
2. Evaluating on test data
3. Extracting feature contexts
4. Interpreting features with GPT-4o

Usage:
    python main.py --model RouteSAE --language_model 1B --model_path /path/to/llm \
        --pipe_data_path /train /eval /apply --pipe_project train eval pipe \
        --hidden_size 2048 --latent_size 16384 --n_layers 16 --k 64 \
        --batch_size 64 --max_length 512 --num_epochs 1 --lr 0.0005 \
        --betas 0.9 0.999 --seed 42 --steps 10 --aggre sum --routing hard \
        --device cuda:0 --use_wandb 1 \
        --api_base https://api.openai.com --api_key YOUR_KEY \
        --api_version 2024-02-01 --engine gpt-4o

Author: RouteSAE Contributors
License: MIT
"""

from utils import parse_args, SAE_pipeline


if __name__ == '__main__':
    cfg = parse_args()
    pipeline = SAE_pipeline(cfg)
    pipeline.run()