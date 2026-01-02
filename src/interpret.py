"""
Feature Interpretation Script

This script uses GPT-4o to automatically interpret the semantic meaning
of SAE features based on their activation contexts.

Usage:
    python interpret.py --language_model 1B --model_path /path/to/llm \
        --data_path ../contexts/model_contexts.json --model TopK \
        --SAE_path ../SAE_models/model.pt --hidden_size 2048 --latent_size 16384 \
        --batch_size 64 --max_length 512 --device cuda:0 --use_wandb 0 \
        --api_base https://api.openai.com --api_key YOUR_KEY \
        --api_version 2024-02-01 --engine gpt-4o

Output Format:
    The script generates a JSON file containing:
    - Feature category (low-level, high-level, undiscernible)
    - Monosemanticity score (1-5)
    - Explanation of the feature's semantic meaning
    - Statistics on low-level vs. high-level features

Note: This requires a valid GPT-4o API key and will incur API costs.

Author: RouteSAE Contributors
License: MIT
"""

from utils import parse_args, Interpreter


if __name__ == '__main__':
    cfg = parse_args()
    interp = Interpreter(cfg)
    interp.run(sample_latents=100)