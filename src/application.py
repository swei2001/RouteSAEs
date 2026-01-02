"""
Feature Extraction and Latent Manipulation Application

This script demonstrates how to:
1. Extract contexts for activated features
2. Manipulate specific latent features (clamping)

Usage:
    python application.py --model_path /path/to/model --SAE_path /path/to/sae [options]
"""

from utils import parse_args, Applier


if __name__ == '__main__':
    cfg = parse_args()
    applier = Applier(cfg)
    
    # Extract contexts for activated features
    # You can customize threshold, max_length, max_per_token, and lines parameters
    applier.get_context(
        threshold=15.0,      # Activation threshold for feature selection
        max_length=64,       # Maximum context length in tokens
        max_per_token=2,     # Maximum contexts per token type
        lines=4              # Minimum lines required to include a feature
    )
    
    # Optional: Perform latent manipulation (clamping)
    # Uncomment and modify the following to manipulate specific latents:
    # 
    # applier.clamp(
    #     max_length=128, 
    #     set_high=[
    #         (13523, 15, 0)  # (latent_idx, value, mode)
    #         # mode: 0 for addition, 1 for multiplication
    #     ],
    #     output_path='../clamp/clamped_output.json'
    # )

