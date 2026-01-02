#!/bin/bash
# Example script for running RouteSAE experiments
# 
# Usage:
#   1. Copy this file: cp run_example.sh run_my_experiment.sh
#   2. Modify the paths and parameters below
#   3. Make executable: chmod +x run_my_experiment.sh
#   4. Run: ./run_my_experiment.sh

set -e  # Exit on error

# ============================================================================
# Configuration - MODIFY THESE PATHS
# ============================================================================

# Navigate to source directory
cd ./src

# Set WandB API key (optional, for experiment tracking)
export WANDB_API_KEY='your_wandb_api_key_here'

# Model and Data Paths
LANGUAGE_MODEL_PATH="/path/to/Llama-3.2-1B-Instruct"
TRAIN_DATA_PATH="/path/to/train_data"
EVAL_DATA_PATH="/path/to/eval_data"
APPLY_DATA_PATH="/path/to/apply_data"

# WandB Project Names
TRAIN_PROJECT="sae_training"
EVAL_PROJECT="sae_evaluation"
PIPE_PROJECT="sae_pipeline"

# GPT-4 API Configuration (for feature interpretation)
API_BASE="https://your-endpoint.openai.azure.com/"
API_KEY="your_api_key_here"
API_VERSION="2024-02-01"
ENGINE="gpt-4o"

# ============================================================================
# Experiment 1: Train and evaluate TopK SAE with different k values
# ============================================================================

echo "========================================="
echo "Running TopK SAE experiments"
echo "========================================="

for k_value in 32 64 128; do
    echo "Training TopK SAE with k=$k_value"
    
    python -u main.py \
        --language_model 1B \
        --model_path $LANGUAGE_MODEL_PATH \
        --hidden_size 2048 \
        --pipe_data_path $TRAIN_DATA_PATH $EVAL_DATA_PATH $APPLY_DATA_PATH \
        --model TopK \
        --layer 12 \
        --latent_size 16384 \
        --batch_size 64 \
        --max_length 512 \
        --lr 0.0005 \
        --betas 0.9 0.999 \
        --num_epochs 1 \
        --seed 42 \
        --steps 10 \
        --use_wandb 1 \
        --pipe_project $TRAIN_PROJECT $EVAL_PROJECT $PIPE_PROJECT \
        --device cuda:0 \
        --k $k_value \
        --api_base $API_BASE \
        --api_key $API_KEY \
        --api_version $API_VERSION \
        --engine $ENGINE
    
    echo "Completed TopK SAE with k=$k_value"
    echo "-----------------------------------------"
done

# ============================================================================
# Experiment 2: Train and evaluate RouteSAE with different k values
# ============================================================================

echo "========================================="
echo "Running RouteSAE experiments"
echo "========================================="

for k_value in 32 64 128; do
    echo "Training RouteSAE with k=$k_value"
    
    python -u main.py \
        --language_model 1B \
        --model_path $LANGUAGE_MODEL_PATH \
        --hidden_size 2048 \
        --pipe_data_path $TRAIN_DATA_PATH $EVAL_DATA_PATH $APPLY_DATA_PATH \
        --model RouteSAE \
        --latent_size 16384 \
        --n_layers 16 \
        --batch_size 64 \
        --max_length 512 \
        --num_epochs 1 \
        --seed 42 \
        --lr 0.0005 \
        --betas 0.9 0.999 \
        --steps 10 \
        --pipe_project $TRAIN_PROJECT $EVAL_PROJECT $PIPE_PROJECT \
        --device cuda:0 \
        --use_wandb 1 \
        --aggre sum \
        --routing hard \
        --api_base $API_BASE \
        --api_key $API_KEY \
        --api_version $API_VERSION \
        --engine $ENGINE \
        --k $k_value
    
    echo "Completed RouteSAE with k=$k_value"
    echo "-----------------------------------------"
done

# ============================================================================
# Experiment 3: Train Vanilla SAE with different lambda values (optional)
# ============================================================================

# Uncomment to run Vanilla SAE experiments
# echo "========================================="
# echo "Running Vanilla SAE experiments"
# echo "========================================="
# 
# for lamda_value in 0.001 0.01 0.1; do
#     echo "Training Vanilla SAE with lambda=$lamda_value"
#     
#     python -u main.py \
#         --language_model 1B \
#         --model_path $LANGUAGE_MODEL_PATH \
#         --hidden_size 2048 \
#         --pipe_data_path $TRAIN_DATA_PATH $EVAL_DATA_PATH $APPLY_DATA_PATH \
#         --model Vanilla \
#         --layer 12 \
#         --latent_size 16384 \
#         --batch_size 64 \
#         --max_length 512 \
#         --lr 0.0005 \
#         --betas 0.9 0.999 \
#         --num_epochs 1 \
#         --seed 42 \
#         --steps 10 \
#         --use_wandb 1 \
#         --pipe_project $TRAIN_PROJECT $EVAL_PROJECT $PIPE_PROJECT \
#         --device cuda:0 \
#         --lamda $lamda_value \
#         --api_base $API_BASE \
#         --api_key $API_KEY \
#         --api_version $API_VERSION \
#         --engine $ENGINE
#     
#     echo "Completed Vanilla SAE with lambda=$lamda_value"
#     echo "-----------------------------------------"
# done

echo "========================================="
echo "All experiments completed!"
echo "========================================="
