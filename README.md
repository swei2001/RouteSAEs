# RouteSAE: Route Sparse Autoencoder to Interpret Large Language Models

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

🎉 **This work has been accepted for Oral presentation at EMNLP 2025.**

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

RouteSAE is a novel approach to interpreting large language models (LLMs) through sparse autoencoders with intelligent routing mechanisms. This repository contains the official implementation of RouteSAE, which enables:

- **Multi-layer Analysis**: Process and interpret activations across multiple layers of LLMs
- **Flexible Routing**: Support both hard and soft routing strategies for feature extraction
- **Comprehensive Evaluation**: Multiple metrics including NormMSE, KLDiv, and DeltaCE
- **Automated Interpretation**: Integration with GPT-4 for automated feature interpretation

## Features

- **Multiple SAE Architectures**: Support for Vanilla, Gated, TopK, JumpReLU, RouteSAE, and Crosscoder models
- **Training Pipeline**: End-to-end training with WandB integration for experiment tracking
- **Evaluation Framework**: Comprehensive evaluation metrics for model performance
- **Feature Interpretation**: Automated feature interpretation using GPT-4 API
- **Application Tools**: Tools for context extraction and latent manipulation

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- PyTorch 2.0 or higher

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/RouteSAEs.git
cd RouteSAEs
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Required Data and Models

Download the required dataset and pre-trained language model:

- [OpenWebText2 Dataset](https://huggingface.co/datasets/segyges/OpenWebText2)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

### Step 4: Create Required Directories

```bash
mkdir -p contexts interpret SAE_models clamp
```

- `contexts/`: Stores extracted contexts for SAE models
- `interpret/`: Stores feature interpretation results
- `SAE_models/`: Stores trained SAE model checkpoints
- `clamp/`: Stores results from latent manipulation experiments

## Quick Start

### Training a Simple TopK SAE

```bash
cd src
python train.py \
    --language_model 1B \
    --model_path /path/to/Llama-3.2-1B-Instruct \
    --data_path /path/to/train_data \
    --model TopK \
    --layer 12 \
    --hidden_size 2048 \
    --latent_size 16384 \
    --k 64 \
    --batch_size 64 \
    --max_length 512 \
    --num_epochs 1 \
    --lr 0.0005 \
    --betas 0.9 0.999 \
    --seed 42 \
    --steps 10 \
    --device cuda:0 \
    --use_wandb 0
```

### Running the Full Pipeline

For a complete training, evaluation, and interpretation pipeline:

```bash
cd src
python main.py \
    --language_model 1B \
    --model_path /path/to/Llama-3.2-1B-Instruct \
    --pipe_data_path /path/to/train /path/to/eval /path/to/apply \
    --model RouteSAE \
    --hidden_size 2048 \
    --latent_size 16384 \
    --n_layers 16 \
    --k 64 \
    --batch_size 64 \
    --max_length 512 \
    --num_epochs 1 \
    --lr 0.0005 \
    --betas 0.9 0.999 \
    --seed 42 \
    --steps 10 \
    --aggre sum \
    --routing hard \
    --device cuda:0 \
    --use_wandb 1 \
    --pipe_project train_project eval_project pipe_project \
    --api_base your_api_base \
    --api_key your_api_key \
    --api_version your_api_version \
    --engine gpt-4o
```

## Usage

### Individual Components

#### 1. Training

Train a sparse autoencoder model:

```bash
python train.py --model [Vanilla|Gated|TopK|JumpReLU|RouteSAE|Crosscoder] [other args]
```

#### 2. Evaluation

Evaluate a trained model:

```bash
python evaluate.py \
    --SAE_path ../SAE_models/your_model.pt \
    --metric [NormMSE|KLDiv|DeltaCE|Recovered] \
    [other args]
```

#### 3. Feature Extraction

Extract contexts for activated features:

```bash
python application.py --SAE_path ../SAE_models/your_model.pt [other args]
```

#### 4. Interpretation

Interpret extracted features using GPT-4:

```bash
python interpret.py \
    --data_path ../contexts/your_contexts.json \
    --api_base your_api_base \
    --api_key your_api_key \
    --engine gpt-4o \
    [other args]
```

### Batch Processing

For batch experiments, you can save the following as a shell script (for Unix/Linux) or batch file (for Windows):

```bash
#!/bin/bash
cd ./src
export WANDB_API_KEY='your_wandb_api_key'

language_model_path=/path/to/Llama-3.2-1B-Instruct
train_data_path=/path/to/train_data
eval_data_path=/path/to/eval_data
apply_data_path=/path/to/apply_data

train_project=your_train_project
eval_project=your_eval_project
pipe_project=your_pipe_project

api_base=your_api_base
api_key=your_api_key
api_version=your_api_version
engine=gpt-4o

# Example: Train and evaluate TopK SAE with different k values
for k_value in 32 64 128; do
    python -u main.py --language_model 1B --model_path $language_model_path --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --model TopK --layer 12 --latent_size 16384 \
        --batch_size 64 --max_length 512 --lr 0.0005 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --k $k_value \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine 
done

# Example: Train and evaluate RouteSAE with different k values
for k_value in 32 64 128; do
    python -u main.py --language_model 1B --model_path $language_model_path --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --model RouteSAE --latent_size 16384 \
        --n_layers 16 --batch_size 64 --max_length 512 --num_epochs 1 --seed 42 --lr 0.0005 --betas 0.9 0.999 --steps 10 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --use_wandb 1 --aggre sum --routing hard \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine --k $k_value
done
```

**Note**: Replace the placeholder paths and API keys with your actual values before running.

## Project Structure

```
RouteSAEs/
│
├── src/                          # Source code directory
│   ├── main.py                   # Main pipeline script
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── application.py            # Feature extraction script
│   ├── interpret.py              # Interpretation script
│   ├── model.py                  # SAE model definitions
│   └── utils.py                  # Utility functions and classes
│
├── SAE_models/                   # Trained model checkpoints (created at runtime)
├── contexts/                     # Extracted feature contexts (created at runtime)
├── interpret/                    # Interpretation results (created at runtime)
├── clamp/                        # Latent manipulation results (created at runtime)
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # License file
```

## Model Architectures

RouteSAE supports multiple sparse autoencoder architectures:

### 1. **Vanilla SAE**
Standard sparse autoencoder with ReLU activation:
```
latents = ReLU(encoder(x - pre_bias) + latent_bias)
reconstruction = decoder(latents) + pre_bias
```

### 2. **Gated SAE**
Autoencoder with gated activation mechanism:
```
latents = gate(pre_acts + gate_bias) * relu(r_mag.exp() * pre_acts + mag_bias)
```

### 3. **TopK SAE**
Autoencoder that only activates top-k features:
```
latents = TopK(encoder(x - pre_bias) + latent_bias, k)
```

### 4. **JumpReLU SAE**
Autoencoder with learned threshold per feature:
```
latents = JumpReLU(pre_acts, threshold, bandwidth)
```

### 5. **RouteSAE** (Our Contribution)
Multi-layer routing mechanism that dynamically selects which layer to process:
- Supports both hard and soft routing
- Processes hidden states across multiple layers
- Learns optimal routing weights

### 6. **Crosscoder**
Processes multiple layers simultaneously with separate encoder/decoder per layer.

## Configuration

### Key Parameters

| Parameter | Description | Default | 
|-----------|-------------|---------|
| `--model` | SAE architecture | Required |
| `--hidden_size` | LLM hidden dimension | Required |
| `--latent_size` | SAE latent dimension | Required |
| `--k` | Number of active features (TopK, RouteSAE) | - |
| `--lamda` | L1 regularization weight | - |
| `--batch_size` | Training batch size | Required |
| `--max_length` | Maximum sequence length | Required |
| `--lr` | Learning rate | Required |
| `--num_epochs` | Number of training epochs | Required |
| `--device` | Device (cuda:0, cpu) | Required |

### Routing Parameters (RouteSAE only)

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--aggre` | Router input aggregation | `sum`, `mean` |
| `--routing` | Routing strategy | `hard`, `soft` |
| `--n_layers` | Number of LLM layers | Required |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `NormMSE` | Normalized mean squared error |
| `KLDiv` | KL divergence between original and reconstructed logits |
| `DeltaCE` | Difference in cross-entropy loss |
| `Recovered` | Recovery percentage from zero ablation |

## Advanced Usage

### Latent Manipulation

You can manipulate specific latent features to study their effects:

```python
from utils import Applier, parse_args

cfg = parse_args()
applier = Applier(cfg)

# Amplify latent 13523 by adding 15
applier.clamp(
    max_length=128,
    set_high=[(13523, 15, 0)],  # (latent_idx, value, mode)
    output_path='../clamp/output.json'
)
```

Mode options:
- `0`: Addition/Subtraction
- `1`: Multiplication/Division

### Custom Interpretation Prompts

Modify the `construct_prompt` method in `utils.py` to customize how features are interpreted by GPT-4o.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{routesae2025,
    title={Route Sparse Autoencoder to Interpret Large Language Models},
    author={Wei Shi, Sihang Li, Tao Liang, Mingyang Wan, Guojun Ma, Xiang Wang, and Xiangnan He},
    booktitle={Proceedings of EMNLP},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source community for the foundational tools
- Inspired by research on sparse autoencoders and mechanistic interpretability
- Built with PyTorch and Hugging Face Transformers

## FAQ

**Q: What GPU memory is required?**  
A: For Llama-3.2-1B with typical configurations, expect 16-24GB VRAM for training.

**Q: Can I use other language models?**  
A: Yes! The code supports any Hugging Face compatible causal LM. Adjust `hidden_size` and `n_layers` accordingly.

**Q: How do I interpret the results?**  
A: Check the `interpret/` directory for JSON files containing feature interpretations with monosemanticity scores.

**Q: What if I don't have a GPT-4o API key?**  
A: You can skip the interpretation step and manually analyze the extracted contexts in the `contexts/` directory.

