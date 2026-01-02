"""
Utility Functions and Classes for RouteSAE Training Pipeline

This module contains core utilities for:
- Data loading and preprocessing
- Training and evaluation
- Feature interpretation
- Model manipulation and analysis

Author: RouteSAE Contributors
License: MIT
"""

import os
import re
import time
import json
import yaml
import torch
import heapq
import wandb
import random
import tiktoken
import argparse
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model import *
from typing import List, Union, Optional, Dict, Tuple, Any
from torch.optim import Adam
from openai import AzureOpenAI
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.hooks import RemovableHandle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for SAE training/evaluation."""
    parser = argparse.ArgumentParser(description='Configuration for sparse autoencoders pipeline')

    # Core model configuration
    parser.add_argument('--language_model', type=str, required=True, help='Language model name (e.g., "Llama-3.2-1B")')
    parser.add_argument('--model_path', type=str, required=True, help='Path to language model')
    parser.add_argument('--model', type=str, required=True, help='SAE model type')
    parser.add_argument('--hidden_size', type=int, required=True, help='LLM hidden dimension')
    parser.add_argument('--latent_size', type=int, required=True, help='SAE latent dimension')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--max_length', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--device', type=str, required=True, help='Device (e.g., "cuda:0", "cpu")')
    parser.add_argument('--use_wandb', type=int, required=True, help='Enable wandb logging (1/0)')

    # Training configuration
    parser.add_argument('--data_path', type=str, required=False, help='Path to dataset')
    parser.add_argument('--wandb_project', type=str, required=False, help='Wandb project name')
    parser.add_argument('--num_epochs', type=int, required=False, help='Number of training epochs')
    parser.add_argument('--k', type=int, required=False, help='TopK parameter')
    parser.add_argument('--lr', type=float, required=False, help='Learning rate')
    parser.add_argument('--betas', type=float, nargs=2, required=False, help='Adam optimizer betas')
    parser.add_argument('--seed', type=int, required=False, help='Random seed')
    parser.add_argument('--layer', type=int, required=False, help='Target layer index (1-indexed)')
    parser.add_argument('--steps', type=int, required=False, help='Decoder normalization interval')
    parser.add_argument('--lamda', type=float, required=False, help='L1 regularization weight')

    # Multi-layer configuration
    parser.add_argument('--n_layers', type=int, required=False, help='Number of LLM layers')
    parser.add_argument('--aggre', type=str, required=False, help='Router aggregation strategy')
    parser.add_argument('--routing', type=str, required=False, help='RouteSAE routing strategy')

    # Evaluation configuration
    parser.add_argument('--SAE_path', type=str, required=False, help='Path to trained SAE model')
    parser.add_argument('--metric', type=str, required=False, help='Evaluation metric')
    parser.add_argument('--infer_k', type=int, required=False, help='Inference-time k')
    parser.add_argument('--theta', type=float, required=False, help='Inference-time threshold')

    # Interpretation configuration (GPT-4o API)
    parser.add_argument('--api_base', type=str, required=False, help='OpenAI API endpoint')
    parser.add_argument('--api_key', type=str, required=False, help='OpenAI API key')
    parser.add_argument('--api_version', type=str, required=False, help='OpenAI API version')
    parser.add_argument('--engine', type=str, required=False, help='GPT model engine')

    # Pipeline configuration
    parser.add_argument('--pipe_data_path', type=str, nargs='+', required=False, help='Pipeline dataset paths')
    parser.add_argument('--pipe_project', type=str, nargs='+', required=False, help='Pipeline wandb projects')

    args = parser.parse_args()
    return args


class Config:
    """Configuration object for loading from YAML files."""
    def __init__(self, config_dict: Dict[str, Any]) -> None:
        for key, value in config_dict.items():
            setattr(self, key, value)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


class OpenWebTextDataset(Dataset):
    """Dataset for loading OpenWebText data in JSONL format."""
    
    def __init__(
        self, 
        folder_path: str, 
        tokenizer: AutoTokenizer, 
        max_length: int,
        keyword: str = 'text'
    ) -> None:
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keyword = keyword
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
        
        if not self.file_list:
            raise ValueError(f"No .jsonl files found in {folder_path}")
        
        logger.info(f"Loading dataset from {folder_path} ({len(self.file_list)} files)")
        self.data = self.load_data()
        logger.info(f"Loaded {len(self.data)} samples")

    def load_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load and tokenize data from JSONL files."""
        data = []
        for file_name in self.file_list:
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        words = record.get(self.keyword, '').split()
                        
                        # Split into chunks of max_length
                        for i in range(0, len(words), self.max_length):
                            chunk = ' '.join(words[i:i + self.max_length])
                            inputs = self.tokenizer(
                                chunk,
                                return_tensors='pt',
                                max_length=self.max_length,
                                padding='max_length',
                                truncation=True
                            )
                            input_ids = inputs['input_ids'].squeeze(0)
                            attention_mask = inputs['attention_mask'].squeeze(0)
                            data.append((input_ids, attention_mask))
                    except json.JSONDecodeError:
                        logger.warning(f'Error decoding JSON in file: {file_path}')
                        continue
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx]
    

def create_dataloader(
    folder_path: str, 
    tokenizer: AutoTokenizer, 
    batch_size: int, 
    max_length: int,
    keyword: str = 'text'
) -> DataLoader:
    """Create DataLoader for OpenWebText dataset."""
    dataset = OpenWebTextDataset(folder_path, tokenizer, max_length, keyword)
    
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        return input_ids, attention_mask

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
    return dataloader


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def wandb_init(project: str, config: Dict[str, Any], name: str) -> None:
    """Initialize Weights & Biases logging."""
    wandb.init(project=project, config=config, name=name)
    logger.info(f"Initialized WandB project: {project}, run: {name}")


def get_language_model(
    model_path: str, 
    device: torch.device
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and language model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    logger.info(f"Loading language model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        return_dict_in_generate=True, 
        output_hidden_states=True
    ).to(device)
    
    logger.info(f"Model loaded on {device}")
    return tokenizer, language_model


def get_outputs(
    cfg: argparse.Namespace, 
    batch: Tuple[torch.Tensor, torch.Tensor], 
    language_model: nn.Module, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
    """Extract model outputs and hidden states from batch."""
    input_ids, attention_mask = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = language_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Select hidden states based on model type
    if cfg.model in ['Vanilla', 'Gated', 'TopK', 'JumpReLU']:
        hidden_states = outputs.hidden_states[cfg.layer]
    elif cfg.model == 'MLSAE':
        hidden_states = outputs.hidden_states[3:12]
    elif cfg.model == 'Random':
        cfg.random_layer = random.randint(1, 16)
        hidden_states = outputs.hidden_states[cfg.random_layer]
    else:  # RouteSAE, Crosscoder
        start_layer = cfg.n_layers // 4
        end_layer = cfg.n_layers * 3 // 4 + 1
        hidden_states = torch.stack(
            outputs.hidden_states[start_layer:end_layer], dim=0
        ).permute(1, 2, 0, 3)
    
    return input_ids, attention_mask, outputs, hidden_states


def pre_process(
    hidden_stats: torch.Tensor, 
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize hidden states to zero mean and unit variance."""
    mean = hidden_stats.mean(dim=-1, keepdim=True)
    std = hidden_stats.std(dim=-1, keepdim=True)
    x = (hidden_stats - mean) / (std + eps)
    return x, mean, std


def L1_loss(latents: torch.Tensor) -> torch.Tensor:
    """Compute mean L1 loss for sparsity regularization."""
    return latents.abs().mean()


def Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Compute normalized MSE loss (normalized by input variance)."""
    return (((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)).mean()


@torch.no_grad()
def unit_norm_decoder(model: nn.Module) -> None:
    """Normalize decoder weights to unit norm (prevents feature suppression)."""
    if isinstance(model, (Vanilla, Gated, TopK, JumpReLU)):
        model.decoder.weight.data /= model.decoder.weight.data.norm(dim=0, keepdim=True)
    elif isinstance(model, RouteSAE):
        model.sae.decoder.weight.data /= model.sae.decoder.weight.data.norm(dim=0, keepdim=True)
    elif isinstance(model, Crosscoder):
        for i in range(len(model.decoder)):
            model.decoder[i].weight.data /= model.decoder[i].weight.data.norm(dim=0, keepdim=True)


def log_layers(layer_weights: np.ndarray) -> None:
    """Log layer weight distribution to WandB."""
    data = [[i, value] for i, value in enumerate(layer_weights[:], start=1)]
    table = wandb.Table(data=data, columns=['Layer', 'Weight'])
    wandb.log({
        'Layer Weights': wandb.plot.bar(
            table=table, label='Layer', value='Weight', title='Layer Weights'
        )
    })


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f'Saved data to {path}')


def convert_hh_jsonl(input_file: str, output_path: str) -> None:
    """Convert HH-RLHF JSONL format to text.jsonl and conversation.json."""
    os.makedirs(output_path, exist_ok=True)
    text_file = os.path.join(output_path, 'text.jsonl')
    conv_file = os.path.join(output_path, 'conversation.json')

    conversations = {}
    index = 1

    def parse_conversation(conv_text: str) -> List[Dict[str, str]]:
        """Parse conversation text into Human-Assistant turns."""
        conv_text = conv_text.strip()
        if not conv_text:
            return []
        
        parts = conv_text.split('Human:')
        turns = []
        for part in parts:
            part = part.strip()
            if not part or 'Assistant:' not in part:
                continue
            
            human_part, assistant_part = part.split('Assistant:', 1)
            turns.append({
                'Human': human_part.strip(),
                'Assistant': assistant_part.strip()
            })
        return turns

    logger.info(f"Converting HH-RLHF data from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(text_file, 'w', encoding='utf-8') as ftext:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            chosen_turns = parse_conversation(data.get('chosen', ''))
            rejected_turns = parse_conversation(data.get('rejected', ''))
            num_turns = min(len(chosen_turns), len(rejected_turns))

            # Build conversation.json entry
            conv_list = []
            for i in range(num_turns):
                conv_list.append({
                    'Human': chosen_turns[i]['Human'],
                    'chosen': chosen_turns[i]['Assistant'],
                    'rejected': rejected_turns[i]['Assistant']
                })
            conversations[str(index)] = conv_list

            # Build text.jsonl entry
            combined_lines = []
            for i in range(num_turns):
                combined_lines.append(f"Human: {chosen_turns[i]['Human']}")
                chosen_line = f"Assistant: {chosen_turns[i]['Assistant']}"
                rejected_line = f"Assistant: {rejected_turns[i]['Assistant']}"
                
                combined_lines.append(chosen_line)
                if chosen_line != rejected_line:
                    combined_lines.append(rejected_line)

            merged_text = '\n'.join(combined_lines)
            ftext.write(json.dumps({'text': merged_text}, ensure_ascii=False) + '\n')
            index += 1
    
    save_json(conversations, conv_file)
    logger.info(f"Conversion complete: {index-1} conversations processed")


def hook_SAE(
    cfg: argparse.Namespace,
    model: Union[TopK, RouteSAE, JumpReLU],
    hooked_module: nn.Module,
    set_high: Optional[List[Tuple[int, float, int]]] = None, 
    set_low: Optional[List[Tuple[int, float, int]]] = None,
    is_zero: bool = False
) -> List[RemovableHandle]:
    """
    Register forward hook to intervene on SAE latents.
    
    Args:
        set_high: List of (latent_idx, val, mode) tuples for upward adjustments
            mode=0: add val, mode=1: multiply by val
        set_low: List of (latent_idx, val, mode) tuples for downward adjustments
            mode=0: subtract val, mode=1: divide by val
        is_zero: If True, zero out all activations instead of SAE intervention
    """
    def hook(module: nn.Module, _, outputs):
        # Handle both single tensor and tuple outputs
        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = [outputs]
        
        if is_zero:
            unpack_outputs[0] = torch.zeros_like(unpack_outputs[0])
        else:
            # Normalize, encode, intervene, decode, denormalize
            x, mu, std = pre_process(unpack_outputs[0])
            latents = model.encode(x, cfg.infer_k, cfg.theta)

            if set_high:
                for (latent_idx, val, mode) in set_high:
                    if mode == 0:
                        latents[..., latent_idx] += val
                    elif mode == 1:
                        latents[..., latent_idx] *= val

            if set_low:
                for (latent_idx, val, mode) in set_low:
                    if mode == 0:
                        latents[..., latent_idx] -= val
                    elif mode == 1 and val != 0:
                        latents[..., latent_idx] /= val

            x_hat = model.decode(latents)
            unpack_outputs[0] = x_hat * std + mu

        return tuple(unpack_outputs) if isinstance(outputs, tuple) else unpack_outputs[0]

    return [hooked_module.register_forward_hook(hook)]


class RouteHook:
    """Forward hook for layer-specific RouteSAE interventions."""
    
    def __init__(
        self,
        cfg: argparse.Namespace,
        layer_idx: int,
        model: RouteSAE,
        batch_layer_weights: torch.Tensor,
        set_high: Optional[List[Tuple[int, float, int]]] = None,
        set_low: Optional[List[Tuple[int, float, int]]] = None,
        is_zero: bool = False  
    ) -> None:
        """
        Args:
            layer_idx: Current layer index
            batch_layer_weights: Shape (batch, seq_len, n_layers), indicates which layers to intervene
            set_high/set_low: Same as hook_SAE
            is_zero: If True, zero out activations instead of SAE intervention
        """
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.model = model
        self.batch_layer_weights = batch_layer_weights
        self.set_high = set_high or []
        self.set_low = set_low or []
        self.is_zero = is_zero  

    def __call__(
        self, 
        module: nn.Module, 
        inputs: tuple, 
        outputs: Union[torch.Tensor, tuple]
    ) -> Union[torch.Tensor, tuple]:
        """Apply SAE intervention to specified layer positions."""
        # Extract layer mask
        layer_mask = self.batch_layer_weights[
            :, :, self.layer_idx - self.model.start_layer + 1
        ].bool()

        if not layer_mask.any():
            return outputs

        # Unpack outputs
        if isinstance(outputs, tuple):
            outputs = list(outputs)
            output_tensor = outputs[0]
        else:
            output_tensor = outputs
        
        if output_tensor.shape[1] != layer_mask.shape[1]:
            return outputs

        if self.is_zero:
            # Zero out masked positions
            replace_mask = layer_mask.unsqueeze(-1).expand_as(output_tensor)
            output_tensor = output_tensor.clone()
            output_tensor[replace_mask] = 0
        else:
            # SAE intervention
            x, mu, std = pre_process(output_tensor)
            latents = self.model.sae.encode(x, self.cfg.infer_k, self.cfg.theta)

            for (idx, val, mode) in self.set_high:
                if mode == 0:
                    latents[..., idx] += val
                elif mode == 1:
                    latents[..., idx] *= val

            for (idx, val, mode) in self.set_low:
                if mode == 0:
                    latents[..., idx] -= val
                elif mode == 1 and val != 0:
                    latents[..., idx] /= val

            x_hat = self.model.sae.decode(latents)
            reconstruct = x_hat * std + mu

            # Replace masked positions with reconstructions
            replace_mask = layer_mask.unsqueeze(-1).expand_as(reconstruct)
            output_tensor = output_tensor.clone()
            output_tensor[replace_mask] = reconstruct[replace_mask]

        return tuple(outputs) if isinstance(outputs, list) else output_tensor


def hook_RouteSAE(
    cfg: argparse.Namespace,
    model: RouteSAE,
    language_model: nn.Module,
    batch_layer_weights: torch.Tensor,
    set_high: Optional[List[Tuple[int, float, int]]] = None,
    set_low: Optional[List[Tuple[int, float, int]]] = None,
    is_zero: bool = False 
) -> List[RemovableHandle]:
    """
    Register forward hooks on multiple layers for RouteSAE interventions.
    
    Args:
        batch_layer_weights: Shape (batch, seq_len, n_layers), routing weights for each layer
        set_high/set_low: Same as hook_SAE
        is_zero: If True, zero out activations instead of SAE intervention
    
    Returns:
        List of registered hook handles
    """
    handles = []
    num_layers = batch_layer_weights.size(-1)

    for layer_idx in range(model.start_layer - 1, num_layers + model.start_layer - 1):
        # Only register hook if this layer has any interventions
        if batch_layer_weights[:, :, layer_idx - model.start_layer + 1].any():
            layer_name = f'model.layers.{layer_idx}'
            try:
                module = language_model.get_submodule(layer_name)
            except AttributeError:
                raise ValueError(f'Submodule {layer_name} not found in language_model')

            hook_fn = RouteHook(
                cfg=cfg,
                layer_idx=layer_idx,
                model=model, 
                batch_layer_weights=batch_layer_weights,
                set_high=set_high,
                set_low=set_low,
                is_zero=is_zero
            )
            handles.append(module.register_forward_hook(hook_fn))
    return handles


class LinearWarmupLR(LambdaLR):
    """Learning rate scheduler with linear warmup and decay."""
    
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        num_warmup_steps: int, 
        num_training_steps: int, 
        max_lr: float
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.max_lr = max_lr
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step: int) -> float:
        """Compute learning rate multiplier for given step."""
        if step < self.num_warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, self.num_warmup_steps))
        elif step < self.num_training_steps - self.num_training_steps // 5:
            # Constant LR
            return 1.0
        else:
            # Linear decay in last 20% of training
            decay_steps = self.num_training_steps // 5
            steps_into_decay = step - (self.num_training_steps - self.num_training_steps // 5)
            return max(0.0, 1.0 - float(steps_into_decay) / float(decay_steps))


class Trainer:
    """Training pipeline for SAE models."""
    
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Load language model and data
        logger.info(f"Initializing Trainer for {cfg.model}")
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
        
        # Build experiment title
        data_name = cfg.data_path.split("/")[-1]
        self.title = f'{cfg.language_model}_{data_name}_{cfg.latent_size}'
        
        self.config_dict = {
            'batch_size': cfg.batch_size,
            'num_epochs': cfg.num_epochs,
            'lr': cfg.lr,
            'steps': cfg.steps
        }

        # Initialize SAE model
        if cfg.model == 'Vanilla':
            self.model = Vanilla(cfg.hidden_size, cfg.latent_size)
            self.title = f'L{cfg.layer}_VL{cfg.lamda}_{self.title}'
        
        elif cfg.model == 'Gated':
            self.model = Gated(cfg.hidden_size, cfg.latent_size)
            self.title = f'L{cfg.layer}_GL{cfg.lamda}_{self.title}'

        elif cfg.model == 'TopK':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
            self.title = f'L{cfg.layer}_K{cfg.k}_{self.title}'

        elif cfg.model == 'JumpReLU':
            self.model = JumpReLU(cfg.hidden_size, cfg.latent_size)
            self.title = f'L{cfg.layer}_JL{cfg.lamda}_{self.title}'

        elif cfg.model == 'RouteSAE':
            self.model = RouteSAE(cfg.hidden_size, cfg.n_layers, cfg.latent_size, cfg.k)
            self.title = f'{cfg.aggre}_{cfg.routing}_K{cfg.k}_{self.title}'
            self.layer_weights = np.zeros(cfg.n_layers // 2 + 1, dtype=float)

        elif cfg.model == 'Crosscoder':
            self.model = Crosscoder(cfg.hidden_size, cfg.n_layers, cfg.latent_size)
            self.title = f'Cross_CL{cfg.lamda}_{self.title}'
        
        elif cfg.model == 'MLSAE':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
            self.title = f'ML_K{cfg.k}_{self.title}'
        
        elif cfg.model == 'Random':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
            self.title = f'RDM_K{cfg.k}_{self.title}'

        else:
            raise ValueError(
                f'Invalid model: {cfg.model}. Expected one of '
                f'[Vanilla, Gated, TopK, JumpReLU, RouteSAE, Crosscoder, MLSAE, Random]'
            )
        
        self.model.to(self.device)
        self.model.train()
        
        # Setup optimizer and scheduler
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, betas=cfg.betas)
        num_training_steps = cfg.num_epochs * len(self.dataloader)
        num_warmup_steps = int(num_training_steps * 0.05)
        self.scheduler = LinearWarmupLR(
            self.optimizer, num_warmup_steps, num_training_steps, cfg.lr
        )
        
        logger.info(f"Model initialized: {self.title}")
    
    def run(self) -> float:
        """Execute training loop."""
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)
        
        logger.info("Starting training")
        curr_loss = 0.0
        unit_norm_decoder(self.model)
        
        for epoch in range(self.cfg.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                _, _, _, hidden_states = get_outputs(
                    self.cfg, batch, self.language_model, self.device
                )
                
                # Handle MLSAE (multi-layer training)
                if self.cfg.model == 'MLSAE':
                    for hidden_state in hidden_states:
                        x, _, _ = pre_process(hidden_state)
                        _, x_hat = self.model(x)
                        mse_loss = Normalized_MSE_loss(x, x_hat)
                        
                        self.optimizer.zero_grad()
                        mse_loss.backward()
                        self.optimizer.step()
                        curr_loss = mse_loss.item()

                    self.scheduler.step()
                    unit_norm_decoder(self.model)

                else:
                    x, _, _ = pre_process(hidden_states)

                    # Forward pass
                    if self.cfg.model == 'RouteSAE':
                        batch_layer_weights, x, _, x_hat, _ = self.model(
                            x, self.cfg.aggre, self.cfg.routing
                        )
                        self.layer_weights += batch_layer_weights.sum(dim=(0, 1)).detach().cpu().numpy()
                    else:
                        latents, x_hat = self.model(x)
                    
                    # Compute loss
                    mse_loss = Normalized_MSE_loss(x, x_hat)

                    if self.cfg.model in ['Vanilla', 'Gated', 'Crosscoder']:
                        l1_loss = L1_loss(latents)
                        loss = mse_loss + self.cfg.lamda * l1_loss
                    elif self.cfg.model == 'JumpReLU':
                        l0_loss = Step_func.apply(
                            latents, self.model.threshold, self.model.bandwidth
                        ).sum(dim=-1).mean()
                        loss = mse_loss + self.cfg.lamda * l0_loss
                    else:
                        loss = mse_loss
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    curr_loss = mse_loss.item()

                    if batch_idx % self.cfg.steps == 0:
                        unit_norm_decoder(self.model)
                
                # Logging
                if self.cfg.use_wandb:
                    wandb.log({'Normalized_MSE': curr_loss})
                    if self.cfg.model in ['Vanilla', 'Gated', 'JumpReLU', 'Crosscoder']:
                        counts = (latents != 0).sum(dim=-1).float().mean().item()
                        wandb.log({'Counts': counts})
                else:
                    if batch_idx % self.cfg.steps == 0:
                        logger.info(f'Epoch {epoch+1}/{self.cfg.num_epochs}, Batch {batch_idx+1}, Loss: {curr_loss:.4f}')
        
        # Finalize
        if self.cfg.use_wandb:
            if self.cfg.model == 'RouteSAE':
                log_layers(self.layer_weights)
            wandb.finish()

        unit_norm_decoder(self.model)
        os.makedirs('../SAE_models', exist_ok=True)
        save_path = f'../SAE_models/{self.title}.pt'
        torch.save(self.model.state_dict(), save_path)
        logger.info(f'Training complete. Model saved to {save_path}')
        
        return curr_loss

class Evaluater:
    """Evaluation pipeline for SAE models."""
    
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Load language model and data
        logger.info(f"Initializing Evaluater for {cfg.model}")
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
        
        sae_name = os.path.splitext(os.path.basename(cfg.SAE_path))[0]
        data_name = os.path.basename(cfg.data_path)
        self.title = f'{sae_name}_{data_name}_{cfg.metric}'
        
        self.config_dict = {
            'batch_size': cfg.batch_size,
            'infer_k': cfg.infer_k,
            'theta': cfg.theta
        }
        
        # Initialize SAE model
        if cfg.model == 'Vanilla':
            self.model = Vanilla(cfg.hidden_size, cfg.latent_size)
        elif cfg.model == 'Gated':
            self.model = Gated(cfg.hidden_size, cfg.latent_size)
        elif cfg.model == 'TopK':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
        elif cfg.model == 'JumpReLU':
            self.model = JumpReLU(cfg.hidden_size, cfg.latent_size)
        elif cfg.model == 'RouteSAE':
            self.model = RouteSAE(cfg.hidden_size, cfg.n_layers, cfg.latent_size, cfg.k)
            self.layer_weights = np.zeros(cfg.n_layers // 2 + 1, dtype=float)
        elif cfg.model == 'Crosscoder':
            self.model = Crosscoder(cfg.hidden_size, cfg.n_layers, cfg.latent_size)
        elif cfg.model == 'MLSAE':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
        elif cfg.model == 'Random':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
            cfg.random_layer = random.randint(4, 12)
        else:
            raise ValueError(
                f'Invalid model: {cfg.model}. Expected one of '
                f'[Vanilla, Gated, TopK, JumpReLU, RouteSAE, Crosscoder, MLSAE, Random]'
            )
        
        # Load pretrained weights
        logger.info(f"Loading SAE weights from {cfg.SAE_path}")
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        if cfg.model in ['Vanilla', 'Gated', 'TopK', 'JumpReLU'] and cfg.metric != 'NormMSE':
            self.hooked_module = self.language_model.get_submodule(f'model.layers.{cfg.layer-1}')
        
        self.num_batches = 0
        self.total_loss = 0.0
        self.total_counts = 0.0

    def DeltaCE(
        self, 
        logits_original: torch.Tensor, 
        logits_reconstruct: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy difference between original and reconstructed."""
        loss_original = F.cross_entropy(
            logits_original[:, :-1].reshape(-1, logits_original.size(-1)),  
            input_ids[:, 1:].reshape(-1)
        )
        loss_reconstruct = F.cross_entropy(
            logits_reconstruct[:, :-1].reshape(-1, logits_reconstruct.size(-1)),  
            input_ids[:, 1:].reshape(-1)
        )
        return loss_reconstruct - loss_original

    def KLDiv(
        self, 
        logits_original: torch.Tensor, 
        logits_reconstruct: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence between original and reconstructed logits."""
        probs_original = F.softmax(logits_original, dim=-1)
        log_probs_reconstruct = F.log_softmax(logits_reconstruct, dim=-1)
        return F.kl_div(log_probs_reconstruct, probs_original, reduction='batchmean')
    
    def Recovered(
        self, 
        logits_original: torch.Tensor, 
        logits_reconstruct: torch.Tensor, 
        logits_zero: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute recovery score: 1 - (reconstruct_loss - original_loss) / (zero_loss - original_loss)."""
        loss_original = F.cross_entropy(
            logits_original[:, :-1].reshape(-1, logits_original.size(-1)),  
            input_ids[:, 1:].reshape(-1)
        )
        loss_reconstruct = F.cross_entropy(
            logits_reconstruct[:, :-1].reshape(-1, logits_reconstruct.size(-1)),  
            input_ids[:, 1:].reshape(-1)
        )
        loss_zero = F.cross_entropy(
            logits_zero[:, :-1].reshape(-1, logits_zero.size(-1)),  
            input_ids[:, 1:].reshape(-1)
        )
        return 1 - (loss_reconstruct - loss_original) / (loss_zero - loss_original + 1e-8)
    
    @torch.no_grad()
    def run(self) -> float:
        """Execute evaluation loop."""
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)

        logger.info(f"Starting evaluation with metric: {self.cfg.metric}")
        
        # Special handling for MLSAE (multi-layer evaluation)
        if self.cfg.model == 'MLSAE':
            loss_vector = torch.zeros(9, device=self.device)
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids, attention_mask, outputs, hidden_states = get_outputs(
                    self.cfg, batch, self.language_model, self.device
                )
                logits_original = outputs.logits
                
                for layer_idx, hidden_state in enumerate(hidden_states):
                    x, _, _ = pre_process(hidden_state)
                    latents, x_hat = self.model(x, self.cfg.infer_k, self.cfg.theta)

                    if self.cfg.metric == 'NormMSE':
                        loss = Normalized_MSE_loss(x, x_hat)
                    else:
                        self.hooked_module = self.language_model.get_submodule(f'model.layers.{layer_idx}')
                        handles = hook_SAE(self.cfg, self.model, self.hooked_module)
                        logits_reconstruct = self.language_model(
                            input_ids=input_ids, attention_mask=attention_mask
                        ).logits
                        
                        for handle in handles:
                            handle.remove()

                        if self.cfg.metric == 'KLDiv':
                            torch.cuda.empty_cache()
                            loss = self.KLDiv(logits_original, logits_reconstruct)
                    
                    if self.cfg.use_wandb:
                        wandb.log({'Batch_loss': loss.item()})
                    loss_vector[layer_idx] += loss.item()
                
                self.num_batches += 1
            
            loss_vector /= self.num_batches
            avg_loss = loss_vector.mean().item()
            
            if self.cfg.use_wandb:
                wandb.log({'Avg_loss': avg_loss})
                wandb.finish()
            
            logger.info(f"Evaluation complete. Average loss: {avg_loss:.4f}")
            return avg_loss
        
        # Standard single-layer evaluation
        else:
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids, attention_mask, outputs, hidden_states = get_outputs(
                    self.cfg, batch, self.language_model, self.device
                )
                x, _, _ = pre_process(hidden_states)

                # Forward pass with SAE encoding
                if self.cfg.model == 'RouteSAE':
                    batch_layer_weights, x, latents, x_hat, _ = self.model(
                        x, self.cfg.aggre, self.cfg.routing, self.cfg.infer_k, self.cfg.theta
                    )
                    self.layer_weights += batch_layer_weights.sum(dim=(0, 1)).detach().cpu().numpy()
                else:
                    latents, x_hat = self.model(x, self.cfg.infer_k, self.cfg.theta)
                    batch_layer_weights = None
                
                # Compute metrics
                if self.cfg.metric == 'NormMSE': 
                    loss = Normalized_MSE_loss(x, x_hat)
                else:
                    if self.cfg.model not in ['Vanilla', 'Gated', 'TopK', 'JumpReLU', 'RouteSAE', 'Random']:
                        raise ValueError('Downstream tasks only supported for Vanilla, TopK, JumpReLU, RouteSAE, Random')
                    
                    logits_original = outputs.logits
                    
                    # Hook SAE intervention
                    if self.cfg.model == 'RouteSAE':
                        if self.cfg.routing == 'soft':
                            raise ValueError('RouteSAE with soft routing not supported on downstream tasks')
                        handles = hook_RouteSAE(
                            self.cfg, self.model, self.language_model, batch_layer_weights
                        )
                    elif self.cfg.model == 'Random':
                        self.hooked_module = self.language_model.get_submodule(
                            f'model.layers.{self.cfg.random_layer-1}'
                        )
                        handles = hook_SAE(self.cfg, self.model, self.hooked_module)
                    else:
                        handles = hook_SAE(self.cfg, self.model, self.hooked_module)
                    
                    logits_reconstruct = self.language_model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits
                    
                    for handle in handles:
                        handle.remove()
                    
                    if self.cfg.metric == 'DeltaCE':
                        loss = self.DeltaCE(logits_original, logits_reconstruct, input_ids)
                        del input_ids, attention_mask
                        torch.cuda.empty_cache()

                    elif self.cfg.metric == 'Recovered':
                        # Get zero-ablation baseline
                        if self.cfg.model == 'RouteSAE':
                            handles = hook_RouteSAE(
                                self.cfg, self.model, self.language_model, 
                                batch_layer_weights, is_zero=True
                            )
                        else:
                            handles = hook_SAE(
                                self.cfg, self.model, self.hooked_module, is_zero=True
                            )
                        
                        logits_zero = self.language_model(
                            input_ids=input_ids, attention_mask=attention_mask
                        ).logits
                        
                        for handle in handles:
                            handle.remove()
                        
                        loss = self.Recovered(
                            logits_original, logits_reconstruct, logits_zero, input_ids
                        )
                    
                    elif self.cfg.metric == 'KLDiv':
                        loss = self.KLDiv(logits_original, logits_reconstruct)
                        del input_ids, attention_mask
                        torch.cuda.empty_cache()

                self.num_batches += 1
                self.total_loss += loss.item()

                if self.cfg.use_wandb:
                    wandb.log({'Batch_loss': loss.item()})
                    if self.cfg.model in ['Vanilla', 'Gated', 'JumpReLU', 'Crosscoder'] or self.cfg.theta is not None:
                        counts = (latents != 0).sum(dim=-1).float().mean().item()
                        self.total_counts += counts
                        wandb.log({'Counts': counts})
                else:
                    if batch_idx % 10 == 0:
                        logger.info(f'Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        avg_loss = self.total_loss / self.num_batches
        
        if self.cfg.use_wandb:
            wandb.log({'Avg_loss': avg_loss})
            if self.cfg.model == 'RouteSAE':
                log_layers(self.layer_weights)
            if self.cfg.model in ['Vanilla', 'Gated', 'JumpReLU', 'Crosscoder'] or self.cfg.theta is not None:
                wandb.log({'Avg_counts': self.total_counts / self.num_batches})
            wandb.finish()
        
        logger.info(f"Evaluation complete. Average loss: {avg_loss:.4f}")
        return avg_loss



class Applier:
    """Feature extraction and intervention pipeline for SAE models."""
    
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Initialize SAE model
        if cfg.model == 'Vanilla':
            self.model = Vanilla(cfg.hidden_size, cfg.latent_size)
        elif cfg.model == 'Gated':
            self.model = Gated(cfg.hidden_size, cfg.latent_size)
        elif cfg.model == 'TopK':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
        elif cfg.model == 'JumpReLU':
            self.model = JumpReLU(cfg.hidden_size, cfg.latent_size)
        elif cfg.model == 'RouteSAE':
            self.model = RouteSAE(cfg.hidden_size, cfg.n_layers, cfg.latent_size, cfg.k)
        elif cfg.model == 'Crosscoder':
            self.model = Crosscoder(cfg.hidden_size, cfg.n_layers, cfg.latent_size)
        elif cfg.model == 'MLSAE':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
        elif cfg.model == 'Random':
            self.model = TopK(cfg.hidden_size, cfg.latent_size, cfg.k)
        else:
            raise ValueError(
                f'Invalid model: {cfg.model}. Expected one of '
                f'[Vanilla, Gated, TopK, JumpReLU, RouteSAE, Crosscoder, MLSAE, Random]'
            )
        
        logger.info(f"Loading SAE weights from {cfg.SAE_path}")
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def validate_triples(
        self, 
        triple_list: Optional[List[Tuple[int, float, int]]], 
        name: str
    ) -> None:
        """Validate intervention triples (latent_idx, val, mode)."""
        if triple_list is not None:
            for i, (a, b, c) in enumerate(triple_list):
                if not (0 <= a < self.cfg.latent_size):
                    raise ValueError(
                        f'{name}[{i}]: latent_idx {a} out of range [0, {self.cfg.latent_size})'
                    )
                if b <= 0:
                    raise ValueError(f'{name}[{i}]: value {b} must be > 0')
                if c not in [0, 1]:
                    raise ValueError(f'{name}[{i}]: mode {c} must be 0 (add/sub) or 1 (mul/div)')
                
    @torch.no_grad()
    def get_context(
        self, 
        threshold: float = 10.0, 
        max_length: int = 64, 
        max_per_token: int = 2, 
        lines: int = 4,  
        output_path: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        Extract top-k activation contexts for each latent feature.
        
        Returns:
            (total_latents, output_path): Number of features found and path to saved JSON
        """
        if output_path is None:
            output_path = f'../contexts/{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}_{threshold}.json'

        logger.info(f"Extracting feature contexts (threshold={threshold})")
        
        sentence_enders = {'.', '!', '?', '<|end_of_text|>', '"'}
        half_length = max_length // 2

        latent_context_map = defaultdict(lambda: defaultdict(list))

        def find_sentence_bounds(seq_pos: int, tokens: List[str]) -> Tuple[int, int]:
            """Find sentence boundaries around activated token."""
            start_pos = seq_pos
            while start_pos > 0 and tokens[start_pos - 1] not in sentence_enders:
                start_pos -= 1
            end_pos = seq_pos
            while end_pos < len(tokens) - 1 and tokens[end_pos] not in sentence_enders:
                end_pos += 1
            if end_pos < len(tokens):
                end_pos += 1  
            return start_pos, end_pos

        def process_and_store_context(
            latent_dim: int, seq_pos: int, activation_value: float, tokens: List[str]
        ) -> None:
            """Process activation and store context in heap."""
            start_pos, end_pos = find_sentence_bounds(seq_pos, tokens)
            sentence_tokens = tokens[start_pos:end_pos]
            sentence_length = len(sentence_tokens)

            if sentence_length > max_length:
                activated_token_idx = seq_pos - start_pos
                left_context_start = max(0, activated_token_idx - half_length)
                right_context_end = min(sentence_length, activated_token_idx + half_length + 1)
                context_tokens = sentence_tokens[left_context_start:right_context_end]
                activated_token_idx -= left_context_start
            else:
                context_tokens = sentence_tokens
                activated_token_idx = seq_pos - start_pos

            if not (0 <= activated_token_idx < len(context_tokens)):
                return

            context_tokens = context_tokens.copy()
            raw_token = context_tokens[activated_token_idx]
            context_tokens[activated_token_idx] = f'<ACTIVATED>{raw_token}</ACTIVATED>'

            while context_tokens and context_tokens[0] in ['<|end_of_text|>', ' ', '']:
                context_tokens.pop(0)
                activated_token_idx -= 1
            while context_tokens and context_tokens[-1] in ['<|end_of_text|>', ' ', '']:
                context_tokens.pop()

            if not context_tokens or not (0 <= activated_token_idx < len(context_tokens)):
                return

            context_text = self.tokenizer.convert_tokens_to_string(context_tokens).strip().strip('"')
            if not context_text:
                return

            activated_token_str = context_tokens[activated_token_idx]
            if activated_token_str.startswith('<ACTIVATED>') and activated_token_str.endswith('</ACTIVATED>'):
                raw_token = activated_token_str[len('<ACTIVATED>'):-len('</ACTIVATED>')].strip()
            else:
                raw_token = activated_token_str.strip()

            token_class = raw_token.lower()

            heap = latent_context_map[latent_dim][token_class]
            heapq.heappush(heap, (activation_value, context_text))
            if len(heap) > max_per_token:
                heapq.heappop(heap)

        self.tokenizer, self.language_model = get_language_model(self.cfg.model_path, self.device)
        dataloader = create_dataloader(self.cfg.data_path, self.tokenizer, self.cfg.batch_size, self.cfg.max_length)

        logger.info(f"Processing {len(dataloader)} batches")
        for batch_idx, batch in enumerate(dataloader):
            input_ids, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
            if self.cfg.model == 'MLSAE':
                hidden_states = hidden_states[-1]
            x, _, _ = pre_process(hidden_states)
            
            if self.cfg.model == 'RouteSAE':
                _, _, latents, _, _ = self.model(x, self.cfg.aggre, self.cfg.routing, self.cfg.infer_k, self.cfg.theta)
            else:
                latents, _ = self.model(x, self.cfg.infer_k, self.cfg.theta)
            batch_size, seq_len, _ = latents.shape
            positions = (latents > threshold)

            for i in range(batch_size):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
                latent_indices = torch.nonzero(positions[i], as_tuple=False)

                for activation in latent_indices:
                    seq_pos, latent_dim = activation.tolist()
                    activation_value = latents[i, seq_pos, latent_dim].item()
                    process_and_store_context(latent_dim, seq_pos, activation_value, tokens)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

        logger.info("Filtering and sorting contexts")
        filtered_latent_context = {}
        for latent_dim, token_dict in latent_context_map.items():
            # Skip latent token categories exceeding 32
            if len(token_dict) > 32:
                continue    
            total_contexts = sum(len(contexts) for contexts in token_dict.values())
            if total_contexts > lines:
                sorted_token_dict = {}
                for t_class, heap in token_dict.items():
                    contexts_list = list(heap)
                    contexts_list.sort(key=lambda x: x[0], reverse=True)
                    sorted_token_dict[t_class] = [
                        {'context': ctx, 'activation': act} for act, ctx in contexts_list
                    ]
                filtered_latent_context[latent_dim] = dict(sorted(sorted_token_dict.items()))

        total_latents = len(filtered_latent_context)
        sorted_latent_context = dict(sorted(filtered_latent_context.items()))

        output_data = {
            'total_latents': total_latents,
            'threshold': threshold,
            'max_length': max_length,
            'max_per_token': max_per_token,
            'lines': lines,
            'latent_context_map': sorted_latent_context,
        }
        save_json(output_data, output_path)
        logger.info(f"Extracted {total_latents} features with sufficient contexts")
        return total_latents, output_path
                
    @torch.no_grad()
    def clamp(
        self,
        max_length: int = 64,
        set_high: Optional[List[Tuple[int, float, int]]] = None, 
        set_low: Optional[List[Tuple[int, float, int]]] = None, 
        output_path: Optional[str] = None
    ) -> None:
        """
        Generate text with SAE feature interventions on multi-turn dialogues.
        
        Args:
            max_length: Maximum tokens to generate per turn
            set_high: List of (latent_idx, val, mode) tuples to amplify
            set_low: List of (latent_idx, val, mode) tuples to suppress
            output_path: Path to save results
        """
        if set_high is None and set_low is None:
            raise ValueError('Both set_high and set_low cannot be None at the same time.')
        if set_high is not None and set_low is not None:
            high_indices = {x[0] for x in set_high}
            low_indices = {x[0] for x in set_low}
            overlap = high_indices.intersection(low_indices)
            if overlap:
                raise ValueError(f'latent_dim index overlap in set_high and set_low: {overlap}')

        self.validate_triples(set_high, 'set_high')
        self.validate_triples(set_low, 'set_low')

        if output_path is None:
            output_path = f'../clamp/clamp_{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}.json'

        logger.info(f"Starting clamping experiment")
        
        conversation_path = os.path.join(self.cfg.data_path, 'conversation.json')
        if not os.path.exists(conversation_path):
            raise FileNotFoundError(f"Conversation file not found: {conversation_path}")
        
        with open(conversation_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)

        logger.info(f"Loaded {len(conversation_data)} conversations")
        
        self.tokenizer, self.language_model = get_language_model(self.cfg.model_path, self.device)
        self.chat_pipe = pipeline(
            task='text-generation',
            model=self.language_model,
            tokenizer=self.tokenizer,
            device=self.device,
            torch_dtype=torch.float32,
            pad_token_id=self.tokenizer.pad_token_id
        )
        if self.cfg.model == 'TopK':
            self.hooked_module = self.chat_pipe.model.get_submodule(f'model.layers.{self.cfg.layer-1}')

        output_data = {
            'sequences': 0,
            'max_length': max_length,
            'set_high': set_high,
            'set_low': set_low,
            'outputs': {}
        }

        def generate_multiturn_dialogue(
            conv_list: List[Dict[str, str]], 
            max_new_tokens: int = 32, 
            is_clamp: bool = False
        ) -> List[Dict[str, str]]:
            """Generate next turn in dialogue with optional SAE intervention."""
            prompt_str = ''
            handles = []
            for item in conv_list:
                role = item['role']
                content = item['content']
                prompt_str += f'{role.capitalize()}: {content}\n'
            
            if is_clamp:
                if self.cfg.model == 'TopK':
                    handles = hook_SAE(
                        cfg=self.cfg,
                        model=self.model,
                        hooked_module=self.hooked_module,
                        set_high=set_high,
                        set_low=set_low
                    )
                
                elif self.cfg.model == 'RouteSAE':
                    if self.cfg.routing == 'soft':
                        raise ValueError('RouteSAE with soft routing not supported for clamping')
                    
                    inputs = self.tokenizer(prompt_str, return_tensors='pt')
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    first_outputs = self.language_model(input_ids=input_ids, attention_mask=attention_mask)

                    hidden_states = torch.stack(first_outputs.hidden_states[1:], dim=0).permute(1, 2, 0, 3)
                    x, _, _ = pre_process(hidden_states)
                    batch_layer_weights, _, _, _, _ = self.model(
                        x, self.cfg.aggre, self.cfg.routing, self.cfg.infer_k, self.cfg.theta
                    )
    
                    handles = hook_RouteSAE(
                        cfg=self.cfg,
                        model=self.model,
                        language_model=self.chat_pipe.model,
                        batch_layer_weights=batch_layer_weights,
                        set_high=set_high,
                        set_low=set_low
                    )
            
            outputs = self.chat_pipe(
                prompt_str,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.1
            )

            if is_clamp:
                for h in handles:
                    h.remove()

            generated_text = outputs[0]['generated_text']
            new_reply = generated_text[len(prompt_str):].strip()
            conv_list.append({
                'role': 'assistant',
                'content': new_reply
            })
            return conv_list

        total_turns = 0
        for conv_id, turns in conversation_data.items():
            output_data['outputs'][conv_id] = []
            base_dialogue = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

            for turn_idx, turn in enumerate(turns):
                human_prompt = turn['Human']
                chosen = turn['chosen']
                rejected = turn['rejected']

                base_dialogue.append({'role': 'user', 'content': human_prompt})
                
                # Generate original response
                original_dialogue = [d.copy() for d in base_dialogue]
                original_dialogue = generate_multiturn_dialogue(original_dialogue, max_length, is_clamp=False)
                original_output_text = original_dialogue[-1]['content']

                # Generate clamped response
                clamped_dialogue = [d.copy() for d in base_dialogue]
                clamped_dialogue = generate_multiturn_dialogue(clamped_dialogue, max_length, is_clamp=True)
                clamped_output_text = clamped_dialogue[-1]['content']

                output_data['outputs'][conv_id].append({
                    'Human': human_prompt,
                    'chosen': chosen,
                    'rejected': rejected,
                    'original_output': original_output_text,  
                    'clamped_output': clamped_output_text     
                })
                base_dialogue.append({'role': 'assistant', 'content': original_output_text})
                total_turns += 1

        output_data['sequences'] = total_turns
        save_json(output_data, output_path)
        logger.info(f"Completed clamping for {len(conversation_data)} conversations ({total_turns} turns)")


class Interpreter:
    """GPT-4o based interpretation of SAE features."""
    
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        logger.info("Initialized Interpreter")

    def calculate_cost(self, input_text: str, output_text: str) -> float:
        """Calculate API cost based on token counts."""
        encoding = tiktoken.encoding_for_model(self.cfg.engine)
        num_input_tokens = len(encoding.encode(input_text))
        num_output_tokens = len(encoding.encode(output_text))
        if self.cfg.engine == 'gpt-4o':
            return num_input_tokens * 2.5 / 1_000_000 + num_output_tokens * 10 / 1_000_000
        elif self.cfg.engine == 'gpt-4o-mini':
            return num_input_tokens * 0.15 / 1_000_000 + num_output_tokens * 0.6 / 1_000_000
        else:
            return 0.0
    
    def construct_prompt(self, tokens_info: List[Dict[str, Any]]) -> str:
        """Construct GPT-4o prompt for feature interpretation."""
        prompt = (
            'We are analyzing the activation levels of features in a neural network, where each feature activates certain tokens in a text.\n'
            'Each token’s activation value indicates its relevance to the feature, with higher values showing stronger association. Features are categorized as:\n'
            'A. Low-level features, which are associated with word-level polysemy disambiguation (e.g., "crushed things", "Europe").\n'
            'B. High-level features, which are associated with long-range pattern formation (e.g., "enumeration", "one of the [number/quantifier]")\n'
            'C. Undiscernible features, which are associated with noise or irrelevant patterns.\n\n'
            'Your task is to classify the feature as low-level, high-level or undiscernible and give this feature a monosemanticity score based on the following scoring rubric:\n'
            'Activation Consistency\n'
            '5: Clear pattern with no deviating examples\n'
            '4: Clear pattern with one or two deviating examples\n'
            '3: Clear overall pattern but quite a few examples not fitting that pattern\n'
            '2: Broad consistent theme but lacking structure\n'
            '1: No discernible pattern\n'
            'Consider the following activations for a feature in the neural network.\n\n'
        )
        for info in tokens_info:
            prompt += f"Token: {info['token']} | Activation: {info['activation']} | Context: {info['context']}\n\n"
        prompt += (
            'Provide your response in the following fixed format:\n'
            'Feature category: [Low-level/High-level/Undiscernible]\n'
            'Score: [5/4/3/2/1]\n'
            'Explanation: [Your brief explanation]\n'
        )
        return prompt

    def chat_completion(
        self, client: AzureOpenAI, prompt: str, max_retry: int = 3
    ) -> str:
        """Call GPT-4o API with retry logic."""
        if client is None:
            raise ValueError('OpenAI client is not initialized')
        
        for attempt in range(1, max_retry + 1):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            'role': 'system',
                            'content': 'You are an assistant that helps explain the latent semantics of language models.',
                        },
                        {'role': 'user', 'content': prompt},
                    ],
                    model=self.cfg.engine,
                    max_tokens=128,  
                    temperature=0.1,
                )
                response_content = chat_completion.choices[0].message.content
                if response_content is None:
                    raise ValueError('API returned None response')
                return response_content.strip()
            except Exception as e:
                logger.warning(f"API call attempt {attempt}/{max_retry} failed: {e}")
                if attempt == max_retry:
                    logger.error('Failed to get response from OpenAI API after all retries')
                    raise
        raise RuntimeError('Failed to get response from OpenAI API')
    
    def run(
        self, 
        data_path: Optional[str] = None, 
        sample_latents: int = 100, 
        output_path: Optional[str] = None
    ) -> Tuple[float, float, float]:
        """
        Run GPT-4o interpretation on sampled features.
        
        Returns:
            (avg_score, low_level_score, high_level_score)
        """
        if data_path is None:
            data_path = self.cfg.data_path

        if output_path is None:
            output_path = f'../interpret/interp_{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}.json'

        logger.info(f"Loading context data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        latent_context_map = data.get('latent_context_map', {})
        all_latents = list(latent_context_map.keys())
        sample_size = min(sample_latents, len(all_latents))
        sampled_indices = random.sample(range(len(all_latents)), sample_size)
        sampled_latents = [all_latents[i] for i in sorted(sampled_indices)]

        logger.info(f"Sampled {sample_size} features for interpretation")
        logger.info(f"Initializing OpenAI client (engine: {self.cfg.engine})")
        
        client = AzureOpenAI(
            azure_endpoint=self.cfg.api_base,
            api_version=self.cfg.api_version,
            api_key=self.cfg.api_key,
        )

        cost = 0.0
        results = {}
        total_score = 0.0
        scored_features = 0

        low_level_features = 0
        concrete_total_score = 0.0
        high_level_features = 0
        abstract_total_score = 0.0

        pattern = re.compile(
            r"Feature category:\s*(?P<category>low-level|high-level|undiscernible)\s*\n"
            r"Score:\s*(?P<score>[1-5])\s*\n"
            r"Explanation:\s*(?P<explanation>.+)",
            re.IGNORECASE | re.DOTALL,
        )

        for idx, latent in enumerate(sampled_latents, 1):
            try:
                latent_id = int(latent)
            except ValueError:
                logger.warning(f"Invalid latent ID {latent}. Skipping.")
                results[latent] = {
                    'category': None,
                    'score': None,
                    'explanation': "Invalid latent ID.",
                }
                continue
            
            token_contexts = latent_context_map[latent]
            tokens_info = []
            for token_class, contexts in token_contexts.items():
                for context in contexts:
                    token = token_class
                    if token.startswith('ġ'):
                        token = ' ' + token[1:]
                    tokens_info.append({
                        'token': token,
                        'context': context['context'],
                        'activation': context['activation'],
                    })

            prompt = self.construct_prompt(tokens_info)
            try:
                response = self.chat_completion(client, prompt)
                cost += self.calculate_cost(prompt, response)

                match = pattern.search(response)
                if match:
                    category = match.group('category').strip().lower()
                    score = int(match.group('score'))
                    explanation = match.group('explanation').strip()

                    if 1 <= score <= 5 and category in ['low-level', 'high-level', 'undiscernible']:
                        results[latent_id] = {
                            'category': category,
                            'score': score,
                            'explanation': explanation,
                        }
                        total_score += score
                        scored_features += 1

                        if category == 'low-level':
                            low_level_features += 1
                            concrete_total_score += score
                        elif category == 'high-level':
                            high_level_features += 1
                            abstract_total_score += score
                    else:
                        logger.warning(f"Invalid category '{category}' or score '{score}' for latent {latent_id}")
                        results[latent_id] = {
                            'category': None,
                            'score': None,
                            'explanation': "Invalid category or score provided.",
                        }
                else:
                    logger.warning(f"Failed to parse response for latent {latent_id}")
                    results[latent_id] = {
                        'category': None,
                        'score': None,
                        'explanation': "Failed to parse response.",
                    }

            except Exception as e:
                logger.error(f"Error processing latent {latent_id}: {e}")
                results[latent_id] = {
                    'category': None,
                    'score': None,
                    'explanation': "Error during processing.",
                }
                continue
            
            if idx % 10 == 0:
                logger.info(f"Processed {idx}/{sample_size} features")

        avg_score = total_score / scored_features if scored_features > 0 else 0.0
        low_level_score = concrete_total_score / low_level_features if low_level_features > 0 else 0.0
        high_level_score = abstract_total_score / high_level_features if high_level_features > 0 else 0.0

        logger.info(f"Interpretation complete: {scored_features} features scored")
        logger.info(f"Avg score: {avg_score:.2f}, Low-level: {low_level_score:.2f}, High-level: {high_level_score:.2f}")
        logger.info(f"Total API cost: ${cost:.4f}")
        
        output_data = {
            'cost': cost,
            'engine': self.cfg.engine,
            'features_scored': scored_features,
            'average_score': avg_score,
            'low_level_features': low_level_features,
            'low_level_score': low_level_score,
            'high_level_features': high_level_features,
            'high_level_score': high_level_score,
            'results': results,
        }
        save_json(output_data, output_path)
        return avg_score, low_level_score, high_level_score


class SAE_pipeline:
    """End-to-end pipeline for SAE training, evaluation, and interpretation."""
    
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.title = f'{cfg.language_model}_{cfg.pipe_data_path[0].split("/")[-1]}_{cfg.latent_size}'

        if cfg.model == 'Vanilla':
            self.title = f'L{cfg.layer}_VL{cfg.lamda}_{self.title}'
        
        elif cfg.model == 'Gated':
            self.title = f'L{cfg.layer}_GL{cfg.lamda}_{self.title}'

        elif cfg.model == 'TopK':
            self.title = f'L{cfg.layer}_K{cfg.k}_{self.title}'

        elif cfg.model == 'JumpReLU':
            self.title = f'L{cfg.layer}_JL{cfg.lamda}_{self.title}'

        elif cfg.model == 'RouteSAE':
            self.title = f'{cfg.aggre}_{cfg.routing}_K{cfg.k}_{self.title}'
        
        elif cfg.model == 'MLSAE':
            self.title = f'ML_K{cfg.k}_{self.title}'
        
        elif cfg.model == 'Random':
            self.title = f'RDM_K{cfg.k}_{self.title}'
        else:
            raise ValueError(
                f'Invalid model: {cfg.model}. Expected one of '
                f'[Vanilla, Gated, TopK, JumpReLU, RouteSAE, MLSAE, Random]'
            )
        
        self.cfg.SAE_path = f'../SAE_models/{self.title}.pt'
        self.result_dict = {}
        logger.info(f"Initialized SAE pipeline: {self.title}")
    
    def train(self) -> None:
        """Execute training phase."""
        logger.info("Starting training phase")
        set_seed(self.cfg.seed)
        self.cfg.data_path = self.cfg.pipe_data_path[0]
        self.cfg.wandb_project = self.cfg.pipe_project[0]
        trainer = Trainer(self.cfg)
        self.result_dict['Train_Loss'] = trainer.run()
        del trainer
        torch.cuda.empty_cache()
        logger.info("Training phase complete")
    
    def evaluate(self) -> None:
        """Execute evaluation phase."""
        logger.info("Starting evaluation phase")
        self.cfg.data_path = self.cfg.pipe_data_path[1]
        self.cfg.wandb_project = self.cfg.pipe_project[1]
        self.cfg.batch_size = self.cfg.batch_size // 2
        for metric in ('NormMSE', 'KLDiv'):
            self.cfg.metric = metric
            if metric != 'NormMSE':
                self.cfg.routing = 'hard'
            evaluater = Evaluater(self.cfg)
            self.result_dict[f'{metric}'] = evaluater.run()
            del evaluater
            torch.cuda.empty_cache()
        logger.info("Evaluation phase complete")

    def apply(self) -> None:
        """Execute feature extraction phase."""
        logger.info("Starting feature extraction phase")
        self.cfg.data_path = self.cfg.pipe_data_path[2]
        applier = Applier(self.cfg)
        self.result_dict[f'Features'], self.context_path = applier.get_context(
            threshold=15, max_length=64, max_per_token=2, lines=4
        )
        del applier
        torch.cuda.empty_cache()
        logger.info("Feature extraction phase complete")

    def interpret(self) -> None:
        """Execute interpretation phase."""
        logger.info("Starting interpretation phase")
        self.cfg.data_path = self.context_path
        interpreter = Interpreter(self.cfg)
        score, low_level_score, high_level_score = interpreter.run(sample_latents=100)
        self.result_dict[f'Score'] = score
        self.result_dict[f'Low_level_Score'] = low_level_score
        self.result_dict[f'High_level_Score'] = high_level_score
        del interpreter
        logger.info("Interpretation phase complete")

    def run(self) -> None:
        """Execute full pipeline: train → evaluate → extract → interpret."""
        logger.info("Starting SAE pipeline execution")
        start_time = time.time()

        self.train()
        self.evaluate()
        self.apply()
        self.interpret()

        end_time = time.time()
        self.result_dict['Runtime'] = (end_time - start_time) / 3600

        logger.info(f"Pipeline complete. Total runtime: {self.result_dict['Runtime']:.2f} hours")
        
        if self.cfg.use_wandb:
            wandb_init(self.cfg.pipe_project[2], self.result_dict, self.title)
            wandb.finish()

