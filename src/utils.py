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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model import *
from typing import List, Union
from torch.optim import Adam
from openai import AzureOpenAI
from collections import defaultdict
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.hooks import RemovableHandle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Configuration on sparse autoencoders pipeline')

    parser.add_argument('--language_model', type=str, required=True, help='Language model name (e.g., "Llama-3.2-1B")')
    parser.add_argument('--model_path', type=str, required=True, help='Language model path')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., "Vanilla", "TopK", "RouteSAE")')
    parser.add_argument('--hidden_size', type=int, required=True, help='Dimensionality of the input residual stream activation')
    parser.add_argument('--latent_size', type=int, required=True, help='Size of the latent space')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--max_length', type=int, required=True, help='Maximum sequence length')
    parser.add_argument('--device', type=str, required=True, help='Device to run the model on (e.g., "cuda:0", "cpu")')
    parser.add_argument('--use_wandb', type=int, required=True, help='Whether to use wandb for logging')

    parser.add_argument('--data_path', type=str, required=False, help='Path to the dataset')
    parser.add_argument('--wandb_project', type=str, required=False, help='Wandb project name')
    parser.add_argument('--num_epochs', type=int, required=False, help='Number of training epochs')
    parser.add_argument('--k', type=int, required=False, help='Hyperparameter k for TopK')
    parser.add_argument('--lr', type=float, required=False, help='Learning rate for training')
    parser.add_argument('--betas', type=float, nargs=2, required=False, help='Beta values for the optimizer')
    parser.add_argument('--seed', type=int, required=False, help='Random seed for reproducibility')
    parser.add_argument('--layer', type=int, required=False, help='Target layer index, start with 1')
    parser.add_argument('--steps', type=int, required=False, help='Number of step batches for unit norm decoder')
    parser.add_argument('--lamda', type=float, required=False, help='Hyperparameter for the L1 regularization')

    parser.add_argument('--n_layers', type=int, required=False, help='Number of layers in the language model')
    parser.add_argument('--aggre', type=str, required=False, help='Aggregation strategy for Router input (e.g., "sum", "mean")')
    parser.add_argument('--routing', type=str, required=False, help='Routing strategy for RouteSAE (e.g., "hard", "soft")')

    parser.add_argument('--SAE_path', type=str, required=False, help='Path to the trained SAE model file')
    parser.add_argument('--metric', type=str, required=False, help='Evaluation metric (e.g., "NormMSE", "DeltaCE", "KLDiv")')
    parser.add_argument('--infer_k', type=int, required=False, help='Number of topK latent activations during inference')
    parser.add_argument('--theta', type=float, required=False, help='Threshold for the JumpReLU activation function during inference')

    parser.add_argument('--api_base', type=str, required=False, help='OpenAI api base')
    parser.add_argument('--api_key', type=str, required=False, help='OpenAI api key')
    parser.add_argument('--api_version', type=str, required=False, help='OpenAI api version')
    parser.add_argument('--engine', type=str, required=False, help='OpenAI api engine (e.g., "gpt-4o", "gpt-4o-mini")')

    parser.add_argument('--pipe_data_path', type=str, nargs='+', required=False, help='Path to the pipe dataset: train, eval and apply')
    parser.add_argument('--pipe_project', type=str, nargs='+', required=False, help='Wandb project name for pipe: train, eval and pipe')

    args = parser.parse_args()
    return args


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def load_config(config_path: str) -> Config:
    # load config from yaml file
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


class OpenWebTextDataset(Dataset):
    def __init__(self, 
                 folder_path: str, 
                 tokenizer: AutoTokenizer, 
                 max_length: int,
                 keyword: str = 'text'):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keyword = keyword
        self.file_list = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
        self.data = self.load_data()

    def load_data(self):
        data = []
        for file_name in self.file_list:
            file_path = os.path.join(self.folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        words = record.get(self.keyword, '').split()
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
                        print(f'Error decoding JSON in file: {file_path}')
                        continue
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def create_dataloader(
    folder_path: str, 
    tokenizer: AutoTokenizer, 
    batch_size: int, 
    max_length: int,
    keyword: str = 'text'
) -> DataLoader:
    dataset = OpenWebTextDataset(folder_path, tokenizer, max_length, keyword)
    
    def collate_fn(batch):
        input_ids = torch.stack([item[0] for item in batch])
        attention_mask = torch.stack([item[1] for item in batch])
        return input_ids, attention_mask

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)
    return dataloader


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wandb_init(project: str, config: dict, name: str) -> None:
    wandb.init(
        project=project,
        config=config,
        name=name
    )


def get_language_model(model_path: str, device: torch.device) -> tuple:
    '''
    Loads and returns a tokenizer and a language model from the specified model path.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    language_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, return_dict_in_generate=True, output_hidden_states=True
    ).to(device)
    return tokenizer, language_model


def get_outputs(
    cfg, batch: tuple, language_model: nn.Module, device: torch.device
) -> tuple:
    '''
    Extracts model outputs and hidden states from a given batch and language model.
    '''
    input_ids, attention_mask = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = language_model(input_ids=input_ids, attention_mask=attention_mask)
    if cfg.model in ['Vanilla', 'Gated', 'TopK', 'JumpReLU']:
        hidden_states = outputs.hidden_states[cfg.layer]
    elif cfg.model == 'MLSAE':
        hidden_states = outputs.hidden_states[3:12]
    elif cfg.model == 'Random':
        cfg.random_layer = random.randint(1, 16)
        hidden_states = outputs.hidden_states[cfg.random_layer]
    else:
        start_layer = cfg.n_layers // 4
        end_layer = cfg.n_layers * 3 // 4 + 1
        hidden_states = torch.stack(outputs.hidden_states[start_layer:end_layer], dim=0).permute(1, 2, 0, 3)
    return input_ids, attention_mask, outputs, hidden_states


def pre_process(hidden_stats: torch.Tensor, eps: float = 1e-6) -> tuple:
    '''
    :param hidden_stats: Hidden states (shape: [batch, max_length, hidden_size]).
    :param eps: Epsilon value for numerical stability.
    '''
    mean = hidden_stats.mean(dim=-1, keepdim=True)
    std = hidden_stats.std(dim=-1, keepdim=True)
    x = (hidden_stats - mean) / (std + eps)
    return x, mean, std


def L1_loss(latents: torch.Tensor) -> torch.Tensor:
    '''
    :param latents: SAE latents activation (shape: [batch, max_length, latent_size])
    :return: scalar Mean L1 loss (shape: [1])
    '''
    return latents.abs().mean()


def Normalized_MSE_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return (((x_hat - x) ** 2).mean(dim=-1) / (x**2).mean(dim=-1)).mean()


@torch.no_grad()
def unit_norm_decoder(model: nn.Module) -> None:
    '''
    Normalize the decoder weights to unit norm
    '''
    if isinstance(model, (Vanilla, Gated, TopK, JumpReLU)):
        model.decoder.weight.data /= model.decoder.weight.data.norm(dim=0)
    elif isinstance(model, RouteSAE):
        model.sae.decoder.weight.data /= model.sae.decoder.weight.data.norm(dim=0)
    elif isinstance(model, Crosscoder):
        for i in range(len(model.decoder)):
            model.decoder[i].weight.data /= model.decoder[i].weight.data.norm(dim=0)


def log_layers(layer_weights: np.ndarray):
    '''
    log the layer weights to wandb
    '''
    data = [[i, value] for i, value in enumerate(layer_weights[:], start=1)]
    table = wandb.Table(data=data, columns=['Layer', 'Weight'])
    wandb.log({'Layer Weights':wandb.plot.bar(table=table, label='Layer', value='Weight', title='Layer Weights')})


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'Saved data to {path}')


def convert_hh_jsonl(input_file: str, output_path: str):
    text_file = os.path.join(output_path, 'text.jsonl')
    conv_file = os.path.join(output_path, 'conversation.json')

    conversations = {}
    index = 1

    def parse_conversation(conv_text: str):
        conv_text = conv_text.strip()
        if not conv_text:
            return []
        
        parts = conv_text.split('Human:')
        turns = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if 'Assistant:' not in part:
                continue
            human_part, assistant_part = part.split('Assistant:', 1)
            human_text = human_part.strip()
            assistant_text = assistant_part.strip()
            turns.append({
                'Human': human_text,
                'Assistant': assistant_text
            })
        return turns

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(text_file, 'w', encoding='utf-8') as ftext:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chosen = data.get('chosen', '')
            rejected = data.get('rejected', '')

            chosen_turns = parse_conversation(chosen)
            rejected_turns = parse_conversation(rejected)
            num_turns = min(len(chosen_turns), len(rejected_turns))

            # conversation.json
            conv_list = []
            for i in range(num_turns):
                conv_list.append({
                    'Human': chosen_turns[i]['Human'],
                    'chosen': chosen_turns[i]['Assistant'],
                    'rejected': rejected_turns[i]['Assistant']
                })
            conversations[str(index)] = conv_list

            # text.jsonl
            combined_lines = []
            for i in range(num_turns):
                human_line = f"Human: {chosen_turns[i]['Human']}"
                chosen_assistant_line = f"Assistant: {chosen_turns[i]['Assistant']}"
                rejected_assistant_line = f"Assistant: {rejected_turns[i]['Assistant']}"

                combined_lines.append(human_line)
                combined_lines.append(chosen_assistant_line)
                if chosen_assistant_line != rejected_assistant_line:
                    combined_lines.append(rejected_assistant_line)

            merged_text = '\n'.join(combined_lines)
            ftext.write(json.dumps({'text': merged_text}, ensure_ascii=False) + '\n')
            index += 1
    save_json(conversations, conv_file)


def hook_SAE(
    cfg,
    model: Union[TopK, RouteSAE, JumpReLU],
    hooked_module: nn.Module,
    set_high: List[tuple[int, float, int]]=None, 
    set_low: List[tuple[int, float, int]]=None,
    is_zero: bool=False
) -> List[RemovableHandle]:
    '''
    :param set_high: A list where each element is a tuple (a, b, c)
        a: Index of the latent variable
        b: Adjustment value
        c: Adjustment method (0 for addition, 1 for multiplication)
        For set_high:
            c=0: Original value + b
            c=1: Original value * b
    :param set_low: A list where each element is a tuple (a, b, c)
        For set_low:
            c=0: Original value - b
            c=1: Original value / b
    '''
    def hook(module: torch.nn.Module, _, outputs):
        if isinstance(outputs, tuple):
            unpack_outputs = list(outputs)
        else:
            unpack_outputs = [outputs]
        
        if is_zero:
            unpack_outputs[0] = torch.zeros_like(unpack_outputs[0])

        else:
            x, mu, std = pre_process(unpack_outputs[0])
            latents = model.encode(x, cfg.infer_k, cfg.theta)

            if set_high:
                for (latent_idx, val, mode) in set_high:
                    if mode == 0:
                        latents[..., latent_idx] = latents[..., latent_idx] + val
                    elif mode == 1:
                        latents[..., latent_idx] = latents[..., latent_idx] * val

            if set_low:
                for (latent_idx, val, mode) in set_low:
                    if mode == 0:
                        latents[..., latent_idx] = latents[..., latent_idx] - val
                    elif mode == 1:
                        if val == 0:
                            continue
                        latents[..., latent_idx] = latents[..., latent_idx] / val

            x_hat = model.decode(latents)
            unpack_outputs[0] = x_hat * std + mu

        if isinstance(outputs, tuple):
            return tuple(unpack_outputs)
        else:
            return unpack_outputs[0]

    handles = [hooked_module.register_forward_hook(hook)]
    return handles


class RouteHook:
    def __init__(
        self,
        cfg,
        layer_idx: int,
        model: RouteSAE,
        batch_layer_weights: torch.Tensor,
        set_high: List[tuple[int, float, int]] = None,
        set_low: List[tuple[int, float, int]] = None,
        is_zero: bool = False  
    ) -> None:
        '''
        :param layer_idx: Index of the current layer
        :param model: Trained RouteSAE instance (optional, depending on is_zero)
        :param batch_layer_weights: A one-hot tensor of shape (batch_size, max_length, num_layers), indicating the layers that need to be replaced
        :param set_high: Same usage as set_high in TopK
        :param set_low: Same usage as set_low in TopK
        :param is_zero: Whether to replace the activations with zeros
        '''
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
        '''
        forward_hook function, used for processing and replacing after the specified layer's output.
        '''
        # Get the mask indicating which layers need to be replaced
        layer_mask = self.batch_layer_weights[:, :, self.layer_idx - self.model.start_layer + 1].bool()

        if not layer_mask.any():
            return outputs

        if isinstance(outputs, tuple):
            outputs = list(outputs)
            output_tensor = outputs[0]
        else:
            output_tensor = outputs
        
        if output_tensor.shape[1] != layer_mask.shape[1]:
            return outputs

        if self.is_zero:  
            replace_mask = layer_mask.unsqueeze(-1).expand(-1, -1, output_tensor.size(-1))
            output_tensor = output_tensor.clone()
            output_tensor[replace_mask] = 0
        else:
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
                elif mode == 1:
                    if val != 0:
                        latents[..., idx] /= val

            x_hat = self.model.sae.decode(latents)
            reconstruct = x_hat * std + mu

            replace_mask = layer_mask.unsqueeze(-1).expand(-1, -1, reconstruct.size(-1))
            output_tensor = output_tensor.clone()
            output_tensor[replace_mask] = reconstruct[replace_mask]

        if isinstance(outputs, list):
            outputs[0] = output_tensor
            return tuple(outputs)
        else:
            return output_tensor


def hook_RouteSAE(
    cfg,
    model: RouteSAE,
    language_model: nn.Module,
    batch_layer_weights: torch.Tensor,
    set_high: List[tuple[int, float, int]] = None,
    set_low: List[tuple[int, float, int]] = None,
    is_zero: bool = False 
) -> List[RemovableHandle]:
    '''
    :param model: Trained RouteSAE instance (optional, depending on is_zero)
    :param language_model: Language model
    :param batch_layer_weights: A tensor of shape (batch_size, max_length, num_layers), indicating the layers that need to be replaced
    :param set_high: Same usage as set_high in TopK
    :param set_low: Same usage as set_low in TopK
    :param is_zero: Whether to replace the activations with zeros
    :return: Registered hook handles
    '''
    handles = []
    num_layers = batch_layer_weights.size(-1)

    for layer_idx in range(model.start_layer - 1, num_layers + model.start_layer - 1):
        if batch_layer_weights[:, :, layer_idx - model.start_layer + 1].any():
            layer_name = f'model.layers.{layer_idx}'
            try:
                module = language_model.get_submodule(layer_name)
            except AttributeError:
                raise ValueError(f'Submodule {layer_name} not found in language_model.')

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
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, max_lr):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.max_lr = max_lr
        lr_lambda = self.lr_lambda
        super().__init__(optimizer, lr_lambda)

    def lr_lambda(self, step: int):
        if step < self.num_warmup_steps:
            return float(step) / float(max(1, self.num_warmup_steps))
        elif step < self.num_training_steps - self.num_training_steps // 5:
            return 1.0
        else:
            decay_steps = self.num_training_steps - self.num_warmup_steps - self.num_training_steps // 5
            return max(0.0, float(self.num_training_steps - step - self.num_warmup_steps) / float(decay_steps))


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
        self.title = f'{cfg.language_model}_{cfg.data_path.split("/")[-1]}_{cfg.latent_size}'
        self.config_dict = {
            'batch_size': self.cfg.batch_size,
            'num_epochs': self.cfg.num_epochs,
            'lr': self.cfg.lr,
            'steps': self.cfg.steps
        }

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
            raise ValueError(f'Invalid model name: {cfg.model} Expected one of [Vanilla, Gated, TopK, JumpReLU, RouteSAE, Crosscoder, MLSAE, Random]')
        
        self.model.to(self.device)
        self.model.train()
        self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, betas=cfg.betas)
        
        num_training_steps = cfg.num_epochs * len(self.dataloader)
        num_warmup_steps = int(num_training_steps * 0.05)
        self.scheduler = LinearWarmupLR(self.optimizer, num_warmup_steps, num_training_steps, cfg.lr)
    
    def run(self):
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)
        curr_loss = 0.0
        unit_norm_decoder(self.model)
        for epoch in range(self.cfg.num_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                _, _, _, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
                if self.cfg.model == 'MLSAE':
                    for hidden_state in hidden_states:
                        x, _, _ = pre_process(hidden_state)
                        _, x_hat = self.model(x)
                        mse_loss = Normalized_MSE_loss(x, x_hat)
                        loss = mse_loss
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        curr_loss = mse_loss.item()

                    self.scheduler.step()
                    unit_norm_decoder(self.model)

                else:
                    x, _, _ = pre_process(hidden_states)

                    if self.cfg.model == 'RouteSAE':
                        batch_layer_weights, x, _, x_hat, _ = self.model(x, self.cfg.aggre, self.cfg.routing)
                        self.layer_weights[:] += batch_layer_weights.sum(dim=(0, 1)).detach().cpu().numpy()
                    else:
                        latents, x_hat = self.model(x)
                    
                    mse_loss = Normalized_MSE_loss(x, x_hat)

                    if self.cfg.model in ['Vanilla', 'Gated', 'Crosscoder']:
                        l1_loss = L1_loss(latents)
                        loss = mse_loss + self.cfg.lamda * l1_loss
                    
                    elif self.cfg.model == 'JumpReLU':
                        l0_loss = Step_func.apply(latents, self.model.threshold, self.model.bandwidth).sum(dim=-1).mean()
                        # l0_loss = Step_func.apply(pre_acts, self.model.threshold, self.model.bandwidth).sum(dim=-1).mean()
                        loss = mse_loss + self.cfg.lamda * l0_loss
                    else:
                        loss = mse_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    self.scheduler.step()
                    curr_loss = mse_loss.item()

                    if batch_idx % self.cfg.steps == 0:
                        unit_norm_decoder(self.model)
                
                if self.cfg.use_wandb:
                    wandb.log({'Normalized_MSE': curr_loss})
                    if self.cfg.model in ['Vanilla', 'Gated', 'JumpReLU', 'Crosscoder']:
                        counts = (latents != 0).sum(dim=-1).float().mean().item()
                        wandb.log({'Counts': counts})
                else:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, Loss: {curr_loss}')
                
        if self.cfg.use_wandb:
            if self.cfg.model == 'RouteSAE':
                log_layers(self.layer_weights)
            wandb.finish()

        unit_norm_decoder(self.model)
        torch.save(self.model.state_dict(), f'../SAE_models/{self.title}.pt')
        return curr_loss

class Evaluater:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.tokenizer, self.language_model = get_language_model(cfg.model_path, self.device)
        self.dataloader = create_dataloader(cfg.data_path, self.tokenizer, cfg.batch_size, cfg.max_length)
        self.title = f'{os.path.splitext(os.path.basename(cfg.SAE_path))[0]}_{os.path.basename(cfg.data_path)}_{cfg.metric}'
        self.config_dict = {
            'batch_size': self.cfg.batch_size,
            'infer_k': self.cfg.infer_k,
            'theta': self.cfg.theta
        }
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
            self.cfg.random_layer = random.randint(4, 12)
        
        else:
            raise ValueError(f'Invalid model name: {cfg.model} Expected one of [Vanilla, Gated, TopK, JumpReLU, RouteSAE, Crosscoder, MLSAE, Random]')
        
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        if cfg.model in ['Vanilla', 'Gated', 'TopK', 'JumpReLU'] and cfg.metric != 'NormMSE':
            self.hooked_module = self.language_model.get_submodule(f'model.layers.{cfg.layer-1}')           
        
        self.num_batches = 0
        self.total_loss = 0.0
        self.total_counts = 0.0

    def DeltaCE(
        self, logits_original: torch.Tensor, logits_reconstruct: torch.Tensor, input_ids
    ) -> torch.Tensor:
        loss_original = F.cross_entropy(
            logits_original[:, :-1, :].reshape(-1, logits_original.size(-1)),  
            input_ids[:, 1:].reshape(-1) 
        )

        loss_reconstruct = F.cross_entropy(
            logits_reconstruct[:, :-1, :].reshape(-1, logits_reconstruct.size(-1)),  
            input_ids[:, 1:].reshape(-1)  
        )
        loss = loss_reconstruct - loss_original
        return loss

    def KLDiv(
        self, logits_original: torch.Tensor, logits_reconstruct: torch.Tensor
    ) -> torch.Tensor:
        probs_original = F.softmax(logits_original, dim=-1)
        log_probs_reconstruct = F.log_softmax(logits_reconstruct, dim=-1)
        loss = F.kl_div(log_probs_reconstruct, probs_original, reduction='batchmean')
        return loss
    
    def Recovered(
        self, 
        logits_original: torch.Tensor, 
        logits_reconstruct: torch.Tensor, 
        logits_zero: torch.Tensor, 
        input_ids
    ) -> torch.Tensor:
        loss_original = F.cross_entropy(
            logits_original[:, :-1, :].reshape(-1, logits_original.size(-1)),  
            input_ids[:, 1:].reshape(-1) 
        )

        loss_reconstruct = F.cross_entropy(
            logits_reconstruct[:, :-1, :].reshape(-1, logits_reconstruct.size(-1)),  
            input_ids[:, 1:].reshape(-1)  
        )

        loss_zero = F.cross_entropy(
            logits_zero[:, :-1, :].reshape(-1, logits_zero.size(-1)),  
            input_ids[:, 1:].reshape(-1)  
        )
        loss = 1 - (loss_reconstruct - loss_original) / (loss_zero - loss_original)
        return loss
    
    @torch.no_grad()
    def run(self):
        if self.cfg.use_wandb:
            wandb_init(self.cfg.wandb_project, self.config_dict, self.title)

        if self.cfg.model == 'MLSAE':
            loss_vector = torch.zeros(9, device=self.device)
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids, attention_mask, outputs, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)
                logits_original = outputs.logits
                for layer_idx, hidden_state in enumerate(hidden_states):
                    x, _, _ = pre_process(hidden_state)
                    latents, x_hat = self.model(x, self.cfg.infer_k, self.cfg.theta)
                    batch_layer_weights = None

                    if self.cfg.metric == 'NormMSE':
                        loss = Normalized_MSE_loss(x, x_hat)
                    else:
                        self.hooked_module = self.language_model.get_submodule(f'model.layers.{layer_idx}')
                        handles = hook_SAE(self.cfg, self.model, self.hooked_module)

                        logits_reconstruct = self.language_model(input_ids=input_ids, attention_mask=attention_mask).logits

                        for handle in handles:
                            handle.remove()

                        if self.cfg.metric == 'KLDiv':
                            torch.cuda.empty_cache()
                            loss = self.KLDiv(logits_original, logits_reconstruct)
                    
                    wandb.log({'Batch_loss': loss.item()})
                    loss_vector[layer_idx] += loss.item()
                self.num_batches += 1
            
            loss_vector /= self.num_batches
            wandb.log({'Avg_loss': loss_vector.mean().item()})
            wandb.finish()
            return loss_vector.mean().item()
        
        else:
            for batch_idx, batch in enumerate(self.dataloader):
                input_ids, attention_mask, outputs, hidden_states = get_outputs(self.cfg, batch, self.language_model, self.device)                   
                x, _, _ = pre_process(hidden_states)

                if self.cfg.model == 'RouteSAE':
                    batch_layer_weights, x, latents, x_hat, _ = self.model(
                        x, self.cfg.aggre, self.cfg.routing, self.cfg.infer_k, self.cfg.theta
                    )
                    self.layer_weights[:] += batch_layer_weights.sum(dim=(0, 1)).detach().cpu().numpy()

                else:
                    latents, x_hat = self.model(x, self.cfg.infer_k, self.cfg.theta)
                    batch_layer_weights = None
                
                if self.cfg.metric == 'NormMSE': 
                    loss = Normalized_MSE_loss(x, x_hat)
                
                else:
                    assert self.cfg.model in ['Vanilla', 'Gated', 'TopK', 'JumpReLU', 'RouteSAE', 'Random'], \
                        'Downstream tasks only supported for Vanilla, TopK, JumpReLU, RouteSAE models'
                    logits_original = outputs.logits
                    if self.cfg.model == 'RouteSAE':
                        assert self.cfg.routing != 'soft', 'RouteSAE with soft routing is not supported on downstream tasks'
                        handles = hook_RouteSAE(self.cfg, self.model, self.language_model, batch_layer_weights)
                    elif self.cfg.model == 'Random':
                        self.hooked_module = self.language_model.get_submodule(f'model.layers.{self.cfg.random_layer-1}')
                        handles = hook_SAE(self.cfg, self.model, self.hooked_module)
                    else:
                        handles = hook_SAE(self.cfg, self.model, self.hooked_module)
                    
                    logits_reconstruct = self.language_model(input_ids=input_ids, attention_mask=attention_mask).logits
                    for handle in handles:
                        handle.remove()
                    
                    if self.cfg.metric == 'DeltaCE':
                        del input_ids, attention_mask
                        torch.cuda.empty_cache()
                        loss = self.DeltaCE(logits_original, logits_reconstruct, input_ids)

                    elif self.cfg.metric == 'Recovered':
                        if self.cfg.model == 'RouteSAE':
                            handles = hook_RouteSAE(self.cfg, self.model, self.language_model, batch_layer_weights, is_zero=True)
                        else:
                            handles = hook_SAE(self.cfg, self.model, self.hooked_module, is_zero=True)
                        logits_zero = self.language_model(input_ids=input_ids, attention_mask=attention_mask).logits
                        for handle in handles:
                            handle.remove()
                        loss = self.Recovered(logits_original, logits_reconstruct, logits_zero, input_ids)
                    
                    elif self.cfg.metric == 'KLDiv':
                        del input_ids, attention_mask
                        torch.cuda.empty_cache()
                        loss = self.KLDiv(logits_original, logits_reconstruct)

                self.num_batches += 1
                self.total_loss += loss.item()

                if self.cfg.use_wandb:
                    wandb.log({'Batch_loss': loss.item()})
                    if self.cfg.model in ['Vanilla', 'Gated', 'JumpReLU', 'Crosscoder'] or self.cfg.theta is not None:
                        counts = (latents != 0).sum(dim=-1).float().mean().item()
                        self.total_counts += counts
                        wandb.log({'Counts': counts})
                else:
                    print(f'Batch: {batch_idx+1}, Loss: {loss.item()}')
        
        if self.cfg.use_wandb:
            wandb.log({'Avg_loss': self.total_loss / self.num_batches})
            if self.cfg.model == 'RouteSAE':
                log_layers(self.layer_weights)
            if self.cfg.model in ['Vanilla', 'Gated', 'JumpReLU', 'Crosscoder'] or self.cfg.theta is not None:
                wandb.log({'Avg_counts': self.total_counts / self.num_batches})
            wandb.finish()
        return self.total_loss / self.num_batches


class Applier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

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
            raise ValueError(f'Invalid model name: {cfg.model} Expected one of [Vanilla, Gated, TopK, JumpReLU, RouteSAE, Crosscoder]')
        
        self.model.load_state_dict(torch.load(cfg.SAE_path, weights_only=True, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def validate_triples(self, triple_list: List[tuple[int, float, int]], name: str) -> None:
        if triple_list is not None:
            for i, (a, b, c) in enumerate(triple_list):
                if not (0 <= a < self.cfg.latent_size):
                    raise ValueError(f'Element {a} in {name} at index {i} is out of latent size range [0, {self.cfg.latent_size}).')
                if b <= 0:
                    raise ValueError(f'Value {b} in {name} at index {i} is not bigger than zero')
                if c not in [0, 1]:
                    raise ValueError(f'Mode {c} in {name} at index {i} must be 0 or 1.')
                
    @torch.no_grad()
    def get_context(
        self, 
        threshold: float = 10.0, 
        max_length: int = 64, 
        max_per_token: int = 2, 
        lines: int = 4,  
        output_path=None
    ):
        if output_path is None:
            output_path = f'../contexts/{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}_{threshold}.json'

        sentence_enders = {'.', '!', '?', '<|end_of_text|>', '"'}
        half_length = max_length // 2

        latent_context_map = defaultdict(lambda: defaultdict(list))

        def find_sentence_bounds(seq_pos: int, tokens: List[str]):
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
        ):
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

        filtered_latent_context = {}
        for latent_dim, token_dict in latent_context_map.items():
            # # Skip latent token categories exceeding 32
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
        return total_latents, output_path
                
    @torch.no_grad()
    def clamp(
        self,
        max_length: int = 64,
        set_high: List[tuple[int, float, int]]=None, 
        set_low: List[tuple[int, float, int]]=None, 
        output_path: str=None
    )-> None:
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

        conversation_path = os.path.join(self.cfg.data_path, 'conversation.json')
        with open(conversation_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)

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

        def generate_multiturn_dialogue(conv_list: List[dict], max_new_tokens=32, is_clamp: bool=False) -> List[dict]:
            prompt_str = ''
            handles=[]
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
                    assert self.cfg.routing != 'soft', 'RouteSAE with soft routing is not supported on downstream tasks'
                    # input_ids = self.tokenizer(prompt_str, return_tensors='pt')['input_ids'].to(self.device)
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

        for conv_id, turns in conversation_data.items():
            output_data['outputs'][conv_id] = []
            base_dialogue = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

            for turn_idx, turn in enumerate(turns):
                human_prompt = turn['Human']
                chosen = turn['chosen']
                rejected = turn['rejected']

                base_dialogue.append({'role': 'user', 'content': human_prompt})
                
                original_dialogue = [d.copy() for d in base_dialogue]
                original_dialogue = generate_multiturn_dialogue(original_dialogue, max_length, is_clamp=False)
                original_output_text = original_dialogue[-1]['content']

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

        output_data['sequences'] = sum(len(turns) for turns in conversation_data.values())
        save_json(output_data, output_path)


class Interpreter:
    def __init__(self, cfg):
        self.cfg = cfg

    def calculate_cost(self, input_text: str, output_text: str) -> float:
        encoding = tiktoken.encoding_for_model(self.cfg.engine)
        num_input_tokens = len(encoding.encode(input_text))
        num_output_tokens = len(encoding.encode(output_text))
        if self.cfg.engine == 'gpt-4o':
            return num_input_tokens * 2.5 / 1_000_000 + num_output_tokens * 10 / 1_000_000
        elif self.cfg.engine == 'gpt-4o-mini':
            return num_input_tokens * 0.15 / 1_000_000 + num_output_tokens * 0.6 / 1_000_000
        else:
            return 0.0
    
    def construct_prompt(self, tokens_info: dict) -> str:
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
        self, client: AzureOpenAI, prompt: str, max_retry: int=3
    ) -> str:
        assert client is not None, 'Client is not set'
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
                assert response_content is not None, 'Response is None'
                return response_content.strip()
            except Exception as e:
                if attempt == max_retry:
                    print('Failed to get a response from the OpenAI API after multiple attempts.')
                    raise e  
        raise Exception('Failed to get a response from the OpenAI API')
    
    def run(
        self, data_path: str=None, sample_latents: int=100, output_path: str=None
    ) -> float:
        if data_path is None:
            data_path = self.cfg.data_path

        if output_path is None:
            output_path = f'../interpret/interp_{os.path.splitext(os.path.basename(self.cfg.SAE_path))[0]}.json'

        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        latent_context_map = data.get('latent_context_map', {})
        all_latents = list(latent_context_map.keys())
        sample_size = min(sample_latents, len(all_latents))
        sampled_indices = random.sample(range(len(all_latents)), sample_size)
        sampled_latents = [all_latents[i] for i in sorted(sampled_indices)]

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

        for latent in sampled_latents:
            try:
                latent_id = int(latent)
            except ValueError:
                print(f"Invalid latent ID {latent}. Skipping.")
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
                        print(f"Invalid category '{category}' or score '{score}' for latent {latent_id}. Skipping.")
                        results[latent_id] = {
                            'category': None,
                            'score': None,
                            'explanation': "Invalid category or score provided.",
                        }
                else:
                    print(f"Failed to parse response for latent {latent_id}. Response: {response}")
                    results[latent_id] = {
                        'category': None,
                        'score': None,
                        'explanation': "Failed to parse response.",
                    }

            except Exception as e:
                print(f"Error processing latent {latent_id}: {e}")
                results[latent_id] = {
                    'category': None,
                    'score': None,
                    'explanation': "Error during processing.",
                }
                continue

        avg_score = total_score / scored_features if scored_features > 0 else 0.0
        low_level_score = concrete_total_score / low_level_features if low_level_features > 0 else 0.0
        high_level_score = abstract_total_score / high_level_features if high_level_features > 0 else 0.0

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
    def __init__(self, cfg):
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
            raise ValueError(f'Invalid model name: {cfg.model} Expected one of [Vanilla, Gated, TopK, JumpReLU, RouteSAE, MLSAE, Random]')
        
        self.cfg.SAE_path = f'../SAE_models/{self.title}.pt'
        self.result_dict = {}
    
    def train(self):
        set_seed(self.cfg.seed)
        self.cfg.data_path = self.cfg.pipe_data_path[0]
        self.cfg.wandb_project = self.cfg.pipe_project[0]
        trainer = Trainer(self.cfg)
        self.result_dict['Train_Loss'] = trainer.run()
        del trainer
        torch.cuda.empty_cache()
    
    def evaluate(self):
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

    def apply(self):
        self.cfg.data_path = self.cfg.pipe_data_path[2]
        applier = Applier(self.cfg)
        self.result_dict[f'Features'], self.context_path = applier.get_context(
            threshold=15, max_length=64, max_per_token=2, lines=4
        )
        del applier
        torch.cuda.empty_cache

    def interpret(self):
        self.cfg.data_path = self.context_path
        interpreter = Interpreter(self.cfg)
        score, Low_level_score, High_Level_score = interpreter.run(sample_latents=100)
        self.result_dict[f'Score'] = score
        self.result_dict[f'Low_level_Score'] = Low_level_score
        self.result_dict[f'High_level_Score'] = High_Level_score
        del interpreter

    def run(self):
        start_time = time.time()

        self.train()
        self.evaluate()
        self.apply()
        self.interpret()

        end_time = time.time()
        self.result_dict['Runtime'] = (end_time - start_time) / 3600

        if self.cfg.use_wandb:
            wandb_init(self.cfg.pipe_project[2], self.result_dict, self.title)
            wandb.finish()
            

