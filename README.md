# Route Sparse Autoencoder to Interpret Large Language Models (RouteSAE)

🎉 **This work has been accepted for Oral presentation at EMNLP 2025.**

## Introduction
This repository contains the official implementation of RouteSAE.

You can download the required dataset and models from the following links:

- [OpenWebText2 Dataset](https://huggingface.co/datasets/segyges/OpenWebText2)
- [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)


The code is written in Python 3.12. The required packages are listed in `requirements.txt`. You can install them by running the following command:

```python
pip install -r requirements.txt
```

## Usage
Ensure that the following directories are created in the root directory of the project:
```bash
./contexts
./interpret
./SAE_models
```
- The `contexts` directory will store the contexts for the SAE models. 
- The `interpret` directory will store the interpretations for the SAE models. 
- The `SAE_models` directory will store the trained SAE models.


We provide a script to run the code. You can save the following command as a `.sh` file and execute it:

```bash
cd ./src
export WANDB_API_KEY='your_wandb_api_key'

language_model_path=your_language_model_path

train_data_path=your_train_data_path
eval_data_path=your_eval_data_path
apply_data_path=your_apply_data_path

train_project=your_train_project
eval_project=your_eval_project
pipe_project=your_pipe_project

api_base=your_api_base
api_key=your_api_key
api_version=your_api_version
engine=your_engine

###--------------------------------------------------pipeline--------------------------------------------------
### TopK
for k_value in ; do
    python -u main.py --language_model 1B --model_path $language_model_path --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --model TopK --layer 12 --latent_size 16384 \
        --batch_size 64 --max_length 512 --lr 0.0005 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --k $k_value \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine 
done

### RouteSAE
for k_value in ; do
    python -u main.py --language_model 1B --model_path $language_model_path --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --model RouteSAE --latent_size 16384 \
        --n_layers 16 --batch_size 64 --max_length 512 --num_epochs 1 --seed 42 --lr 0.0005 --betas 0.9 0.999 --steps 10 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --use_wandb 1 --aggre sum --routing hard \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine --k $k_value
done

### Crosscoder
for lamda_value in ; do
    python -u train.py --language_model 1B --model_path $language_model_path \
        --data_path $train_data_path --model Crosscoder --n_layers 16 --hidden_size 2048 --latent_size 16384 \
        --batch_size 16 --max_length 512 --seed 42 --steps 10 --lr 0.0005 --betas 0.9 0.999 \
        --num_epochs 1 --device cuda:0 --use_wandb 1 --wandb_project $train_project --lamda $lamda_value \
        
    python -u evaluate.py --language_model 1B --model_path $language_model_path \
        --data_path $eval_data_path --model Crosscoder --SAE_path ../SAE_models/Cross_CL${lamda_value}_1B_100M_16384.pt \
        --n_layers 16 --hidden_size 2048 --latent_size 16384 --metric NormMSE --batch_size 16 --max_length 512 \
        --device cuda:0 --use_wandb 1 --wandb_project $eval_project \
    
    python -u application.py --language_model 1B --model_path $language_model_path \
        --data_path $apply_data_path --model Crosscoder --SAE_path ../SAE_models/Cross_CL${lamda_value}_1B_100M_16384.pt \
        --n_layers 16 --hidden_size 2048 --latent_size 16384 --batch_size 16 --max_length 512 \
        --device cuda:0 --use_wandb 0
    
    python -u interpret.py --language_model 1B --model_path $language_model_path \
        --data_path ../contexts/Cross_CL${lamda_value}_1B_100M_16384_15.json --model Crosscoder \
        --SAE_path ../SAE_models/Cross_CL${lamda_value}_1B_100M_16384.pt --n_layers 16 --hidden_size 2048 \
        --latent_size 16384 --batch_size 16 --max_length 512 --device cuda:0 --use_wandb 0 \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine 
done  

### Vanilla
for lamda_value in ; do
    python -u main.py --language_model 1B --model_path $language_model_path --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --model Vanilla --layer 12 --latent_size 16384 \
        --batch_size 64 --max_length 512 --lr 0.0005 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --lamda $lamda_value \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine 
done 

### Gated
for lamda_value in ; do
    python -u main.py --language_model 1B --model_path $language_model_path --hidden_size 2048 \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --model Gated --layer 12 --latent_size 16384 \
        --batch_size 64 --max_length 512 --lr 0.0005 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --pipe_project $train_project $eval_project $pipe_project --device cuda:0 --lamda $lamda_value \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine 
done 

###--------------------------------------------------End--------------------------------------------------
```


