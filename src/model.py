import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class Vanilla(nn.Module):
    '''
    Vanilla Sparse Autoencoder Implements:
    latents = ReLU(encoder(x - pre_bias) + latent_bias)
    reconstruction = decoder(latents) + pre_bias
    '''
    def __init__(self, hidden_size: int, latent_size: int) -> None:
        '''
        :param hidden_size: Dimensionality of the input residual stream activation.
        :param latent_size: Number of latent units.
        '''
        super(Vanilla, self).__init__()
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return self.encoder(x) + self.latent_bias
    
    def get_latents(
        self, pre_acts: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> torch.Tensor:
        assert not (infer_k is not None and theta is not None), 'infer_k and theta cannot both be provided.'
        if theta is not None:
            latents = torch.where(pre_acts>theta, pre_acts, torch.zeros_like(x))
        elif infer_k is not None:
            topk = torch.topk(pre_acts, infer_k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk.indices, topk.values)
        else:
            latents = F.relu(pre_acts)
        return latents
    
    def encode(
        self, x: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> torch.Tensor:
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias
    
    def forward(
        self, x: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> tuple:
        '''
        :param x: Input residual stream activation (shape: [batch_size, max_length, hidden_size]).
        :param infer_k: The number of top-k elements to retain for activation during inference.
        :param theta: Threshold value for jump_relu activation during inference.   
        :return:  latents (shape: [batch_size, max_length, latent_size]).
                  x_hat (shape: [batch_size, max_length, hidden_size]).
        '''
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat


class Gated(nn.Module):
    '''
    Gated Sparse Autoencoder Implements:
    latents = gate(pre_acts + gate_bias) * relu(r_mag.exp() * pre_acts + mag_bias)
    reconstruction = decoder(latents) + pre_bias
    '''
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
    ) -> None:
        '''
        :param hidden_size: Dimensionality of the input residual stream activation.
        :param latent_size: Number of latent units (gated units).
        '''
        super(Gated, self).__init__()
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        
        self.gate_bias = nn.Parameter(torch.zeros(latent_size))
        self.mag_bias = nn.Parameter(torch.zeros(latent_size))
        self.r_mag = nn.Parameter(torch.zeros(latent_size))
        
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x - self.pre_bias)

    def get_latents(self, pre_acts: torch.Tensor, infer_k: int=None, theta: float=None) -> torch.Tensor:
        pi_gate = pre_acts + self.gate_bias
        f_gate = (pi_gate > 0).float()

        pi_mag = torch.exp(self.r_mag) * pre_acts + self.mag_bias
        f_mag = F.relu(pi_mag)

        latents = f_gate * f_mag
        return latents

    def encode(self, x: torch.Tensor, infer_k: int=None, theta: float=None) -> torch.Tensor:
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k, theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

    def forward(self, x: torch.Tensor, infer_k: int=None, theta: float=None) -> tuple:
        '''
        Forward pass of the Gated autoencoder.
        :param x: Input residual stream activation (shape: [batch_size, seq_len, hidden_size]).
        :return:  latents (shape: [batch_size, seq_len, latent_size]),
                  x_hat   (shape: [batch_size, seq_len, hidden_size]).
        '''
        latents = self.encode(x)
        x_hat = self.decode(latents)
        return latents, x_hat


class TopK(nn.Module):
    '''
    TopK Sparse Autoencoder Implements:
    latents = TopK(encoder(x - pre_bias) + latent_bias)
    reconstruction = decoder(latents) + pre_bias
    '''
    def __init__(
        self, hidden_size: int, latent_size: int, k: int
    ) -> None:
        '''
        :param hidden_size: Dimensionality of the input residual stream activation.
        :param latent_size: Number of latent units.
        :param k: Number of activated latents.
        '''
        assert k <= latent_size, f'k should be less than or equal to {latent_size}'
        super(TopK, self).__init__()
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)
        self.k = k

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return self.encoder(x) + self.latent_bias
    
    def get_latents(
        self, pre_acts: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> torch.Tensor:
        assert not (infer_k is not None and theta is not None), 'infer_k and theta cannot both be provided.'
        if theta is not None:
            latents = torch.where(pre_acts>theta, pre_acts, torch.zeros_like(pre_acts))
        else:
            k = infer_k if infer_k is not None else self.k
            topk = torch.topk(pre_acts, k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk.indices, topk.values)
        return latents

    def encode(
        self, x: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> torch.Tensor:
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias
    
    def forward(
        self, x: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> tuple:
        '''
        :param x: Input residual stream activation (shape: [batch_size, max_length, hidden_size]).
        :param infer_k: The number of top-k elements to retain for activation during inference.
        :param theta: Threshold value for jump_relu activation during inference.
        :return:  latents (shape: [batch_size, max_length, latent_size]).
                  x_hat (shape: [batch_size, max_length, hidden_size]).
        '''
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat


class Step_func(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = torch.zeros_like(grad_output)
        threshold_grad = (
            - (1.0 / bandwidth) 
            * ((x - threshold).abs() < bandwidth / 2).float() 
            * grad_output
        )
        return x_grad, threshold_grad, None

class Jump_func(autograd.Function):
    """
    Jump ReLU activation function.
    """
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            - (threshold / bandwidth) 
            * ((x - threshold).abs() < bandwidth / 2).float()
            * grad_output
        )
        return x_grad, threshold_grad, None

class JumpReLU(nn.Module):
    '''
    JumpReLU Sparse Autoencoder Implements:
    latents = JumpReLU(encoder(x - pre_bias) + latent_bias, threshold, bandwidth)
    reconstruction = decoder(latents) + pre_bias
    '''
    def __init__(
        self, hidden_size: int, latent_size: int, threshold: float=1e-3, bandwidth: float=1e-3
    ) -> None:
        '''
        :param hidden_size: Dimensionality of the input residual stream activation.
        :param latent_size: Number of latent units.
        :param threshold: Threshold value for JumpReLU activation.
        :param bandwidth: Bandwidth for JumpReLU activation.
        '''
        super(JumpReLU, self).__init__()
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # "tied" init
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

        self.threshold = nn.Parameter(torch.full((latent_size,), threshold))
        self.bandwidth = bandwidth

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        x = x - self.pre_bias
        return self.encoder(x) + self.latent_bias

    def get_latents(
        self, pre_acts: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> torch.Tensor:
        assert not (infer_k is not None and theta is not None), 'infer_k and theta cannot both be provided.'
        if theta is not None:
            latents = torch.where(pre_acts>theta, pre_acts, torch.zeros_like(pre_acts))
        elif infer_k is not None:
            topk = torch.topk(pre_acts, infer_k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk.indices, topk.values)
        else:
            latents = Jump_func.apply(pre_acts, self.threshold, self.bandwidth)
        return latents

    def encode(
        self, x: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> torch.Tensor:
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

    def forward(
        self, x: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> tuple:
        '''
        :param x: Input residual stream activation (shape: [batch_size, max_length, hidden_size]).
        :param infer_k: The number of top-k elements to retain for activation during inference.
        :param theta: Threshold value for JumpReLU activation during inference.   
        :return:  latents (shape: [batch_size, max_length, latent_size]).
                  x_hat (shape: [batch_size, max_length, hidden_size]).
        '''
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat


class RouteSAE(nn.Module):
    '''
    RoutingSAE Implements:
    Processes hidden states from a language model based on a specified routing strategy.
    '''
    def __init__(
        self, hidden_size: int, n_layers: int, latent_size: int, k: int
    ) -> None:
        '''
        :param hidden_size: Dimensionality of the input residual stream activation.
        :param n_layers: Number of layers in the input hidden_states.
        :param latent_size: Number of latent units.
        :param k: Number of activated latents.
        '''
        super(RouteSAE, self).__init__()
        self.start_layer = n_layers // 4
        self.end_layer = n_layers * 3 // 4 + 1
        self.router = nn.Linear(hidden_size, self.end_layer - self.start_layer, bias=False)
        self.sae = TopK(hidden_size, latent_size, k)

    def get_router_weights(
            self, x: torch.Tensor, aggre: str
        ) -> torch.Tensor:
        if aggre == 'sum':
            router_input = x.sum(dim=2)  # (batch_size, max_length, hidden_size)
        elif aggre == 'mean':
            router_input = x.mean(dim=2) # (batch_size, max_length, hidden_size)
        else:
            raise ValueError(f'Unsupported aggre: {aggre}. Expected one of [sum, mean]. Got: {aggre}.')
        
        router_output = self.router(router_input)   # (batch_size, max_length, n_layers)
        router_weights = torch.softmax(router_output, dim=-1) # (batch_size, max_length, n_layers)
        return router_weights
    
    def get_sae_input(
        self, x: torch.Tensor, router_weights: torch.Tensor, routing: str
    ) -> tuple:
        if routing == 'hard':
            max_weights, target_layer = router_weights.max(dim=-1)  # max_weights: (batch_size, max_length), target_layer: (batch_size, max_length)
            batch_layer_weights = torch.zeros_like(router_weights)  # (batch_size, max_length, n_layers)
            batch_layer_weights.scatter_(2, target_layer.unsqueeze(-1), 1)  # One-hot encoding for the selected layers
            x = torch.gather(x, 2, target_layer.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, x.size(-1)))  # (batch_size, max_length, 1, hidden_size)
            x = x.squeeze(2) * max_weights.unsqueeze(-1)  # (batch_size, max_length, hidden_size)
        
        elif routing == 'soft':
            weighted_hidden_states = x * router_weights.unsqueeze(-1)  # (batch_size, max_length, n_layers, hidden_size)
            x = weighted_hidden_states.sum(dim=2)  # (batch_size, max_length, hidden_size)
            batch_layer_weights = router_weights
        
        else:
            raise ValueError(f'Unsupported routing: {routing}. Expected one of [hard, soft].')

        return batch_layer_weights, x

    def forward(self, 
                x: torch.Tensor, 
                aggre: str, 
                routing: str,
                infer_k: int = None, 
                theta: float = None) -> tuple:
        '''
        :param x: Tensor of activations from all layers (shape: [batch_size, max_length, n_layers, hidden_size]).
        :param aggre: Aggregation method for SAE routing:
                    - 'sum': Sum of all layers' hidden states as router input.
                    - 'mean': Mean pooling of all layers' hidden states as router input.
        :param routing: Routing method for SAE:
                    - 'hard': Hard routing (available for both training and inference).
                    - 'soft': Soft routing (available for training only).
        :param infer_k: Number of top-k elements to retain during inference (optional).
        :param theta: Threshold for jump_relu activation during inference (optional).
        :return: 
            - layer_weights: Weights of selected layers (shape: [batch_size, max_length, n_layers]).
            - x: Routed input (shape: [batch_size, max_length, hidden_size]).
            - latents: Latent representation after processing (shape: [batch_size, max_length, latent_size]).
            - x_hat: Reconstructed input after decoding (shape: [batch_size, max_length, hidden_size]).
            - router_weights: Raw router weights before routing (shape: [batch_size, max_length, n_layers]).
        '''
        # Routing
        router_weights = self.get_router_weights(x, aggre)
        batch_layer_weights, x = self.get_sae_input(x, router_weights, routing)

        # Reconstruct and return
        latents, x_hat = self.sae(x, infer_k=infer_k, theta=theta)
        return batch_layer_weights, x, latents, x_hat, router_weights
    

class Crosscoder(nn.Module):
    '''
    Crosscoder Implements:
    '''
    def __init__(
        self, hidden_size: int, n_layers: int, latent_size: int
    ) -> None:
        '''
        :param hidden_size: Dimensionality of the input residual stream activation.
        :param n_layers: Number of layers in the input hidden_states.
        :param latent_size: Number of latent units.
        :param k: Number of activated latents.
        '''
        super(Crosscoder, self).__init__()
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))

        self.start_layer = n_layers // 4
        self.end_layer = n_layers * 3 // 4 + 1

        self.encoder = nn.ModuleList([
            nn.Linear(hidden_size, latent_size, bias=False) for _ in range(self.end_layer - self.start_layer)
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(latent_size, hidden_size, bias=False) for _ in range(self.end_layer - self.start_layer)
        ])
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # "tied" init
        with torch.no_grad():
            for i in range(len(self.decoder)):
                self.decoder[i].weight.data = self.encoder[i].weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = torch.zeros(x.size(0), x.size(1), len(self.encoder), self.latent_size, device=x.device)
        for i in range(len(self.encoder)):
            pre_acts[:, :, i, :] = self.encoder[i](x[:, :, i, :])
        return pre_acts.sum(dim=-2) + self.latent_bias
    
    def get_latents(
        self, pre_acts: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> torch.Tensor:
        if theta is not None:
            latents = torch.where(pre_acts>theta, pre_acts, torch.zeros_like(pre_acts))
        elif infer_k is not None:
            k = infer_k
            topk = torch.topk(pre_acts, k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk.indices, topk.values)
        else:
            latents = F.relu(pre_acts)
        return latents
    
    def encode(
        self, x: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> torch.Tensor:
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents
    
    def decode(
        self, latents: torch.Tensor
    ) -> torch.Tensor:
        x_hat = torch.zeros(latents.size(0), latents.size(1), len(self.decoder), self.hidden_size, device=latents.device)
        for i in range(len(self.decoder)):
            x_hat[:, :, i, :] = self.decoder[i](latents)
        return x_hat
    
    def forward(
        self, x: torch.Tensor, infer_k: int=None, theta: float=None
    ) -> tuple:
        '''
        :param x: Tensor of activations from all layers (shape: [batch_size, max_length, n_layers, hidden_size]).
        :param infer_k: Number of top-k elements to retain during inference.
        :param theta: Threshold for jump_relu activation during inference.
        :return: latents: Latent representation after processing (shape: [batch_size, max_length, latent_size]).
                 x_hat: Reconstructed input after decoding (shape: [batch_size, max_length, n_layers, hidden_size]).
        '''
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)

        return latents, x_hat
    
