"""
Sparse Autoencoder (SAE) Model Implementations

This module contains various sparse autoencoder architectures for interpreting
large language models, including Vanilla, Gated, TopK, JumpReLU, RouteSAE, and Crosscoder.

Key Concepts:
    - Sparse Autoencoders learn interpretable features by enforcing sparsity
    - Tied weights (decoder.weight = encoder.weight.T) improve training stability
    - Different sparsity mechanisms: L1 penalty, TopK, Gating, JumpReLU
    - RouteSAE adds layer-wise routing for multi-layer analysis

References:
    - Sparse Autoencoder principles
    - Towards Monosemanticity: Decomposing Language Models with Dictionary Learning
    - Gated Sparse Autoencoders

Author: RouteSAE Contributors
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from typing import Optional, Tuple, List
import warnings


class Vanilla(nn.Module):
    """
    Vanilla Sparse Autoencoder with L1 Regularization.
    
    Architecture:
        latents = ReLU(encoder(x - pre_bias) + latent_bias)
        reconstruction = decoder(latents) + pre_bias
    
    This is the standard SAE architecture with:
    - Pre-bias: Centers the input distribution (learned mean subtraction)
    - Latent bias: Allows features to activate more easily
    - Tied weights: decoder.weight = encoder.weight.T for better optimization
    - ReLU activation: Ensures non-negative, sparse latent codes
    
    The sparsity is typically enforced via L1 regularization on latents during training.
    
    Attributes:
        pre_bias: Learnable bias subtracted from input (shape: [hidden_size])
        latent_bias: Learnable bias added to latent activations (shape: [latent_size])
        encoder: Linear layer mapping input to latent space
        decoder: Linear layer reconstructing input from latents
    """
    
    def __init__(self, hidden_size: int, latent_size: int) -> None:
        """
        Initialize Vanilla SAE.
        
        Args:
            hidden_size: Dimensionality of the input residual stream activation
            latent_size: Number of latent features (typically > hidden_size for overcomplete representation)
            
        Raises:
            ValueError: If hidden_size or latent_size is not positive
        """
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError(f"hidden_size and latent_size must be positive, got {hidden_size} and {latent_size}")
            
        super(Vanilla, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Learnable parameters
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        
        # Encoder and decoder layers (without bias for cleaner separation)
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # Initialize with tied weights for better optimization
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize decoder weights as transpose of encoder (tied weights)."""
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activation values (before applying ReLU).
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            
        Returns:
            Pre-activation values (shape: [batch_size, seq_len, latent_size])
        """
        centered_x = x - self.pre_bias
        return self.encoder(centered_x) + self.latent_bias
    
    def get_latents(
        self, 
        pre_acts: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply activation function to pre-activations with optional inference-time modifications.
        
        During training, uses standard ReLU activation.
        During inference, can optionally apply:
        - TopK sparsity: Only keep top-k activations
        - Threshold: Only keep activations above threshold
        
        Args:
            pre_acts: Pre-activation values (shape: [batch_size, seq_len, latent_size])
            infer_k: If provided, only keep top-k activations (inference only)
            theta: If provided, only keep activations above this threshold (inference only)
            
        Returns:
            Activated latent representation with same shape as pre_acts
            
        Raises:
            ValueError: If both infer_k and theta are provided
        """
        if infer_k is not None and theta is not None:
            raise ValueError('Cannot specify both infer_k and theta simultaneously. Choose one.')
        
        if theta is not None:
            # Threshold-based activation
            latents = torch.where(pre_acts > theta, pre_acts, torch.zeros_like(pre_acts))
        elif infer_k is not None:
            # TopK-based activation
            if infer_k > pre_acts.size(-1):
                warnings.warn(f"infer_k ({infer_k}) is larger than latent_size ({pre_acts.size(-1)}), using all latents")
                infer_k = pre_acts.size(-1)
            
            topk_values, topk_indices = torch.topk(pre_acts, infer_k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk_indices, topk_values)
        else:
            # Standard ReLU activation
            latents = F.relu(pre_acts)
        
        return latents
    
    def encode(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            infer_k: Optional top-k constraint for inference
            theta: Optional threshold for inference
            
        Returns:
            Latent representation (shape: [batch_size, seq_len, latent_size])
        """
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            latents: Latent representation (shape: [batch_size, seq_len, latent_size])
            
        Returns:
            Reconstructed input (shape: [batch_size, seq_len, hidden_size])
        """
        return self.decoder(latents) + self.pre_bias
    
    def forward(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input residual stream activation (shape: [batch_size, seq_len, hidden_size])
            infer_k: Optional number of top-k elements to retain during inference
            theta: Optional threshold value for activation during inference
            
        Returns:
            Tuple of:
                - latents: Sparse latent representation (shape: [batch_size, seq_len, latent_size])
                - x_hat: Reconstructed input (shape: [batch_size, seq_len, hidden_size])
                
        Example:
            >>> sae = Vanilla(hidden_size=768, latent_size=4096)
            >>> x = torch.randn(32, 128, 768)
            >>> latents, x_hat = sae(x)
            >>> print(latents.shape, x_hat.shape)
            torch.Size([32, 128, 4096]) torch.Size([32, 128, 768])
        """
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat


class Gated(nn.Module):
    """
    Gated Sparse Autoencoder with separate gating and magnitude components.
    
    Architecture:
        gate = Heaviside(pre_acts + gate_bias)
        magnitude = ReLU(exp(r_mag) * pre_acts + mag_bias)
        latents = gate * magnitude
        reconstruction = decoder(latents) + pre_bias
    
    The gating mechanism separates the decision of WHETHER a feature activates (gate)
    from HOW MUCH it activates (magnitude). This can lead to:
    - Better interpretability (clearer feature activation boundaries)
    - Reduced shrinkage compared to L1 regularization
    - More stable training dynamics
    
    Reference: Gated Sparse Autoencoders paper
    
    Attributes:
        pre_bias: Learnable bias subtracted from input
        gate_bias: Learnable bias for gating function
        mag_bias: Learnable bias for magnitude function
        r_mag: Learnable log-scale parameter for magnitude
        encoder: Shared encoder for both gate and magnitude
        decoder: Decoder for reconstruction
    """
    
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
    ) -> None:
        """
        Initialize Gated SAE.
        
        Args:
            hidden_size: Dimensionality of the input residual stream activation
            latent_size: Number of latent features
            
        Raises:
            ValueError: If hidden_size or latent_size is not positive
        """
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError(f"hidden_size and latent_size must be positive, got {hidden_size} and {latent_size}")
            
        super(Gated, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Input centering
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Gating parameters
        self.gate_bias = nn.Parameter(torch.zeros(latent_size))
        
        # Magnitude parameters
        self.mag_bias = nn.Parameter(torch.zeros(latent_size))
        self.r_mag = nn.Parameter(torch.zeros(latent_size))  # Log-scale multiplier
        
        # Encoder and decoder
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # Initialize with tied weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize decoder weights as transpose of encoder."""
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activation values from centered input.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            
        Returns:
            Pre-activation values (shape: [batch_size, seq_len, latent_size])
        """
        centered_x = x - self.pre_bias
        return self.encoder(centered_x)

    def get_latents(
        self, 
        pre_acts: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply gated activation to pre-activations.
        
        The gating mechanism computes:
        - gate: Binary decision (Heaviside function approximated by step function)
        - magnitude: Continuous value (ReLU of scaled pre-activations)
        - latents: Element-wise product of gate and magnitude
        
        Args:
            pre_acts: Pre-activation values (shape: [batch_size, seq_len, latent_size])
            infer_k: Not used for Gated SAE (kept for interface consistency)
            theta: Not used for Gated SAE (kept for interface consistency)
            
        Returns:
            Gated latent representation (shape: same as pre_acts)
            
        Note:
            infer_k and theta parameters are ignored for Gated SAE as the gating
            mechanism already provides sparse activations.
        """
        # Gate: Determines which features activate (binary decision)
        pi_gate = pre_acts + self.gate_bias
        f_gate = (pi_gate > 0).float()  # Heaviside function (0 or 1)

        # Magnitude: Determines activation strength (continuous value)
        pi_mag = torch.exp(self.r_mag) * pre_acts + self.mag_bias
        f_mag = F.relu(pi_mag)

        # Combine: Gate controls on/off, magnitude controls strength
        latents = f_gate * f_mag
        return latents

    def encode(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            infer_k: Ignored for Gated SAE
            theta: Ignored for Gated SAE
            
        Returns:
            Latent representation (shape: [batch_size, seq_len, latent_size])
        """
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k, theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            latents: Latent representation (shape: [batch_size, seq_len, latent_size])
            
        Returns:
            Reconstructed input (shape: [batch_size, seq_len, hidden_size])
        """
        return self.decoder(latents) + self.pre_bias

    def forward(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the gated autoencoder.
        
        Args:
            x: Input residual stream activation (shape: [batch_size, seq_len, hidden_size])
            infer_k: Ignored for Gated SAE (kept for interface consistency)
            theta: Ignored for Gated SAE (kept for interface consistency)
            
        Returns:
            Tuple of:
                - latents: Sparse gated latent representation (shape: [batch_size, seq_len, latent_size])
                - x_hat: Reconstructed input (shape: [batch_size, seq_len, hidden_size])
                
        Example:
            >>> sae = Gated(hidden_size=768, latent_size=4096)
            >>> x = torch.randn(32, 128, 768)
            >>> latents, x_hat = sae(x)
            >>> # Latents will be sparse due to gating mechanism
            >>> sparsity = (latents == 0).float().mean()
            >>> print(f"Sparsity: {sparsity:.2%}")
        """
        latents = self.encode(x, infer_k, theta)
        x_hat = self.decode(latents)
        return latents, x_hat


class TopK(nn.Module):
    """
    TopK Sparse Autoencoder with fixed sparsity level.
    
    Architecture:
        pre_acts = encoder(x - pre_bias) + latent_bias
        latents = TopK(pre_acts, k)  # Only keep top-k activations
        reconstruction = decoder(latents) + pre_bias
    
    TopK SAE enforces exact sparsity by:
    - Computing all pre-activations
    - Selecting only the top-k largest values
    - Setting all others to zero
    
    This provides:
    - Predictable sparsity level (exactly k non-zero features)
    - No need for L1 regularization hyperparameter tuning
    - Direct control over computational cost
    
    The TopK operation is differentiable via straight-through estimator.
    
    Attributes:
        k: Number of features to activate (sparsity level)
        pre_bias: Learnable bias subtracted from input
        latent_bias: Learnable bias added to latent activations
        encoder: Linear layer mapping input to latent space
        decoder: Linear layer reconstructing input from latents
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        latent_size: int, 
        k: int
    ) -> None:
        """
        Initialize TopK SAE.
        
        Args:
            hidden_size: Dimensionality of the input residual stream activation
            latent_size: Number of latent features
            k: Number of features to activate (must be <= latent_size)
            
        Raises:
            ValueError: If hidden_size, latent_size, or k is invalid
        """
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError(f"hidden_size and latent_size must be positive, got {hidden_size} and {latent_size}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > latent_size:
            raise ValueError(f"k ({k}) cannot be larger than latent_size ({latent_size})")
            
        super(TopK, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.k = k
        
        # Learnable parameters
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        
        # Encoder and decoder layers
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # Initialize with tied weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize decoder weights as transpose of encoder."""
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activation values before TopK selection.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            
        Returns:
            Pre-activation values (shape: [batch_size, seq_len, latent_size])
        """
        centered_x = x - self.pre_bias
        return self.encoder(centered_x) + self.latent_bias
    
    def get_latents(
        self, 
        pre_acts: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply TopK or threshold-based sparsity to pre-activations.
        
        Args:
            pre_acts: Pre-activation values (shape: [batch_size, seq_len, latent_size])
            infer_k: Optional override for k during inference (useful for ablations)
            theta: Optional threshold to use instead of TopK
            
        Returns:
            Sparse latent representation (shape: same as pre_acts)
            
        Raises:
            ValueError: If both infer_k and theta are provided
        """
        if infer_k is not None and theta is not None:
            raise ValueError('Cannot specify both infer_k and theta simultaneously. Choose one.')
        
        if theta is not None:
            # Threshold-based sparsity (for analysis)
            latents = torch.where(pre_acts > theta, pre_acts, torch.zeros_like(pre_acts))
        else:
            # TopK sparsity (default behavior)
            k = infer_k if infer_k is not None else self.k
            
            # Validate k
            if k > pre_acts.size(-1):
                warnings.warn(f"k ({k}) is larger than latent_size ({pre_acts.size(-1)}), using all latents")
                k = pre_acts.size(-1)
            
            # Select top-k activations
            topk_values, topk_indices = torch.topk(pre_acts, k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk_indices, topk_values)
        
        return latents

    def encode(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Encode input to sparse latent representation.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            infer_k: Optional override for k during inference
            theta: Optional threshold for sparsity
            
        Returns:
            Sparse latent representation (shape: [batch_size, seq_len, latent_size])
        """
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            latents: Latent representation (shape: [batch_size, seq_len, latent_size])
            
        Returns:
            Reconstructed input (shape: [batch_size, seq_len, hidden_size])
        """
        return self.decoder(latents) + self.pre_bias
    
    def forward(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the TopK autoencoder.
        
        Args:
            x: Input residual stream activation (shape: [batch_size, seq_len, hidden_size])
            infer_k: Optional override for number of active features during inference
            theta: Optional threshold for activation during inference
            
        Returns:
            Tuple of:
                - latents: Sparse latent representation with exactly k non-zero values per position
                          (shape: [batch_size, seq_len, latent_size])
                - x_hat: Reconstructed input (shape: [batch_size, seq_len, hidden_size])
                
        Example:
            >>> sae = TopK(hidden_size=768, latent_size=4096, k=64)
            >>> x = torch.randn(32, 128, 768)
            >>> latents, x_hat = sae(x)
            >>> # Verify exactly k non-zero per position
            >>> non_zero = (latents != 0).sum(dim=-1)
            >>> assert (non_zero == 64).all()
        """
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat


class Step_func(autograd.Function):
    """
    Custom autograd function for differentiable step (Heaviside) function.
    
    Forward: Returns 1 if x > threshold, 0 otherwise
    Backward: Uses straight-through estimator with bandwidth for smoothing
    
    This allows gradient flow through the discrete step function using
    a smooth approximation in the backward pass.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """
        Forward pass of step function.
        
        Args:
            ctx: Context object for saving tensors
            x: Input tensor
            threshold: Threshold value(s) for step function
            bandwidth: Bandwidth for gradient approximation
            
        Returns:
            Binary tensor: 1.0 where x > threshold, 0.0 elsewhere
        """
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass using straight-through estimator.
        
        Approximates gradient with a box filter of width 'bandwidth'
        centered at the threshold. This provides a smooth gradient
        in a region around the threshold.
        
        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient from subsequent layers
            
        Returns:
            Tuple of gradients for (x, threshold, bandwidth)
        """
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        
        # No gradient flows through x (straight-through)
        x_grad = torch.zeros_like(grad_output)
        
        # Gradient for threshold: box filter approximation
        # Only non-zero when |x - threshold| < bandwidth/2
        in_bandwidth = (x - threshold).abs() < bandwidth / 2
        threshold_grad = -(1.0 / bandwidth) * in_bandwidth.float() * grad_output
        
        # No gradient for bandwidth (it's a hyperparameter)
        return x_grad, threshold_grad, None


class Jump_func(autograd.Function):
    """
    Custom autograd function for JumpReLU activation.
    
    JumpReLU combines:
    - Step function: Determines if feature activates
    - Linear pass-through: Preserves activation magnitude
    
    Forward: x if x > threshold, 0 otherwise
    Backward: Gradient flows when x > threshold, with smooth threshold gradient
    
    This provides sparse activations with magnitude information,
    unlike TopK which normalizes magnitudes.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        """
        Forward pass of JumpReLU.
        
        Args:
            ctx: Context object for saving tensors
            x: Input tensor
            threshold: Learned threshold value(s)
            bandwidth: Bandwidth for gradient approximation
            
        Returns:
            x where x > threshold, 0 elsewhere
        """
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass with gradients for both input and threshold.
        
        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient from subsequent layers
            
        Returns:
            Tuple of gradients for (x, threshold, bandwidth)
        """
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        
        # Gradient flows through x where it's active
        x_grad = (x > threshold).float() * grad_output
        
        # Gradient for threshold: smooth approximation
        # The threshold gradient is weighted by the current value
        # to account for the magnitude preservation
        in_bandwidth = (x - threshold).abs() < bandwidth / 2
        threshold_grad = -(threshold / bandwidth) * in_bandwidth.float() * grad_output
        
        return x_grad, threshold_grad, None


class JumpReLU(nn.Module):
    """
    JumpReLU Sparse Autoencoder with learned per-feature thresholds.
    
    Architecture:
        latents = JumpReLU(encoder(x - pre_bias) + latent_bias, threshold, bandwidth)
        reconstruction = decoder(latents) + pre_bias
    
    JumpReLU combines the benefits of:
    - TopK: Explicit sparsity control
    - Vanilla: Magnitude preservation
    
    Each latent feature learns its own activation threshold. Features only
    activate when their pre-activation exceeds this threshold, preserving
    the magnitude information (unlike TopK which can distort magnitudes).
    
    The bandwidth parameter controls the smoothness of the threshold gradient
    during training, acting as a temperature for the step function approximation.
    
    Reference: JumpReLU SAE paper
    
    Attributes:
        threshold: Learned per-feature activation thresholds
        bandwidth: Fixed bandwidth for gradient smoothing
        pre_bias: Learnable bias subtracted from input
        latent_bias: Learnable bias added to latent activations
        encoder: Linear layer mapping input to latent space
        decoder: Linear layer reconstructing input from latents
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        latent_size: int, 
        threshold: float = 1e-3, 
        bandwidth: float = 1e-3
    ) -> None:
        """
        Initialize JumpReLU SAE.
        
        Args:
            hidden_size: Dimensionality of the input residual stream activation
            latent_size: Number of latent features
            threshold: Initial threshold value for all features
            bandwidth: Bandwidth for gradient smoothing (should be small, e.g., 1e-3)
            
        Raises:
            ValueError: If hidden_size, latent_size, threshold, or bandwidth is invalid
        """
        if hidden_size <= 0 or latent_size <= 0:
            raise ValueError(f"hidden_size and latent_size must be positive, got {hidden_size} and {latent_size}")
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")
            
        super(JumpReLU, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.bandwidth = bandwidth
        
        # Learnable parameters
        self.pre_bias = nn.Parameter(torch.zeros(hidden_size))
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))
        
        # Encoder and decoder
        self.encoder = nn.Linear(hidden_size, latent_size, bias=False)
        self.decoder = nn.Linear(latent_size, hidden_size, bias=False)

        # Per-feature learned thresholds (initialized to small positive value)
        self.threshold = nn.Parameter(torch.full((latent_size,), threshold))
        
        # Initialize with tied weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize decoder weights as transpose of encoder."""
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activation values before applying JumpReLU.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            
        Returns:
            Pre-activation values (shape: [batch_size, seq_len, latent_size])
        """
        centered_x = x - self.pre_bias
        return self.encoder(centered_x) + self.latent_bias

    def get_latents(
        self, 
        pre_acts: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply JumpReLU or alternative sparsity to pre-activations.
        
        Args:
            pre_acts: Pre-activation values (shape: [batch_size, seq_len, latent_size])
            infer_k: Optional top-k constraint for inference (overrides JumpReLU)
            theta: Optional fixed threshold for inference (overrides learned thresholds)
            
        Returns:
            Sparse latent representation (shape: same as pre_acts)
            
        Raises:
            ValueError: If both infer_k and theta are provided
        """
        if infer_k is not None and theta is not None:
            raise ValueError('Cannot specify both infer_k and theta simultaneously. Choose one.')
        
        if theta is not None:
            # Use fixed threshold (for analysis/ablation)
            latents = torch.where(pre_acts > theta, pre_acts, torch.zeros_like(pre_acts))
        elif infer_k is not None:
            # Use TopK sparsity (for comparison)
            if infer_k > pre_acts.size(-1):
                warnings.warn(f"infer_k ({infer_k}) is larger than latent_size ({pre_acts.size(-1)}), using all latents")
                infer_k = pre_acts.size(-1)
            
            topk_values, topk_indices = torch.topk(pre_acts, infer_k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk_indices, topk_values)
        else:
            # Use JumpReLU with learned thresholds (default)
            latents = Jump_func.apply(pre_acts, self.threshold, self.bandwidth)
        
        return latents

    def encode(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Encode input to sparse latent representation.
        
        Args:
            x: Input tensor (shape: [batch_size, seq_len, hidden_size])
            infer_k: Optional top-k constraint for inference
            theta: Optional fixed threshold for inference
            
        Returns:
            Sparse latent representation (shape: [batch_size, seq_len, latent_size])
        """
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to input space.
        
        Args:
            latents: Latent representation (shape: [batch_size, seq_len, latent_size])
            
        Returns:
            Reconstructed input (shape: [batch_size, seq_len, hidden_size])
        """
        return self.decoder(latents) + self.pre_bias

    def forward(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the JumpReLU autoencoder.
        
        Args:
            x: Input residual stream activation (shape: [batch_size, seq_len, hidden_size])
            infer_k: Optional number of top-k elements for inference
            theta: Optional fixed threshold for activation during inference
            
        Returns:
            Tuple of:
                - latents: Sparse latent representation with adaptive sparsity
                          (shape: [batch_size, seq_len, latent_size])
                - x_hat: Reconstructed input (shape: [batch_size, seq_len, hidden_size])
                
        Example:
            >>> sae = JumpReLU(hidden_size=768, latent_size=4096, threshold=0.01)
            >>> x = torch.randn(32, 128, 768)
            >>> latents, x_hat = sae(x)
            >>> # Sparsity is learned per feature
            >>> sparsity_per_feature = (latents == 0).float().mean(dim=(0, 1))
            >>> print(f"Min sparsity: {sparsity_per_feature.min():.2%}")
            >>> print(f"Max sparsity: {sparsity_per_feature.max():.2%}")
        """
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat


class RouteSAE(nn.Module):
    """
    Route Sparse Autoencoder with layer-wise routing mechanism.
    
    RouteSAE extends traditional SAEs to handle multi-layer representations
    by learning to route different tokens to different layers based on their
    semantic needs.
    
    Architecture:
        1. Router: Learns which layer(s) to process for each token
        2. Routing: Selects layer activations (hard or soft)
        3. SAE: Processes routed activations with TopK sparsity
    
    Key innovations:
    - Dynamic layer selection per token
    - Hard routing: Select single best layer (discrete, efficient)
    - Soft routing: Weighted combination of layers (differentiable, flexible)
    - Focuses on middle layers (layers n/4 to 3n/4)
    
    This allows the model to:
    - Capture both low-level and high-level features
    - Adapt processing depth to token semantics
    - Learn which layers contain most informative features
    
    Attributes:
        start_layer: First layer to consider for routing
        end_layer: Last layer to consider for routing (exclusive)
        router: Linear layer predicting layer weights
        sae: TopK SAE for processing routed activations
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        n_layers: int, 
        latent_size: int, 
        k: int
    ) -> None:
        """
        Initialize RouteSAE.
        
        Args:
            hidden_size: Dimensionality of layer activations
            n_layers: Total number of layers in the language model
            latent_size: Number of latent features in the SAE
            k: Number of active features (TopK sparsity)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if hidden_size <= 0 or latent_size <= 0 or n_layers <= 0 or k <= 0:
            raise ValueError("All dimensions must be positive")
        if k > latent_size:
            raise ValueError(f"k ({k}) cannot exceed latent_size ({latent_size})")
        if n_layers < 4:
            raise ValueError(f"n_layers ({n_layers}) should be at least 4 for meaningful routing")
            
        super(RouteSAE, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.latent_size = latent_size
        self.k = k
        
        # Focus on middle layers (typically most informative)
        self.start_layer = n_layers // 4
        self.end_layer = n_layers * 3 // 4 + 1
        self.n_routed_layers = self.end_layer - self.start_layer
        
        # Router: learns layer selection weights
        self.router = nn.Linear(hidden_size, self.n_routed_layers, bias=False)
        
        # SAE: processes routed activations
        self.sae = TopK(hidden_size, latent_size, k)

    def get_router_weights(
        self, 
        x: torch.Tensor, 
        aggre: str
    ) -> torch.Tensor:
        """
        Compute router weights for layer selection.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            aggre: Aggregation method for router input:
                   - 'sum': Sum across layer dimension
                   - 'mean': Average across layer dimension
            
        Returns:
            Normalized router weights (shape: [batch, seq_len, n_layers])
            
        Raises:
            ValueError: If aggre method is not supported
        """
        if aggre == 'sum':
            router_input = x.sum(dim=2)  # Sum across layers
        elif aggre == 'mean':
            router_input = x.mean(dim=2)  # Average across layers
        else:
            raise ValueError(
                f'Unsupported aggregation method: {aggre}. '
                f'Expected one of ["sum", "mean"].'
            )
        
        # Compute router logits and normalize with softmax
        router_output = self.router(router_input)  # (batch, seq_len, n_layers)
        router_weights = F.softmax(router_output, dim=-1)  # Normalize to probabilities
        
        return router_weights
    
    def get_sae_input(
        self, 
        x: torch.Tensor, 
        router_weights: torch.Tensor, 
        routing: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply routing to select layer activations.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            router_weights: Layer selection weights (shape: [batch, seq_len, n_layers])
            routing: Routing strategy:
                     - 'hard': Select single layer with highest weight (discrete)
                     - 'soft': Weighted combination of all layers (continuous)
            
        Returns:
            Tuple of:
                - batch_layer_weights: One-hot or soft layer weights 
                                      (shape: [batch, seq_len, n_layers])
                - routed_x: Selected/combined activations 
                           (shape: [batch, seq_len, hidden_size])
                           
        Raises:
            ValueError: If routing method is not supported
        """
        if routing == 'hard':
            # Hard routing: Select single best layer per token
            max_weights, target_layer = router_weights.max(dim=-1)
            # (batch, seq_len) -> indices of best layers
            
            # Create one-hot encoding for selected layers
            batch_layer_weights = torch.zeros_like(router_weights)
            batch_layer_weights.scatter_(2, target_layer.unsqueeze(-1), 1.0)
            
            # Gather activations from selected layers
            # Expand indices to match x's dimensions
            indices = target_layer.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, x.size(-1)
            )
            routed_x = torch.gather(x, 2, indices).squeeze(2)
            
            # Weight by router confidence
            routed_x = routed_x * max_weights.unsqueeze(-1)
        
        elif routing == 'soft':
            # Soft routing: Weighted combination of all layers
            # Expand router weights for broadcasting
            weights_expanded = router_weights.unsqueeze(-1)  # (batch, seq_len, n_layers, 1)
            
            # Weight each layer's activations
            weighted_hidden_states = x * weights_expanded
            
            # Sum across layers
            routed_x = weighted_hidden_states.sum(dim=2)  # (batch, seq_len, hidden_size)
            
            # Layer weights are the router weights themselves
            batch_layer_weights = router_weights
        
        else:
            raise ValueError(
                f'Unsupported routing method: {routing}. '
                f'Expected one of ["hard", "soft"].'
            )

        return batch_layer_weights, routed_x

    def forward(
        self, 
        x: torch.Tensor, 
        aggre: str, 
        routing: str,
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through RouteSAE.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            aggre: Aggregation method for router: 'sum' or 'mean'
            routing: Routing strategy: 'hard' or 'soft'
            infer_k: Optional override for k during inference
            theta: Optional threshold for activation during inference
            
        Returns:
            Tuple of:
                - layer_weights: Selected/weighted layers (batch, seq_len, n_layers)
                - routed_x: Routed activations (batch, seq_len, hidden_size)
                - latents: SAE latent representation (batch, seq_len, latent_size)
                - x_hat: Reconstructed activations (batch, seq_len, hidden_size)
                - router_weights: Raw router weights before routing (batch, seq_len, n_layers)
                
        Note:
            For hard routing during inference, you can analyze which layers were selected
            most frequently using layer_weights. For soft routing, layer_weights shows
            the contribution of each layer.
            
        Example:
            >>> sae = RouteSAE(hidden_size=768, n_layers=16, latent_size=4096, k=64)
            >>> x = torch.randn(32, 128, 16, 768)  # multi-layer activations
            >>> layer_w, routed, latents, recon, router_w = sae(x, 'sum', 'hard')
            >>> # Analyze layer usage
            >>> layer_usage = layer_w.sum(dim=(0, 1))  # Count selections per layer
            >>> print(f"Most used layer: {layer_usage.argmax().item()}")
        """
        # Step 1: Compute router weights (which layers to use)
        router_weights = self.get_router_weights(x, aggre)
        
        # Step 2: Apply routing to select/combine layer activations
        batch_layer_weights, routed_x = self.get_sae_input(x, router_weights, routing)
        
        # Step 3: Process routed activations through SAE
        latents, x_hat = self.sae(routed_x, infer_k=infer_k, theta=theta)
        
        return batch_layer_weights, routed_x, latents, x_hat, router_weights
    

class Crosscoder(nn.Module):
    """
    Crosscoder: Multi-layer Sparse Autoencoder with cross-layer processing.
    
    Crosscoder processes multiple layers simultaneously by:
    1. Encoding each layer separately with layer-specific encoders
    2. Summing encoded representations into shared latent space
    3. Decoding to reconstruct each layer with layer-specific decoders
    
    This architecture:
    - Captures cross-layer dependencies and shared features
    - Learns features that span multiple processing depths
    - Uses fewer parameters than independent SAEs per layer
    - Identifies features that are consistent across layers
    
    Unlike RouteSAE which routes tokens to specific layers, Crosscoder
    always processes all specified layers, learning to combine information
    across the layer dimension.
    
    Reference: Crosscoder paper
    
    Attributes:
        start_layer: First layer to process
        end_layer: Last layer to process (exclusive)
        latent_bias: Shared bias in latent space
        encoder: ModuleList of per-layer encoders
        decoder: ModuleList of per-layer decoders
        hidden_size: Dimension of layer activations
        latent_size: Dimension of shared latent space
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        n_layers: int, 
        latent_size: int
    ) -> None:
        """
        Initialize Crosscoder.
        
        Args:
            hidden_size: Dimensionality of layer activations
            n_layers: Total number of layers in the language model
            latent_size: Dimension of shared latent space
            
        Raises:
            ValueError: If parameters are invalid
        """
        if hidden_size <= 0 or n_layers <= 0 or latent_size <= 0:
            raise ValueError("All dimensions must be positive")
        if n_layers < 4:
            raise ValueError(f"n_layers ({n_layers}) should be at least 4 for meaningful cross-layer processing")
            
        super(Crosscoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # Focus on middle layers (typically most informative)
        self.start_layer = n_layers // 4
        self.end_layer = n_layers * 3 // 4 + 1
        self.n_processed_layers = self.end_layer - self.start_layer
        
        # Shared latent bias (all layers contribute to same latent space)
        self.latent_bias = nn.Parameter(torch.zeros(latent_size))

        # Per-layer encoders (each layer has its own encoding weights)
        self.encoder = nn.ModuleList([
            nn.Linear(hidden_size, latent_size, bias=False) 
            for _ in range(self.n_processed_layers)
        ])
        
        # Per-layer decoders (each layer has its own decoding weights)
        self.decoder = nn.ModuleList([
            nn.Linear(latent_size, hidden_size, bias=False) 
            for _ in range(self.n_processed_layers)
        ])
        
        # Initialize with tied weights (encoder_i.T = decoder_i for each layer)
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize decoder weights as transpose of encoder for each layer."""
        with torch.no_grad():
            for i in range(len(self.decoder)):
                self.decoder[i].weight.data = self.encoder[i].weight.data.T.clone()
    
    def pre_acts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode each layer and sum into shared latent space.
        
        This is the key innovation of Crosscoder: multiple layers contribute
        to the same latent representation, allowing the model to learn features
        that span across processing depths.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            
        Returns:
            Summed pre-activations in latent space (shape: [batch, seq_len, latent_size])
        """
        batch_size, seq_len, n_layers, _ = x.shape
        
        # Initialize tensor for layer-wise pre-activations
        pre_acts = torch.zeros(
            batch_size, seq_len, self.n_processed_layers, self.latent_size, 
            device=x.device, dtype=x.dtype
        )
        
        # Encode each layer separately
        for i in range(self.n_processed_layers):
            pre_acts[:, :, i, :] = self.encoder[i](x[:, :, i, :])
        
        # Sum across layers and add shared bias
        # This creates a unified latent representation combining all layers
        summed_pre_acts = pre_acts.sum(dim=2) + self.latent_bias
        
        return summed_pre_acts
    
    def get_latents(
        self, 
        pre_acts: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply sparsity to shared latent representation.
        
        Args:
            pre_acts: Summed pre-activations (shape: [batch, seq_len, latent_size])
            infer_k: Optional top-k constraint for inference
            theta: Optional threshold for sparsity
            
        Returns:
            Sparse latent representation (shape: same as pre_acts)
            
        Raises:
            ValueError: If both infer_k and theta are provided
        """
        if infer_k is not None and theta is not None:
            raise ValueError('Cannot specify both infer_k and theta simultaneously.')
        
        if theta is not None:
            # Threshold-based sparsity
            latents = torch.where(pre_acts > theta, pre_acts, torch.zeros_like(pre_acts))
        elif infer_k is not None:
            # TopK sparsity
            if infer_k > pre_acts.size(-1):
                warnings.warn(f"infer_k ({infer_k}) > latent_size ({pre_acts.size(-1)}), using all latents")
                infer_k = pre_acts.size(-1)
            
            topk_values, topk_indices = torch.topk(pre_acts, infer_k, dim=-1)
            latents = torch.zeros_like(pre_acts)
            latents.scatter_(-1, topk_indices, topk_values)
        else:
            # ReLU sparsity (default, used with L1 regularization)
            latents = F.relu(pre_acts)
        
        return latents
    
    def encode(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Encode multi-layer input to shared sparse latent representation.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            infer_k: Optional top-k constraint for inference
            theta: Optional threshold for sparsity
            
        Returns:
            Shared sparse latent representation (shape: [batch, seq_len, latent_size])
        """
        pre_acts = self.pre_acts(x)
        latents = self.get_latents(pre_acts, infer_k=infer_k, theta=theta)
        return latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode shared latents to reconstruct each layer.
        
        Each layer-specific decoder projects the shared latent representation
        back to that layer's activation space. This allows checking whether
        the shared representation captured layer-specific information.
        
        Args:
            latents: Shared latent representation (shape: [batch, seq_len, latent_size])
            
        Returns:
            Reconstructed multi-layer activations 
            (shape: [batch, seq_len, n_layers, hidden_size])
        """
        batch_size, seq_len, _ = latents.shape
        
        # Initialize tensor for layer-wise reconstructions
        x_hat = torch.zeros(
            batch_size, seq_len, self.n_processed_layers, self.hidden_size,
            device=latents.device, dtype=latents.dtype
        )
        
        # Decode to each layer separately
        for i in range(self.n_processed_layers):
            x_hat[:, :, i, :] = self.decoder[i](latents)
        
        return x_hat
    
    def forward(
        self, 
        x: torch.Tensor, 
        infer_k: Optional[int] = None, 
        theta: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Crosscoder.
        
        Args:
            x: Multi-layer activations (shape: [batch, seq_len, n_layers, hidden_size])
            infer_k: Optional number of top-k elements for inference
            theta: Optional threshold for activation
            
        Returns:
            Tuple of:
                - latents: Shared sparse latent representation 
                          (shape: [batch, seq_len, latent_size])
                - x_hat: Reconstructed multi-layer activations 
                        (shape: [batch, seq_len, n_layers, hidden_size])
                        
        Example:
            >>> crosscoder = Crosscoder(hidden_size=768, n_layers=16, latent_size=4096)
            >>> x = torch.randn(32, 128, 16, 768)  # multi-layer activations
            >>> latents, x_hat = crosscoder(x)
            >>> print(latents.shape)  # Shared representation
            torch.Size([32, 128, 4096])
            >>> print(x_hat.shape)    # Layer-wise reconstructions
            torch.Size([32, 128, 16, 768])
            >>> # Check reconstruction quality per layer
            >>> layer_errors = ((x - x_hat) ** 2).mean(dim=(0, 1, 3))
            >>> print("Reconstruction error per layer:", layer_errors)
        """
        latents = self.encode(x, infer_k=infer_k, theta=theta)
        x_hat = self.decode(latents)
        return latents, x_hat
    
