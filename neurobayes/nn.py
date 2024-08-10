from typing import Callable, Dict, List, Tuple, Sequence, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


def get_mlp(hidden_dim: List[int], activation: str = 'tanh', name: str = "main"
            ) -> Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]:
    """Returns a function that represents an MLP for a given hidden_dim"""
    if activation not in ['silu', 'tanh']:
        raise NotImplementedError("Use either 'silu' or 'tanh' for activation")
    activation_fn = jnp.tanh if activation == 'tanh' else jax.nn.silu

    def mlp(X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """MLP for a single MCMC sample of weights and biases, handling arbitrary number of layers"""
        h = X
        for i in range(len(hidden_dim)):
            h = activation_fn(jnp.matmul(h, params[f"{name}_w{i}"]) + params[f"{name}_b{i}"])
        # No non-linearity after the last layer
        z = jnp.matmul(h, params[f"{name}_w{len(hidden_dim)}"]) + params[f"{name}_b{len(hidden_dim)}"]
        return z
    return mlp


def get_heteroskedastic_mlp(hidden_dim: List[int], activation: str = 'tanh'
                            ) -> Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], Tuple[jnp.ndarray, jnp.ndarray]]:
    """Returns a function that represents an MLP for given hidden dimensions, outputting mean and variance"""
    if activation not in ['silu', 'tanh']:
        raise NotImplementedError("Use either 'silu' or 'tanh' for activation")
    activation_fn = jnp.tanh if activation == 'tanh' else jax.nn.silu

    def mlp(X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = X
        for i in range(len(hidden_dim)):
            h = activation_fn(jnp.matmul(h, params[f"w{i}"]) + params[f"b{i}"])
        # Separate output layers for mean and variance
        mean = jnp.matmul(h, params['w_mean']) + params['b_mean']
        log_variance = jnp.matmul(h, params['w_variance']) + params['b_variance']
        variance = jnp.exp(log_variance)  # ensure variance is positive
        return mean, variance
    return mlp


class FlaxMLP(nn.Module):
    # Define the structure of the network
    hidden_dims: Sequence[int]  # List of hidden layer sizes
    output_dim: int             # Number of units in the output layer
    activation: str = 'tanh'    # Type of activation function, default is 'tanh'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the MLP.
        """
        # Set the activation function based on the input parameter
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Build the hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=hidden_dim, name=f"Dense{i}")(x)
            x = activation_fn(x)  # Apply activation function

        # Output layer, no activation function applied here
        if self.output_dim:
            x = nn.Dense(
                features=self.output_dim,
                name=f"Dense{len(self.hidden_dims)}")(x)

        return x


class FlaxMLP2Head(nn.Module):
    # Define the structure of the network
    hidden_dims: Sequence[int]  # List of hidden layer sizes
    output_dim: int             # Number of units in each output head (mean and variance)
    activation: str = 'tanh'    # Type of activation function, default is 'tanh'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the MLP with two heads: one for mean and one for variance.
        """
        # Set the activation function based on the input parameter
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Build the hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=hidden_dim, name=f"Dense{i}")(x)
            x = activation_fn(x)  # Apply activation function

        # Two separate output layers, one for mean and one for variance
        # Mean head
        mean = nn.Dense(features=self.output_dim, name="MeanHead")(x)
        
        # Variance head
        variance = nn.Dense(features=self.output_dim, name="VarianceHead")(x)
        variance = nn.softplus(variance)  # ensure the output is positive

        return mean, variance


class FlaxMultiFidelityMLP(nn.Module):
    input_dim: int
    backbone_dims: Sequence[int]
    output_sizes: Sequence[int]
    num_fidelity_levels: int
    activation: str = 'tanh'

    def setup(self):
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Embedding layer for fidelity level
        self.fidelity_embedding = nn.Embed(
            num_embeddings=self.num_fidelity_levels,
            features=self.backbone_dims[0]
        )

        # Backbone
        layers = []
        for i, dim in enumerate(self.backbone_dims):
            layers.append(nn.Dense(dim, name=f'backbone_dense_{i}'))
            layers.append(activation_fn)
        self.backbone = nn.Sequential(layers)

        # Heads fore different tasks
        self.heads = [nn.Dense(output_size, name=f'head_{i}') 
                      for i, output_size in enumerate(self.output_sizes)]

    def __call__(self, x):
        # Split input features and fidelity level
        features, fidelity = x[:, :-1], x[:, -1].astype(jnp.int32)

        # Get fidelity embedding
        fidelity_emb = self.fidelity_embedding(fidelity)

        # Concatenate features with fidelity embedding
        x = jnp.concatenate([features, fidelity_emb], axis=-1)

        # Pass through backbone
        x = self.backbone(x)

        # Apply heads
        return [head(x) for head in self.heads]