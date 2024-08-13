from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn


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

    
class Embedding(nn.Module):
    embedding_dim: int

    @nn.compact
    def __call__(self, x):
        x_one_hot = jax.nn.one_hot(x, num_classes=x.max() + 1)
        return nn.Dense(self.embedding_dim)(x_one_hot)


class FlaxMultiTaskMLP(nn.Module):
    backbone_dims: Sequence[int]
    output_sizes: Sequence[int]
    num_tasks: int
    activation: str = 'tanh'
    embedding_dim: int = None

    def setup(self):
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Embedding layer for tasks
        self.task_embedding = Embedding(
            self.backbone_dims[-1] if not self.embedding_dim else self.embedding_dim
        )

        # Backbone
        layers = []
        for i, dim in enumerate(self.backbone_dims):
            layers.append(nn.Dense(dim, name=f'backbone_dense_{i}'))
            layers.append(activation_fn)
        self.backbone = nn.Sequential(layers)

        # Heads fore different tasks
        self.heads = [
            nn.Sequential([
                nn.Dense(self.backbone_dims[-1], name=f'head_{i}_dense_1'),
                activation_fn,
                nn.Dense(output_size, name=f'head_{i}_dense_2')
            ]) for i, output_size in enumerate(self.output_sizes)
        ]

    def __call__(self, x):
        # Split input features and task level
        int_dtype = to_int_dtype(x)
        features, task = x[:, :-1], x[:, -1].astype(int_dtype)

        # Pass through backbone
        features = self.backbone(features)

        # Get task embedding
        task_emb = self.task_embedding(task)

        # Concatenate features with task embedding
        x = jnp.concatenate([features, task_emb], axis=-1)

        # Apply heads
        return jnp.hstack([head(x) for head in self.heads])


def to_int_dtype(arr):
    dtype_map = {
        jnp.float32: jnp.int32,
        jnp.float64: jnp.int64
    }
    target_dtype = dtype_map.get(arr.dtype)
    if target_dtype is None:
        raise ValueError(
            "Unsupported data type. Please input an array with float32 or float64 dtype.")
    return target_dtype