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


class FlaxMLP2Head(FlaxMLP):
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = super().__call__(x)
        if self.output_dim:
            mean = nn.Dense(features=self.output_dim, name="MeanHead")(x)
            variance = nn.Dense(features=self.output_dim, name="VarianceHead")(x)
            variance = nn.softplus(variance)
            return mean, variance
        return x

    
class Embedding(nn.Module):
    embedding_dim: int
    num_tasks: int

    @nn.compact
    def __call__(self, x):
        x_one_hot = jax.nn.one_hot(x, num_classes=self.num_tasks)
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
            self.backbone_dims[-1]*2 if not self.embedding_dim else self.embedding_dim,
            self.num_tasks
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
                nn.Dense(self.backbone_dims[-1], name=f'head_{i}_dense_2'),
                activation_fn,
                nn.Dense(output_size, name=f'head_{i}_dense_3')
            ]) for i, output_size in enumerate(self.output_sizes)
        ]

    def __call__(self, x):
        # Split input features and task level
        features, task = x[:, :-1], x[:, -1].astype(jnp.int32)

        # Pass through backbone
        features = self.backbone(features)

        # Get task embedding
        task_emb = self.task_embedding(task)

        # Concatenate features with task embedding
        x = jnp.concatenate([features, task_emb], axis=-1)

        # Apply heads
        return jnp.hstack([head(x) for head in self.heads])
