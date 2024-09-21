from typing import Sequence, Tuple
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
    output_dim: int             # Number of units in each output layer
    activation: str = 'tanh'    # Type of activation function, default is 'tanh'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass of the 2-headed MLP, outputting mean and variance.
        """
        # Set the activation function based on the input parameter
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Build the shared hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=hidden_dim, name=f"Dense{i}")(x)
            x = activation_fn(x)  # Apply activation function

        # Mean head
        mean = nn.Dense(features=self.output_dim, name="MeanHead")(x)

        # Variance head
        variance = nn.Dense(features=self.output_dim, name="VarianceHead")(x)
        variance = nn.softplus(variance)

        return mean, variance

    
class Embedding(nn.Module):
    embedding_dim: int
    num_tasks: int

    @nn.compact
    def __call__(self, x):
        x_one_hot = jax.nn.one_hot(x, num_classes=self.num_tasks)
        return nn.Dense(self.embedding_dim)(x_one_hot)


class EmbeddingBackbone(nn.Module):
    backbone_dims: Sequence[int]
    num_tasks: int
    embedding_dim: int
    activation: str = 'tanh'

    @nn.compact
    def __call__(self, x):
        features, task = x[:, :-1], x[:, -1].astype(jnp.int32)
        
        # Task embedding
        task_emb = Embedding(self.embedding_dim, self.num_tasks)(task)
        
        # Backbone
        backbone = FlaxMLP(self.backbone_dims, self.backbone_dims[-1], self.activation)
        features = backbone(features)
        
        return jnp.concatenate([features, task_emb], axis=-1)


class FlaxMultiTaskMLP(nn.Module):
    backbone_dims: Sequence[int]
    head_dims: Sequence[int]
    output_dim: int
    num_tasks: int
    activation: str = 'tanh'
    embedding_dim: int = None

    def setup(self):
        self.backbone = EmbeddingBackbone(self.backbone_dims, self.num_tasks, self.embedding_dim, self.activation)
        self.head = FlaxMLP(self.head_dims, self.output_dim, self.activation)

    def __call__(self, x):
        x = self.backbone(x)
        return self.head(x)


class FlaxMultiTaskMLP2(nn.Module):
    backbone_dims: Sequence[int]
    output_sizes: Sequence[int]
    task_sizes: Sequence[int]
    num_tasks: int
    embedding_dim: int
    activation: str = 'tanh'
    
    def setup(self):
        #self.num_tasks = len(self.task_sizes)
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Embedding layer for tasks
        self.task_embedding = Embedding(
            self.embedding_dim,
            self.num_tasks
        )

        # Backbone
        layers = []
        for i, dim in enumerate(self.backbone_dims):
            layers.append(nn.Dense(dim, name=f'backbone_dense_{i}'))
            layers.append(activation_fn)
        self.backbone = nn.Sequential(layers)

        # Heads for different tasks
        self.heads = {
            task_idx: nn.Sequential([
                nn.Dense(self.backbone_dims[-1], name=f'head_{task_idx}_dense_1'),
                activation_fn,
                nn.Dense(self.backbone_dims[-1], name=f'head_{task_idx}_dense_2'),
                activation_fn,
                nn.Dense(self.output_sizes[int(task_idx)], name=f'head_{task_idx}_dense_3')
            ]) for task_idx in self.task_sizes.keys()
        }

    def __call__(self, x):

        # Split input features and task level
        features, task = x[:, :-1], x[:, -1].astype(jnp.int32)

        # Get task embedding
        task_emb = self.task_embedding(task)

        # Concatenate features with task embedding
        x = jnp.concatenate([features, task_emb], axis=-1)

        # Pass through backbone
        features = self.backbone(x)

        # Apply heads based on task assignments
        outputs = []
        start = 0
        for task_idx, size in self.task_sizes.items():
            end = start + size
            task_output = self.heads[task_idx](features[start:end])
            outputs.append(task_output)
            start = end

        return jnp.concatenate(outputs)