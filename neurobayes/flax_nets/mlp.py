from typing import Sequence, Tuple, Any
import jax.numpy as jnp
import flax.linen as nn



class MLPLayerModule(nn.Module):
    features: int
    activation: Any = None
    layer_name: str = 'dense'
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features, name=self.layer_name)(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    

class FlaxMLP(nn.Module):
    hidden_dims: Sequence[int]
    target_dim: int
    activation: str = 'tanh'
    classification: bool = False  # Explicit flag for classification tasks

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the MLP"""
        
        # Set the activation function
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Build the hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            layer = MLPLayerModule(
                features=hidden_dim,
                activation=activation_fn,
                layer_name=f"Dense{i}"
            )
            x = layer(x)

        # Output layer
        output_layer = MLPLayerModule(
            features=self.target_dim,
            activation=nn.softmax if self.classification else None,
            layer_name=f"Dense{len(self.hidden_dims)}"
        )
        x = output_layer(x)

        return x
    

class FlaxMLP2Head(nn.Module):
    hidden_dims: Sequence[int]
    target_dim: int
    activation: str = 'tanh'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of the 2-headed MLP"""
        
        # Set the activation function
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Build the shared hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            layer = MLPLayerModule(
                features=hidden_dim,
                activation=activation_fn,
                layer_name=f"Dense{i}"
            )
            x = layer(x)

        # Mean head
        mean_layer = MLPLayerModule(
            features=self.target_dim,
            activation=None,
            layer_name="MeanHead"
        )
        mean = mean_layer(x)

        # Variance head
        var_layer = MLPLayerModule(
            features=self.target_dim,
            activation=nn.softplus,
            layer_name="VarianceHead"
        )
        variance = var_layer(x)

        return mean, variance