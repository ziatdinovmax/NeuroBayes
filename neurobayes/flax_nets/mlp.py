from typing import Sequence, Tuple, Any
import jax.numpy as jnp
import flax.linen as nn



class MLPLayerModule(nn.Module):
    features: int
    activation: Any = None
    dropout: float = 0.0
    layer_name: str = 'dense'
    
    @nn.compact
    def __call__(self, x, enable_dropout: bool = True):
        x = nn.Dense(features=self.features, name=self.layer_name)(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout > 0:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=not enable_dropout)
        return x
    

class FlaxMLP(nn.Module):
    hidden_dims: Sequence[int]
    target_dim: int
    activation: str = 'tanh'
    hidden_dropout: float = 0.0
    output_dropout: float = 0.0
    classification: bool = False
    

    @nn.compact
    def __call__(self, x: jnp.ndarray, enable_dropout: bool = True) -> jnp.ndarray:
        """Forward pass of the MLP"""
        
        # Set the activation function
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Build the hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            layer = MLPLayerModule(
                features=hidden_dim,
                activation=activation_fn,
                dropout=self.hidden_dropout,
                layer_name=f"Dense{i}"
            )
            x = layer(x, enable_dropout)

        # Output layer
        output_layer = MLPLayerModule(
            features=self.target_dim,
            activation=nn.softmax if self.classification else None,
            dropout=self.output_dropout,
            layer_name=f"Dense{len(self.hidden_dims)}"
        )
        x = output_layer(x, enable_dropout)

        return x
    

class FlaxMLP2Head(nn.Module):
    hidden_dims: Sequence[int]
    target_dim: int
    activation: str = 'tanh'
    hidden_dropout: float = 0.0
    output_dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, enable_dropout: bool = True
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass of the 2-headed MLP"""
        
        # Set the activation function
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Build the shared hidden layers
        for i, hidden_dim in enumerate(self.hidden_dims):
            layer = MLPLayerModule(
                features=hidden_dim,
                activation=activation_fn,
                dropout=self.hidden_dropout,
                layer_name=f"Dense{i}"
            )
            x = layer(x, enable_dropout)

        # Mean head
        mean_layer = MLPLayerModule(
            features=self.target_dim,
            activation=None,
            dropout=self.output_dropout,
            layer_name="MeanHead"
        )
        mean = mean_layer(x, enable_dropout)

        # Variance head
        var_layer = MLPLayerModule(
            features=self.target_dim,
            activation=nn.softplus,
            dropout=self.output_dropout,
            layer_name="VarianceHead"
        )
        variance = var_layer(x, enable_dropout)

        return mean, variance