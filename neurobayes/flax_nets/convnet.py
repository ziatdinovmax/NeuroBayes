from typing import Sequence, Tuple, Union, Callable, Dict, Any
import jax.numpy as jnp
import flax.linen as nn

from .mlp import FlaxMLP, FlaxMLP2Head


class ConvLayerModule(nn.Module):
    features: int
    input_dim: int
    kernel_size: Union[int, Tuple[int, ...]]
    activation: Any = None
    layer_name: str = None

    @nn.compact
    def __call__(self, x):
        conv, pool = get_conv_and_pool_ops(self.input_dim, self.kernel_size)
        x = conv(features=self.features, name=self.layer_name)(x)
        if self.activation is not None:
            x = self.activation(x)
        x = pool(x)
        return x

class FlaxConvNet(nn.Module):
    input_dim: int
    conv_layers: Sequence[int]
    fc_layers: Sequence[int]
    target_dim: int
    activation: str = 'tanh'
    kernel_size: Union[int, Tuple[int, ...]] = 3
    hidden_dropout: float = 0.0
    output_dropout: float = 0.0
    classification: bool = False  # Explicit flag for classification tasks

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Convolutional layers
        for i, filters in enumerate(self.conv_layers):
            conv_layer = ConvLayerModule(
                features=filters,
                input_dim=self.input_dim,
                kernel_size=self.kernel_size,
                activation=activation_fn,
                layer_name=f"Conv{i}"
            )
            x = conv_layer(x)

        # Flatten the output for the fully connected layers
        x = x.reshape((x.shape[0], -1))

        # Use FlaxMLP for fully connected layers
        x = FlaxMLP(
            hidden_dims=self.fc_layers,
            target_dim=self.target_dim,
            activation=self.activation,
            hidden_dropout=self.hidden_dropout,
            output_dropout=self.output_dropout,
            classification=self.classification)(x)

        return x
    

class FlaxConvNet2Head(nn.Module):
    input_dim: int
    conv_layers: Sequence[int]
    fc_layers: Sequence[int]
    target_dim: int
    activation: str = 'tanh'
    hidden_dropout: float = 0.0
    output_dropout: float = 0.0
    kernel_size: Union[int, Tuple[int, ...]] = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        # Convolutional layers
        for i, filters in enumerate(self.conv_layers):
            conv_layer = ConvLayerModule(
                features=filters,
                input_dim=self.input_dim,
                kernel_size=self.kernel_size,
                activation=activation_fn,
                layer_name=f"Conv{i}"
            )
            x = conv_layer(x)

        # Flatten the output for the fully connected layers
        x = x.reshape((x.shape[0], -1))

        mean, var = FlaxMLP2Head(
            hidden_dims=self.fc_layers,
            target_dim=self.target_dim,
            activation=self.activation,
            hidden_dropout=self.hidden_dropout,
            output_dropout=self.output_dropout)(x)
        
        return mean, var
    

def get_conv_and_pool_ops(input_dim: int, kernel_size: Union[int, Tuple[int, ...]]) -> Tuple[Callable, Callable]:
    """
    Returns appropriate convolution and pooling operations based on input dimension.
    
    Args:
        input_dim (int): Dimension of input data (1, 2, or 3)
        kernel_size (int or tuple): Size of the convolution kernel
    
    Returns:
        tuple: (conv_op, pool_op) - Convolution and pooling operations
    """
    ops: Dict[int, Tuple[Callable, Callable]] = {
        1: (
            lambda *args, **kwargs: nn.Conv(*args, **kwargs, kernel_size=(kernel_size,), padding='SAME'),
            lambda x: nn.max_pool(x, window_shape=(2,), strides=(2,))
        ),
        2: (
            lambda *args, **kwargs: nn.Conv(*args, **kwargs, kernel_size=(kernel_size, kernel_size), padding='SAME'),
            lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        ),
        3: (
            lambda *args, **kwargs: nn.Conv(*args, **kwargs, kernel_size=(kernel_size, kernel_size, kernel_size), padding='SAME'),
            lambda x: nn.max_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        )
    }
    
    if input_dim not in ops:
        raise ValueError(f"Unsupported input dimension: {input_dim}")
    
    return ops[input_dim]
