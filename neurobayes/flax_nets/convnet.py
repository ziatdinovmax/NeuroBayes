from typing import Sequence, Tuple, Union, Callable, Dict, Any
import jax.numpy as jnp
import flax.linen as nn

from .mlp import FlaxMLP, FlaxMLP2Head


class FlaxConvNet(nn.Module):
    input_dim: int
    conv_layers: Sequence[int]
    fc_layers: Sequence[int]
    output_dim: int
    activation: str = 'tanh'
    kernel_size: Union[int, Tuple[int, ...]] = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu
        conv, pool = get_conv_and_pool_ops(self.input_dim, self.kernel_size)

        # Convolutional layers
        for i, filters in enumerate(self.conv_layers):
            x = conv(features=filters, name=f"Conv{i}")(x)
            x = activation_fn(x)
            x = pool(x)

        # Flatten the output for the fully connected layers
        x = x.reshape((x.shape[0], -1))

        # Use FlaxMLP for fully connected layers
        x = FlaxMLP(
            hidden_dims=self.fc_layers,
            output_dim=self.output_dim,
            activation=self.activation)(x)

        return x


class FlaxConvNet2Head(nn.Module):

    input_dim: int
    conv_layers: Sequence[int]
    fc_layers: Sequence[int]
    output_dim: int
    activation: str = 'tanh'
    kernel_size: Union[int, Tuple[int, ...]] = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu
        conv, pool = get_conv_and_pool_ops(self.input_dim, self.kernel_size)

        # Convolutional layers
        for i, filters in enumerate(self.conv_layers):
            x = conv(features=filters, name=f"Conv{i}")(x)
            x = activation_fn(x)
            x = pool(x)

        # Flatten the output for the fully connected layers
        x = x.reshape((x.shape[0], -1))

        mean, var = FlaxMLP2Head(
            hidden_dims=self.fc_layers,
            output_dim=self.output_dim,
            activation=self.activation)(x)
        
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
