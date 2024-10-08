from typing import Sequence, Tuple, Union, Callable, Dict, Any
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


def split_mlp(model, params, n_layers: int = 1, out_dim: int = None):
    """
    Splits MLP and its weights into two sub-networks: one with last n layers
    (+ output layer) removed and another one consisting only of those n layers.
    """
    out_dim = out_dim if out_dim is not None else model.output_dim  # there will be a mismatch in last_layer_params if out_dim != model.output_dim


    subnet1 = FlaxMLP(
        model.hidden_dims[:-n_layers] if n_layers > 0 else model.hidden_dims,
        output_dim=0, activation=model.activation)
    subnet2 = FlaxMLP(
        model.hidden_dims[-n_layers:] if n_layers > 0 else [],
        output_dim=out_dim, activation=model.activation)

    subnet1_params = {}
    subnet2_params = {}
    for i, (key, val) in enumerate(params.items()):
        if i < len(model.hidden_dims) - n_layers:
            subnet1_params[key] = val
        else:
            #new_key = f"Dense{i}"
            new_key = f"Dense{i - (len(model.hidden_dims) - n_layers)}"
            subnet2_params[new_key] = val

    return subnet1, subnet1_params, subnet2, subnet2_params


def split_mlp2head(model, params, n_layers: int = 1, out_dim: int = None):
    """
    Splits MLP2Head and its weights into two sub-networks: one with last n layers
    (+ output heads) removed and another one consisting of those n layers and the output heads.
    """
    out_dim = out_dim if out_dim is not None else model.output_dim

    subnet1 = FlaxMLP(model.hidden_dims[:-n_layers], output_dim=0, activation=model.activation)
    subnet2 = FlaxMLP2Head(model.hidden_dims[-n_layers:], output_dim=out_dim, activation=model.activation)

    subnet1_params = {}
    subnet2_params = {}
    
    for i, (key, val) in enumerate(params.items()):
        if key in ['MeanHead', 'VarianceHead']:
            subnet2_params[key] = val
        elif i < len(model.hidden_dims) - n_layers:
            subnet1_params[key] = val
        else:
            #new_key = f"Dense{i}"
            new_key = f"Dense{i - (len(model.hidden_dims) - n_layers)}"
            subnet2_params[new_key] = val

    return subnet1, subnet1_params, subnet2, subnet2_params


class FlaxConvNet(nn.Module):
    # Define the structure of the network
    input_dim: int
    conv_layers: Sequence[int]  # List of number of filters for each conv layer
    fc_layers: Sequence[int]    # List of hidden layer sizes for fully connected layers
    output_dim: int             # Number of units in the output layer
    activation: str = 'tanh'    # Type of activation function, default is 'relu'
    kernel_size: Union[int, Tuple[int, ...]] = 3  # Kernel size for conv layers

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Forward pass of the ConvNet.
        """
        # Set the activation function based on the input parameter
        activation_fn = nn.tanh if self.activation == 'tanh' else nn.silu

        conv, pool = get_conv_and_pool_ops(self.input_dim, self.kernel_size)

        # Convolutional layers
        for i, filters in enumerate(self.conv_layers):
            x = conv(features=filters, name=f"Conv{i}")(x)
            x = activation_fn(x)
            x = pool(x)

        # Flatten the output for the fully connected layers
        x = x.reshape((x.shape[0], -1))

        # Fully connected layers
        for i, hidden_dim in enumerate(self.fc_layers):
            x = nn.Dense(features=hidden_dim, name=f"Dense{i}")(x)
            x = activation_fn(x)

        # Output layer, no activation function applied here
        if self.output_dim:
            x = nn.Dense(features=self.output_dim, name=f"Dense{len(self.fc_layers)}")(x)

        return x
    

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


def split_convnet(model: FlaxConvNet, params: Dict[str, Any]):
    """
    Splits ConvNet and its weights into two parts: convolutional layers and MLP (fully connected layers).
    
    Args:
        model (FlaxConvNet): The original model to split
        params (dict): The parameters of the original model
    
    Returns:
        tuple: (conv_model, conv_params, mlp_model, mlp_params)
    """
    conv_model = FlaxConvNet(
        input_dim=model.input_dim,
        conv_layers=model.conv_layers,
        fc_layers=[],
        output_dim=0,
        activation=model.activation,
        kernel_size=model.kernel_size
    )
    
    mlp_model = FlaxMLP(
        hidden_dims=model.fc_layers,
        output_dim=model.output_dim,
        activation=model.activation
    )

    conv_params = {}
    mlp_params = {}
    
    for key, val in params.items():
        if key.startswith('Conv'):
            conv_params[key] = val
        elif key.startswith('Dense'):
            mlp_params[key] = val

    return conv_model, conv_params, mlp_model, mlp_params
