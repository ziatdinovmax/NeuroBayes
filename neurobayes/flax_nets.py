from typing import Sequence, Tuple
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

    subnet1 = FlaxMLP(model.hidden_dims[:-n_layers], output_dim=0, activation=model.activation)
    subnet2 = FlaxMLP(model.hidden_dims[-n_layers:], output_dim=out_dim, activation=model.activation)

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
