from typing import Sequence, Tuple
import jax.numpy as jnp
import flax.linen as nn


class FlaxMLP(nn.Module):
    # Define the structure of the network
    hidden_dims: Sequence[int]  # List of hidden layer sizes
    target_dim: int             # Number of units in the output layer
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
        if self.target_dim:
            x = nn.Dense(
                features=self.target_dim,
                name=f"Dense{len(self.hidden_dims)}")(x)

        return x


class FlaxMLP2Head(nn.Module):
    # Define the structure of the network
    hidden_dims: Sequence[int]  # List of hidden layer sizes
    target_dim: int             # Number of units in each output layer
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
        mean = nn.Dense(features=self.target_dim, name="MeanHead")(x)

        # Variance head
        variance = nn.Dense(features=self.target_dim, name="VarianceHead")(x)
        variance = nn.softplus(variance)

        return mean, variance