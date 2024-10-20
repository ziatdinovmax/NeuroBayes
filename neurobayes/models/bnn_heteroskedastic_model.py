from typing import List, Callable, Optional, Dict
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn_heteroskedastic import HeteroskedasticBNN
from ..flax_nets import FlaxMLP, FlaxConvNet


class VarianceModelHeteroskedasticBNN(HeteroskedasticBNN):
    """
    Variance model based heteroskedastic Bayesian Neural Network

    Args:
        target_dim (int): Dimensionality of the target variable.
        variance_model (Callable): Function to compute the variance given inputs and parameters.
        variance_model_prior (Callable): Function to sample prior parameters for the variance model.
        hidden_dim (List[int], optional): List specifying the number of hidden units in each layer 
            of the neural network architecture. Defaults to [32, 16, 8].
        conv_layers (List[int], optional): List specifying the number of filters in each 
            convolutional layer. If provided, enables a ConvNet architecture with max pooling 
            between each conv layer.
        input_dim (int, optional): Input dimensionality (between 1 and 3). Required only for 
            ConvNet architecture.
        activation (str, optional): Non-linear activation function to use. Defaults to 'tanh'.
    """
    def __init__(self,
                 target_dim: int,
                 variance_model: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]],
                 variance_model_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]],
                 hidden_dim: List[int] = None,
                 conv_layers: List[int] = None,
                 input_dim: int = None,
                 activation: str = 'tanh',
                 ) -> None:
        super().__init__(target_dim, hidden_dim, conv_layers, input_dim, activation)
        if conv_layers:
            hdim = hidden_dim if hidden_dim is not None else [int(conv_layers[-1] * 2),]
            self.nn = FlaxConvNet(input_dim, conv_layers, hdim, target_dim, activation)
        else:
            hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
            self.nn = FlaxMLP(hdim, target_dim, activation)
        self.variance_model = variance_model
        self.variance_model_prior = variance_model_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """Heteroskedastic BNN model"""

        input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],)

        net = random_flax_module(
            "nn", self.nn, input_shape=(1, input_shape),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", net(X))

        # Sample noise variance according to the provided model
        var_params = self.variance_model_prior()
        sig = numpyro.deterministic("sig", self.variance_model(X, var_params))

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)
