from typing import List, Callable
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn_heteroskedastic import HeteroskedasticBNN
from ..flax_nets import FlaxMLP, FlaxConvNet


class VarianceModelHeteroskedasticBNN(HeteroskedasticBNN):
    """
    Variance model based heteroskedastic Bayesian Neural Net
    """
    def __init__(self,
                 target_dim: int,
                 variance_model: Callable,
                 variance_model_prior: Callable,
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
        """BNN probabilistic model"""

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
