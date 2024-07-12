from typing import List, Callable
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .hskbnn import HeteroskedasticBNN
from .nn import FlaxMLP


class HeteroskedasticBNN2(HeteroskedasticBNN):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 variance_model: Callable,
                 variance_model_prior: Callable,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 ) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, activation)

        hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
        self.nn = FlaxMLP(hdim, output_dim, activation)
        self.variance_model = variance_model
        self.variance_model_prior = variance_model_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""
        
        net = random_flax_module(
            "nn", self.nn, input_shape=(1, self.input_dim),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", net(X))

        # Sample noise variance according to the provided model
        var_params = self.variance_model_prior()
        sig = numpyro.deterministic("sig", self.variance_model(X, var_params))

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)
