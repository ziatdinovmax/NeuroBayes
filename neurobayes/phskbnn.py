from typing import List, Optional
import jax.random as jra
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro.contrib.module import random_flax_module

from .hskbnn import HeteroskedasticBNN
from .nn import FlaxMLP2Head
from .utils import put_on_device, split_mlp2head


class HeteroskedasticPartialBNN(HeteroskedasticBNN):

    def __init__(self,
                 deterministic_nn,
                 deterministic_weights,
                 ) -> None:
        super().__init__(1, 1)

        (self.truncated_nn, self.truncated_params,
             self.last_layer_nn) = split_mlp2head(
                 deterministic_nn, deterministic_weights)[:-1]

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        X = self.truncated_nn.apply({'params': self.truncated_params}, X)

        bnn = random_flax_module(
            "nn", self.last_layer_nn, input_shape=(1, self.truncated_nn.hidden_dims[-1]),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))

        # Pass inputs through a NN with the sampled parameters
        mu, sig = bnn(X)
        # Register values with numpyro
        mu = numpyro.deterministic("mu", mu)
        sig = numpyro.deterministic("sig", sig)

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)
