from typing import Dict, Optional, Type
import jax.numpy as jnp
import flax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from .utils import split_mlp


class PartialBNN(BNN):
    """
    Partially stochastic NN
    """

    def __init__(self,
                 deterministic_nn: Type[flax.linen.Module],
                 deterministic_weights: Dict[str, jnp.ndarray],
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        (self.truncated_mlp, self.truncated_params,
         self.last_layer_mlp) = split_mlp(
             deterministic_nn, deterministic_weights)[:-1]
        if noise_prior is None:
            noise_prior = dist.HalfNormal(1.0)
        self.noise_prior = noise_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        X = self.truncated_mlp.apply({'params': self.truncated_params}, X)

        print(X.shape)
        print(self.truncated_mlp.output_dim)

        bnn = random_flax_module(
            "nn", self.last_layer_mlp, input_shape=(1, self.truncated_mlp.hidden_dims[-1]),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", bnn(X))

        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)
