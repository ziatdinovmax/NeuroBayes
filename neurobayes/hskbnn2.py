from typing import List, Callable, Optional
import jax.numpy as jnp
import jax.random as jra
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from .nn import FlaxMLP
from .utils import put_on_device


class HeteroskedasticBNN2(BNN):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 variance_model: Callable,
                 variance_model_prior: Callable,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 ) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, activation)

        # Override the MLP functions with heteroskedastic versions
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

    def predict_noise(self, X_new: jnp.ndarray,
                      device: Optional[str] = None) -> jnp.ndarray:
        X_new = self.set_data(X_new)
        samples = self.get_samples()
        X_new, samples = put_on_device(device, X_new, samples)
        pred = self.sample_from_posterior(
            jra.PRNGKey(0), X_new, samples, return_sites=['sig'])
        return pred['sig'].mean(0)
