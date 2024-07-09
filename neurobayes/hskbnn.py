from typing import List, Optional
import jax.random as jra
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from .nn import FlaxMLP2Head
from .utils import put_on_device


class HeteroskedasticBNN(BNN):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 ) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, activation)

        # Override the MLP functions with heteroskedastic versions
        hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
        self.nn = FlaxMLP2Head(hdim, output_dim, activation)

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        net = random_flax_module(
            "nn", self.nn, input_shape=(1, self.input_dim),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))

        # Pass inputs through a NN with the sampled parameters
        mu, sig = net(X)
        # Register values with numpyro
        mu = numpyro.deterministic("mu", mu)
        sig = numpyro.deterministic("sig", sig)

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

    # def sample_single_posterior_predictive(self, rng_key, X_new, params, n_draws):
    #     loc, sigma = self.nn(X_new, params)
    #     sample = dist.Normal(loc, sigma).sample(rng_key, (n_draws,)).mean(0)
    #     return loc, sample

    # def predict_noise(self, X_new: jnp.ndarray) -> jnp.ndarray:
    #     X_new = self.set_data(X_new)
    #     samples = self.get_samples(chain_dim=False)
    #     predictive = jax.vmap(lambda params: self.nn(X_new, params))
    #     sigma = predictive(samples)[-1]
    #     return sigma.mean(0)

    # def get_prediction_and_noise_stats(self, X_new: jnp.ndarray) -> jnp.ndarray:
    #     X_new = self.set_data(X_new)
    #     samples = self.get_samples(chain_dim=False)
    #     predictive = jax.vmap(lambda params: self.nn(X_new, params))
    #     mu, sig = predictive(samples)
    #     return (mu.mean(0), mu.var(0)), (sig.mean(0), sig.var(0))
