from typing import List, Callable
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .bnn import BNN
from .nn import get_mlp
from .priors import get_mlp_prior


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
        self.nn = get_mlp(hdim, activation)
        self.nn_prior = get_mlp_prior(input_dim, output_dim, hdim)
        self.variance_model = variance_model
        self.variance_model_prior = variance_model_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        # Sample NN parameters
        nn_params = self.nn_prior()
        # Pass inputs through a NN with the sampled parameters
        mu = self.nn(X, nn_params)

        # Sample noise variance according to the provided model
        var_params = self.variance_model_prior()
        sig = self.variance_model(X, var_params)

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def sample_single_posterior_predictive(self, rng_key, X_new, params, n_draws):
        loc = self.nn(X_new, params)
        sigma = self.variance_model(X_new, params)
        sample = dist.Normal(loc, sigma).sample(rng_key, (n_draws,)).mean(0)
        return loc, sample

    def predict_noise(self, X_new: jnp.ndarray) -> jnp.ndarray:
        X_new = self.set_data(X_new)
        samples = self.get_samples(chain_dim=False)
        predictive = jax.vmap(lambda params: self.variance_model(X_new, params))
        sigma = predictive(samples)
        return sigma.mean(0)
