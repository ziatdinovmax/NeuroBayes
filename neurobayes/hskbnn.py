from typing import List
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .bnn import BNN
from .nn import get_heteroskedastic_mlp
from .priors import get_heteroskedastic_mlp_prior


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
        self.nn = get_heteroskedastic_mlp(hdim, activation)
        self.nn_prior = get_heteroskedastic_mlp_prior(input_dim, output_dim, hdim)

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        # Sample NN parameters
        nn_params = self.nn_prior()
        # Pass inputs through a NN with the sampled parameters
        mu, sig = self.nn(X, nn_params)

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def sample_single_posterior_predictive(self, rng_key, X_new, params, n_draws):
        loc, sigma = self.nn(X_new, params)
        sample = dist.Normal(loc, sigma).sample(rng_key, (n_draws,)).mean(0)
        return loc, sample
