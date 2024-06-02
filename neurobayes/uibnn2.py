from typing import List, Optional, Callable
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .bnn import BNN
from .nn import get_mlp
from .priors import get_mlp_prior


class UncertainInputBNN2(BNN):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 noise_prior: Optional[dist.Distribution] = None,
                 input_variance_model: Optional[Callable] = None,
                 input_variance_prior: Optional[Callable] = None,
                 ) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, activation, noise_prior)
        if input_variance_model is None or input_variance_prior is None:
            self.input_variance_model = get_mlp(hidden_dim=[4, 2])
            self.input_variance_prior = get_mlp_prior(
                input_dim, output_dim=input_dim, hidden_dim=[4, 2])
        else:
            self.input_variance_model = input_variance_model
            self.input_variance_prior = input_variance_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        # Sample input X
        X_prime = self._sample_input(X)
        # Sample NN parameters
        nn_params = self.nn_prior()
        # Pass inputs through a NN with the sampled parameters
        mu = self.nn(X_prime, nn_params)
        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def _sample_input(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Samples new input values (X_prime) based on the original inputs (X)
        and a predicted uncertainty in those inputs.
        """
        params = self.input_variance_prior()
        logvar_x = self.input_variance_model(X, params)
        var_x = jnp.exp(logvar_x)
        X_prime = numpyro.sample("X_prime", dist.Normal(X, var_x))
        return X_prime
    

    def sample_single_posterior_predictive(self, rng_key, X_new, params, n_draws):
        logvar_x = self.input_variance_model(X_new, params)
        var_x = jnp.exp(logvar_x)
        X_new_prime = dist.Normal(X_new, var_x).sample(rng_key)
        return super().sample_single_posterior_predictive(rng_key, X_new_prime, params, n_draws)

