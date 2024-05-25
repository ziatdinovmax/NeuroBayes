from typing import List, Optional, Callable
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .bnn import BNN


class InputTransformBNN(BNN):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 noise_prior: Optional[dist.Distribution] = None,
                 input_transform_fn: Callable = None,
                 input_transform_fn_prior: Callable = None
                 ) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, activation, noise_prior)
        self.input_transform_fn = input_transform_fn
        self.input_transform_fn_prior = input_transform_fn_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        # Sample parameters of the input transform function
        transform_fn_params = self.input_transform_fn_prior()
        # transform inputs
        X_prime = self.input_transform_fn(X, transform_fn_params)
        # Sample NN parameters
        nn_params = self.nn_prior()
        # Pass inputs through a NN with the sampled parameters
        mu = self.nn(X_prime, nn_params)
        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def sample_single_posterior_predictive(self, rng_key, X_new, params, n_draws):
        X_new_prime = self.input_transform_fn(X_new, params)
        return super().sample_single_posterior_predictive(rng_key, X_new_prime, params, n_draws)

