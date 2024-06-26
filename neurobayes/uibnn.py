from typing import List, Optional
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .bnn import BNN


class UncertainInputBNN(BNN):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 noise_prior: Optional[dist.Distribution] = None,
                 input_noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        super().__init__(input_dim, output_dim, hidden_dim, activation, noise_prior)
        self.input_noise_prior = input_noise_prior


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
        and prior belief about the uncertainty in those inputs.
        """
        n_samples, n_features = X.shape
        if self.input_noise_prior is not None:
            sigma_x_dist = self.input_noise_prior
        else:
            sigma_x_dist = dist.HalfNormal(.1)
        with numpyro.plate("feature_variance_plate", n_features):
            sigma_x = numpyro.sample("sigma_x", sigma_x_dist)
        # Sample input data using the sampled variances
        X_prime = numpyro.sample("X_prime", dist.Normal(X, sigma_x))
        return X_prime
    
    def sample_single_posterior_predictive(self, rng_key, X_new, params, n_draws):
        sigma_x = params['sigma_x']
        X_new_prime = dist.Normal(X_new, sigma_x).sample(rng_key)
        return super().sample_single_posterior_predictive(rng_key, X_new_prime, params, n_draws)

