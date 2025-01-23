from typing import List, Callable, Optional, Dict, Type
import jax.numpy as jnp
import flax
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn_heteroskedastic import HeteroskedasticBNN

class VarianceModelHeteroskedasticBNN(HeteroskedasticBNN):
    """
    Variance model based heteroskedastic Bayesian Neural Network

    Args:
        architecture: a Flax model.
        variance_model (Callable): Function to compute the variance given inputs and parameters.
        variance_model_prior (Callable): Function to sample prior parameters for the variance model.
    """
    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 variance_model: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]],
                 variance_model_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]],
                 ) -> None:
        super().__init__(architecture)
        self.nn = architecture
        self.variance_model = variance_model
        self.variance_model_prior = variance_model_prior

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, priors_sigma: float = 1.0, **kwargs) -> None:
        """Heteroskedastic BNN model"""

        input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],)

        net = random_flax_module(
            "nn", self.nn, input_shape=(1, *input_shape),
            prior=(lambda name, shape:  dist.Normal(0, priors_sigma)))

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", net(X, enable_dropout=False))

        # Sample noise variance according to the provided model
        var_params = self.variance_model_prior()
        sig = numpyro.deterministic("sig", self.variance_model(X, var_params))
        
        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)
