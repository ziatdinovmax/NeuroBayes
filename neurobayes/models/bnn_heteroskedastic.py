from typing import List, Optional, Dict, Type
import jax.random as jra
import jax.numpy as jnp
import flax
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from ..utils.utils import put_on_device


class HeteroskedasticBNN(BNN):
    """
    Heteroskedastic Bayesian Neural Network for input-dependent observational noise

    Args:
        architecture: a Flax model
        pretrained_priors (Dict, optional):
            Dictionary with pre-trained weights for the provided model architecture.
            These weight values will be used to initialize prior distributions in BNN.
    """
    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 pretrained_priors: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        super().__init__(architecture, pretrained_priors=pretrained_priors)

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              pretrained_priors: Dict = None,
              priors_sigma: float = 1.0,
              **kwargs) -> None:
        """Heteroskedastic BNN model"""

        if self.pretrained_priors is not None:
            pretrained_priors = {}
            for module_dict in self.pretrained_priors.values():
                pretrained_priors.update(module_dict)
            self.pretrained_priors = pretrained_priors

        def prior(name, shape):
            if self.pretrained_priors is not None:
                param_path = name.split('.')
                layer_name = param_path[0]
                param_type = param_path[-1]  # kernel or bias
                return dist.Normal(self.pretrained_priors[layer_name][param_type], priors_sigma)
            return dist.Normal(0., priors_sigma)

        input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],)

        net = random_flax_module(
            "nn", self.nn, input_shape=(1, *input_shape), prior=prior)

        # Pass inputs through a NN with the sampled parameters
        mu, sig = net(X)
        # Register values with numpyro
        mu = numpyro.deterministic("mu", mu)
        sig = numpyro.deterministic("sig", sig)

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def predict_noise(self, X_new: jnp.ndarray,
                      device: Optional[str] = None) -> jnp.ndarray:
        """
        Predict likely values of noise for new data
        and associated uncertainty in the noise prediction
        """
        X_new = self.set_data(X_new)
        samples = self.get_samples()
        X_new, samples = put_on_device(device, X_new, samples)
        pred = self.sample_from_posterior(
            jra.PRNGKey(0), X_new, samples, return_sites=['sig'])
        return pred['sig'].mean(0), pred['sig'].var(0)
