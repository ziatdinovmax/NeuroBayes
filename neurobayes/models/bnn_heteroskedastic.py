from typing import List, Optional, Dict
import jax.random as jra
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from ..flax_nets import FlaxMLP2Head, FlaxConvNet2Head
from ..utils.utils import put_on_device


class HeteroskedasticBNN(BNN):
    """
    Heteroskedastic Bayesian Neural Net
    """
    def __init__(self,
                 target_dim: int,
                 hidden_dim: List[int] = None,
                 conv_layers: List[int] = None,
                 input_dim: int = None,
                 activation: str = 'tanh',
                 ) -> None:
        super().__init__(target_dim, hidden_dim, conv_layers, activation)

        # Override the default mlp/convnet with heteroskedastic versions
        if conv_layers:
            hdim = hidden_dim if hidden_dim is not None else [int(conv_layers[-1] * 2),]
            self.nn = FlaxConvNet2Head(input_dim, conv_layers, hdim, target_dim, activation)
        else:
            hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
            self.nn = FlaxMLP2Head(hdim, target_dim, activation)        

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              pretrained_priors: Dict = None,
              priors_sigma: float = 1.0,
              **kwargs) -> None:
        """Heteroskedastic BNN probabilistic model"""

        def prior(name, shape):
            if pretrained_priors is not None:
                param_path = name.split('.')
                mean = pretrained_priors
                for path in param_path:
                    mean = mean[path]
                return dist.Normal(mean, priors_sigma)
            else:
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
        X_new = self.set_data(X_new)
        samples = self.get_samples()
        X_new, samples = put_on_device(device, X_new, samples)
        pred = self.sample_from_posterior(
            jra.PRNGKey(0), X_new, samples, return_sites=['sig'])
        return pred['sig'].mean(0)
