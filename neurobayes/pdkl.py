from typing import Dict, Optional, Type, Callable, Tuple
import jax.numpy as jnp
import flax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .dkl import DKL
from .priors import GPPriors
from .utils import split_mlp

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class PartialDKL(DKL):
    """
    Partially stochastic NN
    """

    def __init__(self,
                 latent_dim: int,
                 base_kernel: kernel_fn_type,
                 deterministic_nn: Type[flax.linen.Module],
                 deterministic_weights: Dict[str, jnp.ndarray],
                 priors: Optional[GPPriors] = None,
                 jitter: float = 1e-6,
                 ) -> None:
        super(PartialDKL, self).__init__(None, latent_dim, base_kernel, priors, jitter)
        (self.truncated_nn, self.truncated_params,
         self.nn) = split_mlp(
             deterministic_nn, deterministic_weights, latent_dim)[:-1]

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""
        # Get inputs through a deterministic NN part
        X = self.truncated_nn.apply({'params': self.truncated_params}, X)
        # Fully stochastic NN part
        bnn = random_flax_module(
            "nn", self.nn, input_shape=(1, self.truncated_nn.hidden_dims[-1]),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))
        # Latent encoding
        z = bnn(X)
        # GP Part
        f_loc = jnp.zeros(X.shape[0])
        # Sample kernel parameters
        kernel_params = self.sample_kernel_params()
        # Sample observational noise variance
        noise = self.sample_noise()
        # Compute kernel
        k = self.kernel(z, z, kernel_params, noise, self.jitter)
        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def compute_gp_posterior(self,
                             X_new: jnp.ndarray,
                             X_train: jnp.ndarray,
                             y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        # Pass inputs through a deterministc part of NN
        X_new = self.truncated_nn.apply({'params': self.truncated_params}, X_new)
        X_train = self.truncated_nn.apply({'params': self.truncated_params}, X_train)

        # Proceed with the original DKL computations
        return super().compute_gp_posterior(X_new, X_train, y_train, params, noiseless)
