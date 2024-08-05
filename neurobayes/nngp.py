from typing import Dict, Optional, Type, Callable, Tuple
import jax.numpy as jnp
import flax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .gp import GP
from .priors import GPPriors
from .utils import split_mlp

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class NNGP(GP):
    """
    Partially stochastic NN as a mean function in GP
    """

    def __init__(self,
                 input_dim: int, kernel: kernel_fn_type,
                 deterministic_nn: Type[flax.linen.Module],
                 deterministic_weights: Dict[str, jnp.ndarray],
                 priors: Optional[GPPriors] = None,
                 jitter: float = 1e-6
                 ) -> None:
        super(NNGP, self).__init__(input_dim, kernel, priors, jitter)
        (self.truncated_nn, self.truncated_params,
         self.nn) = split_mlp(
             deterministic_nn, deterministic_weights)[:-1]

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """GP model with partiallys stochastic NN as its mean function"""
        # Get inputs through a deterministic NN part
        X_prime = self.truncated_nn.apply({'params': self.truncated_params}, X)
        # Fully stochastic NN part
        bnn = random_flax_module(
            "nn", self.nn, input_shape=(1, self.truncated_nn.hidden_dims[-1]),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))
        # Mean function embedding
        f_loc = bnn(X_prime)
        # Sample kernel parameters
        kernel_params = self.sample_kernel_params()
        # Sample observational noise variance
        noise = self.sample_noise()
        # Compute kernel
        k = self.kernel(X, X, kernel_params, noise, self.jitter)
        # Sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=k),
            obs=y,
        )

    def compute_gp_posterior(self, X_new: jnp.ndarray,
                             X_train: jnp.ndarray, y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True,
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns mean and covariance of multivariate normal
        posterior for a single sample of trained GP parameters
        """                

        noise = params["noise"]
        noise_p = noise * (1 - jnp.array(noiseless, int))
        y_residual = y_train.copy()

        # Compute mean function
        x_prime = self.truncated_nn.apply({'params': self.truncated_params}, X_train)
        f_loc1 = self.nn.apply({'params': params}, x_prime)
        y_residual -= f_loc1.squeeze()

        # compute kernel matrices for train and new/test data
        k_XX = self.kernel(X_train, X_train, params, noise, self.jitter)
        k_pp = self.kernel(X_new, X_new, params, noise_p, self.jitter)
        k_pX = self.kernel(X_new, X_train, params)
        # compute predictive mean covariance
        K_xx_inv = jnp.linalg.inv(k_XX)
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, y_residual))
        cov = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))

        x_prime2 = self.truncated_nn.apply({'params': self.truncated_params}, X_new)
        f_loc2 = self.nn.apply({'params': params}, x_prime2)
        mean += f_loc2.squeeze()

        return mean, cov
