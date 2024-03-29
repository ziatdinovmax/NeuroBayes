from typing import Dict, Tuple, Callable, Optional, List
from jax import vmap
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .gp import GP
from .nn import get_mlp
from .priors import GPPriors, get_mlp_prior


kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class DKL(GP):
    """
    Fully Bayesian Deep Kernel Learning
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 base_kernel: kernel_fn_type,
                 priors: Optional[GPPriors] = None,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 jitter: float = 1e-6
                 ) -> None:
        super(DKL, self).__init__(latent_dim, base_kernel, priors, jitter)
        hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
        self.nn = get_mlp(hdim, activation)
        self.nn_prior = get_mlp_prior(input_dim, latent_dim, hdim)

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              **kwargs: float
              ) -> None:
        """DKL probabilistic model"""
        # BNN part
        nn_params = self.nn_prior()
        z = self.nn(X, nn_params)
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

        # Transform X_new and X_train using the neural network to get embeddings
        z_new = self.nn(X_new, params)
        z_train = self.nn(X_train, params)

        # Proceed with the original GP computations using the embedded inputs
        return super().compute_gp_posterior(z_new, z_train, y_train, params, noiseless)

    def embed(self, X_new: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds data into the latent space using the inferred weights
        of the DKL's Bayesian neural network
        """
        X_new = self.set_data(X_new)
        samples = self.get_samples(chain_dim=False)
        predictive = vmap(lambda params: self.nn(X_new, params))
        z = predictive(samples)
        return z

    def print_summary(self, print_nn_weights: bool = False) -> None:
        samples = self.get_samples(1)
        if print_nn_weights:
            numpyro.diagnostics.print_summary(samples)
        else:
            list_of_keys = ["k_scale", "k_length", "noise"]
        numpyro.diagnostics.print_summary(
            {k: v for (k, v) in samples.items() if k in list_of_keys})
