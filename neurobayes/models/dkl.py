from typing import Dict, Tuple, Callable, Optional, List, Type
from jax import vmap
import jax.numpy as jnp
import flax
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .gp import GP
from ..flax_nets.mlp import FlaxMLP
from ..flax_nets.convnet import FlaxConvNet
from ..utils.priors import GPPriors
from ..utils.utils import get_flax_compatible_dict


kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]


class DKL(GP):
    """
    Fully Bayesian Deep Kernel Learning
    """
    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 base_kernel: kernel_fn_type,
                 priors: Optional[GPPriors] = None,
                 jitter: float = 1e-6
                 ) -> None:
        super(DKL, self).__init__(base_kernel, priors, jitter)
        self.nn = architecture
        
    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              **kwargs: float
              ) -> None:
        """DKL probabilistic model"""
        # BNN part
        input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],)
        net = random_flax_module(
            "nn", self.nn, input_shape=(1, *input_shape),
            prior=(lambda name, shape: dist.Normal(0, 1)))
        z = net(X)
        # GP Part
        f_loc = jnp.zeros(X.shape[0])
        # Sample kernel parameters
        kernel_params = self.sample_kernel_params(kernel_dim=z.shape[-1])
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

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        samples = self.mcmc.get_samples(group_by_chain=chain_dim)
        # Get NN weights and biases
        return get_flax_compatible_dict(samples)

    def compute_gp_posterior(self,
                             X_new: jnp.ndarray,
                             X_train: jnp.ndarray,
                             y_train: jnp.ndarray,
                             params: Dict[str, jnp.ndarray],
                             noiseless: bool = True
                             ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        # Transform X_new and X_train using the neural network to get embeddings
        z_new = self.nn.apply({'params': params}, X_new)
        z_train = self.nn.apply({'params': params}, X_train)

        # Proceed with the original GP computations using the embedded inputs
        return super().compute_gp_posterior(z_new, z_train, y_train, params, noiseless)

    def embed(self, X_new: jnp.ndarray) -> jnp.ndarray:
        """
        Embeds data into the latent space using the inferred weights
        of the DKL's Bayesian neural network
        """
        X_new = self.set_data(X_new)
        samples = self.get_samples()
        predictive = vmap(lambda params: self.nn.apply({'params': params}, X_new))
        z = predictive(samples)
        return z

    def print_summary(self) -> None:
        samples = self.get_samples(1)
        list_of_keys = ["k_scale", "k_length", "noise"]
        numpyro.diagnostics.print_summary(
            {k: v for (k, v) in samples.items() if k in list_of_keys})
