from typing import Dict, Optional, Type
import jax.numpy as jnp
import flax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from .utils import split_mlp
from .detnn import DeterministicNN


class PartialBNN(BNN):
    """
    Partially stochastic NN
    """
    def __init__(self,
                 deterministic_nn: Type[flax.linen.Module],
                 deterministic_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 input_dim: int = None,
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        super().__init__(1, 1)
        if deterministic_weights:
            (self.truncated_mlp, self.truncated_params,
             self.last_layer_mlp) = split_mlp(
                 deterministic_nn, deterministic_weights)[:-1]
        else:
            self.untrained_deterministic_nn = deterministic_nn
            if not input_dim:
                raise ValueError("Please provide input data dimensions or pre-trained model parameters")  
        if noise_prior is None:
            noise_prior = dist.HalfNormal(1.0)
        self.noise_prior = noise_prior
    
    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """BNN probabilistic model"""

        X = self.truncated_mlp.apply({'params': self.truncated_params}, X)

        bnn = random_flax_module(
            "nn", self.last_layer_mlp, input_shape=(1, self.truncated_mlp.hidden_dims[-1]),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", bnn(X))

        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            sgd_epochs: Optional[int] = None,
            progress_bar: bool = True, device: str = None,
            rng_key: Optional[jnp.array] = None,
            ) -> None:
        """
        Run HMC to infer parameters of the BNN

        Args:
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            sgd_epochs:
                number of SGD training epochs for deterministic NN
                (if trained weights are not provided at the initialization stage)
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
        """
        X, y = self.set_data(X, y)
        if hasattr(self, "untrained_deterministic_nn"):
            print("Training deterministic NN...")
            det_nn = DeterministicNN(self.untrained_deterministic_nn, self.input_dim)
            det_nn.train(X, y, 500 if sgd_epochs is None else sgd_epochs)
            (self.truncated_mlp, self.truncated_params,
            self.last_layer_mlp) = split_mlp(
                det_nn.model, det_nn.params)[:-1]
        super().fit(X, y, num_warmup, num_samples, num_chains, chain_method, progress_bar, device, rng_key)

