from typing import List, Optional, Type, Dict
import jax.random as jra
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro.contrib.module import random_flax_module
import flax

from .hskbnn import HeteroskedasticBNN
from .nn import FlaxMLP2Head
from .detnn import DeterministicNN
from .utils import put_on_device, split_mlp2head


class HeteroskedasticPartialBNN(HeteroskedasticBNN):

    def __init__(self,
                 deterministic_nn: Type[flax.linen.Module],
                 deterministic_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 input_dim: int = None,
                 num_stochastic_layers: int = 1
                 ) -> None:
        super().__init__(None, None)
        if deterministic_weights:
            (self.truncated_nn, self.truncated_params,
                self.last_layer_nn) = split_mlp2head(
                    deterministic_nn, deterministic_weights)[:-1]
        else:
            self.untrained_deterministic_nn = deterministic_nn
            self.num_stochastic_layers = num_stochastic_layers
            if not input_dim:
                raise ValueError("Please provide input data dimensions or pre-trained model parameters")  
            self.input_dim = input_dim

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, **kwargs) -> None:
        """Heteroskedastik (partial) BNN probabilistic model"""

        X = self.truncated_nn.apply({'params': self.truncated_params}, X)

        bnn = random_flax_module(
            "nn", self.last_layer_nn, input_shape=(1, self.truncated_nn.hidden_dims[-1]),
            prior=(lambda name, shape: dist.Cauchy() if name == "bias" else dist.Normal()))

        # Pass inputs through a NN with the sampled parameters
        mu, sig = bnn(X)
        # Register values with numpyro
        mu = numpyro.deterministic("mu", mu)
        sig = numpyro.deterministic("sig", sig)

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            sgd_epochs: Optional[int] = None, sgd_lr: Optional[float] = 0.01,
            sgd_batch_size: Optional[int] = None, sgd_swa_epochs: Optional[int] = 10,
            progress_bar: bool = True, device: str = None,
            rng_key: Optional[jnp.array] = None) -> None:
        """
        Run HMC to infer parameters of the heteroskedastic BNN

        Args:
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: choose between 'sequential', 'vectorized', and 'parallel'
            sgd_swa_epochs:
                number of SGD training epochs for deterministic NN
                (if trained weights are not provided at the initialization stage)
            sgd_lr: SGD learning rate (if trained weights are not provided at the initialization stage)
            sgd_batch_size:
                Batch size for SGD training (if trained weights are not provided at the initialization stage).
                Defaults to None, meaning that an entire dataset is passed through an NN.
            sgd_swa_epochs: Number of epochs for stochastic weight averaging at the end of training trajectory (defautls to 10)
            progress_bar: show progress bar
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
        """
        if hasattr(self, "untrained_deterministic_nn"):
            print("Training deterministic NN...")
            det_nn = DeterministicNN(
                self.untrained_deterministic_nn, self.input_dim,
                learning_rate=sgd_lr, swa_epochs=sgd_swa_epochs)
            det_nn.train(X, y, 500 if sgd_epochs is None else sgd_epochs, sgd_batch_size)
            (self.truncated_nn, self.truncated_params,
                self.last_layer_nn) = split_mlp2head(
                    det_nn.model, det_nn.state.params,
                self.num_stochastic_layers)[:-1]
            print("Training partially Bayesian NN")
        super().fit(X, y, num_warmup, num_samples, num_chains, chain_method, progress_bar, device, rng_key)

