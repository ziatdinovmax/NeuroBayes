from typing import Dict, Optional, Type, Tuple, Union, Callable
import jax.numpy as jnp
import flax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from ..flax_nets import FlaxMLP, FlaxConvNet, split_mlp, split_convnet
from ..flax_nets import DeterministicNN

class PartialBNN(BNN):
    """
    Partially stochastic (Bayesian) neural network

    Args:
        deterministic_nn:
            Neural network architecture (MLP, ConvNet, or other supported types)
        deterministic_weights:
            Pre-trained deterministic weights, If not provided,
            the deterministic_nn will be trained from scratch when running .fit() method
        input_dim:
            Number of features in the input data (for MLP) or number of dimensions (for ConvNet)
        num_stochastic_layers:
            Number of layers at the end of deterministic_nn to be treated as fully stochastic (Bayesian)
        noise_prior:
            Custom prior on observational noise distribution
    """
    # Dictionary mapping network types to their corresponding splitter functions
    SPLITTERS = {
        FlaxMLP: split_mlp,
        FlaxConvNet: split_convnet,
        # Add more network types and their splitters here
    }

    def __init__(self,
                 deterministic_nn: Union[Type[FlaxMLP], Type[FlaxConvNet]],
                 deterministic_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 num_stochastic_layers: int = 1,
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        super().__init__(None, noise_prior=noise_prior)
        
        self.nn_type = type(deterministic_nn)
        if self.nn_type not in self.SPLITTERS:
            raise ValueError(f"Unsupported network type: {self.nn_type}")
        self.splitter = self.SPLITTERS[self.nn_type]
        
        if deterministic_weights:
            (self.subnet1, self.subnet1_params,
             self.subnet2, self.subnet2_params) = self.splitter(
                 deterministic_nn, deterministic_weights,
                 num_stochastic_layers)
        else:
            self.untrained_deterministic_nn = deterministic_nn
            self.num_stochastic_layers = num_stochastic_layers
    
    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              pretrained_priors: bool = None,
              priors_sigma: float = 1.0,
              **kwargs) -> None:
        """Partial BNN model"""
        
        def prior(name, shape):
            if pretrained_priors is not None:
                param_path = name.split('.')
                mean = pretrained_priors
                for path in param_path:
                    mean = mean[path]
                return dist.Normal(mean, priors_sigma)
            else:
                return dist.Normal(0., priors_sigma)
            
        X = self.subnet1.apply({'params': self.subnet1_params}, X)

        bnn = random_flax_module(
            "nn", self.subnet2,
            input_shape=(1, X.shape[-1]),
            prior=prior)

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", bnn(X))

        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            sgd_epochs: Optional[int] = None, sgd_lr: Optional[float] = 0.01,
            sgd_batch_size: Optional[int] = None, sgd_wa_epochs: Optional[int] = 10,
            map_sigma: float = 1.0, priors_from_map: bool = False, priors_sigma: float = 1.0,
            progress_bar: bool = True, device: str = None,
            rng_key: Optional[jnp.array] = None,
            extra_fields: Optional[Tuple[str]] = ()
            ) -> None:
        """
        Run HMC to infer parameters of the BNN

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
            sgd_wa_epochs: Number of epochs for stochastic weight averaging at the end of SGD training trajectory (defautls to 10)
            map_sigma: sigma in gaussian prior for regularized SGD training
            priors_from_map: use MAP values to initialize BNN weight priors (Defaults to False)
            priors_sigma: Standard deviation for default or pretrained priors (defaults to 1.0)
            progress_bar: show progress bar
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
            extra_fields:
                Extra fields (e.g. 'accept_prob') to collect during the HMC run.
                The extra fields are accessible from model.mcmc.get_extra_fields() after model training.
        """
        if hasattr(self, "untrained_deterministic_nn"):
            print("Training deterministic NN...")
            X, y = self.set_data(X, y)
            det_nn = DeterministicNN(
                self.untrained_deterministic_nn,
                input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],), # different input shape for ConvNet and MLP
                learning_rate=sgd_lr, swa_epochs=sgd_wa_epochs, sigma=map_sigma)
            det_nn.train(X, y, 500 if sgd_epochs is None else sgd_epochs, sgd_batch_size)
            (self.subnet1, self.subnet1_params,
            self.subnet2, self.subnet2_params) = self.splitter(
                det_nn.model, det_nn.state.params,
                self.num_stochastic_layers)
            print("Training partially Bayesian NN")
        super().fit(
            X, y, num_warmup, num_samples, num_chains, chain_method,
            self.subnet2_params if priors_from_map else None,
            priors_sigma, progress_bar, device, rng_key, extra_fields)
