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
        num_stochastic_layers:
            Number of layers at the end of deterministic_nn to be treated as fully stochastic ('Bayesian')
        noise_prior:
            Custom prior for observational noise distribution
    """
    # Dictionary mapping network types to their corresponding splitter functions
    SPLITTERS = {
        FlaxMLP: split_mlp,
        FlaxConvNet: split_convnet,
        # More network types and their splitters TBA
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
            extra_fields: Optional[Tuple[str, ...]] = ()
            ) -> None:
        """
        Infer parameters of the partial BNN

        Args:
            X (jnp.ndarray): Input features. For MLP: 2D array of shape (n_samples, n_features).
                For ConvNet: N-D array of shape (n_samples, *dims, n_channels), where
                dims = (length,) for spectral data or (height, width) for image data.
            y (jnp.ndarray): Target array. For single-output problems: 1D array of shape (n_samples,).
                For multi-output problems: 2D array of shape (n_samples, target_dim).
            num_warmup (int, optional): Number of NUTS warmup steps. Defaults to 2000.
            num_samples (int, optional): Number of NUTS samples to draw. Defaults to 2000.
            num_chains (int, optional): Number of NUTS chains to run. Defaults to 1.
            chain_method (str, optional): Method for running chains: 'sequential', 'parallel', 
                or 'vectorized'. Defaults to 'sequential'.
            sgd_epochs (Optional[int], optional): Number of SGD training epochs for deterministic NN.
                Defaults to 500 (if no pretrained weights are provided).
            sgd_lr (float, optional): SGD learning rate. Defaults to 0.01.
            sgd_batch_size (Optional[int], optional): Mini-batch size for SGD training. 
                Defaults to None (all input data is processed as a single batch).
            sgd_wa_epochs (int, optional): Number of epochs for stochastic weight averaging. Defaults to 10.
            map_sigma (float, optional): Sigma in Gaussian prior for regularized SGD training. Defaults to 1.0.
            priors_from_map (bool, optional): Use MAP values to initialize BNN weight priors. Defaults to False.
            priors_sigma (float, optional): Standard deviation for default or pretrained priors
                in the Bayesian part of the NN. Defaults to 1.0.
            progress_bar (bool, optional): Show progress bar. Defaults to True.
            device (Optional[str], optional): The device to perform computation on ('cpu', 'gpu'). 
                Defaults to None (JAX default device).
            rng_key (Optional[jnp.ndarray], optional): Random number generator key. Defaults to None.
            extra_fields (Optional[Tuple[str, ...]], optional): Extra fields to collect during the MCMC run. 
                Defaults to ().
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
