from typing import Dict, Optional, Type, Tuple, Union, List
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from ..flax_nets import FlaxMLP, FlaxConvNet
from ..flax_nets import DeterministicNN
from ..flax_nets import MLPLayerModule, ConvLayerModule, extract_configs
from ..utils import flatten_params_dict

class PartialBNN(BNN):
    """
    Partially stochastic (Bayesian) neural network

    Args:
        deterministic_nn:
            Neural network architecture (MLP, ConvNet, or other supported types)
        deterministic_weights:
            Pre-trained deterministic weights, If not provided,
            the deterministic_nn will be trained from scratch when running .fit() method
        num_probabilistic_layers
            Number of layers at the end of deterministic_nn to be treated as fully stochastic ('Bayesian')
        probabilistic_layer_names:
            Names of neural network modules to be treated probabilistically
        num_classes: Number of classes for classification task.
            If None, the model performs regression. Defaults to None.
        noise_prior:
            Custom prior for observational noise distribution
    """

    def __init__(self,
                 deterministic_nn: Union[Type[FlaxMLP], Type[FlaxConvNet]],
                 deterministic_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 num_probabilistic_layers: int = None,
                 probabilistic_layer_names: List[str] = None,
                 num_classes: Optional[int] = None,
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        super().__init__(None, num_classes, noise_prior)
        
        self.deterministic_nn = deterministic_nn
        self.deterministic_weights = deterministic_weights

        self.layer_configs = extract_configs(
            deterministic_nn, probabilistic_layer_names, num_probabilistic_layers)
    
    def model(self,
          X: jnp.ndarray,
          y: jnp.ndarray = None,
          priors_sigma: float = 1.0,
          **kwargs) -> None:
        """Partial BNN model"""
    
        net = self.deterministic_nn
        pretrained_priors = flatten_params_dict(self.deterministic_weights)
        
        def prior(name, shape):
            param_path = name.split('.')
            layer_name = param_path[0]
            param_type = param_path[-1]  # kernel or bias
            return dist.Normal(pretrained_priors[layer_name][param_type], priors_sigma)

        current_input = X

        # Track when we switch from conv to dense layers
        last_conv_idx = max(
            (i for i, c in enumerate(self.layer_configs) if c["layer_type"] == "conv"),
            default=-1
        )
        
        for idx, config in enumerate(self.layer_configs):
            layer_name = config['layer_name']

            # Flatten inputs after last conv layer
            if idx > last_conv_idx and idx-1 == last_conv_idx:
                current_input = current_input.reshape((current_input.shape[0], -1))

            layer_cls = ConvLayerModule if config["layer_type"] == "conv" else MLPLayerModule
            layer = layer_cls(
                features=config['features'],
                activation=config['activation'],
                layer_name=layer_name,
                **({"input_dim": config['input_dim'], 
                    "kernel_size": config['kernel_size']} if config["layer_type"] == "conv" else {})
            )
            print(layer)
            if config['is_probabilistic']:
                net = random_flax_module(
                    layer_name, layer, 
                    input_shape=(1, *current_input.shape[1:]),
                    prior=prior
                )
                current_input = net(current_input)
            else:
                params = {
                    "params": {
                        layer_name: {
                            "kernel": pretrained_priors[layer_name]["kernel"],
                            "bias": pretrained_priors[layer_name]["bias"]
                        }
                    }
                }
                current_input = layer.apply(params, current_input)

        if self.is_regression: # Regression case
            mu = numpyro.deterministic("mu", net(current_input))
            sig = numpyro.sample("sig", self.noise_prior)
            numpyro.sample("y", dist.Normal(mu, sig), obs=y)
        else: # Classification case
            # Note: Even if the original deterministic_nn had softmax,
            # it was overridden to None in extract_mlp_configs, so we  
            # need to apply softmax here
            probs = numpyro.deterministic("probs", softmax(current_input, axis=-1))
            numpyro.sample("y", dist.Categorical(probs=probs), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            sgd_epochs: Optional[int] = None, sgd_lr: Optional[float] = 0.01,
            sgd_batch_size: Optional[int] = None, swa_config: Optional[Dict] = None,
            map_sigma: float = 1.0, priors_sigma: float = 1.0,
            progress_bar: bool = True, device: str = None,
            rng_key: Optional[jnp.array] = None,
            extra_fields: Optional[Tuple[str, ...]] = (),
            **kwargs
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
            swa_config (dict, optional):
                Stochastic weight averaging protocol. Defaults to averaging weights
                at the end of training trajectory (the last 5% of SGD epochs).
            map_sigma (float, optional): Sigma in Gaussian prior for regularized SGD training. Defaults to 1.0.
            priors_sigma (float, optional): Standard deviation for default or pretrained priors
                in the Bayesian part of the NN. Defaults to 1.0.
            progress_bar (bool, optional): Show progress bar. Defaults to True.
            device (Optional[str], optional): The device to perform computation on ('cpu', 'gpu'). 
                Defaults to None (JAX default device).
            rng_key (Optional[jnp.ndarray], optional): Random number generator key. Defaults to None.
            extra_fields (Optional[Tuple[str, ...]], optional): Extra fields to collect during the MCMC run. 
                Defaults to ().
            **max_num_restarts (int, optional): Maximum number of fitting attempts for single chain. 
                Ignored if num_chains > 1. Defaults to 1.
            **min_accept_prob (float, optional): Minimum acceptance probability threshold. 
                Only used if num_chains = 1. Defaults to 0.55.
        """
        if not self.deterministic_weights:
            print("Training deterministic NN...")
            X, y = self.set_data(X, y)
            det_nn = DeterministicNN(
                self.deterministic_nn,
                input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],), # different input shape for ConvNet and MLP
                loss='homoskedastic' if self.is_regression else 'classification',
                learning_rate=sgd_lr, swa_config=swa_config, sigma=map_sigma)
            det_nn.train(X, y, 500 if sgd_epochs is None else sgd_epochs, sgd_batch_size)
            self.deterministic_weights = det_nn.state.params
            print("Training partially Bayesian NN")
        super().fit(
            X, y, num_warmup, num_samples, num_chains, chain_method,
            priors_sigma, progress_bar, device, rng_key, extra_fields, **kwargs)
