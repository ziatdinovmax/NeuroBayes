from typing import Dict, Optional, Type, Tuple, Union, List
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from ..flax_nets import FlaxMLP, FlaxConvNet, split_mlp, split_convnet
from ..flax_nets import DeterministicNN
from ..flax_nets import extract_mlp_configs, MLPLayerModule

class PartialBNN2(BNN):
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
        stochastic_layer_names:
            Names of neural network modules to be treated probabilistically
        noise_prior:
            Custom prior for observational noise distribution
    """

    def __init__(self,
                 deterministic_nn: Union[Type[FlaxMLP], Type[FlaxConvNet]],
                 deterministic_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 num_stochastic_layers: int = 1,
                 stochastic_layer_names: List[str] = None,
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        super().__init__(None, noise_prior=noise_prior)
        
        self.deterministic_nn = deterministic_nn
        self.deterministic_weights = deterministic_weights

        self.probabilistic_layers = stochastic_layer_names
    
    def model(self,
          X: jnp.ndarray,
          y: jnp.ndarray = None,
          priors_sigma: float = 1.0,
          **kwargs) -> None:
        """Partial BNN model"""
    
        net = self.deterministic_nn
        pretrained_priors = self.deterministic_weights

        # Extract layer configurations
        layer_configs = extract_mlp_configs(net, self.probabilistic_layers)
        
        def prior(name, shape):
            param_path = name.split('.')
            layer_name = param_path[0]
            param_type = param_path[-1]  # kernel or bias
            return dist.Normal(pretrained_priors[layer_name][param_type], priors_sigma)

        current_input = X
        
        for idx, config in enumerate(layer_configs):
            layer_name = f"Dense{idx}"
            layer = MLPLayerModule(
                features=config['features'],
                activation=config['activation'],
                layer_name=layer_name
            )
            
            if config['is_probabilistic']:
                net = random_flax_module(
                    layer_name, layer, 
                    input_shape=(1, current_input.shape[-1]),
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

        # Register final output
        mu = numpyro.deterministic("mu", current_input)

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
            priors_sigma (float, optional): Standard deviation for default or pretrained priors
                in the Bayesian part of the NN. Defaults to 1.0.
            progress_bar (bool, optional): Show progress bar. Defaults to True.
            device (Optional[str], optional): The device to perform computation on ('cpu', 'gpu'). 
                Defaults to None (JAX default device).
            rng_key (Optional[jnp.ndarray], optional): Random number generator key. Defaults to None.
            extra_fields (Optional[Tuple[str, ...]], optional): Extra fields to collect during the MCMC run. 
                Defaults to ().
        """
        if not self.deterministic_weights:
            print("Training deterministic NN...")
            X, y = self.set_data(X, y)
            det_nn = DeterministicNN(
                self.untrained_deterministic_nn,
                input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],), # different input shape for ConvNet and MLP
                learning_rate=sgd_lr, swa_epochs=sgd_wa_epochs, sigma=map_sigma)
            det_nn.train(X, y, 500 if sgd_epochs is None else sgd_epochs, sgd_batch_size)
            self.deterministic_weights = det_nn.state.params
            print("Training partially Bayesian NN")
        super().fit(
            X, y, num_warmup, num_samples, num_chains, chain_method,
            priors_sigma, progress_bar, device, rng_key, extra_fields)
