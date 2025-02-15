from typing import Dict, Optional, Type, List, Tuple, Any
import jax.numpy as jnp
import jax.random as random
from jax.nn import softmax
import jax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from .hybrid_layers import partial_bayesian_dense
from ..flax_nets import FlaxMLP, DeterministicNN
from ..flax_nets import MLPLayerModule
from ..flax_nets import extract_configs
from ..utils import flatten_params_dict


class PartialBayesianMLP(BNN):
    """
    Partially stochastic (Bayesian) MLP network, with the extra option that in
    selected layers only a subset of neurons are treated as stochastic.

    Args:
        mlp: Neural network architecture (FlaxMLP)
        deterministic_weights: Pre-trained deterministic weights.
        num_probabilistic_layers: Number of layers at the end to be treated as stochastic.
        probabilistic_layer_names: Names of MLP modules to be treated probabilistically.
        probabilistic_neurons: Optional dict mapping layer names to lists of
            (input_neuron, output_neuron) tuples specifying which weight connections
            should be Bayesian. For layers not in this dict, the entire layer will be treated
            as Bayesian. Can be automatically generated using select_bayesian_components() 
            utility function which supports various selection methods including magnitude,
            gradient, variance, and clustering-based approaches.
        num_classes: Number of classes for classification tasks.
        noise_prior: Custom prior for observational noise distribution.

    Example:
        # Automatically select Bayesian weights using variance-based selection
        prob_neurons = select_bayesian_components(
            mlp,  # DeterministicNN class
            layer_names=['Dense0', 'Dense2'],
            method='variance',
            num_pairs_per_layer=2
        )
        
        # Or manually specify weight connections
        probabilistic_neurons = {
            'Dense0': [(0, 1), (2, 3)],  # Make these input-output connections Bayesian
            'Dense2': [(0, 5), (1, 10)]
        }

        # Note that you would typically need to make far more than 2 weights probabilistic
    """
    def __init__(self,
                 mlp: Type[FlaxMLP],
                 deterministic_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 num_probabilistic_layers: int = None,
                 probabilistic_layer_names: List[str] = None,
                 probabilistic_neurons: Optional[Dict[str, List[Tuple[int]]]] = None,
                 num_classes: Optional[int] = None,
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        super().__init__(None, num_classes, noise_prior)
        
        self.deterministic_nn = mlp
        self.deterministic_weights = deterministic_weights
        # Extract configuration from the architecture
        self.layer_configs = extract_configs(mlp, probabilistic_layer_names, num_probabilistic_layers)
        # For each layer, if a subset of neurons is specified, add that to the config.
        if probabilistic_neurons is not None:
            for config in self.layer_configs:
                layer_name = config['layer_name']
                if layer_name in probabilistic_neurons:
                    config['probabilistic_neurons'] = probabilistic_neurons[layer_name]
                else:
                    config['probabilistic_neurons'] = None
        else:
            for config in self.layer_configs:
                config['probabilistic_neurons'] = None

    def model(self, X: jnp.ndarray, y: jnp.ndarray = None, priors_sigma: float = 1.0, **kwargs) -> None:
        """MLP model with partial Bayesian inference"""

        pretrained_priors = flatten_params_dict(self.deterministic_weights)
        
        def prior(name, shape):
            param_path = name.split('.')
            layer_name = param_path[0]
            param_type = param_path[-1]  # kernel or bias
            return dist.Normal(pretrained_priors[layer_name][param_type], priors_sigma)
        
        current_input = X
        
        for config in self.layer_configs:
            layer_name = config['layer_name']
            
            if config['is_probabilistic']:
                if config['probabilistic_neurons'] is not None:
                    # Use custom partial Bayesian implementation for neuron-level control
                    current_input = partial_bayesian_dense(
                        current_input,
                        pretrained_kernel=pretrained_priors[layer_name]["kernel"],
                        pretrained_bias=pretrained_priors[layer_name]["bias"],
                        prob_neurons=config['probabilistic_neurons'],
                        priors_sigma=priors_sigma,
                        layer_name=layer_name,
                        activation=config['activation']
                    )
                else:
                    # Use standard random_flax_module for full layer Bayesian treatment
                    layer = MLPLayerModule(
                        features=config['features'],
                        activation=config['activation'],
                        layer_name=layer_name
                    )
                    net = random_flax_module(
                        layer_name, layer, 
                        input_shape=(1, *current_input.shape[1:]),
                        prior=prior
                    )
                    current_input = net(current_input, enable_dropout=False)
            else:
                # Deterministic layer
                params = {
                    "params": {
                        layer_name: {
                            "kernel": pretrained_priors[layer_name]["kernel"],
                            "bias": pretrained_priors[layer_name]["bias"]
                        }
                    }
                }
                layer = MLPLayerModule(
                    features=config['features'],
                    activation=config['activation'],
                    layer_name=layer_name
                )
                current_input = layer.apply(params, current_input, enable_dropout=False)

        # Rest of the model implementation remains the same
        if self.is_regression:
            mu = numpyro.deterministic("mu", current_input)
            sig = numpyro.sample("sig", self.noise_prior)
            numpyro.sample("y", dist.Normal(mu, sig), obs=y)
        else:
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
        Infer parameters of the partially Bayesian MLP

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
            **run_diagnostics (bool, optional): Run Gelman-Rubin diagnostics layer-by-layer at the end.
                Defaults to False.
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
