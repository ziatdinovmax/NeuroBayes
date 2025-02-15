from typing import Dict, Optional, Type, Union, List, Tuple
import jax.numpy as jnp
import numpyro.distributions as dist

from .partial_bnn_mlp import PartialBayesianMLP
from .partial_bnn_conv import PartialBayesianConvNet
from .partial_bnn_transformer import PartialBayesianTransformer
from ..flax_nets import FlaxMLP, FlaxConvNet, FlaxTransformer

class PartialBNN:
    """
    A unified wrapper for partially Bayesian neural networks that supports MLPs, 
    ConvNets, and Transformers with a consistent API.

    Args:
        architecture: Neural network architecture (FlaxMLP, FlaxConvNet, or FlaxTransformer)
        deterministic_weights: Pre-trained deterministic weights. If not provided,
            the network will be trained from scratch when running .fit() method
        num_probabilistic_layers: Number of layers at the end to be treated as fully stochastic
        probabilistic_layer_names: Names of neural network modules to be treated probabilistically
        num_classes: Number of classes for classification task.
            If None, the model performs regression. Defaults to None.
        noise_prior: Custom prior for observational noise distribution
    """
    def __init__(self,
                 architecture: Union[Type[FlaxMLP], Type[FlaxConvNet], Type[FlaxTransformer]],
                 deterministic_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 num_probabilistic_layers: Optional[int] = None,
                 probabilistic_layer_names: Optional[List[str]] = None,
                 probabilistic_neurons: Optional[Dict[str, List[Tuple[int]]]] = None,
                 num_classes: Optional[int] = None,
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        
        self.architecture = architecture
        
        arch_class = architecture if isinstance(architecture, type) else architecture.__class__
        
        if issubclass(arch_class, FlaxTransformer):
            self._model = PartialBayesianTransformer(
                transformer=architecture,
                deterministic_weights=deterministic_weights,
                num_probabilistic_layers=num_probabilistic_layers,
                probabilistic_layer_names=probabilistic_layer_names,
                probabilistic_neurons=probabilistic_neurons,
                num_classes=num_classes,
                noise_prior=noise_prior
            )
        elif issubclass(arch_class, FlaxMLP):
            self._model = PartialBayesianMLP(
                mlp=architecture,
                deterministic_weights=deterministic_weights,
                num_probabilistic_layers=num_probabilistic_layers,
                probabilistic_layer_names=probabilistic_layer_names,
                probabilistic_neurons=probabilistic_neurons,
                num_classes=num_classes,
                noise_prior=noise_prior
            )
        elif issubclass(arch_class, FlaxConvNet):
            self._model = PartialBayesianConvNet(
                convnet=architecture,
                deterministic_weights=deterministic_weights,
                num_probabilistic_layers=num_probabilistic_layers,
                probabilistic_layer_names=probabilistic_layer_names,
                probabilistic_neurons=probabilistic_neurons,
                num_classes=num_classes,
                noise_prior=noise_prior
            )
        else:
            raise ValueError(
                f"Unsupported architecture type: {arch_class}. "
                "Must be one of: FlaxMLP, FlaxConvNet, or FlaxTransformer"
            )

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            sgd_epochs: Optional[int] = None, sgd_lr: Optional[float] = 0.01,
            sgd_batch_size: Optional[int] = None, swa_config: Optional[Dict] = None,
            map_sigma: float = 1.0, priors_sigma: float = 1.0,
            progress_bar: bool = True, device: str = None,
            rng_key: Optional[jnp.array] = None,
            extra_fields: Optional[Tuple[str, ...]] = (),
            max_num_restarts: int = 1,
            min_accept_prob: float = 0.55,
            run_diagnostics: bool = False
            ) -> None:
        """
        Fit the partially Bayesian neural network.
        
        Args:
            X: Input data
                - For MLP: 2D array (n_samples, n_features)
                - For ConvNet: ND array (n_samples, *dims, n_channels)
                - For Transformer: 2D array (n_samples, seq_length)
            y: Target values
                - For regression: 1D array (n_samples,) or 2D array (n_samples, target_dim)
                - For classification: 1D array (n_samples,) with class labels
            num_warmup: Number of NUTS warmup steps. Defaults to 2000.
            num_samples: Number of NUTS samples to draw. Defaults to 2000.
            num_chains: Number of NUTS chains to run. Defaults to 1.
            chain_method: Method for running chains: 'sequential', 'parallel', 
                or 'vectorized'. Defaults to 'sequential'.
            sgd_epochs: Number of SGD training epochs for deterministic NN.
                Defaults to 500 (if no pretrained weights are provided).
            sgd_lr: SGD learning rate. Defaults to 0.01.
            sgd_batch_size: Mini-batch size for SGD training. 
                Defaults to None (all input data is processed as a single batch).
            swa_config: Stochastic weight averaging protocol. Defaults to averaging weights
                at the end of training trajectory (the last 5% of SGD epochs).
            map_sigma: Sigma in Gaussian prior for regularized SGD training. Defaults to 1.0.
            priors_sigma: Standard deviation for default or pretrained priors
                in the Bayesian part of the NN. Defaults to 1.0.
            progress_bar: Show progress bar. Defaults to True.
            device: The device to perform computation on ('cpu', 'gpu'). 
                Defaults to None (JAX default device).
            rng_key: Random number generator key. Defaults to None.
            extra_fields: Extra fields to collect during the MCMC run. 
                Defaults to ().
            max_num_restarts: Maximum number of fitting attempts for single chain. 
                Ignored if num_chains > 1. Defaults to 1.
            min_accept_prob: Minimum acceptance probability threshold. 
                Only used if num_chains = 1. Defaults to 0.55.
            run_diagnostics: Run Gelman-Rubin diagnostics layer-by-layer at the end.
                Defaults to False.
        """
        return self._model.fit(
            X, y,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            sgd_epochs=sgd_epochs,
            sgd_lr=sgd_lr,
            sgd_batch_size=sgd_batch_size,
            swa_config=swa_config,
            map_sigma=map_sigma,
            priors_sigma=priors_sigma,
            progress_bar=progress_bar,
            device=device,
            rng_key=rng_key,
            extra_fields=extra_fields,
            max_num_restarts=max_num_restarts,
            min_accept_prob=min_accept_prob,
            run_diagnostics=run_diagnostics
        )

    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Make predictions using the fitted model."""
        return self._model.predict(X)
    
    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self._model.get_samples(chain_dim)
    
    def predict_classes(self, X_new: jnp.ndarray,
                       samples: Optional[Dict[str, jnp.ndarray]] = None,
                       device: Optional[str] = None,
                       rng_key: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Predict class labels for classification tasks"""
        return self._model.predict_classes(X_new, samples, device, rng_key)
    
    def sample_from_posterior(self,
                            rng_key: jnp.ndarray,
                            X_new: jnp.ndarray,
                            samples: Dict[str, jnp.ndarray],
                            return_sites: Optional[List[str]] = None) -> jnp.ndarray:
        """Sample from posterior distribution at new inputs X_new"""
        return self._model.sample_from_posterior(rng_key, X_new, samples, return_sites)
    
    def set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
             ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        """Prepare data for model fitting or prediction"""
        return self._model.set_data(X, y)
    
    @property
    def is_regression(self) -> bool:
        """Check if the model is performing regression"""
        return self._model.is_regression
    
    @property
    def mcmc(self):
        """Get the MCMC sampler"""
        return self._model.mcmc
    
    @property
    def diagnostic_results(self):
        """Get diagnostic results if run_diagnostics was True during fitting"""
        return self._model.diagnostic_results