import logging
logger = logging.getLogger(__name__)

from typing import Dict, Tuple, Optional, Union, List, Type
from enum import Enum

import jax
import jax.random as jra
import jax.numpy as jnp
import flax
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive
from numpyro.contrib.module import random_flax_module

from ..utils import put_on_device, flatten_params_dict


class BNN:
    """
    A Bayesian Neural Network (BNN) for both regression and classification tasks.

    This model automatically determines the task type based on num_classes:
    - If num_classes is None: Regression task
    - If num_classes >= 2: Classification task with specified number of classes

    Args:
        architecture: a Flax model
        num_classes (int, optional): Number of classes for classification task.
            If None, the model performs regression. Defaults to None.
        noise_prior (dist.Distribution, optional): Prior probability distribution over 
            observational noise for regression. Defaults to HalfNormal(1.0).
        pretrained_priors (Dict, optional): Dictionary with pre-trained weights.
            
    Note:
        For input labels y:
        - Regression (num_classes=None): y should be a 1D array of shape (n_samples,) for single output
          or 2D array of shape (n_samples, n_outputs) for multiple outputs
        - Binary classification (num_classes=2): y should be a 1D array of shape (n_samples,)
          containing 0s and 1s
        - Multi-class classification (num_classes>2): y should be a 1D array of shape (n_samples,)
          containing class indices from 0 to num_classes-1
    """
    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 num_classes: Optional[int] = None,
                 noise_prior: Optional[dist.Distribution] = None,
                 pretrained_priors: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        self.nn = architecture
        self.num_classes = num_classes
        
        if num_classes is not None and num_classes < 2:
            raise ValueError("num_classes must be at least 2 for classification or None for regression")
        
        # Set noise prior only for regression tasks
        self.noise_prior = noise_prior if num_classes is None else None
        if self.noise_prior is None and num_classes is None:
            self.noise_prior = dist.HalfNormal(1.0)
            
        self.pretrained_priors = pretrained_priors

    @property
    def is_regression(self) -> bool:
        """Check if the model is performing regression"""
        return self.num_classes is None

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              priors_sigma: float = 1.0,
              **kwargs) -> None:
        """Unified BNN model for both regression and classification"""

        pretrained_priors = (flatten_params_dict(self.pretrained_priors) 
                           if self.pretrained_priors is not None else None)
        
        def prior(name, shape):
            if pretrained_priors is not None:
                param_path = name.split('.')
                layer_name = param_path[-2]
                param_type = param_path[-1]  # kernel or bias
                return dist.Normal(pretrained_priors[layer_name][param_type], priors_sigma)
            return dist.Normal(0., priors_sigma)
        
        input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],)
        net = random_flax_module(
            "nn", self.nn, input_shape=(1, *input_shape), prior=prior)

        if self.is_regression:
            # Regression case
            mu = numpyro.deterministic("mu", net(X))
            sig = numpyro.sample("sig", self.noise_prior)
            numpyro.sample("y", dist.Normal(mu, sig), obs=y)
        else:
            # Classification case
            logits = net(X)
            probs = numpyro.deterministic("probs", softmax(logits, axis=-1))
            numpyro.sample("y", dist.Categorical(probs=probs), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            priors_sigma: Optional[float] = 1.0,
            progress_bar: bool = True, device: Optional[str] = None,
            rng_key: Optional[jnp.array] = None,
            extra_fields: Optional[Tuple[str, ...]] = (),
            max_num_restarts: int = 1,
            min_accept_prob: float = 0.55
            ) -> None:
        """
        Run No-U-Turn Sampler (NUTS) to infer parameters of the Bayesian Neural Network.

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
            priors_sigma (float, optional): Standard deviation for default or pretrained priors. 
                Defaults to 1.0.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.
            device (str, optional): The device to perform computation on ('cpu', 'gpu'). 
                If None, uses the JAX default device.
            rng_key (jnp.ndarray, optional): Random number generator key. If None, uses a default key.
            extra_fields (Tuple[str, ...], optional): Extra fields (e.g. 'accept_prob') to collect 
                during the MCMC run. Accessible via model.mcmc.get_extra_fields() after training.
            max_num_restarts (int, optional): Maximum number of fitting attempts for single chain. 
                Ignored if num_chains > 1. Defaults to 1.
            min_accept_prob (float, optional): Minimum acceptance probability threshold. 
                Only used if num_chains = 1. Defaults to 0.55.

        Returns:
            None: The method updates the model's internal state but does not return a value.

        Note:
            After running this method, the MCMC samples are stored in the `mcmc` attribute
            of the model and can be accessed via .get_samples() method for further analysis.
        """
        
        X, y = self.set_data(X, y)
        X, y = put_on_device(device, X, y)
        kernel = NUTS(self.model, init_strategy=init_to_median(num_samples=10))

        # Multiple chains: single run with split keys
        if num_chains > 1:
            key = jra.split(rng_key if rng_key is not None else jra.PRNGKey(0), num_chains)
            self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                            num_chains=num_chains, chain_method=chain_method,
                            progress_bar=progress_bar, jit_model_args=False)
            self.mcmc.run(key, X, y, priors_sigma, extra_fields=extra_fields)
            return

        # Single chain with restarts
        best_mcmc = None
        best_prob = -1.0
        fields = ('accept_prob',) if 'accept_prob' not in extra_fields else extra_fields

        for i in range(max_num_restarts):
            key = rng_key if i == 0 and rng_key is not None else jra.PRNGKey(i)
            self.mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                            num_chains=1, progress_bar=progress_bar, jit_model_args=False)
            
            self.mcmc.run(key, X, y, priors_sigma, extra_fields=fields)
            prob = self.mcmc.get_extra_fields()['accept_prob'].mean()
            
            if prob > best_prob:
                best_mcmc, best_prob = self.mcmc, prob
                
            if prob > min_accept_prob:
                return
            
            if i < max_num_restarts - 1:
                logger.warning(f"MCMC restart {i+1}/{max_num_restarts}: acceptance rate {prob:.3f} below {min_accept_prob}")
        
        self.mcmc = best_mcmc
        logger.warning(f"Using best run (acceptance rate: {best_prob:.3f})")

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def predict(self,
                X_new: jnp.ndarray,
                samples: Optional[Dict[str, jnp.ndarray]] = None,
                device: Optional[str] = None,
                rng_key: Optional[jnp.ndarray] = None
                ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], 
                         Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Predict outputs for new inputs.

        Args:
            X_new (jnp.ndarray): New input data for predictions
            samples (Dict[str, jnp.ndarray], optional): Dictionary of posterior samples
            device (str, optional): The device to perform computation on
            rng_key (jnp.ndarray, optional): Random number generator key

        Returns:
                Tuple[jnp.ndarray, jnp.ndarray]: (posterior_mean, posterior_var)
        """
        X_new = self.set_data(X_new)

        if rng_key is None:
            rng_key = jra.PRNGKey(0)
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        X_new, samples = put_on_device(device, X_new, samples)

        if self.is_regression: # Regression
            predictions = self.sample_from_posterior(
                rng_key, X_new, samples, return_sites=["mu", "y"])
            posterior_mean = predictions["mu"].mean(0)
            posterior_var = predictions["y"].var(0)
        
        else:  # Classification
            predictions = self.sample_from_posterior(
                rng_key, X_new, samples, return_sites=["probs"])
            predictive_probs = predictions["probs"]
            posterior_mean = predictive_probs.mean(0)
            posterior_var = predictive_probs.var(0)
                
        return posterior_mean, posterior_var
        
    def predict_classes(self,
                      X_new: jnp.ndarray,
                      samples: Optional[Dict[str, jnp.ndarray]] = None,
                      device: Optional[str] = None,
                      rng_key: Optional[jnp.ndarray] = None
                      ) -> jnp.ndarray:
        """
        Predict class labels for classification tasks
        """
        if self.is_regression:
            raise ValueError("predict_classes() is only for classification tasks")
            
        probs_mean, _ = self.predict(X_new, samples, device, rng_key)
        
        return jnp.argmax(probs_mean, axis=-1)

    def sample_from_posterior(self,
                            rng_key: jnp.ndarray,
                            X_new: jnp.ndarray,
                            samples: Dict[str, jnp.ndarray],
                            return_sites: Optional[List[str]] = None
                            ) -> jnp.ndarray:
        """Sample from posterior distribution at new inputs X_new"""
        predictive = Predictive(
            self.model, samples,
            return_sites=return_sites
        )
        return predictive(rng_key, X_new)

    def set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
             ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        """
        Prepare data for model fitting or prediction.
        
        Ensures consistent shapes for both X and y:
        - X: Always at least 2D
        - y: Shape depends on task type:
            - Regression: 2D array (n_samples, n_outputs)
            - Classification (binary or multi-class): 1D array (n_samples,)
        """
        X = X if X.ndim > 1 else X[:, None]
        
        if y is not None:
            if self.is_regression:
                y = y[:, None] if y.ndim < 2 else y  # Regression
            else:
                y = y.reshape(-1)  # Classification (both binary and multi-class)
            return X, y
        return X