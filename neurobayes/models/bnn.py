from typing import Dict, Tuple, Optional, Union, List, Type
import jax
import jax.random as jra
import jax.numpy as jnp
import flax

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive
from numpyro.contrib.module import random_flax_module

from ..utils.utils import put_on_device, split_dict


class BNN:
    """
    A Fully Bayesian Neural Network (BNN).

    This model treats the weights in a neural network as probabilistic distributions and utilizes 
    the No-U-Turn Sampler to sample directly from the posterior distribution. This approach allows 
    the BNN to account for all plausible weight configurations, enabling it to make probabilistic 
    predictions. Instead of single-point estimates, it provides entire distributions of possible 
    outcomes, thus quantifying the inherent uncertainty.

    Args:
        architecture: a Flax model
        noise_prior (dist.Distribution, optional): Prior probability distribution over 
            observational noise. Defaults to HalfNormal(1.0).
        pretrained_priors (Dict, optional):
            Dictionary with pre-trained weights for the provided model architecture.
            These weight values will be used to initialize prior distributions in BNN.
    """
    def __init__(self,
                 architecture: Type[flax.linen.Module],
                 noise_prior: Optional[dist.Distribution] = None,
                 pretrained_priors: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        if noise_prior is None:
            noise_prior = dist.HalfNormal(1.0)
        self.nn = architecture
        self.noise_prior = noise_prior
        self.pretrained_priors = pretrained_priors

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              priors_sigma: float = 1.0,
              **kwargs) -> None:
        """BNN model"""

        if self.pretrained_priors is not None:
            pretrained_priors = {}
            for module_dict in self.pretrained_priors.values():
                pretrained_priors.update(module_dict)
            self.pretrained_priors = pretrained_priors
        
        def prior(name, shape):
            if self.pretrained_priors is not None:
                param_path = name.split('.')
                layer_name = param_path[0]
                param_type = param_path[-1]  # kernel or bias
                return dist.Normal(self.pretrained_priors[layer_name][param_type], priors_sigma)
            return dist.Normal(0., priors_sigma)
        
        input_shape = X.shape[1:] if X.ndim > 2 else (X.shape[-1],)

        net = random_flax_module(
            "nn", self.nn, input_shape=(1, *input_shape), prior=prior)

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", net(X))

        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            priors_sigma: Optional[float] = 1.0,
            progress_bar: bool = True, device: Optional[str] = None,
            rng_key: Optional[jnp.array] = None,
            extra_fields: Optional[Tuple[str, ...]] = (),
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

        Returns:
            None: The method updates the model's internal state but does not return a value.

        Note:
            After running this method, the MCMC samples are stored in the `mcmc` attribute
            of the model and can be accessed via .get_samples() method for further analysis.
        """
        key = rng_key if rng_key is not None else jra.PRNGKey(0)
        X, y = self.set_data(X, y)
        X, y = put_on_device(device, X, y)
        
        init_strategy = init_to_median(num_samples=10)
        kernel = NUTS(self.model, init_strategy=init_strategy)
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            jit_model_args=False
        )
        self.mcmc.run(
            key, X, y,
            priors_sigma,
            extra_fields=extra_fields)

    def sample_noise(self) -> jnp.ndarray:
        """
        Sample observational noise variance
        """
        return numpyro.sample("sig", self.noise_prior)
    
    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def predict(self,
                X_new: jnp.ndarray,
                samples: Optional[Dict[str, jnp.ndarray]] = None,
                device: Optional[str] = None,
                rng_key: Optional[jnp.ndarray] = None
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict the mean and variance of the target values for new inputs.

        Args:
            X_new (jnp.ndarray): New input data for predictions. Should have the same structure
                as the training input X: 2D array of shape (n_samples, n_features) for MLP, or
                N-D array of shape (n_samples, *dims, n_channels) for ConvNet.
            samples (Dict[str, jnp.ndarray], optional): Dictionary of posterior samples with 
                inferred model parameters (weights and biases). Uses samples from 
                the last MCMC run by default.
            device (str, optional): The device to perform computation on ('cpu', 'gpu'). 
                If None, uses the JAX default device.
            rng_key (jnp.ndarray, optional): Random number generator key for JAX operations. 
                If None, a default key is used.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
                - posterior_mean (jnp.ndarray): Mean of the posterior predictive distribution.
                Shape: (n_samples, target_dim).
                - posterior_var (jnp.ndarray): Variance of the posterior predictive distribution.
                Shape: (n_samples, target_dim).
        """
        X_new = self.set_data(X_new)

        if rng_key is None:
            rng_key = jra.PRNGKey(0)
        if samples is None:
            samples = self.get_samples(chain_dim=False)
        X_new, samples = put_on_device(device, X_new, samples)

        predictions = self.sample_from_posterior(
            rng_key, X_new, samples, return_sites=["mu", "y"])
        posterior_mean = predictions["mu"].mean(0)
        posterior_var = predictions["y"].var(0)
        return posterior_mean, posterior_var

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
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X
