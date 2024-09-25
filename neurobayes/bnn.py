from typing import Dict, Tuple, Optional, Union, List
import jax
import jax.random as jra
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median, Predictive
from numpyro.contrib.module import random_flax_module

from .flax_nets import FlaxMLP
from .utils import put_on_device, split_dict


class BNN:
    """
    A Fully Bayesian Neural Network.
    This approach employs a probabilistic treatment of all neural network weights,
    treating them as random variables with specified prior distributions
    and utilizing advanced Markov Chain Monte Carlo techniques to sample directly
    from the posterior distribution, allowing to account for all plausible weight configurations.
    This approach enables the network to make probabilistic predictions,
    not just single-point estimates but entire distributions of possible outcomes,
    quantifying the inherent uncertainty.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: List[int] = None,
                 activation: str = 'tanh',
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        if noise_prior is None:
            noise_prior = dist.HalfNormal(1.0)
        hdim = hidden_dim if hidden_dim is not None else [32, 16, 8]
        self.nn = FlaxMLP(hdim, output_dim, activation)
        self.input_dim = input_dim
        self.noise_prior = noise_prior

    def model(self,
              X: jnp.ndarray,
              y: jnp.ndarray = None,
              pretrained_priors: Dict = None,
              **kwargs) -> None:
        """BNN probabilistic model"""
        
        def prior(name, shape):
            if pretrained_priors is not None:
                param_path = name.split('.')
                mean = pretrained_priors
                for path in param_path:
                    mean = mean[path]
                return dist.Normal(mean, 1.0)
            else:
                return dist.Normal(0., 1.0)
            
        net = random_flax_module(
            "nn", self.nn, input_shape=(1, self.input_dim), prior=prior)

        # Pass inputs through a NN with the sampled parameters
        mu = numpyro.deterministic("mu", net(X))

        # Sample noise
        sig = self.sample_noise()

        # Score against the observed data points
        numpyro.sample("y", dist.Normal(mu, sig), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            pretrained_priors: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
            progress_bar: bool = True, device: str = None,
            rng_key: Optional[jnp.array] = None,
            extra_fields: Optional[Tuple[str]] = (),
            ) -> None:
        """
        Run HMC to infer parameters of the BNN

        Args:
            X: 2D feature vector
            y: 1D target vector
            num_warmup: number of HMC warmup states
            num_samples: number of HMC samples
            num_chains: number of HMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            progress_bar: show progress bar
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key: random number generator key
            pretrained_priors: Dictionary with mean values for Normal prior distributions over model weights and biases
            extra_fields:
                Extra fields (e.g. 'accept_prob') to collect during the HMC run.
                The extra fields are accessible from model.mcmc.get_extra_fields() after model training.
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
        self.mcmc.run(key, X, y, pretrained_priors, extra_fields=extra_fields)

    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    def sample_noise(self) -> jnp.ndarray:
        """
        Sample observational noise variance
        """
        return numpyro.sample("sig", self.noise_prior)

    def predict(self,
                X_new: jnp.ndarray,
                samples: Optional[Dict[str, jnp.ndarray]] = None,
                device: Optional[str] = None,
                rng_key: Optional[jnp.ndarray] = None
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict the mean and variance of the target values for new inputs.

        Args:
            X_new:
                New input data for predictions.
            samples:
                Dictionary of posterior samples with inferred model parameters (weights and biases)
            device:
                The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
                is performed on the JAX default device.
            rng_key:
                Random number generator key for JAX operations.

        Returns:
            Tuple containing the means and samples from the posterior predictive distribution.
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
   
        predictive = Predictive(
            self.model, samples,
            return_sites=return_sites
        )
        return predictive(rng_key, X_new)

    # def predict_in_batches(self, X_new: jnp.ndarray,
    #                        batch_size: int = 100,
    #                        device: Optional[str] = None,
    #                        rng_key: Optional[jnp.ndarray] = None
    #                        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #     """
    #     Make prediction in batches (to avoid memory overflow)
    #     at X_new points a trained BNN model
    #     """
    #     X_new = self.set_data(X_new)
    #     if rng_key is None:
    #         rng_key = jra.PRNGKey(0)
    #     samples = self.get_samples(chain_dim=False)
    #     X_new, samples = put_on_device(device, X_new, samples)
    #     mu_chunks, y_chunks = [], []
    #     for batch in split_dict(samples, batch_size):
    #         predictions = self.sample_from_posterior(
    #             rng_key, X_new, batch, return_sites=['mu', 'y']).values()
    #         mu_i, y_i = predictions["mu"], predictions["y"]
    #         mu_i = jax.device_put(mu_i, jax.devices("cpu")[0])
    #         y_i = jax.device_put(y_i, jax.devices("cpu")[0])
    #         mu_chunks.append(mu_i)
    #         y_chunks.append(y_i)
    #     mu_chunks = jnp.concatenate(mu_chunks, axis=0)
    #     y_chunks = jnp.concatenate(y_chunks, axis=0)
        
    #     return mu_chunks.mean(0), y_chunks.var(0)

    def set_data(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None
                 ) -> Union[Tuple[jnp.ndarray], jnp.ndarray]:
        X = X if X.ndim > 1 else X[:, None]
        if y is not None:
            y = y[:, None] if y.ndim < 2 else y
            return X, y
        return X
