from typing import Callable, Dict, List
from dataclasses import dataclass
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


@dataclass
class GPPriors:
    lengthscale_prior: dist.Distribution = dist.LogNormal(0.0, 1.0)
    noise_prior: dist.Distribution = dist.HalfNormal(1.0)
    output_scale_prior: dist.Distribution = dist.LogNormal(0.0, 1.0)


def sample_weights(name: str, in_channels: int, out_channels: int, scale: float = 1.0) -> jnp.ndarray:
    """Sampling weights matrix"""
    w = numpyro.sample(name=name, fn=dist.Normal(
        jnp.zeros((in_channels, out_channels)),
        scale * jnp.ones((in_channels, out_channels)))
    )
    return w


def sample_biases(name: str, channels: int, scale: float = 1.0) -> jnp.ndarray:
    """Sampling bias vector"""
    b = numpyro.sample(name=name, fn=dist.Cauchy(
        jnp.zeros((channels)),
        scale * jnp.ones((channels)))
    )
    return b


def get_mlp_prior(input_dim: int, output_dim: int,
                  architecture: List[int], name: str = "main",
                  scale: float = 1.0
                  ) -> Callable[[], Dict[str, jnp.ndarray]]:
    """Priors over weights and biases for a Bayesian MLP"""
    def mlp_prior():
        params = {}
        in_channels = input_dim
        for i, out_channels in enumerate(architecture):
            params[f"{name}_w{i}"] = sample_weights(
                f"{name}_w{i}", in_channels, out_channels, scale)
            params[f"{name}_b{i}"] = sample_biases(
                f"{name}_b{i}", out_channels, scale)
            in_channels = out_channels
        # Output layer
        params[f"{name}_w{len(architecture)}"] = sample_weights(
            f"{name}_w{len(architecture)}", in_channels, output_dim, scale)
        params[f"{name}_b{len(architecture)}"] = sample_biases(
            f"{name}_b{len(architecture)}", output_dim, scale)
        return params
    return mlp_prior


def get_heteroskedastic_mlp_prior(input_dim: int, output_dim: int,
                                  architecture: List[int],
                                  scale: float = 1.0
                                  ) -> Callable[[], Dict[str, jnp.ndarray]]:
    """Priors over weights and biases for a Bayesian MLP with heteroskedastic outputs"""
    def mlp_prior():
        params = {}
        in_channels = input_dim
        for i, out_channels in enumerate(architecture):
            params[f"w{i}"] = sample_weights(f"w{i}", in_channels, out_channels, scale)
            params[f"b{i}"] = sample_biases(f"b{i}", out_channels, scale)
            in_channels = out_channels
        # Output layers for mean and variance
        params['w_mean'] = sample_weights('w_mean', in_channels, output_dim, scale)
        params['b_mean'] = sample_biases('b_mean', output_dim, scale)
        params['w_variance'] = sample_weights('w_variance', in_channels, output_dim, scale)
        params['b_variance'] = sample_biases('b_variance', output_dim, scale)
        return params
    return mlp_prior
