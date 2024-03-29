from typing import Callable, Dict, List
from dataclasses import dataclass
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


@dataclass
class GPPriors:
    lengthscale_prior: dist.Distribution = dist.LogNormal(0.0, 1.0)
    noise_prior: dist.Distribution = dist.LogNormal(0.0, 1.0)
    output_scale_prior: dist.Distribution = dist.LogNormal(0.0, 1.0)


def sample_weights(name: str, in_channels: int, out_channels: int) -> jnp.ndarray:
    """Sampling weights matrix"""
    w = numpyro.sample(name=name, fn=dist.Normal(
        loc=jnp.zeros((in_channels, out_channels)),
        scale=jnp.ones((in_channels, out_channels))))
    return w


def sample_biases(name: str, channels: int) -> jnp.ndarray:
    """Sampling bias vector"""
    b = numpyro.sample(name=name, fn=dist.Cauchy(
        loc=jnp.zeros((channels)), scale=jnp.ones((channels))))
    return b


def get_mlp_prior(input_dim: int, output_dim: int, architecture: List[int]) -> Callable[[], Dict[str, jnp.ndarray]]:
    """Priors over weights and biases for a Bayesian MLP"""
    def mlp_prior():
        params = {}
        in_channels = input_dim
        for i, out_channels in enumerate(architecture):
            params[f"w{i}"] = sample_weights(f"w{i}", in_channels, out_channels)
            params[f"b{i}"] = sample_biases(f"b{i}", out_channels)
            in_channels = out_channels
        # Output layer
        params[f"w{len(architecture)}"] = sample_weights(f"w{len(architecture)}", in_channels, output_dim)
        params[f"b{len(architecture)}"] = sample_biases(f"b{len(architecture)}", output_dim)
        return params
    return mlp_prior
