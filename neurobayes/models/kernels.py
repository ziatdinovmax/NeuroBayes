from typing import Union, Dict

import jax.numpy as jnp


def _sqrt(x, eps=1e-12):
    return jnp.sqrt(x + eps)


def square_scaled_distance(X: jnp.ndarray, Z: jnp.ndarray,
                           lengthscale: Union[jnp.ndarray, float] = 1.
                           ) -> jnp.ndarray:
    r"""
    Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
    between X and Z are vectors with :math:`n x num_features` dimensions
    """
    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X ** 2).sum(1, keepdims=True)
    Z2 = (scaled_Z ** 2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)


def RBFKernel(X: jnp.ndarray, Z: jnp.ndarray,
              params: Dict[str, jnp.ndarray],
              noise: int = 0, jitter: float = 1e-6,
              ) -> jnp.ndarray:
    """
    Radial basis function kernel

    Args:
        X: 2D vector with *(number of points, number of features)* dimension
        Z: 2D vector with *(number of points, number of features)* dimension
        params: Dictionary with kernel hyperparameters 'k_length' and 'k_scale'
        noise: optional noise vector with dimension (n,)

    Returns:
        Computed kernel matrix betwenen X and Z
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    k = params["k_scale"] * jnp.exp(-0.5 * r2)
    if Z is X:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


def MaternKernel(X: jnp.ndarray, Z: jnp.ndarray,
                 params: Dict[str, jnp.ndarray],
                 noise: int = 0, jitter: float = 1e-6,
                 ) -> jnp.ndarray:
    """
    Matern52 kernel

    Args:
        X: 2D vector with *(number of points, number of features)* dimension
        Z: 2D vector with *(number of points, number of features)* dimension
        params: Dictionary with kernel hyperparameters 'k_length' and 'k_scale'
        noise: optional noise vector with dimension (n,)

    Returns:
        Computed kernel matrix between X and Z
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    r = _sqrt(r2)
    sqrt5_r = 5**0.5 * r
    k = params["k_scale"] * (1 + sqrt5_r + (5/3) * r2) * jnp.exp(-sqrt5_r)
    if Z is X:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k
