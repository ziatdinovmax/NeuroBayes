---
layout: default
title: Kernels
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.kernels"></a>

# neurobayes.models.kernels

<a id="neurobayes.models.kernels.square_scaled_distance"></a>

#### square\_scaled\_distance

```python
def square_scaled_distance(
        X: jnp.ndarray,
        Z: jnp.ndarray,
        lengthscale: Union[jnp.ndarray, float] = 1.) -> jnp.ndarray
```

Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
between X and Z are vectors with :math:`n x num_features` dimensions

<a id="neurobayes.models.kernels.RBFKernel"></a>

#### RBFKernel

```python
def RBFKernel(X: jnp.ndarray,
              Z: jnp.ndarray,
              params: Dict[str, jnp.ndarray],
              noise: int = 0,
              jitter: float = 1e-6) -> jnp.ndarray
```

Radial basis function kernel

**Arguments**:

- `X` - 2D vector with *(number of points, number of features)* dimension
- `Z` - 2D vector with *(number of points, number of features)* dimension
- `params` - Dictionary with kernel hyperparameters 'k_length' and 'k_scale'
- `noise` - optional noise vector with dimension (n,)
  

**Returns**:

  Computed kernel matrix betwenen X and Z

<a id="neurobayes.models.kernels.MaternKernel"></a>

#### MaternKernel

```python
def MaternKernel(X: jnp.ndarray,
                 Z: jnp.ndarray,
                 params: Dict[str, jnp.ndarray],
                 noise: int = 0,
                 jitter: float = 1e-6) -> jnp.ndarray
```

Matern52 kernel

**Arguments**:

- `X` - 2D vector with *(number of points, number of features)* dimension
- `Z` - 2D vector with *(number of points, number of features)* dimension
- `params` - Dictionary with kernel hyperparameters 'k_length' and 'k_scale'
- `noise` - optional noise vector with dimension (n,)
  

**Returns**:

  Computed kernel matrix between X and Z


