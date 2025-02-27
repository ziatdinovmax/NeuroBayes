---
layout: default
title: VIDKL
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.vidkl"></a>

# neurobayes.models.vidkl

<a id="neurobayes.models.vidkl.VIDKL"></a>

## VIDKL Objects

```python
class VIDKL(DKL)
```

Variational Inference-based Deep Kernel Learning

<a id="neurobayes.models.vidkl.VIDKL.fit"></a>

#### fit

```python
def fit(X: jnp.ndarray,
        y: jnp.ndarray,
        num_steps: int = 1000,
        step_size: float = 5e-3,
        progress_bar: bool = True,
        print_summary: bool = True,
        device: str = None,
        rng_key: jnp.array = None,
        **kwargs: float) -> None
```

Run variational inference to learn DKL (hyper)parameters

**Arguments**:

- `rng_key` - random number generator key
- `X` - 2D feature vector with *(number of points, number of features)* dimensions
- `y` - 1D target vector with *(n,)* dimensions
- `num_steps` - number of SVI steps
- `step_size` - step size schedule for Adam optimizer
- `progress_bar` - show progress bar
  device:
  The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
  is performed on the JAX default device.
- `print_summary` - print summary at the end of training
- `rng_key` - random number generator key


