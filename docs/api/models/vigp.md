---
layout: default
title: VIGP
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.vigp"></a>

# neurobayes.models.vigp

<a id="neurobayes.models.vigp.VIGP"></a>

## VIGP Objects

```python
class VIGP(GP)
```

Variational Inference-based Gaussian process

<a id="neurobayes.models.vigp.VIGP.fit"></a>

#### fit

```python
def fit(X: jnp.ndarray,
        y: jnp.ndarray,
        num_steps: int = 1000,
        step_size: float = 5e-3,
        progress_bar: bool = True,
        device: str = None,
        rng_key: jnp.array = None,
        **kwargs: float) -> None
```

Run variational inference to learn GP (hyper)parameters

**Arguments**:

- `rng_key` - random number generator key
- `X` - 2D feature vector with *(number of points, number of features)* dimensions
- `y` - 1D target vector with *(n,)* dimensions
- `num_steps` - number of SVI steps
- `step_size` - step size schedule for Adam optimizer
- `progress_bar` - show progress bar
- `print_summary` - print summary at the end of training
- `rng_key` - random number generator key


