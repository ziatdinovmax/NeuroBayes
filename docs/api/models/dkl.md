---
layout: default
title: DKL
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.dkl"></a>

# neurobayes.models.dkl

<a id="neurobayes.models.dkl.DKL"></a>

## DKL Objects

```python
class DKL(GP)
```

Fully Bayesian Deep Kernel Learning

<a id="neurobayes.models.dkl.DKL.model"></a>

#### model

```python
def model(X: jnp.ndarray, y: jnp.ndarray = None, **kwargs: float) -> None
```

DKL probabilistic model

<a id="neurobayes.models.dkl.DKL.embed"></a>

#### embed

```python
def embed(X_new: jnp.ndarray) -> jnp.ndarray
```

Embeds data into the latent space using the inferred weights
of the DKL's Bayesian neural network


