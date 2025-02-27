---
layout: default
title: HeteroskedasticBNN
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.bnn_heteroskedastic"></a>

# neurobayes.models.bnn\_heteroskedastic

<a id="neurobayes.models.bnn_heteroskedastic.HeteroskedasticBNN"></a>

## HeteroskedasticBNN Objects

```python
class HeteroskedasticBNN(BNN)
```

Heteroskedastic Bayesian Neural Network for input-dependent observational noise

**Arguments**:

- `architecture` - a Flax model
  pretrained_priors (Dict, optional):
  Dictionary with pre-trained weights for the provided model architecture.
  These weight values will be used to initialize prior distributions in BNN.

<a id="neurobayes.models.bnn_heteroskedastic.HeteroskedasticBNN.model"></a>

#### model

```python
def model(X: jnp.ndarray,
          y: jnp.ndarray = None,
          pretrained_priors: Dict = None,
          priors_sigma: float = 1.0,
          **kwargs) -> None
```

Heteroskedastic BNN model

<a id="neurobayes.models.bnn_heteroskedastic.HeteroskedasticBNN.predict_noise"></a>

#### predict\_noise

```python
def predict_noise(X_new: jnp.ndarray,
                  device: Optional[str] = None) -> jnp.ndarray
```

Predict likely values of noise for new data
and associated uncertainty in the noise prediction


