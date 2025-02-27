---
layout: default
title: VarianceModelHeteroskedasticBNN
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.bnn_heteroskedastic_model"></a>

# neurobayes.models.bnn\_heteroskedastic\_model

<a id="neurobayes.models.bnn_heteroskedastic_model.VarianceModelHeteroskedasticBNN"></a>

## VarianceModelHeteroskedasticBNN Objects

```python
class VarianceModelHeteroskedasticBNN(HeteroskedasticBNN)
```

Variance model based heteroskedastic Bayesian Neural Network

**Arguments**:

- `architecture` - a Flax model.
- `variance_model` _Callable_ - Function to compute the variance given inputs and parameters.
- `variance_model_prior` _Callable_ - Function to sample prior parameters for the variance model.

<a id="neurobayes.models.bnn_heteroskedastic_model.VarianceModelHeteroskedasticBNN.model"></a>

#### model

```python
def model(X: jnp.ndarray,
          y: jnp.ndarray = None,
          priors_sigma: float = 1.0,
          **kwargs) -> None
```

Heteroskedastic BNN model


