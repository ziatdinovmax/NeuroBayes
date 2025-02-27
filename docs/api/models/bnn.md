---
layout: default
title: BNN
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.bnn"></a>

# neurobayes.models.bnn

<a id="neurobayes.models.bnn.BNN"></a>

## BNN Objects

```python
class BNN()
```

A Bayesian Neural Network (BNN) for both regression and classification tasks.

This model automatically determines the task type based on num_classes:
- If num_classes is None: Regression task
- If num_classes >= 2: Classification task with specified number of classes

**Arguments**:

- `architecture` - a Flax model
- `num_classes` _int, optional_ - Number of classes for classification task.
  If None, the model performs regression. Defaults to None.
- `noise_prior` _dist.Distribution, optional_ - Prior probability distribution over
  observational noise for regression. Defaults to HalfNormal(1.0).
- `pretrained_priors` _Dict, optional_ - Dictionary with pre-trained weights.
  

**Notes**:

  For input labels y:
  - Regression (num_classes=None): y should be a 1D array of shape (n_samples,) for single output
  or 2D array of shape (n_samples, n_outputs) for multiple outputs
  - Binary classification (num_classes=2): y should be a 1D array of shape (n_samples,)
  containing 0s and 1s
  - Multi-class classification (num_classes>2): y should be a 1D array of shape (n_samples,)
  containing class indices from 0 to num_classes-1

<a id="neurobayes.models.bnn.BNN.is_regression"></a>

#### is\_regression

```python
@property
def is_regression() -> bool
```

Check if the model is performing regression

<a id="neurobayes.models.bnn.BNN.fit"></a>

#### fit

```python
def fit(X: jnp.ndarray,
        y: jnp.ndarray,
        num_warmup: int = 2000,
        num_samples: int = 2000,
        num_chains: int = 1,
        chain_method: str = 'sequential',
        priors_sigma: Optional[float] = 1.0,
        progress_bar: bool = True,
        device: Optional[str] = None,
        rng_key: Optional[jnp.array] = None,
        extra_fields: Optional[Tuple[str, ...]] = (),
        max_num_restarts: int = 1,
        min_accept_prob: float = 0.55,
        run_diagnostics: bool = False) -> None
```

Run No-U-Turn Sampler (NUTS) to infer parameters of the Bayesian Neural Network.

**Arguments**:

- `X` _jnp.ndarray_ - Input features. For MLP: 2D array of shape (n_samples, n_features).
  For ConvNet: N-D array of shape (n_samples, *dims, n_channels), where
  dims = (length,) for spectral data or (height, width) for image data.
- `y` _jnp.ndarray_ - Target array. For single-output problems: 1D array of shape (n_samples,).
  For multi-output problems: 2D array of shape (n_samples, target_dim).
- `num_warmup` _int, optional_ - Number of NUTS warmup steps. Defaults to 2000.
- `num_samples` _int, optional_ - Number of NUTS samples to draw. Defaults to 2000.
- `num_chains` _int, optional_ - Number of NUTS chains to run. Defaults to 1.
- `chain_method` _str, optional_ - Method for running chains: 'sequential', 'parallel',
  or 'vectorized'. Defaults to 'sequential'.
- `priors_sigma` _float, optional_ - Standard deviation for default or pretrained priors.
  Defaults to 1.0.
- `progress_bar` _bool, optional_ - Whether to show a progress bar. Defaults to True.
- `device` _str, optional_ - The device to perform computation on ('cpu', 'gpu').
  If None, uses the JAX default device.
- `rng_key` _jnp.ndarray, optional_ - Random number generator key. If None, uses a default key.
- `extra_fields` _Tuple[str, ...], optional_ - Extra fields (e.g. 'accept_prob') to collect
  during the MCMC run. Accessible via model.mcmc.get_extra_fields() after training.
- `max_num_restarts` _int, optional_ - Maximum number of fitting attempts for single chain.
  Ignored if num_chains > 1. Defaults to 1.
- `min_accept_prob` _float, optional_ - Minimum acceptance probability threshold.
  Only used if num_chains = 1. Defaults to 0.55.
- `run_diagnostics` _bool, optional_ - Run Gelman-Rubin diagnostics layer-by-layer at the end.
  Defaults to False.
  

**Returns**:

- `None` - The method updates the model's internal state but does not return a value.
  

**Notes**:

  After running this method, the MCMC samples are stored in the `mcmc` attribute
  of the model and can be accessed via .get_samples() method for further analysis.

<a id="neurobayes.models.bnn.BNN.get_samples"></a>

#### get\_samples

```python
def get_samples(chain_dim: bool = False) -> Dict[str, jnp.ndarray]
```

Get posterior samples (after running the MCMC chains)

<a id="neurobayes.models.bnn.BNN.predict"></a>

#### predict

```python
def predict(
        X_new: jnp.ndarray,
        samples: Optional[Dict[str, jnp.ndarray]] = None,
        device: Optional[str] = None,
        rng_key: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]
```

Predict outputs for new inputs.

**Arguments**:

- `X_new` _jnp.ndarray_ - New input data for predictions
- `samples` _Dict[str, jnp.ndarray], optional_ - Dictionary of posterior samples
- `device` _str, optional_ - The device to perform computation on
- `rng_key` _jnp.ndarray, optional_ - Random number generator key
  

**Returns**:

  Tuple[jnp.ndarray, jnp.ndarray]: (predictive mean and uncertainty)

<a id="neurobayes.models.bnn.BNN.predict_classes"></a>

#### predict\_classes

```python
def predict_classes(X_new: jnp.ndarray,
                    samples: Optional[Dict[str, jnp.ndarray]] = None,
                    device: Optional[str] = None,
                    rng_key: Optional[jnp.ndarray] = None) -> jnp.ndarray
```

Predict class labels for classification tasks

<a id="neurobayes.models.bnn.BNN.sample_from_posterior"></a>

#### sample\_from\_posterior

```python
def sample_from_posterior(
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        samples: Dict[str, jnp.ndarray],
        return_sites: Optional[List[str]] = None) -> jnp.ndarray
```

Sample from posterior distribution at new inputs X_new

<a id="neurobayes.models.bnn.BNN.set_data"></a>

#### set\_data

```python
def set_data(
        X: jnp.ndarray,
        y: Optional[jnp.ndarray] = None
) -> Union[Tuple[jnp.ndarray], jnp.ndarray]
```

Prepare data for model fitting or prediction.

Ensures consistent shapes for both X and y:
- X: Always at least 2D
- y: Shape depends on task type:
    - Regression: 2D array (n_samples, n_outputs)
    - Classification (binary or multi-class): 1D array (n_samples,)


