---
layout: default
title: PartialBNN
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.partial_bnn"></a>

# neurobayes.models.partial\_bnn

<a id="neurobayes.models.partial_bnn.PartialBNN"></a>

## PartialBNN Objects

```python
class PartialBNN()
```

A unified wrapper for partially Bayesian neural networks that supports MLPs,
ConvNets, and Transformers with a consistent API.

**Arguments**:

- `architecture` - Neural network architecture (FlaxMLP, FlaxConvNet, or FlaxTransformer)
- `deterministic_weights` - Pre-trained deterministic weights. If not provided,
  the network will be trained from scratch when running .fit() method
- `num_probabilistic_layers` - Number of layers at the end to be treated as fully stochastic
- `probabilistic_layer_names` - Names of neural network modules to be treated probabilistically
- `num_classes` - Number of classes for classification task.
  If None, the model performs regression. Defaults to None.
- `noise_prior` - Custom prior for observational noise distribution

<a id="neurobayes.models.partial_bnn.PartialBNN.fit"></a>

#### fit

```python
def fit(X: jnp.ndarray,
        y: jnp.ndarray,
        num_warmup: int = 2000,
        num_samples: int = 2000,
        num_chains: int = 1,
        chain_method: str = 'sequential',
        sgd_epochs: Optional[int] = None,
        sgd_lr: Optional[float] = 0.01,
        sgd_batch_size: Optional[int] = None,
        swa_config: Optional[Dict] = None,
        map_sigma: float = 1.0,
        priors_sigma: float = 1.0,
        progress_bar: bool = True,
        device: str = None,
        rng_key: Optional[jnp.array] = None,
        extra_fields: Optional[Tuple[str, ...]] = (),
        select_neurons_config: Optional[Dict] = None,
        max_num_restarts: int = 1,
        min_accept_prob: float = 0.55,
        run_diagnostics: bool = False) -> None
```

Fit the partially Bayesian neural network.

**Arguments**:

- `X` - Input data
  - For MLP: 2D array (n_samples, n_features)
  - For ConvNet: ND array (n_samples, *dims, n_channels)
  - For Transformer: 2D array (n_samples, seq_length)
- `y` - Target values
  - For regression: 1D array (n_samples,) or 2D array (n_samples, target_dim)
  - For classification: 1D array (n_samples,) with class labels
- `num_warmup` - Number of NUTS warmup steps. Defaults to 2000.
- `num_samples` - Number of NUTS samples to draw. Defaults to 2000.
- `num_chains` - Number of NUTS chains to run. Defaults to 1.
- `chain_method` - Method for running chains: 'sequential', 'parallel',
  or 'vectorized'. Defaults to 'sequential'.
- `sgd_epochs` - Number of SGD training epochs for deterministic NN.
  Defaults to 500 (if no pretrained weights are provided).
- `sgd_lr` - SGD learning rate. Defaults to 0.01.
- `sgd_batch_size` - Mini-batch size for SGD training.
  Defaults to None (all input data is processed as a single batch).
- `swa_config` - Stochastic weight averaging protocol. Defaults to averaging weights
  at the end of training trajectory (the last 5% of SGD epochs).
- `map_sigma` - Sigma in Gaussian prior for regularized SGD training. Defaults to 1.0.
- `priors_sigma` - Standard deviation for default or pretrained priors
  in the Bayesian part of the NN. Defaults to 1.0.
- `progress_bar` - Show progress bar. Defaults to True.
- `device` - The device to perform computation on ('cpu', 'gpu').
  Defaults to None (JAX default device).
- `rng_key` - Random number generator key. Defaults to None.
- `extra_fields` - Extra fields to collect during the MCMC run.
  Defaults to ().
- `select_neurons_config` _Optional[Dict], optional_ - Configuration for selecting
  probabilistic neurons after deterministic training. Should contain:
  - method: str - Selection method ('variance', 'gradient', etc.)
  - layer_names: List[str] - Names of layers to make partially Bayesian
  - num_pairs_per_layer: int - Number of weight pairs to select per layer
  - Additional method-specific parameters
- `max_num_restarts` - Maximum number of fitting attempts for single chain.
  Ignored if num_chains > 1. Defaults to 1.
- `min_accept_prob` - Minimum acceptance probability threshold.
  Only used if num_chains = 1. Defaults to 0.55.
- `run_diagnostics` - Run Gelman-Rubin diagnostics layer-by-layer at the end.
  Defaults to False.
  

**Example**:

  model.fit(
  X, y, num_warmup=1000, num_samples=1000,
  sgd_lr=1e-3, sgd_epochs=100,
  select_neurons_config={
- `'method'` - 'variance',
- `'layer_names'` - ['Dense0', 'Dense2'],
- `'num_pairs_per_layer'` - 5
  }
  )

<a id="neurobayes.models.partial_bnn.PartialBNN.predict"></a>

#### predict

```python
def predict(X: jnp.ndarray) -> jnp.ndarray
```

Make predictions using the fitted model.

<a id="neurobayes.models.partial_bnn.PartialBNN.get_samples"></a>

#### get\_samples

```python
def get_samples(chain_dim: bool = False) -> Dict[str, jnp.ndarray]
```

Get posterior samples (after running the MCMC chains)

<a id="neurobayes.models.partial_bnn.PartialBNN.predict_classes"></a>

#### predict\_classes

```python
def predict_classes(X_new: jnp.ndarray,
                    samples: Optional[Dict[str, jnp.ndarray]] = None,
                    device: Optional[str] = None,
                    rng_key: Optional[jnp.ndarray] = None) -> jnp.ndarray
```

Predict class labels for classification tasks

<a id="neurobayes.models.partial_bnn.PartialBNN.sample_from_posterior"></a>

#### sample\_from\_posterior

```python
def sample_from_posterior(
        rng_key: jnp.ndarray,
        X_new: jnp.ndarray,
        samples: Dict[str, jnp.ndarray],
        return_sites: Optional[List[str]] = None) -> jnp.ndarray
```

Sample from posterior distribution at new inputs X_new

<a id="neurobayes.models.partial_bnn.PartialBNN.set_data"></a>

#### set\_data

```python
def set_data(
        X: jnp.ndarray,
        y: Optional[jnp.ndarray] = None
) -> Union[Tuple[jnp.ndarray], jnp.ndarray]
```

Prepare data for model fitting or prediction

<a id="neurobayes.models.partial_bnn.PartialBNN.is_regression"></a>

#### is\_regression

```python
@property
def is_regression() -> bool
```

Check if the model is performing regression

<a id="neurobayes.models.partial_bnn.PartialBNN.mcmc"></a>

#### mcmc

```python
@property
def mcmc()
```

Get the MCMC sampler

<a id="neurobayes.models.partial_bnn.PartialBNN.diagnostic_results"></a>

#### diagnostic\_results

```python
@property
def diagnostic_results()
```

Get diagnostic results if run_diagnostics was True during fitting


