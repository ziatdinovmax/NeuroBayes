---
layout: default
title: GP
parent: Models
grand_parent: API Reference
---

<a id="neurobayes.models.gp"></a>

# neurobayes.models.gp

<a id="neurobayes.models.gp.GP"></a>

## GP Objects

```python
class GP()
```

Fully Bayesian exact Gaussian process

<a id="neurobayes.models.gp.GP.model"></a>

#### model

```python
def model(X: jnp.ndarray, y: jnp.ndarray = None) -> None
```

GP probabilistic model with inputs X and targets y

<a id="neurobayes.models.gp.GP.fit"></a>

#### fit

```python
def fit(
    X: jnp.ndarray,
    y: jnp.ndarray,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = 1,
    chain_method: str = "sequential",
    progress_bar: bool = True,
    print_summary: bool = True,
    device: str = None,
    rng_key: jnp.array = None,
    extra_fields: Optional[Tuple[str]] = ()) -> None
```

Run Hamiltonian Monter Carlo to infer the GP parameters

**Arguments**:

- `X` - 2D feature vector
- `y` - 1D target vector
- `num_warmup` - number of HMC warmup states
- `num_samples` - number of HMC samples
- `num_chains` - number of HMC chains
- `chain_method` - 'sequential', 'parallel' or 'vectorized'
- `progress_bar` - show progress bar
- `print_summary` - print summary at the end of sampling
  device:
  The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
  is performed on the JAX default device.
- `rng_key` - random number generator key
  extra_fields:
  Extra fields (e.g. 'accept_prob') to collect during the HMC run.
  The extra fields are accessible from model.mcmc.get_extra_fields() after model training.

<a id="neurobayes.models.gp.GP.sample_noise"></a>

#### sample\_noise

```python
def sample_noise() -> jnp.ndarray
```

Sample observational noise variance

<a id="neurobayes.models.gp.GP.sample_kernel_params"></a>

#### sample\_kernel\_params

```python
def sample_kernel_params(kernel_dim: int) -> Dict[str, jnp.ndarray]
```

Sample kernel parameters

<a id="neurobayes.models.gp.GP.compute_gp_posterior"></a>

#### compute\_gp\_posterior

```python
def compute_gp_posterior(
        X_new: jnp.ndarray,
        X_train: jnp.ndarray,
        y_train: jnp.ndarray,
        params: Dict[str, jnp.ndarray],
        noiseless: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]
```

Returns mean and covariance of multivariate normal
posterior for a single sample of trained GP parameters

<a id="neurobayes.models.gp.GP.predict"></a>

#### predict

```python
def predict(X_new: jnp.ndarray,
            noiseless: bool = True,
            device: str = None) -> Tuple[jnp.ndarray, jnp.ndarray]
```

Make prediction at X_new points a trained GP model

**Arguments**:

  X_new:
  New inputs with *(number of points, number of features)* dimensions
  noiseless:
  Noise-free prediction. It is set to False by default as new/unseen data is assumed
  to follow the same distribution as the training data. Hence, since we introduce a model noise
  by default for the training data, we also want to include that noise in our prediction.
  device:
  The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
  is performed on the JAX default device.
  

**Returns**:

  Posterior mean and variance

<a id="neurobayes.models.gp.GP.predict_in_batches"></a>

#### predict\_in\_batches

```python
def predict_in_batches(X_new: jnp.ndarray,
                       batch_size: int = 200,
                       noiseless: bool = True,
                       device: str = None) -> Tuple[jnp.ndarray, jnp.ndarray]
```

Make prediction in batches (to avoid memory overflow) 
at X_new points a trained GP model

<a id="neurobayes.models.gp.GP.draw_from_mvn"></a>

#### draw\_from\_mvn

```python
def draw_from_mvn(rng_key: jnp.ndarray, X_new: jnp.ndarray,
                  params: Dict[str, jnp.ndarray], n_draws: int,
                  noiseless: bool) -> jnp.ndarray
```

Draws predictive samples from multivariate normal distribution
at X_new for a single estimate of GP posterior parameters

<a id="neurobayes.models.gp.GP.sample_from_posterior"></a>

#### sample\_from\_posterior

```python
def sample_from_posterior(X_new: jnp.ndarray,
                          noiseless: bool = True,
                          n_draws: int = 100,
                          device: str = None,
                          rng_key: jnp.ndarray = None) -> jnp.ndarray
```

Sample from the posterior predictive distribution at X_new

**Arguments**:

  X_new:
  New inputs with *(number of points, number of features)* dimensions
  noiseless:
  Noise-free prediction. It is set to False by default as new/unseen data is assumed
  to follow the same distribution as the training data. Hence, since we introduce a model noise
  by default for the training data, we also want to include that noise in our prediction.
  n_draws:
  Number of MVN distribution samples to draw for each sample with GP parameters
  device:
  The device (e.g. "cpu" or "gpu") perform computation on ('cpu', 'gpu'). If None, computation
  is performed on the JAX default device.
  rng_key:
  Optional random number generator key
  

**Returns**:

  A set of samples from the posterior predictive distribution.

<a id="neurobayes.models.gp.GP.get_samples"></a>

#### get\_samples

```python
def get_samples(chain_dim: bool = False) -> Dict[str, jnp.ndarray]
```

Get posterior samples (after running the MCMC chains)


