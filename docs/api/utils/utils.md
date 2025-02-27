---
layout: default
title: Utils
parent: Utils
grand_parent: API Reference
---

<a id="neurobayes.utils.utils"></a>

# neurobayes.utils.utils

<a id="neurobayes.utils.utils.infer_device"></a>

#### infer\_device

```python
def infer_device(device_preference: str = None)
```

Returns a JAX device based on the specified preference.
Defaults to the first available device if no preference is given, or if the specified
device type is not available.

**Arguments**:

  - device_preference (str, optional): The preferred device type ('cpu' or 'gpu').
  

**Returns**:

  - A JAX device.

<a id="neurobayes.utils.utils.put_on_device"></a>

#### put\_on\_device

```python
def put_on_device(device=None, *data_items)
```

Places multiple data items on the specified device.

**Arguments**:

- `device` - The target device as a string (e.g., 'cpu', 'gpu'). If None, the default device is used.
- `*data_items` - Variable number of data items (such as JAX array or dictionary) to be placed on the device.
  

**Returns**:

  A tuple of the data items placed on the specified device. The structure of each data item is preserved.

<a id="neurobayes.utils.utils.split_in_batches"></a>

#### split\_in\_batches

```python
def split_in_batches(array: jnp.ndarray,
                     batch_size: int = 200) -> List[jnp.ndarray]
```

Splits array into batches

<a id="neurobayes.utils.utils.split_dict"></a>

#### split\_dict

```python
def split_dict(data: Dict[str, jnp.ndarray],
               chunk_size: int) -> List[Dict[str, jnp.ndarray]]
```

Splits a dictionary of arrays into a list of smaller dictionaries.

**Arguments**:

- `data` - Dictionary containing numpy arrays.
- `chunk_size` - Desired size of the smaller arrays.
  

**Returns**:

  List of dictionaries with smaller numpy arrays.

<a id="neurobayes.utils.utils.monitor_dnn_loss"></a>

#### monitor\_dnn\_loss

```python
def monitor_dnn_loss(loss: np.ndarray) -> None
```

Checks whether current change in loss is greater than a 25% decrease

<a id="neurobayes.utils.utils.mse"></a>

#### mse

```python
def mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray
```

Calculates the mean squared error between true and predicted values.

<a id="neurobayes.utils.utils.rmse"></a>

#### rmse

```python
def rmse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray
```

Calculates the root mean squared error between true and predicted values.

<a id="neurobayes.utils.utils.mae"></a>

#### mae

```python
def mae(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray
```

Calculates the mean absolute error between true and predicted values.

<a id="neurobayes.utils.utils.nlpd"></a>

#### nlpd

```python
def nlpd(y: jnp.ndarray,
         mu: jnp.ndarray,
         sigma_squared: jnp.ndarray,
         eps: float = 1e-6) -> jnp.ndarray
```

Computes the Negative Log Predictive Density (NLPD) for observed data points
given the predictive mean and variance.

**Arguments**:

- `y` _np.array_ - Array of observed values
- `mu` _np.array_ - Array of predictive means from the model
- `sigma_squared` _np.array_ - Array of predictive variances from the model
  

**Returns**:

  The NLPD value

<a id="neurobayes.utils.utils.get_flax_compatible_dict"></a>

#### get\_flax\_compatible\_dict

```python
def get_flax_compatible_dict(
    params_numpyro: Dict[str,
                         jnp.ndarray]) -> Dict[str, Dict[str, jnp.ndarray]]
```

Takes a dictionary with MCMC samples produced by numpyro
and creates a dictionary with weights and biases compatible
with flax .apply() method.

<a id="neurobayes.utils.utils.get_prior_means_from_samples"></a>

#### get\_prior\_means\_from\_samples

```python
def get_prior_means_from_samples(
    params_numpyro: Dict[str,
                         jnp.ndarray]) -> Dict[str, Dict[str, jnp.ndarray]]
```

Takes a dictionary with MCMC samples produced by numpyro
and creates a dictionary with mean of weights and biases
that can be used to set priors in BNNs

<a id="neurobayes.utils.utils.flatten_params_dict"></a>

#### flatten\_params\_dict

```python
def flatten_params_dict(params_dict: Dict[str, Any]) -> Dict[str, Any]
```

Recursively flatten a nested parameter dictionary into a flat dictionary
where each key maps to a parameter dictionary with 'kernel' and 'bias'.

<a id="neurobayes.utils.utils.flatten_transformer_params_dict"></a>

#### flatten\_transformer\_params\_dict

```python
def flatten_transformer_params_dict(params_dict)
```

Properly flatten transformer parameter dictionary to match our layer naming scheme.

<a id="neurobayes.utils.utils.set_fn"></a>

#### set\_fn

```python
def set_fn(func: Callable) -> Callable
```

Transforms a given deterministic function to use a params dictionary
for its parameters, excluding the first one (assumed to be the dependent variable).

**Arguments**:

  - func (Callable): The deterministic function to be transformed.
  

**Returns**:

  - Callable: The transformed function where parameters are accessed
  from a `params` dictionary.


