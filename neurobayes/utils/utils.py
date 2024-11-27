import inspect
import re

from typing import List, Dict, Any, Callable

import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

import numpyro

import warnings


def infer_device(device_preference: str = None):
    """
    Returns a JAX device based on the specified preference.
    Defaults to the first available device if no preference is given, or if the specified
    device type is not available.

    Args:
    - device_preference (str, optional): The preferred device type ('cpu' or 'gpu').

    Returns:
    - A JAX device.
    """
    if device_preference:
        # Normalize the input to lowercase to ensure compatibility.
        device_preference = device_preference.lower()
        # Try to get devices of the specified type.
        devices_of_type = jax.devices(device_preference)
        if devices_of_type:
            # If there are any devices of the requested type, return the first one.
            return devices_of_type[0]
        else:
            print(f"No devices of type '{device_preference}' found. Falling back to the default device.")

    # If no preference is specified or no devices of the specified type are found, return the default device.
    return jax.devices()[0]


def put_on_device(device=None, *data_items):
    """
    Places multiple data items on the specified device.

    Args:
        device: The target device as a string (e.g., 'cpu', 'gpu'). If None, the default device is used.
        *data_items: Variable number of data items (such as JAX array or dictionary) to be placed on the device.

    Returns:
        A tuple of the data items placed on the specified device. The structure of each data item is preserved.
    """
    if device is not None:
        device = infer_device(device)
        return tuple(jax.device_put(item, device) for item in data_items)
    return data_items


def split_in_batches(array: jnp.ndarray, batch_size: int = 200) -> List[jnp.ndarray]:
    """Splits array into batches"""
    num_batches = (array.shape[0] + batch_size - 1) // batch_size
    return [array[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def split_dict(data: Dict[str, jnp.ndarray], chunk_size: int
               ) -> List[Dict[str, jnp.ndarray]]:
    """Splits a dictionary of arrays into a list of smaller dictionaries.

    Args:
        data: Dictionary containing numpy arrays.
        chunk_size: Desired size of the smaller arrays.

    Returns:
        List of dictionaries with smaller numpy arrays.
    """
    N = len(next(iter(data.values())))
    num_chunks = int(np.ceil(N / chunk_size))
    result = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i+1) * chunk_size, N)

        chunk = {key: value[start_idx:end_idx] for key, value in data.items()}
        result.append(chunk)

    return result

def monitor_dnn_loss(loss: np.ndarray) -> None:
    """Checks whether current change in loss is greater than a 25% decrease"""
    loss = loss[loss != 0]
    if len(loss) > 1:
      if np.diff(loss)[-1] / loss[0] < -0.25:
          warnings.warn('The deterministic training loss is decreasing rapidly - learning and accuracy may be improved by increasing the batch size, increasing MAP sigma, or adjusting the learning rate.', stacklevel=2)
    return

def mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the mean squared error between true and predicted values.
    """
    mse = jnp.mean((y_true - y_pred) ** 2)
    return mse


def rmse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the root mean squared error between true and predicted values.
    """
    mse_ = mse(y_pred, y_true)
    return jnp.sqrt(mse_)


def mae(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the mean absolute error between true and predicted values.
    """
    mae = jnp.mean(jnp.abs(y_true - y_pred))
    return mae


def nlpd(y: jnp.ndarray, mu: jnp.ndarray, sigma_squared: jnp.ndarray,
         eps: float = 1e-6) -> jnp.ndarray:
    """
    Computes the Negative Log Predictive Density (NLPD) for observed data points
    given the predictive mean and variance.

    Parameters:
        y (np.array): Array of observed values
        mu (np.array): Array of predictive means from the model
        sigma_squared (np.array): Array of predictive variances from the model

    Returns:
        The NLPD value
    """
    sigma_squared += eps
    # Constants for the normal distribution's probability density function
    const = -0.5 * jnp.log(2 * jnp.pi * sigma_squared)
    # The squared differences divided by the variance
    diff_squared = (y - mu) ** 2
    probability_density = -0.5 * diff_squared / sigma_squared

    # Log probability is the sum of the constant and probability density components
    log_prob = const + probability_density

    # Compute the NLPD by averaging and negating the log probabilities
    nlpd = -jnp.mean(log_prob)

    return nlpd


def get_flax_compatible_dict(params_numpyro: Dict[str, jnp.ndarray]) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Takes a dictionary with MCMC samples produced by numpyro
    and creates a dictionary with weights and biases compatible
    with flax .apply() method.
    """
    params_all = {}
    module_params = {} 
    
    # First pass: organize parameters by module
    for key, val in params_numpyro.items():
        if key.startswith('nn'):
            module, layer, param = key.split('/')[-1].split('.')
            if module not in module_params:
                module_params[module] = {'weights': {}, 'biases': {}}
            if param == 'bias':
                module_params[module]['biases'][layer] = val
            else:
                module_params[module]['weights'][layer] = val
        else:
            params_all[key] = val
    
    # Second pass: combine weights and biases for each module
    for module, params in module_params.items():
        params_all[module] = {}
        for (layer, w), (_, b) in zip(params['weights'].items(), params['biases'].items()):
            params_all[module][layer] = {
                "kernel": w,
                "bias": b
            }
            
    return params_all


def get_prior_means_from_samples(params_numpyro: Dict[str, jnp.ndarray]) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Takes a dictionary with MCMC samples produced by numpyro
    and creates a dictionary with mean of weights and biases
    that can be used to set priors in BNNs
    """
    params_all = {}
    weights, biases = {}, {}
    for key, val in params_numpyro.items():
        if key.startswith('nn'):
            layer, param = key.split('/')[-1].split('.')
            if param == 'bias':
                biases[layer] = val.mean(0)
            else:
                weights[layer] = val.mean(0)
    for (k, v1), (_, v2) in zip(weights.items(), biases.items()):
        params_all[k] = {"kernel": v1, "bias": v2}
    return params_all


def get_init_vals_dict(nn_params):
    if jax.config.x64_enabled:
        nn_params = jax.tree_map(promote_to_x64, nn_params)
    nn_params_ref = {}
    for k, layer in nn_params.items():
        for param_name, param_vals in layer.items():
            if param_name == 'bias':
                nn_params_ref['nn/' + k + '.bias'] = param_vals
            else:
                nn_params_ref['nn/' + k + '.kernel'] = param_vals
    return nn_params_ref


def promote_to_x64(x):
    return x.astype(jnp.float64)


def calculate_sigma(X):
    if X.ndim == 1:
        X = X[:, None]
    n_features = X.shape[1]
    avg_squared_norm = np.mean(np.sum(X**2, axis=1))
    sigma = np.sqrt(avg_squared_norm / n_features)
    return sigma


def flatten_params_dict(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively flatten a nested parameter dictionary into a flat dictionary
    where each key maps to a parameter dictionary with 'kernel' and 'bias'.
    """
    flattened = {}
    for module_dict in params_dict.values():
        for key, value in module_dict.items():
            if isinstance(value, dict) and 'kernel' in value and 'bias' in value:
                # Found a parameter dictionary
                flattened[key] = value
            elif isinstance(value, dict):
                # Keep searching deeper
                flattened.update(flatten_params_dict({key: value}))
    return flattened


def set_fn(func: Callable) -> Callable:
    """
    Transforms a given deterministic function to use a params dictionary
    for its parameters, excluding the first one (assumed to be the dependent variable).

    Args:
    - func (Callable): The deterministic function to be transformed.

    Returns:
    - Callable: The transformed function where parameters are accessed
                from a `params` dictionary.
    """
    # Extract parameter names excluding the first one (assumed to be the dependent variable)
    params_names = list(inspect.signature(func).parameters.keys())[1:]

    # Create the transformed function definition
    transformed_code = f"def {func.__name__}(x, params):\n"

    # Retrieve the source code of the function and indent it to be a valid function body
    source = inspect.getsource(func).split("\n", 1)[1]
    source = "    " + source.replace("\n", "\n    ")

    # Replace each parameter name with its dictionary lookup using regex
    for name in params_names:
        source = re.sub(rf'\b{name}\b', f'params["{name}"]', source)

    # Combine to get the full source
    transformed_code += source

    # Define the transformed function in the local namespace
    local_namespace = {}
    exec(transformed_code, globals(), local_namespace)

    # Return the transformed function
    return local_namespace[func.__name__]


def plot_rhats(samples):
    sgr = numpyro.diagnostics.split_gelman_rubin
    rhats = [sgr(v).flatten() for (k, v) in samples.items() if k.endswith('kernel')]
    rhats = np.concatenate(rhats)
    plt.hist(rhats, bins=20, color='green', alpha=0.6)
    plt.xlabel('r_hat', fontsize=14)
    plt.ylabel('Count', fontsize=14)