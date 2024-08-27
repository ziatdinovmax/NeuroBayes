from typing import List, Dict

import jax
import jax.numpy as jnp

import numpy as np

from .nn import FlaxMLP, FlaxMLP2Head, EmbeddingBackbone


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


def mse(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the mean squared error between true and predicted values.
    """
    # Compute the mean squared error
    mse = jnp.mean((y_true - y_pred) ** 2)
    return mse


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


def split_mlp(model, params, out_dim: int = None):
    """
    Splits MLP and its weights into two sub-networks: one with last two layers
    (last hidden layer + output layer) removed and another one consisting only of those two layers.
    """
    out_dim = out_dim if out_dim is not None else model.output_dim  # there will be a mismatch in last_layer_params if out_dim != model.output_dim

    truncated_mlp = FlaxMLP(model.hidden_dims[:-1], output_dim=0)
    last_layer_mlp = FlaxMLP(model.hidden_dims[-1:], output_dim=out_dim)

    truncated_params = {}
    last_layer_params = {}
    for i, (key, val) in enumerate(params.items()):
        if i < len(model.hidden_dims) - 1:
            truncated_params[key] = val
        else:
            new_key = f"Dense{i - len(model.hidden_dims[:-1])}"
            last_layer_params[new_key] = val

    return truncated_mlp, truncated_params, last_layer_mlp, last_layer_params


def split_mlp2head(model, params, out_dim: int = None):
    """
    Splits MLP2Head and its weights into two sub-networks: one with last hidden layer
    and output heads removed, and another consisting only of the last hidden layer and output heads.
    """
    out_dim = out_dim if out_dim is not None else model.output_dim

    truncated_mlp = FlaxMLP2Head(model.hidden_dims[:-1], output_dim=0)
    last_layer_mlp = FlaxMLP2Head(model.hidden_dims[-1:], output_dim=out_dim)

    truncated_params = {}
    last_layer_params = {}
    for key, val in params.items():
        if key.startswith('Dense') and int(key[5:]) < len(model.hidden_dims) - 1:
            truncated_params[key] = val
        else:
            if key.startswith('Dense'):
                new_key = f"Dense0"
            else:  # MeanHead or VarianceHead
                new_key = key
            last_layer_params[new_key] = val

    return truncated_mlp, truncated_params, last_layer_mlp, last_layer_params


def split_multitask_model(trained_model, params):
    # Extract the backbone and head components
    embedding_backbone = EmbeddingBackbone(
        backbone_dims=trained_model.backbone_dims,
        num_tasks=trained_model.num_tasks,
        embedding_dim=trained_model.embedding_dim,
        activation=trained_model.activation
    )
    head = FlaxMLP(
        hidden_dims=trained_model.head_dims,
        output_dim=trained_model.output_dim,
        activation=trained_model.activation
    )

    # Extract relevant parameters
    embedding_backbone_params = params['backbone']
    head_params = params['head']

    return embedding_backbone, embedding_backbone_params, head, head_params


def get_flax_compatible_dict(params_numpyro: Dict[str, jnp.ndarray]) -> Dict[str, Dict[str, jnp.ndarray]]:
    """
    Takes a dictionary with MCMC samples produced by numpyro
    and creates a dictionary with weights and biases compatible
    with flax .apply() method
    """
    params_all = {}
    weights, biases = {}, {}
    for key, val in params_numpyro.items():
        if key.startswith('nn'):
            layer, param = key.split('/')[-1].split('.')
            if param == 'bias':
                biases[layer] = val
            else:
                weights[layer] = val
        else:
            params_all[key] = val
    for (k, v1), (_, v2) in zip(weights.items(), biases.items()):
        params_all[k] = {"kernel": v1, "bias": v2}
    return params_all
