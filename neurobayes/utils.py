from typing import List, Dict

import jax
import jax.numpy as jnp

import numpy as np


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
