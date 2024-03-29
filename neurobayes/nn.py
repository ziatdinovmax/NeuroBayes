from typing import Callable, Dict, List
import jax
import jax.numpy as jnp


def get_mlp(hidden_dim: List[int], activation: str = 'tanh'
            ) -> Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]:
    """Returns a function that represents an MLP for a given hidden_dim"""
    if activation not in ['relu', 'tanh']:
        raise NotImplementedError("Use either 'relu' or 'tanh' for activation")
    activation_fn = jnp.tanh if activation == 'tanh' else jax.nn.relu

    def mlp(X: jnp.ndarray, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """MLP for a single MCMC sample of weights and biases, handling arbitrary number of layers"""
        h = X
        for i in range(len(hidden_dim)):
            h = activation_fn(jnp.matmul(h, params[f"w{i}"]) + params[f"b{i}"])
        # No non-linearity after the last layer
        z = jnp.matmul(h, params[f"w{len(hidden_dim)}"]) + params[f"b{len(hidden_dim)}"]
        return z
    return mlp
