from typing import List, Callable, Union, Tuple, Optional

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp

from ..flax_nets import get_conv_and_pool_ops


def partial_bayesian_dense(x, pretrained_kernel, pretrained_bias, prob_neurons, 
                          priors_sigma, layer_name, activation=None):
    """
    A dense layer function where only specified neurons are Bayesian.
    """
    if prob_neurons is not None:
        # Convert to array for indexing
        prob_neurons = jnp.array(prob_neurons)
        
        # Start with pretrained parameters
        kernel = pretrained_kernel
        bias = pretrained_bias
        
        # Sample parameters for the Bayesian neurons
        kernel_prob = numpyro.sample(
            f"{layer_name}_kernel_prob",
            dist.Normal(kernel[:, prob_neurons], priors_sigma).to_event(2)
        )
        bias_prob = numpyro.sample(
            f"{layer_name}_bias_prob",
            dist.Normal(bias[prob_neurons], priors_sigma).to_event(1)
        )
        
        # Update the parameters
        kernel = kernel.at[:, prob_neurons].set(kernel_prob)
        bias = bias.at[prob_neurons].set(bias_prob)
    else:
        # Full Bayesian layer
        kernel = numpyro.sample(
            f"{layer_name}_kernel",
            dist.Normal(pretrained_kernel, priors_sigma).to_event(2)
        )
        bias = numpyro.sample(
            f"{layer_name}_bias",
            dist.Normal(pretrained_bias, priors_sigma).to_event(1)
        )
    
    # Compute output
    y = jnp.dot(x, kernel) + bias
    if activation is not None:
        y = activation(y)
    
    return y


def partial_bayesian_embed(x, pretrained_embedding, prob_indices, 
                          priors_sigma, layer_name, dtype=None):
    """
    A partial Bayesian embedding layer that matches Flax's Embed functionality.
    """
    # Type checking
    if not jnp.issubdtype(x.dtype, jnp.integer):
        raise ValueError('Input type must be an integer or unsigned integer.')
    
    num_embeddings, features = pretrained_embedding.shape
    
    # Handle special case when num_embeddings = 1
    if num_embeddings == 1:
        if prob_indices is not None and len(prob_indices) > 0 and 0 in prob_indices:
            # Sample the single embedding vector
            embedding_matrix = numpyro.sample(
                f"{layer_name}_embedding",
                dist.Normal(pretrained_embedding, priors_sigma).to_event(2)
            )
        else:
            embedding_matrix = pretrained_embedding
            
        # Match Flax's behavior for num_embeddings = 1
        return jnp.where(
            jnp.broadcast_to(x[..., None], x.shape + (features,)) == 0,
            embedding_matrix,
            jnp.nan
        )
    
    # Regular case: num_embeddings > 1
    # Start with pretrained embedding matrix
    embedding_matrix = pretrained_embedding
    
    if prob_indices is not None and len(prob_indices) > 0:
        # Convert indices to array for proper indexing
        prob_indices = jnp.array(prob_indices)
        
        # Sample parameters only for the Bayesian embedding vectors
        embeddings_prob = numpyro.sample(
            f"{layer_name}_embedding_prob",
            dist.Normal(embedding_matrix[prob_indices], priors_sigma).to_event(2)
        )
        
        # Update only the probabilistic embeddings
        embedding_matrix = embedding_matrix.at[prob_indices].set(embeddings_prob)
    
    # Use take for proper indexing behavior (matching Flax)
    return jnp.take(embedding_matrix, x, axis=0)


def partial_bayesian_conv(x: jnp.ndarray,
                         pretrained_kernel: jnp.ndarray,
                         pretrained_bias: jnp.ndarray,
                         prob_channels: List[int],
                         priors_sigma: float,
                         layer_name: str,
                         activation: Callable = None,
                         input_dim: int = None,
                         kernel_size: Union[int, Tuple[int, ...]] = 3) -> jnp.ndarray:
    """
    Implements a partially Bayesian convolutional layer where only specified channels
    are treated as probabilistic.
    """
    # Get total number of output channels
    n_channels = pretrained_kernel.shape[-1]
    
    # Convert prob_channels to array for easier handling
    prob_channels = jnp.array(prob_channels)
    
    # Sample probabilistic parameters
    kernel_prob = numpyro.sample(
        f"{layer_name}.kernel_prob",
        dist.Normal(pretrained_kernel[..., prob_channels], priors_sigma)
    )
    bias_prob = numpyro.sample(
        f"{layer_name}.bias_prob",
        dist.Normal(pretrained_bias[prob_channels], priors_sigma)
    )
    
    # Initialize output arrays with deterministic values
    kernel = pretrained_kernel
    bias = pretrained_bias
    
    # Update probabilistic channels using scatter operations
    kernel = kernel.at[..., prob_channels].set(kernel_prob)
    bias = bias.at[prob_channels].set(bias_prob)
    
    # Apply convolution using JAX operations
    if input_dim == 1:
        padding = [(kernel_size // 2, kernel_size // 2)]
        x = jax.lax.conv_general_dilated(
            x[..., None],  # Add channel dim
            kernel[..., None, :],  # Reshape kernel for 1D conv
            window_strides=(1,),
            padding=padding,
            dimension_numbers=('NWHC', 'WHIO', 'NWHC')
        )
    
    elif input_dim == 2:
        padding = [(kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)]
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding=padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC')
        )
    
    # Add bias
    x = x + bias
    
    # Apply activation if provided
    if activation is not None:
        x = activation(x)
    
    # Apply max pooling
    if input_dim == 1:
        x = jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, 2, 1),
            window_strides=(1, 2, 1),
            padding='VALID'
        )
    elif input_dim == 2:
        x = jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, 2, 2, 1),
            window_strides=(1, 2, 2, 1),
            padding='VALID'
        )
    
    return x