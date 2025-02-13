from typing import List, Callable, Union, Tuple, Optional

import numpyro
import numpyro.distributions as dist
import jax
import jax.numpy as jnp



def partial_bayesian_dense(x, pretrained_kernel, pretrained_bias, prob_neurons, 
                          priors_sigma, layer_name, activation=None):
    """
    A dense layer function where only specified pairs of neurons are Bayesian.
    """
    prob_weights = prob_neurons
    if prob_weights is not None:
        # Initialize with pretrained values
        kernel = pretrained_kernel
        bias = pretrained_bias
        
        # Create array indices for probabilistic weights
        input_idx = jnp.array([i for i, _ in prob_weights])
        output_idx = jnp.array([j for _, j in prob_weights])
        
        # Get current values for probabilistic weights
        prob_weight_values = pretrained_kernel[input_idx, output_idx]
        
        # Sample new values
        new_prob_weights = numpyro.sample(
            f"{layer_name}_kernel_prob",
            dist.Normal(prob_weight_values, priors_sigma)
        )
        
        # Update only the probabilistic weights
        kernel = kernel.at[input_idx, output_idx].set(new_prob_weights)
        
        # Sample new bias values for those outputs
        new_prob_biases = numpyro.sample(
            f"{layer_name}_bias_prob",
            dist.Normal(pretrained_bias[output_idx], priors_sigma)
        )
        
        # Update the biases
        bias = bias.at[output_idx].set(new_prob_biases)
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
                          priors_sigma, layer_name):
    """
    A connection-specific partial Bayesian embedding layer that matches Flax's Embed functionality.
    prob_pairs should be a list of tuples (embedding_idx, feature_idx) specifying which
    specific embedding-feature connections should be Bayesian.
    """

    prob_pairs = prob_indices
    if not jnp.issubdtype(x.dtype, jnp.integer):
        raise ValueError('Input type must be an integer or unsigned integer.')
    
    num_embeddings, features = pretrained_embedding.shape
    
    # Handle special case when num_embeddings = 1
    if num_embeddings == 1:
        if prob_pairs is not None and len(prob_pairs) > 0 and any(idx == 0 for idx, _ in prob_pairs):
            # For num_embeddings = 1, we need to handle the specific feature indices
            embedding_matrix = pretrained_embedding.copy()
            feature_indices = [f_idx for _, f_idx in prob_pairs if f_idx < features]
            
            # Sample only the specified feature connections
            prob_features = numpyro.sample(
                f"{layer_name}_embedding_features",
                dist.Normal(embedding_matrix[0, feature_indices], priors_sigma)
            )
            
            # Update only the probabilistic features
            embedding_matrix = embedding_matrix.at[0, feature_indices].set(prob_features)
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
    
    if prob_pairs is not None and len(prob_pairs) > 0:
        # Convert pairs to arrays for proper indexing
        embedding_idx = jnp.array([i for i, _ in prob_pairs])
        feature_idx = jnp.array([j for _, j in prob_pairs])
        
        # Get current values for probabilistic connections
        prob_values = embedding_matrix[embedding_idx, feature_idx]
        
        # Sample new values for specific connections
        new_prob_values = numpyro.sample(
            f"{layer_name}_embedding_prob",
            dist.Normal(prob_values, priors_sigma)
        )
        
        # Update only the specific probabilistic connections
        embedding_matrix = embedding_matrix.at[embedding_idx, feature_idx].set(new_prob_values)
    
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
        x = jax.lax.conv_general_dilated(
            x[..., None],  # Add channel dim
            kernel[..., None, :],  # Reshape kernel for 1D conv
            window_strides=(1,),
            padding='SAME',
            dimension_numbers=('NWHC', 'WHIO', 'NWHC')
        )
    
    elif input_dim == 2:
        x = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding='SAME',
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
            padding='SAME'
        )
    elif input_dim == 2:
        x = jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, 2, 2, 1),
            window_strides=(1, 2, 2, 1),
            padding='SAME'
        )
    
    return x