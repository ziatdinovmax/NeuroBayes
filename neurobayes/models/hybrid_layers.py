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
                          prob_channels: List[Tuple[int, int]],  # [(in_c, out_c), ...]
                          priors_sigma: float,
                          layer_name: str,
                          activation: Callable = None,
                          input_dim: int = None,
                          ) -> jnp.ndarray:
    """
    Implements a partially Bayesian convolutional layer where specific input-output
    channel pairs can be specified as probabilistic.
    """
    if prob_channels is not None:
        # Initialize with pretrained values
        kernel = pretrained_kernel
        bias = pretrained_bias
        
        # Create array indices for probabilistic weights
        in_c_idx = jnp.array([ic for ic, _ in prob_channels])
        out_c_idx = jnp.array([oc for _, oc in prob_channels])
        
        # For each kernel spatial location, sample the specified channel pairs
        kernel_shape = kernel.shape
        for h in range(kernel_shape[0]):
            for w in range(kernel_shape[1]):
                # Get current values for probabilistic weights at this spatial location
                prob_weight_values = kernel[h, w, in_c_idx, out_c_idx]
                
                # Sample new values for weights
                new_prob_weights = numpyro.sample(
                    f"{layer_name}_kernel_prob_{h}_{w}",
                    dist.Normal(prob_weight_values, priors_sigma)
                )
                
                # Update only the probabilistic weights
                kernel = kernel.at[h, w, in_c_idx, out_c_idx].set(new_prob_weights)
        
        # Sample new bias values for outputs
        new_prob_biases = numpyro.sample(
            f"{layer_name}_bias_prob",
            dist.Normal(pretrained_bias[out_c_idx], priors_sigma)
        )

        # Update the biases
        bias = bias.at[out_c_idx].set(new_prob_biases)
        
    else:
        # Full Bayesian layer
        kernel = numpyro.sample(
            f"{layer_name}_kernel",
            dist.Normal(pretrained_kernel, priors_sigma).to_event(4)
        )
        bias = numpyro.sample(
            f"{layer_name}_bias",
            dist.Normal(pretrained_bias, priors_sigma).to_event(1)
        )
    
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


def partial_bayesian_attention(x, pretrained_params, prob_neurons, priors_sigma, 
                               num_heads, dropout_rate, layer_name, enable_dropout=False):
    """
    A partial Bayesian attention layer that updates only selected weights within the 
    attention module's subcomponents (query, key, value, out).

    Args:
        x: Input tensor of shape (batch, seq_length, input_dim).
        pretrained_params: Dictionary of pretrained parameters for the attention layer.
                           Expected keys: 'query', 'key', 'value', 'out', each a dict with
                           keys 'kernel' and 'bias'.
        prob_neurons: A dict mapping subcomponent names (e.g. "query", "key", "value", "out")
                      to a list of (input_idx, output_idx) tuples indicating which connections 
                      to treat as Bayesian.
        priors_sigma: Standard deviation for the Normal priors.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate to apply on attention weights (if enabled).
        layer_name: Base name used for naming the numpyro samples.
        enable_dropout: If True, apply dropout (you can customize dropout sampling as needed).
        
    Returns:
        The attention output tensor.
    """
    # Get the total feature dimension (qkv_features) from the query kernel shape.
    qkv_features = pretrained_params['query']['kernel'].shape[1]
    head_dim = qkv_features // num_heads

    def partial_dense(x, sublayer, sample_name):
        """
        Performs a dense operation using the pretrained weights from a given subcomponent.
        If probabilistic indices are provided for that subcomponent, only those weight entries 
        (and corresponding biases) are updated via sampling.
        """
        kernel = pretrained_params[sublayer]['kernel']
        bias = pretrained_params[sublayer]['bias']
        sub_prob = prob_neurons.get(sublayer, None)
        if sub_prob is not None:
            # Extract indices from the provided list of (input_idx, output_idx) tuples.
            input_idx = jnp.array([i for i, _ in sub_prob])
            output_idx = jnp.array([j for _, j in sub_prob])
            # Get the current values for these connections.
            prob_weight_values = kernel[input_idx, output_idx]
            new_prob_weights = numpyro.sample(
                f"{layer_name}_{sample_name}_kernel_prob",
                dist.Normal(prob_weight_values, priors_sigma)
            )
            kernel = kernel.at[input_idx, output_idx].set(new_prob_weights)
            # For the bias, update only the entries corresponding to the selected output indices.
            prob_bias_values = bias[output_idx]
            new_prob_bias = numpyro.sample(
                f"{layer_name}_{sample_name}_bias_prob",
                dist.Normal(prob_bias_values, priors_sigma)
            )
            bias = bias.at[output_idx].set(new_prob_bias)
        # Compute the dense operation.
        return jnp.dot(x, kernel) + bias

    # Compute the projections.
    q = partial_dense(x, 'query', 'query')
    k = partial_dense(x, 'key', 'key')
    v = partial_dense(x, 'value', 'value')

    # Reshape the projections to multi-head format.
    def reshape_to_heads(t):
        batch, seq_len, _ = t.shape
        # Reshape to (batch, seq_len, num_heads, head_dim) and then transpose to (batch, num_heads, seq_len, head_dim)
        return t.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    q_heads = reshape_to_heads(q)
    k_heads = reshape_to_heads(k)
    v_heads = reshape_to_heads(v)

    # Compute scaled dot–product attention.
    scale = 1.0 / jnp.sqrt(head_dim)
    attn_logits = jnp.einsum("bhqd, bhkd -> bhqk", q_heads, k_heads) * scale
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)
    
    if enable_dropout and dropout_rate > 0.0:
        # (Optionally) apply dropout on attention weights.
        # You can further customize dropout here with a proper random key.
        dropout_mask = numpyro.sample(
            f"{layer_name}_attn_dropout_mask",
            dist.Bernoulli(probs=1 - dropout_rate).to_event(attn_weights.ndim)
        )
        attn_weights = attn_weights * dropout_mask / (1 - dropout_rate)
    
    # Compute the weighted sum of the values.
    attn_output_heads = jnp.einsum("bhqk, bhkd -> bhqd", attn_weights, v_heads)
    # Combine the heads.
    attn_output = attn_output_heads.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], qkv_features)

    # Apply the output projection.
    out = partial_dense(attn_output, 'out', 'out')

    return out
