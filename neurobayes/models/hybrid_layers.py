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


def partial_bayesian_attention(x, pretrained_params, prob_neurons, priors_sigma, layer_name):
    """
    Implements a partial Bayesian attention layer compatible with Flax's MultiHeadDotProductAttention.

    Args:
        x: Input tensor of shape (batch, seq_len, in_features).
        pretrained_params: Dictionary containing pretrained weights for attention components.
        prob_neurons: Dictionary mapping each subcomponent to probabilistic indices.
        priors_sigma: Standard deviation for the Normal prior.
        layer_name: Base name for the layer.

    Returns:
        The output of the attention layer.
    """

    def partial_bayesian_dense_3d(x, kernel, bias, prob_indices, priors_sigma, sample_name, layer_name):
        """
        Applies a dense projection with a 3D kernel. If prob_indices is provided, then only
        those weights are updated via sampling.
        """
        if prob_indices is not None:
            in_idx = jnp.array([idx[0] for idx in prob_indices])
            head_idx = jnp.array([idx[1] for idx in prob_indices])
            out_idx = jnp.array([idx[2] for idx in prob_indices])
            prob_vals = kernel[in_idx, head_idx, out_idx]
            new_prob_vals = numpyro.sample(
                f"{layer_name}_{sample_name}_kernel_prob",
                dist.Normal(prob_vals, priors_sigma)
            )
            kernel = kernel.at[in_idx, head_idx, out_idx].set(new_prob_vals)
            prob_bias_vals = bias[head_idx, out_idx]
            new_prob_bias = numpyro.sample(
                f"{layer_name}_{sample_name}_bias_prob",
                dist.Normal(prob_bias_vals, priors_sigma)
            )
            bias = bias.at[head_idx, out_idx].set(new_prob_bias)

        return jnp.tensordot(x, kernel, axes=([2], [0])) + bias

    # Compute query, key, and value projections
    q = partial_bayesian_dense_3d(
        x,
        pretrained_params['query']['kernel'],
        pretrained_params['query']['bias'],
        prob_neurons.get('query', None),
        priors_sigma, 'query', layer_name
    )
    k = partial_bayesian_dense_3d(
        x,
        pretrained_params['key']['kernel'],
        pretrained_params['key']['bias'],
        prob_neurons.get('key', None),
        priors_sigma, 'key', layer_name
    )
    v = partial_bayesian_dense_3d(
        x,
        pretrained_params['value']['kernel'],
        pretrained_params['value']['bias'],
        prob_neurons.get('value', None),
        priors_sigma, 'value', layer_name
    )

    # Transpose to shape (batch, num_heads, seq_len, head_dim) for attention computation
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    # Compute scaled dot-product attention
    scale = 1.0 / jnp.sqrt(q.shape[-1])
    attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    # Compute the weighted sum of the values
    attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

    # Reshape attention output to match Flax's structure for the output projection
    # From [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads*head_dim]
    batch_size, num_heads, seq_len, head_dim = attn_output.shape
    attn_output_reshaped = attn_output.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_heads * head_dim)

    # Reshape the output kernel to match Flax's approach
    # For Flax, the output projection combines across both the head and feature dimensions
    out_kernel = pretrained_params['out']['kernel']
    out_kernel_reshaped = out_kernel.reshape(num_heads * head_dim, -1)
    out_bias = pretrained_params['out']['bias']

    # Apply the partial Bayesian treatment to the reshaped output projection
    if prob_neurons.get('out', None) is not None:
        # Convert 3D indices to 2D indices for the reshaped kernel
        out_prob_indices = []
        for idx in prob_neurons['out']:
            # Map (in_idx, head_idx, out_idx) to flattened index
            in_idx, head_idx, out_idx = idx
            flattened_in_idx = head_idx * head_dim + out_idx
            out_prob_indices.append((flattened_in_idx, out_idx))

        # Apply Bayesian treatment to selected weights in flattened kernel
        in_idx = jnp.array([idx[0] for idx in out_prob_indices])
        out_idx = jnp.array([idx[1] for idx in out_prob_indices])
        prob_vals = out_kernel_reshaped[in_idx, out_idx]
        new_prob_vals = numpyro.sample(
            f"{layer_name}_out_kernel_prob",
            dist.Normal(prob_vals, priors_sigma)
        )
        out_kernel_reshaped = out_kernel_reshaped.at[in_idx, out_idx].set(new_prob_vals)

        prob_bias_vals = out_bias[out_idx]
        new_prob_bias = numpyro.sample(
            f"{layer_name}_out_bias_prob",
            dist.Normal(prob_bias_vals, priors_sigma)
        )
        out_bias = out_bias.at[out_idx].set(new_prob_bias)

    # Apply the output projection
    out = jnp.matmul(attn_output_reshaped, out_kernel_reshaped) + out_bias

    return out
