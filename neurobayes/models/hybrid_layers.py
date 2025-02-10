
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp


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