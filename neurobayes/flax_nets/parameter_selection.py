import jax.numpy as jnp
from sklearn.cluster import KMeans
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from .transformer import FlaxTransformer
from ..utils import flatten_params_dict, flatten_transformer_params_dict


class SelectionMethod(Enum):
    MAGNITUDE = "magnitude"
    GRADIENT = "gradient"
    CLUSTERING = "clustering"
    VARIANCE = "variance"

def select_bayesian_weights(
    weights: jnp.ndarray,
    method: Union[str, SelectionMethod],
    layer_type: str,
    num_pairs: int = 100,
    grads: Optional[jnp.ndarray] = None,
    param_history: Optional[jnp.ndarray] = None,
    threshold: Optional[float] = None,
    n_clusters: int = 10,
    token_frequencies: Optional[jnp.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    Select neuron pairs (weights) for Bayesian treatment from either embedding or dense layers.
    """
    if isinstance(method, str):
        method = SelectionMethod(method.lower())
    
    if method == SelectionMethod.MAGNITUDE:
        return _select_by_magnitude(weights, num_pairs, threshold)
    elif method == SelectionMethod.GRADIENT:
        if grads is None:
            raise ValueError("grads must be provided for gradient-based selection")
        return _select_by_gradient_direct(grads, num_pairs)
    elif method == SelectionMethod.VARIANCE:
        if param_history is None:
            raise ValueError("param_history must be provided for variance-based selection")
        return _select_by_variance(param_history, num_pairs)
    elif method == SelectionMethod.CLUSTERING:
        return _select_by_clustering(
            weights, 
            n_clusters, 
            num_pairs_per_cluster=num_pairs // n_clusters,
            layer_type=layer_type,
            token_frequencies=token_frequencies if layer_type == 'embedding' else None
        )
    else:
        raise ValueError(f"Unknown method: {method}")

def _select_by_magnitude(
    weights: jnp.ndarray,
    num_pairs: int,
    threshold: Optional[float] = None
) -> List[Tuple[int, int]]:
    """Select pairs based on weight magnitude."""
    magnitudes = jnp.abs(weights)
    
    if threshold is not None:
        indices = jnp.where(magnitudes > threshold)
        pairs = list(zip(indices[0].tolist(), indices[1].tolist()))
        if len(pairs) > num_pairs:
            magnitudes_flat = magnitudes.ravel()
            selected_indices = np.argsort(magnitudes_flat)[-num_pairs:]
            rows, cols = np.unravel_index(selected_indices, weights.shape)
            pairs = list(zip(rows.tolist(), cols.tolist()))
    else:
        magnitudes_flat = magnitudes.ravel()
        selected_indices = np.argsort(magnitudes_flat)[-num_pairs:]
        rows, cols = np.unravel_index(selected_indices, weights.shape)
        pairs = list(zip(rows.tolist(), cols.tolist()))
    
    return pairs

def _select_by_variance(
    weight_history: jnp.ndarray,
    num_pairs: int
) -> List[Tuple[int, int]]:
    """Select pairs based on weight variance across training iterations."""
    # Compute variance across the history dimension
    variances = jnp.var(weight_history, axis=0)
    
    # Select top-k by variance magnitude
    var_flat = variances.ravel()
    selected_indices = np.argsort(var_flat)[-num_pairs:]
    rows, cols = np.unravel_index(selected_indices, variances.shape)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    
    return pairs

def _select_by_gradient_direct(
    grads: jnp.ndarray,
    num_pairs: int
) -> List[Tuple[int, int]]:
    """Select pairs based on gradient magnitude using pre-computed gradients."""
    grad_magnitudes = jnp.abs(grads)
    
    # Select top-k by gradient magnitude
    grad_flat = grad_magnitudes.ravel()
    selected_indices = np.argsort(grad_flat)[-num_pairs:]
    rows, cols = np.unravel_index(selected_indices, grads.shape)
    pairs = list(zip(rows.tolist(), cols.tolist()))
    
    return pairs

def _select_by_clustering(
    weights: jnp.ndarray,
    n_clusters: int,
    num_pairs_per_cluster: int,
    layer_type: str,
    token_frequencies: Optional[jnp.ndarray] = None
) -> List[Tuple[int, int]]:
    """Select pairs using clustering approach."""
    weights_np = np.array(weights)
    
    if layer_type == 'embedding':
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(weights_np)
        
        pairs = []
        for cluster_idx in range(n_clusters):
            mask = clusters == cluster_idx
            cluster_emb_indices = np.where(mask)[0]
            
            if token_frequencies is not None and len(cluster_emb_indices) > 0:
                cluster_frequencies = token_frequencies[cluster_emb_indices]
                cluster_probs = cluster_frequencies / cluster_frequencies.sum()
                selected_embs = np.random.choice(
                    cluster_emb_indices,
                    size=min(num_pairs_per_cluster, len(cluster_emb_indices)),
                    p=cluster_probs
                )
            else:
                selected_embs = np.random.choice(
                    cluster_emb_indices,
                    size=min(num_pairs_per_cluster, len(cluster_emb_indices))
                )
            
            selected_feats = np.random.randint(
                0, weights.shape[1],
                size=len(selected_embs)
            )
            
            pairs.extend(zip(selected_embs.tolist(), selected_feats.tolist()))
    else:
        kmeans_in = KMeans(n_clusters=n_clusters)
        clusters_in = kmeans_in.fit_predict(weights_np)
        
        pairs = []
        pairs_per_input_cluster = num_pairs_per_cluster // 2
        
        for cluster_idx in range(n_clusters):
            mask = clusters_in == cluster_idx
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_indices) > 0:
                selected_inputs = np.random.choice(
                    cluster_indices,
                    size=min(pairs_per_input_cluster, len(cluster_indices))
                )
                selected_outputs = np.random.randint(
                    0, weights.shape[1],
                    size=len(selected_inputs)
                )
                
                pairs.extend(zip(selected_inputs.tolist(), selected_outputs.tolist()))
        
        kmeans_out = KMeans(n_clusters=n_clusters)
        clusters_out = kmeans_out.fit_predict(weights_np.T)
        
        pairs_per_output_cluster = num_pairs_per_cluster - pairs_per_input_cluster
        for cluster_idx in range(n_clusters):
            mask = clusters_out == cluster_idx
            cluster_indices = np.where(mask)[0]
            
            if len(cluster_indices) > 0:
                selected_outputs = np.random.choice(
                    cluster_indices,
                    size=min(pairs_per_output_cluster, len(cluster_indices))
                )
                selected_inputs = np.random.randint(
                    0, weights.shape[0],
                    size=len(selected_outputs)
                )
                
                pairs.extend(zip(selected_inputs.tolist(), selected_outputs.tolist()))
    
    return pairs


def select_bayesian_weights_conv(
    weights: jnp.ndarray,
    method: Union[str, SelectionMethod],
    layer_type: str,  # Added for consistency
    num_pairs: int = 100,
    grads: Optional[jnp.ndarray] = None,
    param_history: Optional[jnp.ndarray] = None,
    threshold: Optional[float] = None,
    n_clusters: int = 10,
    token_frequencies: Optional[jnp.ndarray] = None,  # Added for consistency
) -> List[Tuple[int, int]]:
    """
    Select input-output channel pairs for Bayesian treatment from convolutional layers.
    
    Args:
        weights: Array of shape:
            - 3D (kernel_size, in_channels, out_channels) for 1D convolution
            - 4D (height, width, in_channels, out_channels) for 2D convolution
        method: Selection method (magnitude, gradient, clustering, variance)
        layer_type: Type of layer ('conv1d' or 'conv2d')
        num_pairs: Number of channel pairs to select
        grads: Optional gradients for gradient-based selection
        param_history: Optional parameter history for variance-based selection
        threshold: Optional magnitude threshold
        n_clusters: Number of clusters for clustering-based selection
        token_frequencies: Not used for conv layers, added for API consistency
    """
    if isinstance(method, str):
        method = SelectionMethod(method.lower())
    
    is_1d = layer_type == 'conv1d'
    spatial_axes = (0,) if is_1d else (0, 1)
    
    if method == SelectionMethod.MAGNITUDE:
        channel_magnitudes = jnp.mean(jnp.abs(weights), axis=spatial_axes)
        return _select_by_magnitude_conv(channel_magnitudes, num_pairs, threshold)
    
    elif method == SelectionMethod.GRADIENT:
        if grads is None:
            raise ValueError("grads must be provided for gradient-based selection")
        channel_grads = jnp.mean(jnp.abs(grads), axis=spatial_axes)
        return _select_by_gradient_conv(channel_grads, num_pairs)
    
    elif method == SelectionMethod.VARIANCE:
        if param_history is None:
            raise ValueError("param_history must be provided for variance-based selection")
        channel_variances = jnp.mean(jnp.var(param_history, axis=0), axis=spatial_axes)
        return _select_by_variance_conv(channel_variances, num_pairs)
    
    elif method == SelectionMethod.CLUSTERING:
        if is_1d:
            kernel_size, in_c, out_c = weights.shape
            weights_in = weights.transpose(1, 0, 2).reshape(in_c, -1)
            weights_out = weights.transpose(2, 0, 1).reshape(out_c, -1)
        else:
            h, w, in_c, out_c = weights.shape
            weights_in = weights.transpose(2, 0, 1, 3).reshape(in_c, -1)
            weights_out = weights.transpose(3, 0, 1, 2).reshape(out_c, -1)
        
        return _select_by_clustering_conv(
            weights_in=weights_in,
            weights_out=weights_out,
            n_clusters=n_clusters,
            num_pairs_per_cluster=num_pairs // n_clusters,
            layer_type=layer_type  # Pass through layer_type
        )
    else:
        raise ValueError(f"Unknown method: {method}")

def _select_by_clustering_conv(
    weights_in: jnp.ndarray,
    weights_out: jnp.ndarray,
    n_clusters: int,
    num_pairs_per_cluster: int,
    layer_type: str  # Added for consistency with dense/embedding
) -> List[Tuple[int, int]]:
    """
    Select channel pairs using clustering approach.
    
    Args:
        weights_in: Reshaped weights (in_channels, -1)
        weights_out: Reshaped weights (out_channels, -1)
        n_clusters: Number of clusters
        num_pairs_per_cluster: Number of pairs to select per cluster
        layer_type: Type of convolution layer ('conv1d' or 'conv2d')
    """
    # Cluster input channels
    kmeans_in = KMeans(n_clusters=n_clusters)
    clusters_in = kmeans_in.fit_predict(weights_in)
    
    # Cluster output channels
    kmeans_out = KMeans(n_clusters=n_clusters)
    clusters_out = kmeans_out.fit_predict(weights_out)
    
    pairs = []
    pairs_per_input_cluster = num_pairs_per_cluster // 2
    
    # Select from input channel clusters
    in_channels = weights_in.shape[0]
    out_channels = weights_out.shape[0]
    
    for cluster_idx in range(n_clusters):
        # Input channel selection
        in_mask = clusters_in == cluster_idx
        in_indices = np.where(in_mask)[0]
        
        if len(in_indices) > 0:
            selected_inputs = np.random.choice(
                in_indices,
                size=min(pairs_per_input_cluster, len(in_indices))
            )
            selected_outputs = np.random.randint(
                0, out_channels,
                size=len(selected_inputs)
            )
            pairs.extend(zip(selected_inputs.tolist(), selected_outputs.tolist()))
        
        # Output channel selection
        out_mask = clusters_out == cluster_idx
        out_indices = np.where(out_mask)[0]
        
        if len(out_indices) > 0:
            selected_outputs = np.random.choice(
                out_indices,
                size=min(pairs_per_input_cluster, len(out_indices))
            )
            selected_inputs = np.random.randint(
                0, in_channels,
                size=len(selected_outputs)
            )
            pairs.extend(zip(selected_inputs.tolist(), selected_outputs.tolist()))
    
    return pairs

def _select_by_magnitude_conv(
    channel_magnitudes: jnp.ndarray,
    num_pairs: int,
    threshold: Optional[float] = None
) -> List[Tuple[int, int]]:
    """Select channel pairs based on aggregated magnitude."""
    if threshold is not None:
        indices = jnp.where(channel_magnitudes > threshold)
        channel_pairs = list(zip(indices[0].tolist(), indices[1].tolist()))
        if len(channel_pairs) > num_pairs:
            magnitudes_flat = channel_magnitudes.ravel()
            selected_indices = np.argsort(magnitudes_flat)[-num_pairs:]
            in_channels, out_channels = np.unravel_index(selected_indices, channel_magnitudes.shape)
            channel_pairs = list(zip(in_channels.tolist(), out_channels.tolist()))
    else:
        magnitudes_flat = channel_magnitudes.ravel()
        selected_indices = np.argsort(magnitudes_flat)[-num_pairs:]
        in_channels, out_channels = np.unravel_index(selected_indices, channel_magnitudes.shape)
        channel_pairs = list(zip(in_channels.tolist(), out_channels.tolist()))
    
    return channel_pairs

def _select_by_variance_conv(
    channel_variances: jnp.ndarray,
    num_pairs: int
) -> List[Tuple[int, int]]:
    """Select channel pairs based on aggregated variance."""
    var_flat = channel_variances.ravel()
    selected_indices = np.argsort(var_flat)[-num_pairs:]
    in_channels, out_channels = np.unravel_index(selected_indices, channel_variances.shape)
    channel_pairs = list(zip(in_channels.tolist(), out_channels.tolist()))
    
    return channel_pairs

def _select_by_gradient_conv(
    channel_grads: jnp.ndarray,
    num_pairs: int
) -> List[Tuple[int, int]]:
    """Select channel pairs based on aggregated gradient magnitude."""
    grad_flat = channel_grads.ravel()
    selected_indices = np.argsort(grad_flat)[-num_pairs:]
    in_channels, out_channels = np.unravel_index(selected_indices, channel_grads.shape)
    channel_pairs = list(zip(in_channels.tolist(), out_channels.tolist()))
    
    return channel_pairs


def select_bayesian_components(
    model,
    layer_names: Union[str, List[str]],
    method: Union[str, SelectionMethod] = 'magnitude',
    num_pairs_per_layer: int = 100,
    threshold: Optional[float] = None,
    n_clusters: int = 10,
    param_info: Optional[Dict] = None
) -> Dict[str, List[tuple]]:
    """
    Select weight pairs for Bayesian treatment from multiple layers.
    Supports dense, embedding, 1D conv, and 2D conv layers.
    Uses model's stored gradients when gradient-based selection is chosen.
    """
    # Convert single layer name to list
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    # Convert method to SelectionMethod if string
    if isinstance(method, str):
        method = SelectionMethod(method.lower())
    
    # Determine model type and get appropriate flatten function
    flatten_fn = flatten_transformer_params_dict if isinstance(model.model, FlaxTransformer) else flatten_params_dict
    
    # Get flattened parameters
    params = flatten_fn(model.get_params())
    
    # Get flattened gradients if needed
    if method == SelectionMethod.GRADIENT:
        if not hasattr(model, 'grad_history') or not model.grad_history:
            raise ValueError("No gradient history available. Run training with gradient collection.")
        grads = flatten_fn(model.average_grads)
    
    # Get parameter history if needed
    if method == SelectionMethod.VARIANCE:
        if not hasattr(model, 'params_history') or not model.params_history:
            raise ValueError("No parameter history available for variance-based selection.")
    
    bayesian_components = {}
    for layer_name in layer_names:
        if layer_name not in params:
            raise ValueError(f"Layer {layer_name} not found in model parameters")
        
        # Determine layer type and get weights
        is_embedding = 'embed' in layer_name.lower()
        param_key = 'embedding' if is_embedding else 'kernel'
        
        weights = params[layer_name][param_key]
        weight_dims = len(weights.shape)
        
        # Determine layer type based on weight dimensions
        if weight_dims == 2:
            layer_type = 'embedding' if is_embedding else 'dense'
        elif weight_dims in [3, 4]:
            layer_type = 'conv1d' if weight_dims == 3 else 'conv2d'
        else:
            raise ValueError(f"Unexpected weight dimensions: {weight_dims}")
        
        # Prepare layer-specific parameters
        layer_params = {
            'weights': weights,
            'method': method,
            'layer_type': layer_type,
            'num_pairs': num_pairs_per_layer,
            'threshold': threshold,
            'n_clusters': n_clusters
        }
        
        # Add method-specific parameters
        if method == SelectionMethod.GRADIENT:
            layer_grads = grads[layer_name][param_key]
            layer_params['grads'] = layer_grads
            
        elif method == SelectionMethod.VARIANCE:
            # Flatten each params dict in the history
            layer_param_history = jnp.stack([
                flatten_fn(params)[layer_name][param_key]
                for params in model.params_history
            ])
            layer_params['param_history'] = layer_param_history
        
        # Add embedding-specific parameters
        if is_embedding and param_info and 'token_frequencies' in param_info:
            layer_params['token_frequencies'] = param_info['token_frequencies'].get(layer_name)
        
        # Select pairs based on layer type
        try:
            if layer_type in ['conv1d', 'conv2d']:
                pairs = select_bayesian_weights_conv(**layer_params)
            else:
                pairs = select_bayesian_weights(**layer_params)
                
            bayesian_components[layer_name] = pairs
            
        except Exception as e:
            print(f"Warning: Failed to process layer {layer_name}: {str(e)}")
            continue
    
    return bayesian_components