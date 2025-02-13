import jax.numpy as jnp
import jax
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

def select_bayesian_weights(
    weights: jnp.ndarray,
    method: Union[str, SelectionMethod],
    layer_type: str,
    num_pairs: int = 100,
    loss_fn: Optional[callable] = None,
    threshold: Optional[float] = None,
    n_clusters: int = 10,
    activation_stats: Optional[Dict] = None,
    token_frequencies: Optional[jnp.ndarray] = None,
) -> List[Tuple[int, int]]:
    """
    Select weight pairs for Bayesian treatment from either embedding or dense layers.
    For gradient method, loss_fn should directly return the gradient values.
    """
    if isinstance(method, str):
        method = SelectionMethod(method.lower())
    
    if method == SelectionMethod.MAGNITUDE:
        return _select_by_magnitude(weights, num_pairs, threshold)
    elif method == SelectionMethod.GRADIENT:
        if loss_fn is None:
            raise ValueError("loss_fn must be provided for gradient-based selection")
        # Now loss_fn returns gradients directly
        grads = loss_fn(weights)  # No need for jax.grad anymore
        return _select_by_gradient_direct(grads, num_pairs)
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
    Uses model's stored gradients when gradient-based selection is chosen.
    """
    # Convert single layer name to list
    if isinstance(layer_names, str):
        layer_names = [layer_names]
    
    # Determine model type and get appropriate flatten function
    flatten_fn = flatten_transformer_params_dict if isinstance(model.model, FlaxTransformer) else flatten_params_dict
    
    # Get flattened parameters
    params = flatten_fn(model.get_params())
    
    # Get flattened gradients if needed
    if method == 'gradient':
        if not hasattr(model, 'grad_history') or not model.grad_history:
            raise ValueError("No gradient history available. Run training with gradient collection.")
        grads = flatten_fn(model.average_grads)
    
    bayesian_components = {}
    for layer_name in layer_names:
        if layer_name not in params:
            raise ValueError(f"Layer {layer_name} not found in model parameters")
        
        # Determine layer type and get weights
        is_embedding = 'embed' in layer_name.lower()
        param_key = 'embedding' if is_embedding else 'kernel'
        layer_type = 'embedding' if is_embedding else 'dense'
        
        weights = params[layer_name][param_key]
        
        # Get layer-specific parameters
        layer_params = {
            'weights': weights,
            'method': method,
            'layer_type': layer_type,
            'num_pairs': num_pairs_per_layer,
            'threshold': threshold,
            'n_clusters': n_clusters
        }
        
        # Add method-specific parameters
        if method == 'gradient':
            layer_grads = grads[layer_name][param_key]
            layer_params['loss_fn'] = lambda _: layer_grads
            
        if is_embedding and param_info and 'token_frequencies' in param_info:
            layer_params['token_frequencies'] = param_info['token_frequencies'].get(layer_name)
            
        # Select pairs for this layer
        try:
            pairs = select_bayesian_weights(**layer_params)
            bayesian_components[layer_name] = pairs
        except Exception as e:
            print(f"Warning: Failed to process layer {layer_name}: {str(e)}")
            continue
            
    return bayesian_components