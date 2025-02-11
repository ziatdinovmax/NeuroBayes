from typing import Dict, List
from functools import singledispatch
import jax
import jax.numpy as jnp
import numpy as np

from .convnet import FlaxConvNet, FlaxConvNet2Head
from .mlp import FlaxMLP, FlaxMLP2Head
from .transformer import FlaxTransformer
from .configs import extract_mlp_configs, extract_convnet_configs, extract_mlp2head_configs, extract_convnet2head_configs, extract_transformer_configs
from ..utils import flatten_params_dict, flatten_transformer_params_dict

@singledispatch
def extract_configs(net, probabilistic_layers: List[str] = None, 
                   num_probabilistic_layers: int = None) -> List[Dict]:
    """Generic config extractor dispatch function"""
    raise NotImplementedError(f"No config extractor implemented for {type(net)}")

@extract_configs.register
def _(net: FlaxMLP, probabilistic_layers: List[str] = None, 
      num_probabilistic_layers: int = None) -> List[Dict]:
    return extract_mlp_configs(net, probabilistic_layers, num_probabilistic_layers)

@extract_configs.register
def _(net: FlaxMLP2Head, probabilistic_layers: List[str] = None, 
      num_probabilistic_layers: int = None) -> List[Dict]:
    return extract_mlp2head_configs(net, probabilistic_layers, num_probabilistic_layers)

@extract_configs.register
def _(net: FlaxConvNet, probabilistic_layers: List[str] = None, 
      num_probabilistic_layers: int = None) -> List[Dict]:
    return extract_convnet_configs(net, probabilistic_layers, num_probabilistic_layers)

@extract_configs.register
def _(net: FlaxConvNet2Head, probabilistic_layers: List[str] = None, 
      num_probabilistic_layers: int = None) -> List[Dict]:
    return extract_convnet2head_configs(net, probabilistic_layers, num_probabilistic_layers)

@extract_configs.register
def _(net: FlaxTransformer, probabilistic_layers: List[str] = None, 
      num_probabilistic_layers: int = None) -> List[Dict]:
    return extract_transformer_configs(net, probabilistic_layers, num_probabilistic_layers)


def get_prob_indices(model, layer_names, prob_ratio=0.2):
    indices_dict = {}
    configs = extract_configs(model, [])
    for config in configs:
        layer_name = config["layer_name"]
        if layer_name in layer_names:
            if layer_name in ['TokenEmbed', 'PosEmbed']:
                features = config['num_embeddings']
            else:
                features = config["features"]
            indices_dict[layer_name] = np.random.choice(
                range(0, features + 1), size=int(prob_ratio*features), replace=False)
    return indices_dict


def _select_probabilistic_components(weights, method='magnitude', threshold_percentile=90, top_k_percent=10, grads=None, for_embedding=False):
    """Helper function to select components based on weights/gradients."""
    if method == 'magnitude':
        axis = 1 if for_embedding else 0
        magnitudes = jnp.linalg.norm(weights, axis=axis)
        threshold = jnp.percentile(magnitudes, threshold_percentile)
        indices = jnp.where(magnitudes > threshold)[0]

    elif method == 'variance':
        axis = 1 if for_embedding else 0
        variances = jnp.var(weights, axis=axis)
        k = int(len(variances) * top_k_percent / 100)
        indices = jnp.argsort(variances)[-k:]

    elif method == 'gradient':
        if grads is None:
            raise ValueError("Gradient information required for gradient-based selection")
        sensitivities = jnp.mean(jnp.abs(grads), axis=0)
        k = int(len(sensitivities) * top_k_percent / 100)
        indices = jnp.argsort(sensitivities)[-k:]

    else:
        raise ValueError(f"Unknown selection method: {method}")

    return indices


def select_probabilistic_components(model,
                                 layer_names,
                                 method='magnitude',
                                 threshold_percentile=90,
                                 top_k_percent=10):
    """Select which components (neurons/embeddings) to make probabilistic within specified layers."""
    if isinstance(layer_names, str):
        layer_names = [layer_names]
        
    # Determine model type based on the architecture
    is_transformer = isinstance(model.model, FlaxTransformer)
    flatten_fn = flatten_transformer_params_dict if is_transformer else flatten_params_dict
    
    # Get flattened parameters
    flat_params = flatten_fn(model.get_params())
    
    # Get flattened gradients if needed
    if method == 'gradient':
        if not hasattr(model, 'grad_history') or not model.grad_history:
            raise ValueError("No gradient history available. Run training with gradient collection.")
        avg_grads = jax.tree_map(lambda *x: jnp.mean(jnp.stack(x), axis=0), *model.grad_history)
        flat_grads = flatten_fn(avg_grads)
    else:
        flat_grads = None
    
    prob_components = {}
    for layer_name in layer_names:
        if layer_name not in flat_params:
            raise ValueError(f"Layer {layer_name} not found in model parameters")
            
        # For transformer, only token/pos embeddings are embedding layers
        is_embedding = is_transformer and 'Embed' in layer_name
        weights = flat_params[layer_name]['embedding' if is_embedding else 'kernel']
        grads = None if flat_grads is None else flat_grads[layer_name]['embedding' if is_embedding else 'kernel']
        
        indices = _select_probabilistic_components(
            weights=weights,
            method=method,
            threshold_percentile=threshold_percentile,
            top_k_percent=top_k_percent,
            grads=grads,
            for_embedding=is_embedding
        )
        
        prob_components[layer_name] = indices.tolist()
    
    return prob_components