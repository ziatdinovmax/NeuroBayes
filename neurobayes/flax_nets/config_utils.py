from typing import Dict, List
from functools import singledispatch
import numpy as np

from .convnet import FlaxConvNet, FlaxConvNet2Head
from .mlp import FlaxMLP, FlaxMLP2Head
from .transformer import FlaxTransformer
from .configs import extract_mlp_configs, extract_convnet_configs, extract_mlp2head_configs, extract_convnet2head_configs, extract_transformer_configs

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