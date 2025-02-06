from typing import Dict, List
from functools import singledispatch

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