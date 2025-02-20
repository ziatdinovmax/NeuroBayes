from typing import Dict, List
from functools import singledispatch
from collections import OrderedDict

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


def get_configs(model, pretty_print: bool = True) -> List[Dict]:
    """
    Get basic layer configurations using the existing extract_configs function.
    
    Args:
        model: The neural network model (FlaxMLP, FlaxConvNet, etc.)
        pretty_print: If True, prints a formatted version of the configs
        
    Returns:
        List of dictionaries with layer configurations
    """
    configs = extract_configs(model, probabilistic_layers=['None'])
    
    # Keep only essential fields
    essential_fields = [
        'layer_name',
        'layer_type',
        'features',
        'num_embeddings'
    ]
    
    # Function to simplify config dict
    def simplify_config(config: Dict) -> OrderedDict:
        ordered = OrderedDict()
        for field in essential_fields:
            if field in config:
                ordered[field] = config[field]
        return ordered
    
    # Simplify configs
    ordered_configs = [simplify_config(config) for config in configs]
    
    if pretty_print:
        _pretty_print_configs(ordered_configs, model.__class__.__name__)
    
    return ordered_configs

def _pretty_print_configs(configs: List[OrderedDict], model_type: str) -> None:
    """
    Pretty print the layer configurations.
    """
    # Get the maximum width for each column
    widths = {}
    for config in configs:
        for key, value in config.items():
            val_str = str(value) if value is not None else "-"
            widths[key] = max(widths.get(key, len(key)), len(val_str))
    
    # Print header
    print(f"\n{'=' * 80}")
    print(f"Model Architecture: {model_type}")
    print(f"{'=' * 80}\n")
    
    # Print column headers
    header = "  ".join(
        f"{key:<{widths[key]}}" 
        for key in configs[0].keys()
    )
    print(header)
    print("-" * len(header))
    
    # Print each layer's config
    for config in configs:
        row = []
        for key, value in config.items():
            val_str = str(value) if value is not None else "-"
            row.append(f"{val_str:<{widths[key]}}")
        print("  ".join(row))
    
    # Print simple summary
    print(f"\n{'=' * 80}")
    print(f"Total layers: {len(configs)}")
    print(f"{'=' * 80}\n")

def print_layer_configs(model):
    """
    Print the layer configurations for a given model.
    
    Args:
        model: The neural network model (FlaxMLP, FlaxConvNet, etc.)
    """
    _ = get_configs(model, pretty_print=True)