from typing import Dict, List, Any
import flax.linen as nn
from .mlp import FlaxMLP, FlaxMLP2Head


def extract_mlp_configs(
    mlp: FlaxMLP,
    probabilistic_layers: List[str] = None,
    num_probabilistic_layers: int = None
    ) -> List[Dict]:
    """
    Extract layer configurations from a FlaxMLP model.
    
    Args:
        mlp: The FlaxMLP instance
        probabilistic_layers: List of layer names to be treated as probabilistic
        num_probabilistic_layers: Number of hidden layers to be probabilistic
                                (0 means only output layer is probabilistic)
    """
    if (probabilistic_layers is None) == (num_probabilistic_layers is None):
        raise ValueError(
            "Exactly one of 'probabilistic_layers' or 'num_probabilistic_layers' must be specified"
        )
    
    # Get all layer names
    layer_names = [f"Dense{i}" for i in range(len(mlp.hidden_dims))]
    if mlp.target_dim:
        layer_names.append(f"Dense{len(mlp.hidden_dims)}")
    
    # If using num_probabilistic_layers, create probabilistic_layers list
    if num_probabilistic_layers is not None:
        hidden_layers = layer_names[:-1][-num_probabilistic_layers:] if num_probabilistic_layers > 0 else []
        probabilistic_layers = hidden_layers + [layer_names[-1]]  # Always add output layer

    # Get activation function
    activation_fn = nn.tanh if mlp.activation == 'tanh' else nn.silu
    
    configs = []
    # Process hidden layers
    for i, hidden_dim in enumerate(mlp.hidden_dims):
        layer_name = f"Dense{i}"
        configs.append({
            "features": hidden_dim,
            "activation": activation_fn,
            "is_probabilistic": layer_name in probabilistic_layers
        })
    
    # Process output layer if it exists
    if mlp.target_dim:
        layer_name = f"Dense{len(mlp.hidden_dims)}"
        configs.append({
            "features": mlp.target_dim,
            "activation": None,
            "is_probabilistic": layer_name in probabilistic_layers
        })
    
    return configs


def extract_mlp2head_configs(
    mlp: FlaxMLP2Head,
    probabilistic_layers: List[str] = None,
    num_probabilistic_layers: int = None
) -> List[Dict]:
    """
    Extract layer configurations from a two-headed FlaxMLP model.
    
    Args:
        mlp: The FlaxMLP2Head instance
        probabilistic_layers: List of layer names to be treated as probabilistic
        num_probabilistic_layers: Number of final layers to be probabilistic (including heads)
    
    Raises:
        ValueError: If neither or both parameters are specified
    """
    if (probabilistic_layers is None) == (num_probabilistic_layers is None):
        raise ValueError(
            "Exactly one of 'probabilistic_layers' or 'num_probabilistic_layers' must be specified"
        )
    
    # Get all layer names
    layer_names = [f"Dense{i}" for i in range(len(mlp.hidden_dims))]
    head_names = ["MeanHead", "VarianceHead"]
    
    # If using num_probabilistic_layers, create probabilistic_layers list
    if num_probabilistic_layers is not None:
        hidden_layers = layer_names[-num_probabilistic_layers:] if num_probabilistic_layers > 0 else []
        probabilistic_layers = hidden_layers + head_names  # Always add both heads

     # Get activation function
    activation_fn = nn.tanh if mlp.activation == 'tanh' else nn.silu
    
    configs = []
    # Process hidden layers
    for i, hidden_dim in enumerate(mlp.hidden_dims):
        layer_name = f"Dense{i}"
        configs.append({
            "features": hidden_dim,
            "activation": activation_fn,
            "is_probabilistic": layer_name in probabilistic_layers
        })
    
    # Process head layers
    configs.append({
        "features": mlp.target_dim,
        "activation": None,
        "is_probabilistic": "MeanHead" in probabilistic_layers
    })
    
    configs.append({
        "features": mlp.target_dim,
        "activation": nn.softplus,
        "is_probabilistic": "VarianceHead" in probabilistic_layers
    })
    
    return configs