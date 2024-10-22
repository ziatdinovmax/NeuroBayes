from typing import Dict, List, Any
import flax.linen as nn
from .mlp import FlaxMLP

def extract_mlp_configs(
    mlp: FlaxMLP,
    probabilistic_layers: List[str]
) -> List[Dict]:
    """
    Extract layer configurations from a FlaxMLP model.
    
    Args:
        mlp: The FlaxMLP instance
        probabilistic_layers: List of layer names to be treated as probabilistic
        
    Returns:
        List of layer configurations for BNN
    """
    configs = []
    
    # Get activation function
    activation_fn = nn.tanh if mlp.activation == 'tanh' else nn.silu
    
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
            "activation": None,  # No activation for output layer
            "is_probabilistic": layer_name in probabilistic_layers
        })
    
    return configs


class MLPLayerModule(nn.Module):
    features: int
    activation: Any = None
    layer_name: str = 'dense'
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features, name=self.layer_name)(x)
        if self.activation is not None:
            x = self.activation(x)
        return x