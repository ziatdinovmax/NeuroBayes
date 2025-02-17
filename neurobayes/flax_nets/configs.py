from typing import Dict, List, Any
import flax.linen as nn
from .mlp import FlaxMLP, FlaxMLP2Head
from .convnet import FlaxConvNet, FlaxConvNet2Head
from .transformer import FlaxTransformer


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
            "is_probabilistic": layer_name in probabilistic_layers,
            "layer_type": "fc",
            "layer_name": layer_name
        })
    
    # Process output layer if it exists
    if mlp.target_dim:
        layer_name = f"Dense{len(mlp.hidden_dims)}"
        configs.append({
            "features": mlp.target_dim,
            # Note: activation is explicitly None here, overriding any softmax 
            # in the original FlaxMLP. For classification tasks, softmax will 
            # be applied later in PartialBNN.model()
            "activation": None,
            "is_probabilistic": layer_name in probabilistic_layers,
            "layer_type": "fc",
            "layer_name": layer_name
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
            "is_probabilistic": layer_name in probabilistic_layers,
            "layer_type": "fc",
            "layer_name": layer_name
        })
    
    # Process head layers
    layer_name = "MeanHead"
    configs.append({
        "features": mlp.target_dim,
        "activation": None,
        "is_probabilistic": layer_name in probabilistic_layers,
        "layer_type": "fc",
        "layer_name": layer_name
    })
    layer_name = "VarianceHead"
    configs.append({
        "features": mlp.target_dim,
        "activation": nn.softplus,
        "is_probabilistic": layer_name in probabilistic_layers,
        "layer_type": "fc",
        "layer_name": layer_name
    })
    
    return configs


def extract_convnet_configs(
    net: FlaxConvNet,
    probabilistic_layers: List[str] = None,
    num_probabilistic_layers: int = None
) -> List[Dict]:
    """
    Extract layer configurations from a ConvNet model.
    
    Args:
        net: The FlaxConvNet instance
        probabilistic_layers: List of layer names to be treated as probabilistic
        num_probabilistic_layers: Number of final layers to be probabilistic
                                (0 means only output layer is probabilistic)
    """
    if (probabilistic_layers is None) == (num_probabilistic_layers is None):
        raise ValueError(
            "Exactly one of 'probabilistic_layers' or 'num_probabilistic_layers' must be specified"
        )
    
    # Get activation function
    activation_fn = nn.tanh if net.activation == 'tanh' else nn.silu
    
    # Get all layer names
    conv_names = [f"Conv{i}" for i in range(len(net.conv_layers))]
    fc_names = [f"Dense{i}" for i in range(len(net.fc_layers))]
    output_name = f"Dense{len(net.fc_layers)}"
    all_layer_names = conv_names + fc_names + [output_name]
    
    # If using num_probabilistic_layers, create probabilistic_layers list
    if num_probabilistic_layers is not None:
        hidden_layers = all_layer_names[:-1][-num_probabilistic_layers:] if num_probabilistic_layers > 0 else []
        probabilistic_layers = hidden_layers + [output_name]
    
    configs = []
    
    # Add conv layer configs
    for i, filters in enumerate(net.conv_layers):
        layer_name = f"Conv{i}"
        configs.append({
            "features": filters,
            "input_dim": net.input_dim,
            "kernel_size": net.kernel_size,
            "activation": activation_fn,
            "is_probabilistic": layer_name in probabilistic_layers,
            "layer_type": "conv",
            "layer_name": layer_name
        })
    
    # Add FC layer configs
    for i, hidden_dim in enumerate(net.fc_layers):
        layer_name = f"Dense{i}"
        configs.append({
            "features": hidden_dim,
            "activation": activation_fn,
            "is_probabilistic": layer_name in probabilistic_layers,
            "layer_type": "fc",
            "layer_name": layer_name
        })
    
    # Add output layer config
    configs.append({
        "features": net.target_dim,
        "activation": None,
        "is_probabilistic": output_name in probabilistic_layers,
        "layer_type": "fc",
        "layer_name": output_name
    })
    
    return configs


def extract_convnet2head_configs(
    net: FlaxConvNet2Head,
    probabilistic_layers: List[str] = None,
    num_probabilistic_layers: int = None
    ) -> List[Dict]:
    """
    Extract layer configurations from a two-headed ConvNet model.
    
    Args:
        net: The FlaxConvNet2Head instance
        probabilistic_layers: List of layer names to be treated as probabilistic
        num_probabilistic_layers: Number of final layers to be probabilistic
                                (0 means only output heads are probabilistic)
    """
    if (probabilistic_layers is None) == (num_probabilistic_layers is None):
        raise ValueError(
            "Exactly one of 'probabilistic_layers' or 'num_probabilistic_layers' must be specified"
        )
    
    # Get activation function
    activation_fn = nn.tanh if net.activation == 'tanh' else nn.silu
    
    # Get all layer names
    conv_names = [f"Conv{i}" for i in range(len(net.conv_layers))]
    fc_names = [f"Dense{i}" for i in range(len(net.fc_layers))]
    head_names = ["MeanHead", "VarianceHead"]
    all_layer_names = conv_names + fc_names + head_names
    
    # If using num_probabilistic_layers, create probabilistic_layers list
    if num_probabilistic_layers is not None:
        hidden_layers = all_layer_names[:-2][-num_probabilistic_layers:] if num_probabilistic_layers > 0 else []
        probabilistic_layers = hidden_layers + head_names
    
    configs = []
    
    # Add conv layer configs
    for i, filters in enumerate(net.conv_layers):
        layer_name = f"Conv{i}"
        configs.append({
            "features": filters,
            "input_dim": net.input_dim,
            "kernel_size": net.kernel_size,
            "activation": activation_fn,
            "is_probabilistic": layer_name in probabilistic_layers,
            "layer_type": "conv",
            "layer_name": layer_name
        })
    
    # Add FC layer configs
    for i, hidden_dim in enumerate(net.fc_layers):
        layer_name = f"Dense{i}"
        configs.append({
            "features": hidden_dim,
            "activation": activation_fn,
            "is_probabilistic": layer_name in probabilistic_layers,
            "layer_type": "fc",
            "layer_name": layer_name
        })
    
    # Add head layer configs
    layer_name = "MeanHead"
    configs.append({
        "features": net.target_dim,
        "activation": None,
        "is_probabilistic": layer_name in probabilistic_layers,
        "layer_type": "fc",
        "layer_name": layer_name
    })
    layer_name = "VarianceHead"
    configs.append({
        "features": net.target_dim,
        "activation": nn.softplus,
        "is_probabilistic": layer_name in probabilistic_layers,
        "layer_type": "fc",
        "layer_name": layer_name
    })
    
    return configs


def extract_transformer_configs(
    transformer: FlaxTransformer,
    probabilistic_layers: List[str] = None,
    num_probabilistic_layers: int = None
) -> List[Dict]:
    """
    Extract layer configurations from a Transformer model.
    
    Args:
        transformer: The FlaxTransformer instance
        probabilistic_layers: List of layer names to be treated as probabilistic
        num_probabilistic_layers: Number of final layers to be probabilistic
    """
    if (probabilistic_layers is None) == (num_probabilistic_layers is None):
        raise ValueError(
            "Exactly one of 'probabilistic_layers' or 'num_probabilistic_layers' must be specified"
        )
    
    # Get activation function
    activation_fn = nn.silu if transformer.activation == 'silu' else nn.tanh
    
    configs = []
    
    # Embedding layer configs
    configs.append({
        "features": transformer.d_model,
        "num_embeddings": transformer.vocab_size,
        "is_probabilistic": "TokenEmbed" in (probabilistic_layers or []),
        "layer_type": "embedding",
        "layer_name": "TokenEmbed"
    })
    
    configs.append({
        "features": transformer.d_model,
        "num_embeddings": transformer.max_seq_length,
        "is_probabilistic": "PosEmbed" in (probabilistic_layers or []),
        "layer_type": "embedding",
        "layer_name": "PosEmbed"
    })
    
    # For each transformer block
    for i in range(transformer.num_layers):
        configs.append({
            "num_heads": transformer.nhead,
            "qkv_features": transformer.d_model,
            "dropout_rate": transformer.dropout_rate,
            "is_probabilistic": f"Block{i}_Attention" in (probabilistic_layers or []),
            "layer_type": "attention",
            "layer_name": f"Block{i}_Attention"
        })
        
        configs.append({
            "is_probabilistic": f"Block{i}_LayerNorm1" in (probabilistic_layers or []),
            "layer_type": "layernorm",
            "layer_name": f"Block{i}_LayerNorm1"
        })
        
        configs.append({
            "features": transformer.dim_feedforward,
            "activation": activation_fn,
            "is_probabilistic": f"Block{i}_MLP_dense1" in (probabilistic_layers or []),
            "layer_type": "fc",
            "layer_name": f"Block{i}_MLP_dense1"
        })
        
        configs.append({
            "features": transformer.d_model,
            "activation": None,
            "is_probabilistic": f"Block{i}_MLP_dense2" in (probabilistic_layers or []),
            "layer_type": "fc",
            "layer_name": f"Block{i}_MLP_dense2"
        })
        
        configs.append({
            "is_probabilistic": f"Block{i}_LayerNorm2" in (probabilistic_layers or []),
            "layer_type": "layernorm",
            "layer_name": f"Block{i}_LayerNorm2"
        })
    
    # Final layers
    configs.append({
        "features": transformer.dim_feedforward,
        "activation": activation_fn,
        "is_probabilistic": "FinalDense1" in (probabilistic_layers or []),
        "layer_type": "fc",
        "layer_name": "FinalDense1"
    })
    
    configs.append({
        "features": transformer.target_dim,
        "activation": None,
        "is_probabilistic": "FinalDense2" in (probabilistic_layers or []),
        "layer_type": "fc",
        "layer_name": "FinalDense2"
    })
    
    # If using num_probabilistic_layers, update is_probabilistic flags
    if num_probabilistic_layers is not None:
        total_layers = len(configs)
        for i, config in enumerate(configs):
            config["is_probabilistic"] = i >= (total_layers - num_probabilistic_layers)
    
    return configs
