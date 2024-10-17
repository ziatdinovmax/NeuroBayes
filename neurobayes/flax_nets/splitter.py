from typing import Dict, Any
from .mlp import FlaxMLP, FlaxMLP2Head
from .convnet import FlaxConvNet, FlaxConvNet2Head



def split_mlp(model, params, n_layers: int = 1, out_dim: int = None):
    """
    Splits MLP and its weights into two sub-networks: one with last n layers
    (+ output layer) removed and another one consisting only of those n layers.
    """
    out_dim = out_dim if out_dim is not None else model.output_dim  # there will be a mismatch in last_layer_params if out_dim != model.output_dim


    subnet1 = FlaxMLP(
        model.hidden_dims[:-n_layers] if n_layers > 0 else model.hidden_dims,
        output_dim=0, activation=model.activation)
    subnet2 = FlaxMLP(
        model.hidden_dims[-n_layers:] if n_layers > 0 else [],
        output_dim=out_dim, activation=model.activation)

    subnet1_params = {}
    subnet2_params = {}
    for i, (key, val) in enumerate(params.items()):
        if i < len(model.hidden_dims) - n_layers:
            subnet1_params[key] = val
        else:
            #new_key = f"Dense{i}"
            new_key = f"Dense{i - (len(model.hidden_dims) - n_layers)}"
            subnet2_params[new_key] = val

    return subnet1, subnet1_params, subnet2, subnet2_params


def split_mlp2head(model, params, n_layers: int = 1, out_dim: int = None):
    """
    Splits MLP2Head and its weights into two sub-networks: one with last n layers
    (+ output heads) removed and another one consisting of those n layers and the output heads.
    """
    out_dim = out_dim if out_dim is not None else model.output_dim

    subnet1 = FlaxMLP(model.hidden_dims[:-n_layers], output_dim=0, activation=model.activation)
    subnet2 = FlaxMLP2Head(model.hidden_dims[-n_layers:], output_dim=out_dim, activation=model.activation)

    subnet1_params = {}
    subnet2_params = {}
    
    for i, (key, val) in enumerate(params.items()):
        if key in ['MeanHead', 'VarianceHead']:
            subnet2_params[key] = val
        elif i < len(model.hidden_dims) - n_layers:
            subnet1_params[key] = val
        else:
            #new_key = f"Dense{i}"
            new_key = f"Dense{i - (len(model.hidden_dims) - n_layers)}"
            subnet2_params[new_key] = val

    return subnet1, subnet1_params, subnet2, subnet2_params


def split_convnet(model: FlaxConvNet, params: Dict[str, Any], n_layers: int = 1):
    """
    Splits FlaxConvNet and its weights into two parts: deterministic (conv + (optionally) deterministic MLP) and 
    stochastic MLP layers.
    
    Args:
        model (FlaxConvNet): The original model to split
        params (dict): The parameters of the original model
        n_layers (int): Number of MLP layers to be considered stochastic (from the end)
    
    Returns:
        tuple: (det_model, det_params, stoch_model, stoch_params)
    """
    det_fc_layers = model.fc_layers[:-n_layers] if n_layers > 0 else model.fc_layers
    stoch_fc_layers = model.fc_layers[-n_layers:] if n_layers > 0 else []
    
    det_model = FlaxConvNet(
        input_dim=model.input_dim,
        conv_layers=model.conv_layers,
        fc_layers=det_fc_layers,
        output_dim=0,  # No output layer in deterministic part
        activation=model.activation,
        kernel_size=model.kernel_size
    )
    
    stoch_model = FlaxMLP(
        hidden_dims=stoch_fc_layers,
        output_dim=model.output_dim,
        activation=model.activation
    )

    det_params = {}
    stoch_params = {}
    
    for key, val in params.items():
        if key.startswith('Conv'):
            det_params[key] = val
        elif key == 'FlaxMLP_0':
            mlp_params = val
            det_mlp_params = {}
            stoch_mlp_params = {}
            for layer_key, layer_val in mlp_params.items():
                layer_num = int(layer_key[5:])  # Extract number from 'DenseX'
                if layer_num < len(det_fc_layers):
                    det_mlp_params[layer_key] = layer_val
                else:
                    new_key = f"Dense{layer_num - len(det_fc_layers)}"
                    stoch_mlp_params[new_key] = layer_val
            if det_mlp_params:
                det_params['FlaxMLP_0'] = det_mlp_params
            if stoch_mlp_params:
                stoch_params = stoch_mlp_params

    return det_model, det_params, stoch_model, stoch_params


def split_convnet_2head(model: FlaxConvNet2Head, params: Dict[str, Any], n_layers: int = 1):
    """
    Splits FlaxConvNet2Head and its weights into two parts: deterministic (conv + (optionally) deterministic MLP) and 
    stochastic MLP layers.
    
    Args:
        model (FlaxConvNet2Head): The original model to split
        params (dict): The parameters of the original model
        n_layers (int): Number of MLP layers to be considered stochastic (from the end)
    
    Returns:
        tuple: (det_model, det_params, stoch_model, stoch_params)
    """
    det_fc_layers = model.fc_layers[:-n_layers] if n_layers > 0 else model.fc_layers
    stoch_fc_layers = model.fc_layers[-n_layers:] if n_layers > 0 else []
    
    det_model = FlaxConvNet2Head(
        input_dim=model.input_dim,
        conv_layers=model.conv_layers,
        fc_layers=det_fc_layers,
        output_dim=0,  # No output layer in deterministic part
        activation=model.activation,
        kernel_size=model.kernel_size
    )
    
    stoch_model = FlaxMLP2Head(
        hidden_dims=stoch_fc_layers,
        output_dim=model.output_dim,
        activation=model.activation
    )

    det_params = {}
    stoch_params = {}
    
    for key, val in params.items():
        if key.startswith('Conv'):
            det_params[key] = val
        elif key == 'FlaxMLP2Head_0':
            mlp_params = val
            det_mlp_params = {}
            stoch_mlp_params = {}
            for layer_key, layer_val in mlp_params.items():
                if layer_key.startswith('Dense'):
                    layer_num = int(layer_key[5:])  # Extract number from 'DenseX'
                    if layer_num < len(det_fc_layers):
                        det_mlp_params[layer_key] = layer_val
                    else:
                        new_key = f"Dense{layer_num - len(det_fc_layers)}"
                        stoch_mlp_params[new_key] = layer_val
                elif layer_key in ['mean', 'var']:
                    stoch_mlp_params[layer_key] = layer_val
            if det_mlp_params:
                det_params['FlaxMLP2Head_0'] = det_mlp_params
            if stoch_mlp_params:
                stoch_params = stoch_mlp_params

    return det_model, det_params, stoch_model, stoch_params