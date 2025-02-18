from typing import Dict, Optional, Type, Tuple, List
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from .hybrid_layers import partial_bayesian_dense, partial_bayesian_embed, partial_bayesian_attention
from ..flax_nets import FlaxTransformer, DeterministicNN
from ..flax_nets import MLPLayerModule, TransformerAttentionModule, EmbedModule, LayerNormModule
from ..flax_nets import extract_configs, select_bayesian_components
from ..utils import flatten_transformer_params_dict


class PartialBayesianTransformer(BNN):
    """
    Partially stochastic (Bayesian) Transformer network.

    Args:
        transformer: FlaxTransformer architecture
        deterministic_weights: Pre-trained deterministic weights. If not provided,
            the transformer will be trained from scratch when running .fit() method
        probabilistic_layer_names: Names of transformer modules to be treated probabilistically.
            Valid names include: "TokenEmbed_0", "PosEmbed_0", "Block{i}_Attention",
            "Block{i}_MLP_dense1", "Block{i}_MLP_dense2", "FinalDense1", "FinalDense2"
        num_probabilistic_layers: Alternative to probabilistic_layer_names.
            Number of final layers to be treated as probabilistic
        probabilistic_neurons: Optional dict mapping layer names to lists of (input_idx, output_idx) 
            tuples specifying which weight connections should be Bayesian.
            For layers not in this dict, the entire layer will be treated as Bayesian.
            For dense layers, pairs represent (input_neuron, output_neuron) connections.
            For embedding layers, pairs represent (embedding_idx, feature_idx) connections.
            Can be automatically generated using select_bayesian_components() utility function
            which supports various selection methods including magnitude, gradient, variance,
            and clustering-based approaches.
        num_classes: Number of classes for classification task.
            If None, the model performs regression. Defaults to None.
        noise_prior: Custom prior for observational noise distribution

    Example:
        # Automatically select Bayesian weights using variance-based selection
        prob_neurons = select_bayesian_components(
            transformer,  # DeterministicNN class
            layer_names=['TokenEmbed', 'FinalDense1'],
            method='variance',
            num_pairs_per_layer=2
        )

        # Or manually specify weight connections
        probabilistic_neurons = {
            'TokenEmbed': [(0, 1), (2, 3)],  # Make these embedding-feature connections Bayesian
            'FinalDense1': [(0, 5), (1, 10)]  # Make these input-output connections Bayesian
        }

        # Note that you would typically need to make far more than 2 weights probabilistic
    """

    def __init__(self,
                 transformer: Type[FlaxTransformer],
                 deterministic_weights: Optional[Dict[str, jnp.ndarray]] = None,
                 probabilistic_layer_names: List[str] = None,
                 num_probabilistic_layers: int = None,
                 probabilistic_neurons: Optional[Dict[str, List[Tuple[int]]]] = None,
                 num_classes: Optional[int] = None,
                 noise_prior: Optional[dist.Distribution] = None
                 ) -> None:
        super().__init__(None, num_classes, noise_prior)
        
        self.deterministic_nn = transformer
        self.deterministic_weights = deterministic_weights
        
        # Extract configurations
        self.layer_configs = extract_configs(
            transformer, probabilistic_layer_names, num_probabilistic_layers)
        
        # For each layer, if a subset of neurons is specified, add that to the config.
        if probabilistic_neurons is not None:
            for config in self.layer_configs:
                layer_name = config['layer_name']
                if layer_name in probabilistic_neurons:
                    config['probabilistic_neurons'] = probabilistic_neurons[layer_name]
                else:
                    config['probabilistic_neurons'] = None
        else:
            for config in self.layer_configs:
                config['probabilistic_neurons'] = None
    
    def model(self, X, y=None, priors_sigma=1.0, **kwargs):
        net = self.deterministic_nn
        pretrained_priors = flatten_transformer_params_dict(self.deterministic_weights)
        
        def prior(name, shape):
            param_path = name.split('.')
            layer_name = param_path[0]
            
            if len(param_path) == 3:  # Attention parameters
                component = param_path[1]  # 'query', 'key', or 'value'
                param_type = param_path[2]  # 'kernel' or 'bias'
                return dist.Normal(pretrained_priors[layer_name][component][param_type], priors_sigma)
            else:  # Other parameters
                param_type = param_path[-1]
                return dist.Normal(pretrained_priors[layer_name][param_type], priors_sigma)
        
        current_input = X
        positions = jnp.arange(X.shape[1])[None, :]
        token_embedding = None
        pos_embedding = None
        
        for config in self.layer_configs:
            layer_name = config['layer_name']
            layer_type = config['layer_type']
            
            if layer_type == "embedding":
                layer = EmbedModule(
                    features=config['features'],
                    num_embeddings=config['num_embeddings'],
                    layer_name=layer_name
                )
                input_data = positions if layer_name == 'PosEmbed' else current_input
                
                if config['is_probabilistic']:
                    # Get probabilistic indices from config if specified
                    prob_indices = config.get('probabilistic_neurons', None)
                    if prob_indices is not None:
                        # Use custom partial Bayesian implementation for neuron-level control
                        embedding = partial_bayesian_embed(
                            input_data,
                            pretrained_embedding=pretrained_priors[layer_name]['embedding'],
                            prob_indices=prob_indices,
                            priors_sigma=priors_sigma,
                            layer_name=layer_name
                        )
                    else:
                        # Use random_flax_module when making the entire layer Bayesian
                        net = random_flax_module(
                            layer_name, 
                            layer,
                            input_shape=(1, *input_data.shape[1:]),
                            prior=prior
                        )
                        embedding = net(input_data)
                else:
                    # Use deterministic embedding
                    params = {"params": {layer_name: pretrained_priors[layer_name]}}
                    embedding = layer.apply(params, input_data)
                
                if layer_name == 'TokenEmbed':
                    token_embedding = embedding
                else:  # PosEmbed
                    pos_embedding = embedding
                    current_input = token_embedding + pos_embedding
                
            elif layer_type == "attention":
                # Save input for the residual connection.
                residual = current_input
            
                block_idx = int(layer_name.split('_')[0][5:])
                layer = TransformerAttentionModule(
                    num_heads=config['num_heads'],
                    qkv_features=config['qkv_features'],
                    dropout_rate=config.get('dropout_rate', 0.1),
                    layer_name="Attention",
                    block_idx=block_idx
                )
                if config['is_probabilistic']:
                    if config.get('probabilistic_neurons') is not None:
                        # Use our custom partial Bayesian attention implementation.
                        current_input = partial_bayesian_attention(
                            current_input,
                            pretrained_params=pretrained_priors[layer_name],
                            prob_neurons=config['probabilistic_neurons'],
                            priors_sigma=priors_sigma,
                            num_heads=config['num_heads'],
                            layer_name=layer_name,
                            enable_dropout=False
                        )
                    else:
                        # Use full Bayesian attention via the random_flax_module.
                        net = random_flax_module(layer_name, layer,
                                                 input_shape=(1, *current_input.shape[1:]), prior=prior)
                        current_input = net(current_input, enable_dropout=False)
                else:
                    params = {"params": {f"Block{block_idx}_Attention": pretrained_priors[layer_name]}}
                    current_input = layer.apply(params, current_input, enable_dropout=False)
            
                # Add the residual connection.
                current_input = current_input + residual
                
            elif layer_type == "layernorm":
                layer = LayerNormModule(layer_name=layer_name)
                params = {"params": {layer_name: pretrained_priors[layer_name]}}
                current_input = layer.apply(params, current_input)
                
                # Save residual after first layer norm in each block
                if layer_name.endswith('LayerNorm1'):
                    residual = current_input
                    
            else:  # fc layers
                layer = MLPLayerModule(
                    features=config['features'],
                    activation=config.get('activation'),
                    layer_name=layer_name
                )
                if config['is_probabilistic']:
                    prob_neurons = config.get('probabilistic_neurons')
                    if prob_neurons is not None:
                        # Use custom partial Bayesian implementation for neuron-level control
                        current_input = partial_bayesian_dense(
                            current_input,
                            pretrained_kernel=pretrained_priors[layer_name]['kernel'],
                            pretrained_bias=pretrained_priors[layer_name]['bias'],
                            prob_neurons=prob_neurons,
                            priors_sigma=priors_sigma,
                            layer_name=layer_name,
                            activation=config.get('activation')
                        )
                    else:
                        # Use random_flax_module when making the entire layer Bayesian
                        net = random_flax_module(layer_name, layer,
                                            input_shape=(1, *current_input.shape[1:]), prior=prior)
                        current_input = net(current_input, enable_dropout=False)
                else:
                    if layer_name.startswith('Block'):
                        block_idx = int(layer_name.split('_')[0][5:])
                        params = {"params": {f"Block{block_idx}_MLP_{layer_name.split('_')[-1]}": pretrained_priors[layer_name]}}
                    else:
                        params = {"params": {layer_name: pretrained_priors[layer_name]}}
                    current_input = layer.apply(params, current_input, enable_dropout=False)
                
                # Add residual after second dense layer in each block
                if layer_name.endswith('dense2'):
                    current_input = current_input + residual

        current_input = jnp.mean(current_input, axis=1)
        
        if self.is_regression:
            mu = numpyro.deterministic("mu", current_input)
            sig = numpyro.sample("sig", self.noise_prior)
            numpyro.sample("y", dist.Normal(mu, sig), obs=y)
        else:
            probs = numpyro.deterministic("probs", softmax(current_input, axis=-1))
            numpyro.sample("y", dist.Categorical(probs=probs), obs=y)

    def fit(self, X: jnp.ndarray, y: jnp.ndarray,
            num_warmup: int = 2000, num_samples: int = 2000,
            num_chains: int = 1, chain_method: str = 'sequential',
            sgd_epochs: Optional[int] = None, sgd_lr: Optional[float] = 0.01,
            sgd_batch_size: Optional[int] = None, swa_config: Optional[Dict] = None,
            map_sigma: float = 1.0, priors_sigma: float = 1.0,
            progress_bar: bool = True, device: str = None,
            rng_key: Optional[jnp.array] = None,
            extra_fields: Optional[Tuple[str, ...]] = (),
            select_neurons_config: Optional[Dict] = None,
            **kwargs
            ) -> None:
        """
        Fit the partially Bayesian transformer.
        
        Args:
            X (jnp.ndarray): Input sequences of shape (batch_size, seq_length).
            y (jnp.ndarray): Target array. For single-output problems: 1D array of shape (n_samples,).
                For multi-output problems: 2D array of shape (n_samples, target_dim).
            num_warmup (int, optional): Number of NUTS warmup steps. Defaults to 2000.
            num_samples (int, optional): Number of NUTS samples to draw. Defaults to 2000.
            num_chains (int, optional): Number of NUTS chains to run. Defaults to 1.
            chain_method (str, optional): Method for running chains: 'sequential', 'parallel', 
                or 'vectorized'. Defaults to 'sequential'.
            sgd_epochs (Optional[int], optional): Number of SGD training epochs for deterministic NN.
                Defaults to 500 (if no pretrained weights are provided).
            sgd_lr (float, optional): SGD learning rate. Defaults to 0.01.
            sgd_batch_size (Optional[int], optional): Mini-batch size for SGD training. 
                Defaults to None (all input data is processed as a single batch).
            swa_config (dict, optional):
                Stochastic weight averaging protocol. Defaults to averaging weights
                at the end of training trajectory (the last 5% of SGD epochs).
            map_sigma (float, optional): Sigma in Gaussian prior for regularized SGD training. Defaults to 1.0.
            priors_sigma (float, optional): Standard deviation for default or pretrained priors
                in the Bayesian part of the NN. Defaults to 1.0.
            progress_bar (bool, optional): Show progress bar. Defaults to True.
            device (Optional[str], optional): The device to perform computation on ('cpu', 'gpu'). 
                Defaults to None (JAX default device).
            rng_key (Optional[jnp.ndarray], optional): Random number generator key. Defaults to None.
            extra_fields (Optional[Tuple[str, ...]], optional): Extra fields to collect during the MCMC run. 
                Defaults to ().
            select_neurons_config (Optional[Dict], optional): Configuration for selecting 
                probabilistic neurons after deterministic training. Should contain:
                - method: str - Selection method ('variance', 'gradient', etc.)
                - layer_names: List[str] - Names of layers to make partially Bayesian
                - num_pairs_per_layer: int - Number of weight pairs to select per layer
                - Additional method-specific parameters
            **max_num_restarts (int, optional): Maximum number of fitting attempts for single chain. 
                Ignored if num_chains > 1. Defaults to 1.
            **min_accept_prob (float, optional): Minimum acceptance probability threshold. 
                Only used if num_chains = 1. Defaults to 0.55.
            **run_diagnostics (bool, optional): Run Gelman-Rubin diagnostics layer-by-layer at the end.
                Defaults to False.

        Example:
            model.fit(
                X, y, num_warmup=1000, num_samples=1000,
                sgd_lr=1e-3, sgd_epochs=100,
                select_neurons_config={
                    'method': 'variance',
                    'layer_names': ['TokenEmbed', 'FinalDense1'],
                    'num_pairs_per_layer': 10
                }
            )
        """        
        if not self.deterministic_weights:
            print("Training deterministic transformer...")
            X, y = self.set_data(X, y)
            collect_gradients = (select_neurons_config is not None and 
                                 select_neurons_config.get('method') == 'gradient')
            det_nn = DeterministicNN(
                self.transformer,
                input_shape=X.shape[1:],
                loss='homoskedastic' if self.is_regression else 'classification',
                learning_rate=sgd_lr, 
                swa_config=swa_config, 
                sigma=map_sigma,
                collect_gradients=collect_gradients
            )
            det_nn.train(
                X, y, 
                500 if sgd_epochs is None else sgd_epochs,
                sgd_batch_size
            )
            self.deterministic_weights = det_nn.state.params

            # If neuron selection config is provided, select probabilistic neurons
            if select_neurons_config is not None:
                print(f"Selecting probabilistic neurons using {select_neurons_config['method']} method...")
                prob_neurons = select_bayesian_components(
                    det_nn,  # Pass the just trained deterministic model
                    layer_names=select_neurons_config['layer_names'],
                    method=select_neurons_config['method'],
                    num_pairs_per_layer=select_neurons_config['num_pairs_per_layer'],
                    **{k: v for k, v in select_neurons_config.items() 
                       if k not in ['method', 'layer_names', 'num_pairs_per_layer']}
                )
                
                # Update layer configs with newly selected neurons
                for config in self.layer_configs:
                    layer_name = config['layer_name']
                    if layer_name in prob_neurons:
                        config['probabilistic_neurons'] = prob_neurons[layer_name]
                    else:
                        config['probabilistic_neurons'] = None

            print("Training partially Bayesian transformer")
            
        super().fit(
            X, y, num_warmup, num_samples, num_chains, chain_method,
            priors_sigma, progress_bar, device, rng_key, extra_fields, **kwargs
        )
