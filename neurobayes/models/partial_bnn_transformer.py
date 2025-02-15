from typing import Dict, Optional, Type, Tuple, List
import jax.numpy as jnp
from jax.nn import softmax

import numpyro
import numpyro.distributions as dist
from numpyro.contrib.module import random_flax_module

from .bnn import BNN
from .hybrid_layers import partial_bayesian_dense, partial_bayesian_embed
from ..flax_nets import FlaxTransformer, DeterministicNN
from ..flax_nets import MLPLayerModule, TransformerAttentionModule, EmbedModule, LayerNormModule
from ..flax_nets import extract_configs
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
        num_classes: Number of classes for classification task.
            If None, the model performs regression. Defaults to None.
        noise_prior: Custom prior for observational noise distribution
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
                # Save input for residual
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
                    net = random_flax_module(layer_name, layer,
                                        input_shape=(1, *current_input.shape[1:]), prior=prior)
                    current_input = net(current_input, enable_dropout=False)
                else:
                    params = {"params": {f"Block{block_idx}_Attention": pretrained_priors[layer_name]}}
                    current_input = layer.apply(params, current_input, enable_dropout=False)
                
                # Add residual after attention
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
            **kwargs
            ) -> None:
        """
        Fit the partially Bayesian transformer.
        
        Args:
            X (jnp.ndarray): Input sequences of shape (batch_size, seq_length).
            For other parameters, see BNN.fit() documentation.
        """
        
        if not self.deterministic_weights:
            print("Training deterministic transformer...")
            X, y = self.set_data(X, y)
            det_nn = DeterministicNN(
                self.transformer,
                input_shape=X.shape[1:],
                loss='homoskedastic' if self.is_regression else 'classification',
                learning_rate=sgd_lr, 
                swa_config=swa_config, 
                sigma=map_sigma
            )
            det_nn.train(
                X, y, 
                500 if sgd_epochs is None else sgd_epochs,
                sgd_batch_size
            )
            self.deterministic_weights = det_nn.state.params
            print("Training partially Bayesian transformer")
            
        super().fit(
            X, y, num_warmup, num_samples, num_chains, chain_method,
            priors_sigma, progress_bar, device, rng_key, extra_fields, **kwargs
        )
