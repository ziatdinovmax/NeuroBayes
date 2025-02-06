from typing import List, Dict
import jax.numpy as jnp
import flax.linen as nn


class EmbedModule(nn.Module):
    features: int
    num_embeddings: int
    layer_name: str = 'embed'
    
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.int32)
        return nn.Embed(
            num_embeddings=self.num_embeddings,
            features=self.features,
            name=self.layer_name
        )(x)

class LayerNormModule(nn.Module):
    layer_name: str = 'layernorm'
    
    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm(name=self.layer_name)(x)
    

class TransformerAttentionModule(nn.Module):
    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.1
    layer_name: str = 'attention'
    block_idx: int = 0
    
    @nn.compact
    def __call__(self, x, enable_dropout: bool = True):
        return nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            dropout_rate=self.dropout_rate,
            deterministic=not enable_dropout,
            name=f"Block{self.block_idx}_{self.layer_name}"
        )(x, x)


class TransformerMLPModule(nn.Module):
    features: int
    output_dim: int
    activation: str = 'silu'
    dropout_rate: float = 0.1
    layer_name: str = 'mlp'
    block_idx: int = 0
    
    @nn.compact
    def __call__(self, x, enable_dropout: bool = True):
        activation_fn = nn.silu if self.activation == 'silu' else nn.tanh
        x = nn.Dense(
            features=self.features, 
            name=f"Block{self.block_idx}_{self.layer_name}_dense1"
        )(x)
        x = activation_fn(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not enable_dropout)(x)
        x = nn.Dense(
            features=self.output_dim, 
            name=f"Block{self.block_idx}_{self.layer_name}_dense2"
        )(x)
        return x


class TransformerBlock(nn.Module):
    d_model: int
    nhead: int
    dim_feedforward: int
    activation: str = 'silu'
    dropout_rate: float = 0.1
    block_idx: int = 0

    @nn.compact
    def __call__(self, x, enable_dropout: bool = True):
        # Multi-head self-attention
        attention = TransformerAttentionModule(
            num_heads=self.nhead,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            layer_name="Attention",
            block_idx=self.block_idx
        )(x, enable_dropout)

        # First residual and norm
        x = x + attention
        x = LayerNormModule(
            layer_name=f"Block{self.block_idx}_LayerNorm1"
        )(x)

        # MLP block
        mlp = TransformerMLPModule(
            features=self.dim_feedforward,
            output_dim=self.d_model,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            layer_name="MLP",
            block_idx=self.block_idx
        )(x, enable_dropout)

        # Second residual and norm
        x = x + mlp
        x = LayerNormModule(
            layer_name=f"Block{self.block_idx}_LayerNorm2"
        )(x)
        
        return x
    

class FlaxTransformer(nn.Module):
    """Transformer model"""
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    activation: str = 'silu'
    dropout_rate: float = 0.1
    classification: bool = False
    max_seq_length: int = 1024

    @nn.compact
    def __call__(self, x, enable_dropout: bool = True):
        # Embedding layers
        token_embed = EmbedModule(
            features=self.d_model,
            num_embeddings=self.vocab_size,
            layer_name="TokenEmbed"
        )(x)

        positions = jnp.arange(x.shape[1])[None, :]
        position_embedding = EmbedModule(
            features=self.d_model,
            num_embeddings=self.max_seq_length,
            layer_name="PosEmbed"
        )(positions)
        
        x = token_embed + position_embedding

        # Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                block_idx=i
            )(x, enable_dropout=enable_dropout)

        # Pooling and final layers
        activation_fn = nn.silu if self.activation == 'silu' else nn.tanh
        x = jnp.mean(x, axis=1)
        x = nn.Dense(
            features=self.dim_feedforward,
            name="FinalDense1"
        )(x)
        x = activation_fn(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not enable_dropout)(x)
        x = nn.Dense(
            features=1,
            name="FinalDense2"
        )(x)
        if self.classification:
            x = nn.softmax(x)

        return x