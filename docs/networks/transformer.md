---
layout: default
title: Transformer Architectures
parent: Networks
nav_order: 3
---

# Transformer Architectures

NeuroBayes provides Transformer-based neural network implementations for sequential data such as text, SMILES strings, or time series.

## FlaxTransformer

The Transformer architecture in NeuroBayes follows the standard encoder-only design with self-attention mechanisms:

```python
import neurobayes as nb

# Create a Transformer for sequence data
transformer = nb.FlaxTransformer(
    vocab_size=10000,         # Size of vocabulary for embeddings
    d_model=256,              # Model dimension
    nhead=8,                  # Number of attention heads
    num_layers=4,             # Number of transformer blocks
    dim_feedforward=1024,     # Hidden dimension in MLP blocks
    activation='silu',        # Activation function
    dropout_rate=0.1,         # Dropout rate
    target_dim=1,             # Output dimension
    classification=False,     # Set to True for classification
    max_seq_length=1024       # Maximum sequence length
)
```

## Internal Structure
The FlaxTransformer architecture consists of:

- Embedding Layers:

    - Token embeddings that map token indices to vectors
    - Position embeddings that encode position information


- Transformer Blocks:
    - Multi-head self-attention
    - Layer normalization
    - Feedforward networks


- Output Head:

    - Global pooling (mean)
    - Final feedforward layers

Transformers are too computationally intensive to use them with full BNNs so the primary focus is Partially Bayesian transformer NNs:

```python
transformer = nb.FlaxTransformer(vocab_size=10000, d_model=128, nhead=4, num_layers=2, target_dim=1)
model = nb.PartialBNN(
    transformer,
    probabilistic_layer_names=['TokenEmbed', 'Block0_Attention', 'FinalDense2']
)
```

## Comparing Different Bayesian Configurations
When using Partial Bayesian Transformers, you can make different components probabilistic to achieve various trade-offs:

### Token Embedding Layer (TokenEmbed):

Good for capturing uncertainty in the representation of input tokens
Particularly useful when input tokens have variable reliability


### Attention Layers (Block{N}_Attention):

Captures uncertainty in how tokens relate to each other
Effective for sequence data where relationships between tokens are uncertain


### Final Dense Layers (FinalDense1, FinalDense2):

Focuses uncertainty on the output mapping
More efficient when you're confident about representation but uncertain about final predictions

## Performance Considerations

For many transformer models, even doing Bayesian inference over a single layer/module can be too computationally expensive. Hence, it is a good idea to make only a small subset of weights probabilistic. In addition, one may consider data thinning procedure where only a subset of initial training data is used during the Bayesian conversion phase. Use the ```nb.print_layer_configs()``` function to help identify layer names for selective Bayesian treatment.

### Selective Weight Bayesianization

```python
# Select a small subset of weights for Bayesian treatment
prob_indices = nb.select_bayesian_components(
    det_model,
    layer_names=['TokenEmbed'],
    method='variance',         # Available methods: 'variance', 'gradient', 'magnitude', 'clustering'
    num_pairs_per_layer=32    # Use a small number relative to total parameters
)
```

### Data Thinning
For large datasets, you can employ data thinning when training Bayesian components:
```python
# Train on a subset of data to speed up computation
pbnn_model.fit(
    train_data["input_ids"][::10],   # Use only every 10th example
    train_data['solubility'][::10],
    num_warmup=1000,
    num_samples=1000
)
```
