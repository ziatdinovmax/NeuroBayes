---
layout: default
title: MLP Architectures
parent: Networks
nav_order: 1
---

# Multi-Layer Perceptron (MLP) Architectures

NeuroBayes provides flexible MLP (Multi-Layer Perceptron) implementations built on Flax, which can be used with various Bayesian and non-Bayesian models.

## Available MLP Architectures

NeuroBayes offers two types of MLP architectures:

1. **FlaxMLP**: Standard MLP with a single output head
2. **FlaxMLP2Head**: MLP with two output heads (for heteroskedastic models)

## FlaxMLP

The standard MLP architecture is suitable for both regression and classification tasks:

```python
import neurobayes as nb

# Create an MLP for regression
mlp = nb.FlaxMLP(
    hidden_dims=[64, 32],     # Hidden layer dimensions
    target_dim=1,             # Output dimension (number of classes for classification)
    activation='tanh',        # Activation function ('tanh' or 'silu')
    hidden_dropout=0.1,       # Dropout rate for hidden layers
    classification=False      # Set to True for classification tasks
)
```

## FlaxMLP2Head
The two-headed MLP is designed for heteroskedastic models, where one head predicts the mean and the other predicts the variance:
```python
import neurobayes as nb

# Create a two-headed MLP for heteroskedastic modeling
mlp2head = nb.FlaxMLP2Head(
    hidden_dims=[64, 32],     # Hidden layer dimensions
    target_dim=1,             # Output dimension for each head
    activation='tanh',        # Activation function
    hidden_dropout=0.1,       # Dropout rate for hidden layers
)
```

## Usage with Different Models
MLPs can be used with various model types in NeuroBayes:

### With BNN
```python
mlp = nb.FlaxMLP(hidden_dims=[64, 32], target_dim=1)
model = nb.BNN(mlp)
```

### With Partial BNN
```python
mlp = nb.FlaxMLP(hidden_dims=[64, 32], target_dim=1)
model = nb.PartialBNN(mlp, num_probabilistic_layers=1)
```

### With Heteroskedastic BNN
```python
mlp2head = nb.FlaxMLP2Head(hidden_dims=[64, 32], target_dim=1)
model = nb.HeteroskedasticBNN(mlp2head)
```

### With Deep Kernel Learning
For DKL, the target_dim is the dimension of the GP input space
```python
mlp = nb.FlaxMLP(hidden_dims=[64, 32], target_dim=2)
model = nb.DKL(mlp, kernel=nb.kernels.RBFKernel)
```
