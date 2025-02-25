---
layout: default
title: DeterministicNN
parent: Networks
nav_order: 4
---

# DeterministicNN

The `DeterministicNN` class provides a convenient wrapper for training deterministic neural networks in NeuroBayes, which can be used independently or as a first step before Bayesian conversion.

## Overview

`DeterministicNN` handles standard (non-Bayesian) neural network training using JAX and Flax. It provides optimization, regularization, and model evaluation capabilities. This class is particularly useful for:

1. Training standalone deterministic neural networks
2. Pre-training networks before converting to Bayesian models
3. Analyzing neural network behavior before applying Bayesian treatment

## Basic Usage

```python
import neurobayes as nb
import jax.numpy as jnp

# Create a neural network architecture
mlp = nb.FlaxMLP(hidden_dims=[64, 32, 16], target_dim=1, activation='tanh')

# Create a deterministic neural network
det_nn = nb.DeterministicNN(
    architecture=mlp,
    input_shape=(10,),           # Input feature dimension
    loss='homoskedastic',        # 'homoskedastic', 'heteroskedastic', or 'classification'
    learning_rate=0.01,          # Learning rate for optimizer
    map=True,                    # Use maximum a posteriori estimation (regularization)
    sigma=1.0,                   # Prior sigma for MAP regularization
    collect_gradients=False      # Whether to collect gradients during training
)

# Train the model
det_nn.train(
    X_train,
    y_train,
    epochs=200,
    batch_size=32
)

# Make predictions
predictions = det_nn.predict(X_test)
```

## Stochastic Weight Averaging
DeterministicNN supports Stochastic Weight Averaging (SWA), which can improve model generalization and provide more robust priors by averaging weights from different points in the training trajectory:
```python
# Configure SWA
swa_config = {
    'schedule': 'linear',    # Options: 'constant', 'linear', 'cyclic'
    'start_pct': 0.80,         # When to start collecting weights (fraction of total epochs)
    'swa_lr': 0.001            # Learning rate during SWA period
}

det_nn = nb.DeterministicNN(
    architecture=mlp,
    input_shape=(10,),
    learning_rate=0.01,
    swa_config=swa_config
)
```
