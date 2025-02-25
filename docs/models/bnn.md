---
layout: default
title: Bayesian Neural Networks
parent: Models
nav_order: 1
---

# Bayesian Neural Networks

Bayesian Neural Networks (BNNs) place prior distributions over all network weights, enabling full uncertainty quantification but at high computational cost.

## Overview

Unlike traditional neural networks that learn point estimates for weights, BNNs learn a distribution over all weights. While this approach provides comprehensive uncertainty quantification, it becomes computationally expensive for networks with many parameters. For most practical applications, [Partially Bayesian Neural Networks](partial_bnn.md) offer a better balance between uncertainty estimation and computational efficiency.

## Basic Usage

```python
import neurobayes as nb
import jax.numpy as jnp

# Create neural network architecture
mlp = nb.FlaxMLP(
    hidden_dims=[32, 16],  # Keep architecture small for BNNs
    target_dim=1,
    activation='tanh'
)

# Create BNN model
bnn = nb.BNN(mlp)

# Train the model (this may take considerable time)
bnn.fit(X, y, 
        num_warmup=500,
        num_samples=1000)

# Make predictions
mean, variance = bnn.predict(X_test)
```