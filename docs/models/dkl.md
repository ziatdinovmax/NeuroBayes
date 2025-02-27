---
layout: default
title: Deep Kernel Learning
parent: Models
nav_order: 5
---

# Deep Kernel Learning

Deep Kernel Learning (DKL) combines neural networks with Gaussian processes, leveraging the strengths of both approaches.

## Overview

DKL uses a neural network to learn a feature representation that is then fed into a Gaussian Process. This approach:

- Scales better to large datasets than standard GPs
- Handles non-stationary functions and discontinuities better than GPs
- Provides well-calibrated uncertainty estimates

One of the drawbacks is potential training instabilities due to conflicting optimization dynamics between its GP and neural network components.

## Available DKL Models

- **DKL**: Deep Kernel Learning with MCMC inference
- **VIDKL**: Deep Kernel Learning with Variational Inference (faster)

## Basic Usage

```python
import neurobayes as nb
import jax.numpy as jnp

# Define neural network feature extractor
mlp = nb.FlaxMLP(hidden_dims=[64, 32, 16], target_dim=2, activation='tanh')

# MCMC-based DKL
dkl = nb.DKL(mlp, kernel=nb.kernels.RBFKernel)
dkl.fit(X_train, y_train, num_warmup=500, num_samples=1000)
mean, variance = dkl.predict(X_test)

# Variational Inference DKL (faster)
vi_dkl = nb.VIDKL(mlp, kernel=nb.kernels.RBFKernel)
vi_dkl.fit(X_train, y_train, num_steps=1000)
mean, variance = vi_dkl.predict(X_test)
```