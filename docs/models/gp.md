---
layout: default
title: Gaussian Processes
parent: Models
nav_order: 4
---

# Gaussian Processes

Gaussian Processes (GPs) are non-parametric probabilistic models that provide well-calibrated uncertainty estimates.

## Overview

GPs define a prior over functions and update this prior based on observed data. They excel at uncertainty quantification for small to medium datasets but struggle with discontinuities, non-stationarities, and scaling to large datasets.

## Available GP Models

- **GP**: Standard Gaussian Process with MCMC inference
- **VIGP**: Gaussian Process with Variational Inference (faster but approximate)

## Basic Usage

```python
import neurobayes as nb

# MCMC-based GP
gp = nb.GP(kernel=nb.kernels.MaternKernel)
gp.fit(X, y, num_warmup=500, num_samples=1000)
mean, variance = gp.predict(X_test)

# Variational Inference GP (faster)
vi_gp = nb.VIGP(kernel=nb.kernels.MaternKernel)
vi_gp.fit(X, y, num_steps=1000, step_size=1e-3)
mean, variance = vi_gp.predict(X_test)
```