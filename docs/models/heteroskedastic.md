---
layout: default
title: Heteroskedastic Models
parent: Models
nav_order: 3
---

# Heteroskedastic Models

Heteroskedastic models account for input-dependent noise, allowing uncertainty to vary across the input space. NeuroBayes provides multiple approaches to handling heteroskedastic noise.

## Overview

In many real-world scenarios, especially scientific experiments, noise levels aren't constant across inputs:

- Measurement precision may vary with experimental conditions
- Some regions of the input space may have intrinsically higher uncertainty
- Domain experts often have prior knowledge about how noise varies with inputs

NeuroBayes offers three approaches to heteroskedastic modeling:

1. **Neural Network with Two Heads**: One head predicts the mean, another predicts the variance
2. **Noise Model-Based Approach**: Incorporates domain knowledge about noise behavior
3. **Partial Bayesian Heteroskedastic Models**: Efficiency-optimized version of the two-headed approach

## Neural Network with Two Heads

This approach uses neural networks with two output heads to learn both the mean and variance functions:

```python
import neurobayes as nb
import jax.numpy as jnp

# Create a two-headed MLP architecture
mlp_2head = nb.FlaxMLP2Head(
    hidden_dims=[32, 16],
    target_dim=1,
    activation='tanh'
)

# Create heteroskedastic BNN model
het_bnn = nb.HeteroskedasticBNN(mlp_2head)

# Train the model
het_bnn.fit(X, y, 
           num_warmup=500,
           num_samples=1000)

# Make predictions
mean, variance = het_bnn.predict(X_test)

# Separately access the predicted noise
noise_mean, noise_var = het_bnn.predict_noise(X_test)
```

## Noise Model-Based Approach
When domain knowledge about noise behavior is available, you can directly incorporate this knowledge using the ```VarianceModelHeteroskedasticBNN```:

```python
import neurobayes as nb

# Define a noise model based on domain knowledge
def noise_model_fn(x, a, b):
    """Noise that grows exponentially with x"""
    return a * jnp.exp(b * x)

# Convert to parameter-based function
noise_model = nb.utils.set_fn(noise_model_fn)

# Define prior distributions over the noise model parameters
noise_model_prior = nb.priors.auto_normal_priors(noise_model_fn)

# Create standard neural network for the mean function
architecture = nb.FlaxMLP(hidden_dims=[32, 16], target_dim=1)

# Create noise model-based heteroskedastic BNN
model = nb.VarianceModelHeteroskedasticBNN(
    architecture,
    noise_model,
    noise_model_prior
)

# Train the model
model.fit(X, y, num_warmup=1000, num_samples=1000)

# Make predictions
mean, variance = model.predict(X_test)

# Get the inferred noise
noise_mean, noise_variance = model.predict_noise(X_test)
```
## Partial Bayesian Heteroskedastic Models
For better computational efficiency, you can use a partial Bayesian approach:
```python
import neurobayes as nb

# Create a two-headed CNN architecture
cnn_2head = nb.FlexConvNet2Head(
    input_dim=2,           # 2D input data
    conv_layers=[32, 64],
    fc_layers=[128, 64],
    target_dim=1,
    activation='tanh'
)

# Create heteroskedastic partial BNN
het_pbn = nb.HeteroskedasticPartialBNN(
    cnn_2head,
    probabilistic_layer_names=['MeanHead', 'VarianceHead']  # Make only the output heads Bayesian
)

# Train model
het_pbn.fit(X, y,
           sgd_epochs=200,     # Deterministic pre-training
           num_warmup=500,     # MCMC warmup steps
           num_samples=1000)   # MCMC samples

# Make predictions
mean, variance = het_pbn.predict(X_test)
```

### Applications
Heteroskedastic models are particularly valuable for:

- Scientific measurements with variable precision
- Systems with known physical noise dependencies
- Sensor data with environment-dependent noise levels