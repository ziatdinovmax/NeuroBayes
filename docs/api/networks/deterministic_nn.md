---
layout: default
title: DeterministicNN
parent: Networks
grand_parent: API Reference
---

<a id="neurobayes.flax_nets.deterministic_nn"></a>

# neurobayes.flax\_nets.deterministic\_nn

<a id="neurobayes.flax_nets.deterministic_nn.DeterministicNN"></a>

## DeterministicNN Objects

```python
class DeterministicNN()
```

**Arguments**:

- `architecture` - a Flax model
- `input_shape` - (n_features,) or (*dims, n_channels)
- `loss` - type of loss, 'homoskedastic' (default) or 'heteroskedastic'
- `learning_rate` - Initial learning rate
- `map` - Uses maximum a posteriori approximation
- `sigma` - Standard deviation for Gaussian prior
- `swa_config` - Dictionary configuring the Stochastic Weight Averaging behavior:
  - 'schedule': Type of learning rate schedule and weight collection strategy.
  Options:
  - 'constant': Uses constant learning rate, collects weights after start_pct
  - 'linear': Linearly decays learning rate to swa_lr, then collects weights
  - 'cyclic': Cycles learning rate between swa_lr and a peak value, collecting at cycle starts
  - 'start_pct': When to start SWA as fraction of total epochs (default: 0.95)
  - 'swa_lr': Final/SWA learning rate (default: same as initial learning_rate)
  Additional parameters based on schedule type:
  - For 'linear' and 'cyclic':
  - 'decay_fraction': Fraction of total epochs for decay period (default: 0.05)
  - For 'cyclic' only:
  - 'cycle_length': Number of epochs per cycle (required)

<a id="neurobayes.flax_nets.deterministic_nn.DeterministicNN.train_step"></a>

#### train\_step

```python
@partial(jax.jit, static_argnums=(0, ))
def train_step(state, inputs, targets)
```

JIT-compiled training step

<a id="neurobayes.flax_nets.deterministic_nn.DeterministicNN.update_learning_rate"></a>

#### update\_learning\_rate

```python
def update_learning_rate(learning_rate: float)
```

Update the optimizer with a new learning rate

<a id="neurobayes.flax_nets.deterministic_nn.DeterministicNN.average_params"></a>

#### average\_params

```python
def average_params() -> Dict
```

Average model parameters, excluding normalization layers

<a id="neurobayes.flax_nets.deterministic_nn.DeterministicNN.reset_swa"></a>

#### reset\_swa

```python
def reset_swa()
```

Reset SWA collections


