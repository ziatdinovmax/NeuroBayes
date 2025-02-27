---
layout: default
title: FlaxMLP
parent: Networks
grand_parent: API Reference
---

<a id="neurobayes.flax_nets.mlp"></a>

# neurobayes.flax\_nets.mlp

<a id="neurobayes.flax_nets.mlp.FlaxMLP"></a>

## FlaxMLP Objects

```python
class FlaxMLP(nn.Module)
```

<a id="neurobayes.flax_nets.mlp.FlaxMLP.__call__"></a>

#### \_\_call\_\_

```python
@nn.compact
def __call__(x: jnp.ndarray, enable_dropout: bool = True) -> jnp.ndarray
```

Forward pass of the MLP

<a id="neurobayes.flax_nets.mlp.FlaxMLP2Head"></a>

## FlaxMLP2Head Objects

```python
class FlaxMLP2Head(nn.Module)
```

<a id="neurobayes.flax_nets.mlp.FlaxMLP2Head.__call__"></a>

#### \_\_call\_\_

```python
@nn.compact
def __call__(x: jnp.ndarray,
             enable_dropout: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]
```

Forward pass of the 2-headed MLP


