---
layout: default
title: FlaxConvNet
parent: Networks
grand_parent: API Reference
---

<a id="neurobayes.flax_nets.convnet"></a>

# neurobayes.flax\_nets.convnet

<a id="neurobayes.flax_nets.convnet.get_conv_and_pool_ops"></a>

#### get\_conv\_and\_pool\_ops

```python
def get_conv_and_pool_ops(
        input_dim: int,
        kernel_size: Union[int, Tuple[int, ...]]) -> Tuple[Callable, Callable]
```

Returns appropriate convolution and pooling operations based on input dimension.

**Arguments**:

- `input_dim` _int_ - Dimension of input data (1, 2, or 3)
- `kernel_size` _int or tuple_ - Size of the convolution kernel
  

**Returns**:

- `tuple` - (conv_op, pool_op) - Convolution and pooling operations


