Now, let's create `convnet.md`:

```markdown
---
layout: default
title: ConvNet Architectures
parent: Networks
nav_order: 2
---

# Convolutional Neural Network (ConvNet) Architectures

NeuroBayes provides convolutional neural network implementations that support 1D, 2D, and 3D convolutional operations for different types of data.

## Available ConvNet Architectures

NeuroBayes offers two types of ConvNet architectures:

1. **FlaxConvNet**: Standard ConvNet with a single output head
2. **FlaxConvNet2Head**: ConvNet with two output heads (for heteroskedastic models)

## FlaxConvNet

The standard ConvNet architecture can handle 1D, 2D, or 3D input data:

```python
import neurobayes as nb

# Create a ConvNet for 2D image data
cnn = nb.FlaxConvNet(
    input_dim=2,              # Dimensionality (1=1D, 2=2D, 3=3D)
    conv_layers=[32, 64],     # Filters in convolutional layers
    fc_layers=[128, 64],      # Hidden dimensions for fully connected layers
    target_dim=1,             # Output dimension (number of classes for classification)
    activation='tanh',        # Activation function
    kernel_size=3,            # Kernel size for convolutions
    conv_dropout=0.1,         # Dropout rate for conv layers
    hidden_dropout=0.1,       # Dropout rate for FC hidden layers
    classification=False      # Set to True for classification
)
```

## FlaxConvNet2Head
The two-headed ConvNet is designed for heteroskedastic models:
```python
import neurobayes as nb

# Create a two-headed ConvNet for heteroskedastic modeling of 2D data
cnn2head = nb.FlaxConvNet2Head(
    input_dim=2,              # Dimensionality (1=1D, 2=2D, 3=3D)
    conv_layers=[32, 64],     # Filters in convolutional layers
    fc_layers=[128, 64],      # Hidden dimensions for fully connected layers
    target_dim=1,             # Output dimension for each head
    activation='tanh',        # Activation function
    kernel_size=3,            # Kernel size for convolutions
    conv_dropout=0.1,         # Dropout rate for conv layers
    hidden_dropout=0.1        # Dropout rate for FC hidden layers
)
```

## Usage with Different Models
ConvNets can be used with various model types in NeuroBayes:

### With BNN
```python
cnn = nb.FlaxConvNet(input_dim=2, conv_layers=[8, 32, 64], fc_layers=[128, 16], target_dim=1)
model = nb.BNN(cnn)
```
### With Partial BNN
```python
cnn = nb.FlaxConvNet(input_dim=2, conv_layers=[8, 32, 64], fc_layers=[128, 16], target_dim=1)
model = nb.PartialBNN(cnn, num_probabilistic_layers=1)
```
### With Heteroskedastic BNN
```python
cnn2head = nb.FlaxConvNet2Head(input_dim=2, conv_layers=[8, 32, 64], fc_layers=[128, 16], target_dim=1)
model = nb.HeteroskedasticBNN(cnn2head)
```

### With Deep Kernel Learning
```python
# For DKL, the target_dim is the dimension of the GP input space
cnn = nb.FlaxConvNet(input_dim=2, conv_layers=[8, 32, 64], fc_layers=[128, 16], target_dim=2)
model = nb.DKL(cnn, kernel=nb.kernels.RBFKernel)
```