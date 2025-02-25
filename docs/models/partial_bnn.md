---
layout: default
title: Partially Bayesian Neural Networks
parent: Models
nav_order: 2
---

# Partially Bayesian Neural Networks

Partially Bayesian Neural Networks (PBNNs) enhance computational efficiency by applying Bayesian inference selectively to specific layers or weights, while treating other parts deterministically. This approach offers a balance between uncertainty quantification and computational tractability.

## PartialBNN Class

NeuroBayes provides a unified `PartialBNN` class that supports various neural network architectures (MLP, ConvNet, Transformer) with a consistent API.

```python
import neurobayes as nb

# Create neural network architecture
mlp = nb.FlaxMLP(
    hidden_dims=[64, 32, 16],  # Hidden layer dimensions
    target_dim=1,              # Output dimension (1 for regression)
    activation='tanh'          # Activation function
)

# Create PartialBNN model with only the last layer being Bayesian
pbn = nb.PartialBNN(mlp, num_probabilistic_layers=1)
```

## Specifying Bayesian Components

There are several ways to define which parts of the neural network should be treated as Bayesian:
### 1. Number of Probabilistic Layers
Specify how many layers from the end of the network should be Bayesian:
```python
# Make the last 2 layers Bayesian
pbn = nb.PartialBNN(mlp, num_probabilistic_layers=2)
```
### 2. Named Layers
Explicitly name the layers that should be Bayesian:
```python
# Make specific layers Bayesian
pbn = nb.PartialBNN(mlp, probabilistic_layer_names=['Dense1', 'Dense3'])
```
### 3. Specific Neurons/Weights
For even more fine-grained control, specify which individual weights should be Bayesian:
```python
# Define probabilistic layer names
prob_layers = ['Dense1', 'Dense3']
# Define specific weights to be Bayesian
prob_neurons = {
    'Dense0': [(0, 5), (2, 10)],  # Make weights connecting input 0 → output 5 and 2→10 Bayesian
    'Dense2': [(1, 0), (3, 2)]    # Similar for another layer
}

pbn = nb.PartialBNN(mlp, probabilistic_layer_names=prob_layers, probabilistic_neurons=prob_neurons)
```

## Automatic Selection of Bayesian Weights
NeuroBayes provides methods to automatically select which weights should be treated as Bayesian based on various criteria:
```python
# Train with automatic selection of Bayesian weights
pbn.fit(X, y,
        sgd_epochs=100,    # Deterministic pre-training
        num_warmup=500,
        num_samples=1000,
        # Configuration for selecting Bayesian weights
        select_neurons_config={
            'method': 'variance',  # Method for selection
            'layer_names': ['Dense0', 'Dense2'],  # Layers to partially Bayesianize
            'num_pairs_per_layer': 10  # Number of weight pairs per layer
        })
```
### Selection Methods
NeuroBayes supports several methods for automatically selecting important weights:

- variance: Select weights with highest variance during training
- gradient: Select weights with largest gradients
- magnitude: Select weights with largest magnitude
- clustering: Group weights and select representatives from each cluster

## Classification with Partial BNNs

For classification tasks, specify the `num_classes` parameter when creating the model:

```python
import neurobayes as nb

# Create architecture for a 10-class classification problem
mlp = nb.FlaxMLP(
    hidden_dims=[64, 32],
    target_dim=10,         # 10 output units for 10 classes
    activation='tanh'
)

# Create model for classification
model = nb.PartialBNN(mlp, num_classes=10, num_probabilistic_layers=1)

# Train model
model.fit(X_train, y_train,  # y_train should contain class indices
          sgd_epochs=200,
          num_warmup=500,
          num_samples=1000)

# Predict class probabilities and uncertainty
probs, uncertainty = model.predict(X_test)

# Predict class labels
predicted_classes = model.predict_classes(X_test)
```

## Manual two-stage training
Sometimes you may need to have more control over the deterministic pre-training phase. In such case, you can explicitly separate the pre-training and Bayesian conversion phases:

```python
import neurobayes as nb

# 1. Train a deterministic model with extensive experimentation
mlp = nb.FlaxMLP(hidden_dims=[64, 32, 16, 8], target_dim=1, activation='silu')
det_model = nb.DeterministicNN(
    mlp, 
    input_shape=(X.shape[-1],),
    learning_rate=0.001,
    map=True,                # Use MAP training (with regularization)
    sigma=1.0               # Prior sigma for regularization
)

# Experiment with different training strategies
det_model.train(X, y, epochs=500, batch_size=64)

# Evaluate deterministic model performance
y_pred = det_model.predict(X_val)
print(f"Validation MSE: {(nb.utils.mse(y_pred, y_val)}")

# Try different architectures, hyperparameters, etc.
# ...

# When satisfied, get the final weights
pretrained_weights = det_model.get_params()

# 2. Create Partial BNN with these carefully tuned weights
pbn = nb.PartialBNN(
    mlp,
    deterministic_weights=pretrained_weights,
    probabilistic_layer_names=['Dense1']
)

# 3. Train with just MCMC 
# (automatically skips deterministic pre-training if pretrained weights are provided)
pbn.fit(X, y, 
        num_warmup=500, 
        num_samples=1000)
```

The manual approach is particularly valuable when:

- You need to extensively fine-tune the deterministic model before Bayesian treatment
- You want to try different architectures, optimizers, or regularization strategies
- The deterministic pre-training requires careful validation, early stopping, or learning rate scheduling
- You're working with complex datasets where finding a good deterministic model is challenging in itself
- You want to experiment with different weight selection strategies based on the deterministic model's behavior
- You need to reuse models trained in other frameworks or incorporate domain-specific pre-trained models
