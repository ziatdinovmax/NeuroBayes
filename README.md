# NeuroBayes

### *Important: This package is under active development and breaking changes are expected on weekly and sometimes even daily basis*

![NB2](https://github.com/user-attachments/assets/a78fbcac-23e5-4073-afbd-d424a8f7d2dc)

## What is it for
Machine learning, at its core, is about approximating unknown functions – mapping inputs to outputs based on observed data. In scientific and engineering applications, this often means modeling complex relationships between process parameters and target properties. Traditionally, Gaussian Processes (GPs) have been favored in these domains for their ability to provide robust uncertainty estimates. However, GPs struggle with systems featuring discontinuities and non-stationarities, common in physical science problems, as well as with high dimensional data. **NeuroBayes** bridges this gap by combining the flexibility and scalability of neural networks with the rigorous uncertainty quantification of Bayesian methods. This repository enables the use of full BNNs and partial BNNs with the No-U-Turn Sampler for intermediate size datasets, making it a powerful tool for a wide range of scientific and engineering applications.


## How to use
NeuroBayes provides two main approaches for implementing Bayesian Neural Networks: Fully Bayesian Neural Networks (BNN) and Partially Bayesian Neural Networks (PBNN). Both approaches currently support MLP and ConvNet architectures, with more architectures on the way. Here's how to use BNN and PBNN:

### Fully Bayesian Neural Nets
Fully Bayesian Neural Networks replace all constant weights in the network with probabilistic distributions. This approach provides comprehensive uncertainty quantification but may be computationally intensive for large models.

```python3
# Initialize model
model = BNN(target_dim=1, hidden_dim=[32, 16, 8, 4])
# Train model
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
# Make a prediction on full domain
posterior_mean, posterior_var = model.predict(X_domain)
```

### Partially Bayesian Neural Network
Partially Bayesian Neural Networks replace constant weights with probabilistic distributions only in a subset of the network's layers. This approach ismore computationally efficient while still providing good uncertainty estimates. By default, the deterministic part of PBNNs is trained using Maximum A Posteriori approximation, with stochastic weight averaging applied at the end of each training trajectory.

```python3
# Number of probabilistic ('Bayesian') layers
num_stochastic_layers = 2

# Initialize a determinsitc neural net
net = FlaxMLP(hidden_dims=[32, 16, 8, 4], target_dim=1)
# Intitalize and train a PBNN model
model = PartialBNN(net, num_stochastic_layers=num_stochastic_layers)
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
# Make a prediction on unmeasured points or the full domain
posterior_mean, posterior_var = model.predict(X_domain)
```

![BNN_PBNN](https://github.com/user-attachments/assets/af377d84-3a57-4d4c-9880-fe3ca931bcf9)

The obtained posterior means and variances can be used in active learning and Bayesian optimization frameworks. See example of BNN-powered active learning [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/bnn_example1d.ipynb) and example of PBNN-powered active learning [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/pbnn_example1d.ipynb).
    
### Heteroskedastic noise
By default, we assume constant observation noise across all inputs. However, this assumption often doesn't hold in real-world datasets which may exhibit input-dependent levels of noise. NeuroBayes offers heteroskedastic BNNs that can capture varying levels of noise in different regions of the data, allowing for more accurate uncertainty quantification.

The usage of a heteroskedastic BNN is straightforward and follows the same pattern as the standard BNN models:

For fully Bayesian heteroskedastic NN:
```python3
# Initialize HeteroskedasticBNN model
model = HeteroskedasticBNN(target_dim=1)
# Train
model.fit(X_measured, y_measured, num_warmup=2000, num_samples=2000)
# Make a prediction
posterior_mean, posterior_var = model.predict(X_domain)
```

For partially Bayesian heteroskedastic NN:
```python3
# Initialize model architecture
hidden_dims = [64, 32, 16, 8, 8]
net = FlaxMLP2Head(hidden_dims, 1)
# Pass it to HeteroskedasticPartialBNN module and perform training
model = HeteroskedasticPartialBNN(net, num_stochastic_layers=2)
model.fit(X_measured, y_measured, sgd_epochs=5000, sgd_lr=5e-3, num_warmup=1000, num_samples=1000)
# Make a prediction
posterior_mean, posterior_var = model.predict(X_domain)
```

![hsk](https://github.com/user-attachments/assets/5a619361-74c0-4d03-9b1a-4aa995f1c540)

See example [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/heteroskedastic.ipynb).

### Pre-trained priors
NeuroBayes extends the concept of partial BNNs to leverage pre-existing knowledge or simulations, allowing for more informed priors in Bayesian Neural Networks. This approach is particularly useful when you have theoretical models or simulations that can guide the learning process for experimental data.
The process involves two main steps:

Pre-training a deterministic neural network on theoretical or simulated data.
Using the weights from this pre-trained network to center the prior distributions for a Fully Bayesian Neural Network (FBNN) or Partially Bayesian Neural Network (PBNN).

Here's how to implement this approach:
First, fit a deterministic NN to theoretical data:
```python3
import neurobayes as nb
from neurobayes.flax_nets import FlaxMLP

hidden_dims = [64, 32, 16, 8]
net = FlaxMLP(hidden_dims=hidden_dims, target_dim=1)
detnn = nb.DeterministicNN(net, input_shape=(1,), learning_rate=5e-3, map=True, sigma=nb.utils.calculate_sigma(X1))
detnn.train(X1, y1, epochs=5000, batch_size=None)
```

Note: In practice, you should use proper train-test-validation splits for robust model development.

Next, train a BNN on experimental data, using the pre-trained weights to set theory-informed BNN priors:

```python3
model = nb.BNN(target_dim=1, hidden_dim=hidden_dims)
model.fit(
    X2, y2, num_warmup=1000, num_samples=1000, num_chains=1,
    pretrained_priors=detnn.state.params  # use trained weights to set priors for BNN
)
```

Make a prediction as ususal
```python3
posterior_mean, posterior_var = model.predict(X_test)
```

This approach allows you to incorporate domain knowledge or theoretical models into your Bayesian Neural Network, potentially leading to better generalization and more accurate uncertainty estimates, especially in cases where experimental data is limited. 

![pretrained_priors](https://github.com/user-attachments/assets/33f80877-4a5c-46d2-ba5d-ee540418e21b)

See examples [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/pretrained_priors.ipynb) (full BNN) and [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/pretrained_priors_partial.ipynb) (Partial BNN).
