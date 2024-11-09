# NeuroBayes

> [!IMPORTANT]
This package is actively in development, and breaking changes may occur frequently
> 

![NB_logo2](https://github.com/user-attachments/assets/36e1ea05-83cb-41a9-a52e-cf73335dd499)

## What is it for
Machine learning, at its core, is about approximating unknown functions – mapping inputs to outputs based on observed data. In scientific and engineering applications, this often means modeling complex relationships between process parameters and target properties. Traditionally, Gaussian Processes (GPs) have been favored in these domains for their ability to provide robust uncertainty estimates. However, GPs struggle with systems featuring discontinuities and non-stationarities, common in physical science problems, as well as with high dimensional data. **NeuroBayes** bridges this gap by combining the flexibility and scalability of neural networks with the rigorous uncertainty quantification of Bayesian methods. This repository enables the use of full BNNs and partial BNNs with the No-U-Turn Sampler for intermediate size datasets, making it a powerful tool for a wide range of scientific and engineering applications.


## How to use
NeuroBayes provides two main approaches for implementing Bayesian Neural Networks: Fully Bayesian Neural Networks (BNN) and Partially Bayesian Neural Networks (PBNN). Both approaches currently support MLP and ConvNet architectures, with more architectures on the way. Here's how to use BNN and PBNN:

![NN_types](https://github.com/user-attachments/assets/9b58d9ec-cb7f-49de-aa58-e75990b08b83)


### Fully Bayesian Neural Nets
Fully Bayesian Neural Networks replace all constant weights in the network with probabilistic distributions. This approach provides comprehensive uncertainty quantification but may be computationally intensive for large models.

```python3
import neurobayes as nb

# Initialize NN architecture
architecture = nb.FlaxMLP(hidden_dims = [64, 32, 16, 8], target_dim=1)

# Initialize Bayesian model
model = nb.BNN(architecture)
# Train model
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
# Make a prediction on full domain
posterior_mean, posterior_var = model.predict(X_domain)
```

### Partially Bayesian Neural Network
Partially Bayesian Neural Networks replace constant weights with probabilistic distributions only in a subset of the network's layers. This approach ismore computationally efficient while still providing good uncertainty estimates. By default, the deterministic part of PBNNs is trained using Maximum A Posteriori approximation, with stochastic weight averaging applied at the end of each training trajectory.

```python3
# Number of probabilistic ('Bayesian') layers
num_probabilistic_layers = 2 # two last learnable layers + output layer

# Intitalize a PBNN model
model = nb.PartialBNN(architecture, num_probabilistic_layers=num_probabilistic_layers)
# Train
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
# Make a prediction on unmeasured points or the full domain
posterior_mean, posterior_var = model.predict(X_domain)
```

Alternatively, we can specify directly the names of the layers we want to be probabilistic

```python3
# Specify the names of probabilistic layers (output layer, 'Dense4', needs to be specified explicitly)
probabilistic_layer_names = ['Dense2', 'Dense3', 'Dense4']

# Intitalize and train a PBNN model
model = nb.PartialBNN(architecture, probabilistic_layer_names=probabilistic_layer_names)
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
```

![BNN_PBNN](https://github.com/user-attachments/assets/8281b071-4f05-4432-8e23-babcaaad6b5d)

The obtained posterior means and variances can be used in active learning and Bayesian optimization frameworks. See example of BNN-powered active learning [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/bnn_example1d.ipynb) and example of PBNN-powered active learning [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/pbnn_example1d.ipynb).
    
### Heteroskedastic noise
By default, we assume constant observation noise across all inputs. However, this assumption often doesn't hold in real-world datasets which may exhibit input-dependent levels of noise. NeuroBayes offers heteroskedastic BNNs that can capture varying levels of noise in different regions of the data, allowing for more accurate uncertainty quantification.

The usage of a heteroskedastic BNN is straightforward and follows the same pattern as the standard BNN models:

For fully Bayesian heteroskedastic NN:
```python3
# Initialize HeteroskedasticBNN model
model = nb.HeteroskedasticBNN(architecture)
# Train
model.fit(X_measured, y_measured, num_warmup=2000, num_samples=2000)
# Make a prediction
posterior_mean, posterior_var = model.predict(X_domain)
```

For partially Bayesian heteroskedastic NN:
```python3
# Initialize and train
model = nb.HeteroskedasticPartialBNN(architecture, num_probabilistic_layers=2)
model.fit(X_measured, y_measured, sgd_epochs=5000, sgd_lr=5e-3, num_warmup=1000, num_samples=1000)
# Make a prediction
posterior_mean, posterior_var = model.predict(X_domain)
```

![hsk](https://github.com/user-attachments/assets/5a619361-74c0-4d03-9b1a-4aa995f1c540)

See example [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/heteroskedastic.ipynb).

### Pre-trained priors

![Transfer_learning](https://github.com/user-attachments/assets/b3e2d7da-7bb1-4919-8140-899795ac042d)

NeuroBayes extends the concept of partial BNNs to leverage pre-existing knowledge or simulations, allowing for more informed priors in Bayesian Neural Networks. This approach is particularly useful when you have theoretical models or simulations that can guide the learning process for experimental data.
The process involves two main steps:

Pre-training a deterministic neural network on theoretical or simulated data.
Using the weights from this pre-trained network to center the prior distributions for a Fully Bayesian Neural Network (FBNN) or Partially Bayesian Neural Network (PBNN).

Here's how to implement this approach:
First, fit a deterministic NN to theoretical data:
```python3
hidden_dims = [64, 32, 16, 8]
net = nb.FlaxMLP(hidden_dims, target_dim=1)
detnn = nb.DeterministicNN(net, input_shape=(1,), learning_rate=5e-3, map=True, sigma=nb.utils.calculate_sigma(X1))
detnn.train(X1, y1, epochs=5000, batch_size=None)
```

Note: In practice, you should use proper train-test-validation splits for robust model development.

Next, train a BNN on experimental data, using the pre-trained weights to set theory-informed BNN priors:

```python3
model = nb.BNN(
    net,
    pretrained_priors=detnn.state.params  # use pre-trained weights to set priors for BNN
) 
model.fit(X2, y2, num_warmup=1000, num_samples=1000, num_chains=1)
```

Make a prediction as ususal
```python3
posterior_mean, posterior_var = model.predict(X_test)
```

This approach allows you to incorporate domain knowledge or theoretical models into your Bayesian Neural Network, potentially leading to better generalization and more accurate uncertainty estimates, especially in cases where experimental data is limited. 

![pretrained_priors](https://github.com/user-attachments/assets/33f80877-4a5c-46d2-ba5d-ee540418e21b)

See example [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/pretrained_priors.ipynb).

### Comparison with GP-based methods
NeuroBayes provides implementations of Gaussian Process (GP) and Deep Kernel Learning (DKL) models for comparison with BNN approaches. These implementations support both fully Bayesian and variational inference methods.

Gaussian Process:

```python3
# Specify kernel
kernel = nb.kernels.MaternKernel
# Initialize GP model
model = nb.GP(kernel)
# Train the same way as BNN
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
# Make a prediction the same way as with BNN
posterior_mean, posterior_var = model.predict(X_domain)
```

Deep Kernel Learning:

```python3
# Set a number of latent dimensions
latent_dim = 2
# Initialize NN architecture for the feature extractor part of the DKL
architecture = nb.FlaxMLP(hidden_dims = [64, 32, 16, 8], target_dim=latent_dim)
# Specify kernel for the GP part of DKL
kernel = nb.kernels.MaternKernel

# Initialize DKL model
model = nb.DKL(net, kernel)
# Train and make a prediction the same way as with GP and BNN
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
posterior_mean, posterior_var = model.predict(X_domain)
```
The training and prediction interface is consistent across all model types (BNN, PBNN, GP, and DKL) in NeuroBayes, making it easy to compare different approaches for your specific use case.

### Surrogate model recommender

I made a simple tool to guide the selection of a surrogate model (between BNN, PBNN, DKL, and GP) in the active learning setting:

https://surrogate-model-selector.vercel.app/

Note: It reflects typical behaviors based on active learning requirements like training time per iteration, but model performance can vary significantly based on implementation details, hyperparameter tuning, and specific problem characteristics.

## Installation
To install NeuroBayes, use either
```bash
git clone https://github.com/ziatdinovmax/NeuroBayes.git
cd NeuroBayes
pip install -e .
```

or

```bash
pip install git+https://github.com/ziatdinovmax/NeuroBayes.git
```
