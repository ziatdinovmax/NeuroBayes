# NeuroBayes

## What is it for
![actove_learning](https://github.com/user-attachments/assets/203fb3ab-cfbe-4882-9cab-ec8269359110)


Active learning optimizes the exploration of large parameter spaces by strategically selecting experiments or simulations, reducing resource use and accelerating discovery. A crucial element is a probabilistic surrogate model, typically a Gaussian Process (GP), which approximates the relationship between control parameters and a target property. However, GPs struggle with systems featuring discontinuities and non-stationarities, common in physical science problems. Fully Bayesian Neural Networks (FBNNs) offer a promising alternative by treating all network weights probabilistically and using advanced Markov Chain Monte Carlo techniques for for direct sampling from the posterior distribution. This approach provides reliable predictive distributions, crucial for decision-making under uncertainty. Although FBNNs are traditionally seen as computationally expensive for big data, many physical science problems involve small data sets, making FBNNs feasible. For more complex parameter spaces, Partially Bayesian Neural Networks (PBNNs) can be used, where only some neurons are Bayesian. This repository enables the use of FBNNs and PBNNs with the No-U-Turn Sampler in active and transfer learning tasks for small and intermediate data volumes, demonstrating their potential in physical science applications.

## How to use
### Fully Bayesian Neural Nets
Initialize a simulator
```python3
import neurobayes as nb

(x_start, x_stop), fn = genfunc.nonstationary2()

# Generate ground truth data
X_domain = np.linspace(x_start, x_stop, 500)
y_true = fn(X_domain)

# Create a measurement function
measure = lambda x: fn(x) + np.random.normal(0, 0.02, size=len(x))

# Generate initial dataset
X_measured = np.random.uniform(x_start, x_stop, 50)
y_measured = measure(X_measured)
```

Run a single shot Bayesian neural network
```python3
# Initialize model
model = BNN(target_dim=1)
# Train model
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
# Make a prediction on full domain
posterior_mean, posterior_var = model.predict(X_domain)
```

Run active learning with Bayesian neural network
```python3
for step in range(exploration_steps):
    # Intitalize and train model
    model = BNN(target_dim=1)
    model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
    # Make a prediction on unmeasured points or the full domain
    posterior_mean, posterior_var = model.predict(X_domain)
    # Select next point to evaluate
    next_point_idx = posterior_var.argmax(0)
    X_next = X_domain[next_point_idx]
    # Evaluate function in this point
    y_next = measure(X_next)
    # Update training and test set
    X_measured = np.append(X_measured, X_next[None])
    y_measured = np.append(y_measured, y_next)
```
See full active learning example [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/bnn_example1d.ipynb).
    
### Partially Bayesian Neural Nets
PBNNs follow a similar approach, with the key difference being that a deterministic model must first be defined, along with the specification of stochastic gradient descent parameters. By default, PBNNs are trained using Maximum A Posteriori approximation, with stochastic weight averaging applied at the end of each training trajectory. This allows PBNNs to balance computational efficiency with uncertainty quantification by only treating a subset of neurons probabilistically.
```python3
sgd_epochs = 2000
sgd_lr = 5e-3
num_stochastic_layers = 1

# Initialize a determinsitc neural net
net = FlaxMLP(hidden_dims=[32, 16, 8, 8], output_dim=1)

# Run active learning
for step in range(exploration_steps):
    print('step {}'.format(step))
    # Intitalize and train model
    model = PartialBNN(net, num_stochastic_layers=num_stochastic_layers)
    model.fit(X_measured, y_measured, sgd_epochs=sgd_epochs, sgd_lr=sgd_lr, sgd_batch_size=16, num_warmup=1000, num_samples=1000)
    # Make a prediction on unmeasured points or the full domain
    posterior_mean, posterior_var = model.predict(X_domain)
    # Select next point to evaluate
    next_point_idx = posterior_var.argmax(0)
    X_next = X_domain[next_point_idx]
    # Evaluate function in this point
    y_next = measure(X_next)
    # Update training and test set
    X_measured = np.append(X_measured, X_next[None])
    y_measured = np.append(y_measured, y_next)
```
See full example [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/pbnn_example1d.ipynb).

### Other applications
#### Heteroskedastic noise
By default, we assume constant observation noise across all inputs. However, this assumption often doesn't hold in real-world datasets which may exhibit input-dependent levels of noise. NeuroBayes offers heteroskedastic BNNs that can capture varying levels of noise in different regions of the data, allowing for more accurate uncertainty quantification.

The usage of a heteroskedastic BNN is straightforward and follows the same pattern as the standard BNN models:

For fully Bayesian heteroskedastic BNN:
```python3
# Initialize HeteroskedasticBNN model
model = HeteroskedasticBNN(target_dim=1)
# Train
model.fit(X_measured, y_measured, num_warmup=2000, num_samples=2000)
# Make a prediction
posterior_mean, posterior_var = model.predict(X_domain)
```

For partially Bayesian heteroskedastic BNN:
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

#### Pre-trained priors
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
net = FlaxMLP(hidden_dims=hidden_dims, output_dim=1)
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
