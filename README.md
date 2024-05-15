# NeuroBayes

## What is it for

Active learning optimizes the exploration of large parameter spaces by strategically selecting which experiments or simulations to conduct, thus reducing resource consumption and potentially accelerating scientific discovery. A key component of this approach is a probabilistic surrogate model, typically a Gaussian Process (GP), which approximates an unknown functional relationship between control parameters and a target property. However, conventional GPs often struggle when applied to systems with discontinuities and non-stationarities, prompting the exploration of alternative models. This limitation becomes particularly relevant in physical science problems, which are often characterized by abrupt transitions between different system states and rapid changes in physical property behavior. Fully Bayesian Neural Networks (FBNNs) serve as a promising substitute, treating all neural network weights probabilistically and leveraging advanced Markov Chain Monte Carlo techniques for direct sampling from the posterior distribution. This approach enables FBNNs to provide reliable predictive distributions, crucial for making informed decisions under uncertainty in the active learning setting. Although traditionally considered too computationally expensive for 'big data' applications, many physical sciences problems involve small amounts of data in relatively low-dimensional parameter spaces. This repository was designed to assess the suitability and performance of FBNNs with the No-U-Turn Sampler for active learning tasks in the 'small data' regime, highlighting their potential to enhance predictive accuracy and reliability on test functions relevant to problems in physical sciences.

## How to use
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

Run a single shot Gaussian process
```python3
# Initialize model
model = GP(input_dim=1, kernel=nb.kernels.MaternKernel)
# Train model
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
# Make a prediction on full domain
posterior_mean, posterior_var = model.predict(X_domain)
```

Run a single shot Bayesian neural network
```python3
# Initialize model
model = BNN(input_dim=1, output_dim=1)
# Train model
model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
# Make a prediction on full domain
posterior_mean, posterior_var = model.predict(X_domain)
```

Run active learning with Bayesian neural network
```python3
for step in range(exploration_steps):
    # Intitalize and train model
    model = BNN(1, 1)
    model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
    # make a prediction on unmeasured points or the full domain
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
See full active learning example [here](https://github.com/ziatdinovmax/NeuroBayes/blob/main/example1d.ipynb).
    
    


