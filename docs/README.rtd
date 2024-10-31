NeuroBayes
==========

.. warning::
   This package is actively in development, and breaking changes may occur frequently.

.. image:: _static/NB_logo2.png
   :alt: NeuroBayes Logo

Overview
--------

Machine learning, at its core, is about approximating unknown functions – mapping inputs to outputs based on observed data. In scientific and engineering applications, this often means modeling complex relationships between process parameters and target properties. 

**NeuroBayes** bridges the gap between traditional Gaussian Processes (GPs) and neural networks by combining the flexibility and scalability of neural networks with the rigorous uncertainty quantification of Bayesian methods. This package enables the use of full BNNs and partial BNNs with the No-U-Turn Sampler for intermediate size datasets, making it a powerful tool for scientific and engineering applications.

Key Features
-----------

- Full Bayesian Neural Networks (BNN)
- Partial Bayesian Neural Networks (PBNN)
- Support for MLP and ConvNet architectures
- Heteroskedastic noise modeling
- Pre-trained priors integration

.. image:: _static/NN_types.png
   :alt: Neural Network Types

Installation
-----------

.. code-block:: bash

   pip install neurobayes

Usage Guide
----------

Fully Bayesian Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fully Bayesian Neural Networks replace all constant weights in the network with probabilistic distributions:

.. code-block:: python

   import neurobayes as nb

   # Initialize NN architecture
   architecture = nb.FlaxMLP(hidden_dims=[64, 32, 16, 8], target_dim=1)

   # Initialize Bayesian model
   model = nb.BNN(architecture)
   
   # Train model
   model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
   
   # Make a prediction on full domain
   posterior_mean, posterior_var = model.predict(X_domain)

Partially Bayesian Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Partially Bayesian Neural Networks apply probabilistic distributions to only a subset of layers:

.. code-block:: python

   # Number of probabilistic ('Bayesian') layers
   num_probabilistic_layers = 2  # two last learnable layers + output layer

   # Initialize a PBNN model
   model = nb.PartialBNN(architecture, num_probabilistic_layers=num_probabilistic_layers)
   
   # Train
   model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)
   
   # Make a prediction
   posterior_mean, posterior_var = model.predict(X_domain)

Alternative layer specification:

.. code-block:: python

   # Specify the names of probabilistic layers
   probabilistic_layer_names = ['Dense2', 'Dense3', 'Dense4']

   # Initialize and train a PBNN model
   model = nb.PartialBNN(architecture, probabilistic_layer_names=probabilistic_layer_names)
   model.fit(X_measured, y_measured, num_warmup=1000, num_samples=1000)

.. image:: _static/BNN_PBNN.png
   :alt: BNN vs PBNN Comparison

Heteroskedastic Noise Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For datasets with input-dependent noise levels:

.. code-block:: python

   # Fully Bayesian heteroskedastic NN
   model = nb.HeteroskedasticBNN(architecture)
   model.fit(X_measured, y_measured, num_warmup=2000, num_samples=2000)
   posterior_mean, posterior_var = model.predict(X_domain)

   # Partially Bayesian heteroskedastic NN
   model = nb.HeteroskedasticPartialBNN(architecture, num_probabilistic_layers=2)
   model.fit(X_measured, y_measured, sgd_epochs=5000, sgd_lr=5e-3, 
            num_warmup=1000, num_samples=1000)
   posterior_mean, posterior_var = model.predict(X_domain)

.. image:: _static/hsk.png
   :alt: Heteroskedastic Noise Example

Pre-trained Priors
~~~~~~~~~~~~~~~~

Leverage existing knowledge through pre-trained models:

.. code-block:: python

   # Train deterministic NN on theoretical data
   hidden_dims = [64, 32, 16, 8]
   net = nb.FlaxMLP(hidden_dims, target_dim=1)
   detnn = nb.DeterministicNN(net, input_shape=(1,), learning_rate=5e-3, 
                             map=True, sigma=nb.utils.calculate_sigma(X1))
   detnn.train(X1, y1, epochs=5000, batch_size=None)

   # Use pre-trained weights for BNN priors
   model = nb.BNN(net, pretrained_priors=detnn.state.params)
   model.fit(X2, y2, num_warmup=1000, num_samples=1000, num_chains=1)
   posterior_mean, posterior_var = model.predict(X_test)

.. image:: _static/Transfer_learning.png
   :alt: Transfer Learning
   
.. image:: _static/pretrained_priors.png
   :alt: Pre-trained Priors Example

Examples
--------

- `BNN-powered active learning example <https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/bnn_example1d.ipynb>`_
- `PBNN-powered active learning example <https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/pbnn_example1d.ipynb>`_
- `Heteroskedastic modeling example <https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/heteroskedastic.ipynb>`_
- `Pre-trained priors example <https://github.com/ziatdinovmax/NeuroBayes/blob/main/examples/pretrained_priors.ipynb>`_
