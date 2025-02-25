import sys
import pytest
import jax.numpy as jnp
import jax
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../neurobayes/")


from neurobayes.models.partial_bnn import PartialBNN
from neurobayes.models.partial_bnn_mlp import PartialBayesianMLP
from neurobayes.models.partial_bnn_conv import PartialBayesianConvNet
from neurobayes.flax_nets.mlp import FlaxMLP
from neurobayes.flax_nets.convnet import FlaxConvNet

@pytest.fixture
def setup_partial_bnn_mlp():
    # Set fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create simple dataset
    n_samples = 30
    n_features = 2
    X = jax.random.normal(key, (n_samples, n_features))
    y = jnp.sin(X[:, 0]) + 0.1 * jax.random.normal(key, (n_samples,))
    
    # Create a simple MLP
    mlp = FlaxMLP(
        hidden_dims=[10, 8, 5],
        target_dim=1,
        activation='tanh'
    )
    
    # Create PartialBNN model with the last layer as probabilistic
    partial_bnn = PartialBNN(
        architecture=mlp,
        num_probabilistic_layers=1
    )
    
    return {'X': X, 'y': y, 'partial_bnn': partial_bnn, 'key': key, 'mlp': mlp}

@pytest.fixture
def setup_partial_bnn_convnet():
    # Set fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create simple dataset for convnet (assume 2D data)
    n_samples = 20
    height, width, channels = 8, 8, 1
    X = jax.random.normal(key, (n_samples, height, width, channels))
    y = jnp.sum(X, axis=(1, 2, 3))  # Simple target
    
    # Create a simple ConvNet
    convnet = FlaxConvNet(
        input_dim=2,  # 2D images
        conv_layers=[16, 8],
        fc_layers=[16, 8],
        target_dim=1,
        kernel_size=3,
        activation='tanh'
    )
    
    # Create PartialBNN model with probabilistic layer names specified
    partial_bnn = PartialBNN(
        architecture=convnet,
        probabilistic_layer_names=['Dense1', 'Dense2']
    )
    
    return {'X': X, 'y': y, 'partial_bnn': partial_bnn, 'key': key, 'convnet': convnet}

def test_partial_bnn_initialization_mlp(setup_partial_bnn_mlp):
    partial_bnn = setup_partial_bnn_mlp['partial_bnn']
    
    assert partial_bnn.architecture is not None
    assert partial_bnn._model is not None
    assert isinstance(partial_bnn._model, PartialBayesianMLP)
    assert partial_bnn.is_regression  # Default is regression

def test_partial_bnn_initialization_convnet(setup_partial_bnn_convnet):
    partial_bnn = setup_partial_bnn_convnet['partial_bnn']
    
    assert partial_bnn.architecture is not None
    assert partial_bnn._model is not None
    assert isinstance(partial_bnn._model, PartialBayesianConvNet)
    assert partial_bnn.is_regression  # Default is regression

def test_partial_bnn_with_prob_neurons():
    """Test initialization with probabilistic neurons specified"""
    # Set fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create a simple MLP
    mlp = FlaxMLP(
        hidden_dims=[10, 5],
        target_dim=1,
        activation='tanh'
    )
    
    # Specify probabilistic neurons
    prob_neurons = {
        'Dense0': [(0, 1), (2, 3)],  # These connections in Dense0 will be Bayesian
        'Dense1': [(1, 0), (4, 2)]   # These connections in Dense1 will be Bayesian
    }
    
    # Create PartialBNN model with specific probabilistic neurons
    partial_bnn = PartialBNN(
        architecture=mlp,
        probabilistic_layer_names=['Dense0', 'Dense1'],
        probabilistic_neurons=prob_neurons
    )
    
    assert partial_bnn._model is not None
    # Check that probabilistic neurons were properly passed to the underlying model
    for config in partial_bnn._model.layer_configs:
        layer_name = config['layer_name']
        if layer_name in prob_neurons:
            assert 'probabilistic_neurons' in config
            assert config['probabilistic_neurons'] == prob_neurons[layer_name]

def test_partial_bnn_invalid_architecture():
    """Test that PartialBNN raises error with unsupported architecture"""
    # Create a non-supported architecture
    class UnsupportedArchitecture:
        pass
    
    with pytest.raises(ValueError):
        PartialBNN(architecture=UnsupportedArchitecture())

@pytest.mark.parametrize("num_warmup,num_samples,sgd_epochs", [(5, 5, 2)])
def test_partial_bnn_fit_mlp(setup_partial_bnn_mlp, num_warmup, num_samples, sgd_epochs):
    """Test that fit runs without error for PartialBNN with MLP"""
    partial_bnn = setup_partial_bnn_mlp['partial_bnn']
    X = setup_partial_bnn_mlp['X']
    y = setup_partial_bnn_mlp['y']
    key = setup_partial_bnn_mlp['key']
    
    try:
        # First fit with deterministic training
        partial_bnn.fit(
            X=X, 
            y=y, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            sgd_epochs=sgd_epochs,  # Very few epochs for testing
            progress_bar=False,
            rng_key=key
        )
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"PartialBNN fit function raised an exception: {e}")
    
    assert success
    assert hasattr(partial_bnn._model, 'mcmc'), "MCMC object should be created after fitting"
    assert partial_bnn._model.deterministic_weights is not None, "Deterministic weights should be set"

@pytest.mark.parametrize("select_method", ["magnitude"])
def test_partial_bnn_fit_with_neuron_selection(setup_partial_bnn_mlp, select_method):
    """Test that fit runs with neuron selection"""
    partial_bnn = setup_partial_bnn_mlp['partial_bnn']
    X = setup_partial_bnn_mlp['X']
    y = setup_partial_bnn_mlp['y']
    key = setup_partial_bnn_mlp['key']
    
    # Configuration for neuron selection
    select_neurons_config = {
        'method': select_method,
        'layer_names': ['Dense1', 'Dense2'],
        'num_pairs_per_layer': 2
    }
    
    try:
        # Fit with neuron selection
        partial_bnn.fit(
            X=X, 
            y=y, 
            num_warmup=2, 
            num_samples=2,
            sgd_epochs=2,
            progress_bar=False,
            rng_key=key,
            select_neurons_config=select_neurons_config
        )
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"PartialBNN fit with neuron selection raised an exception: {e}")
    
    assert success
    
    # Check that probabilistic neurons were selected for the specified layers
    for config in partial_bnn._model.layer_configs:
        if config['layer_name'] in select_neurons_config['layer_names']:
            assert 'probabilistic_neurons' in config
            if config['probabilistic_neurons'] is not None:
                assert len(config['probabilistic_neurons']) <= select_neurons_config['num_pairs_per_layer']

@pytest.mark.parametrize("num_warmup,num_samples,sgd_epochs", [(5, 5, 2)])
def test_partial_bnn_predict(setup_partial_bnn_mlp, num_warmup, num_samples, sgd_epochs):
    """Test that predict returns the correct output shapes"""
    partial_bnn = setup_partial_bnn_mlp['partial_bnn']
    X = setup_partial_bnn_mlp['X']
    y = setup_partial_bnn_mlp['y']
    key = setup_partial_bnn_mlp['key']
    
    # Fit the model with minimal steps
    partial_bnn.fit(
        X=X, 
        y=y, 
        num_warmup=num_warmup, 
        num_samples=num_samples,
        sgd_epochs=sgd_epochs,
        progress_bar=False,
        rng_key=key
    )
    
    # Predict on the same data for simplicity
    mean, uncertainty = partial_bnn.predict(X)
    
    assert mean.shape == (X.shape[0], 1), "Mean predictions should have shape (n_samples, 1)"
    assert uncertainty.shape == (X.shape[0], 1), "Uncertainties should have shape (n_samples, 1)"

@pytest.mark.parametrize("num_warmup,num_samples,sgd_epochs", [(5, 5, 2)])
def test_partial_bnn_get_samples(setup_partial_bnn_mlp, num_warmup, num_samples, sgd_epochs):
    """Test that get_samples returns the correct samples after fitting"""
    partial_bnn = setup_partial_bnn_mlp['partial_bnn']
    X = setup_partial_bnn_mlp['X']
    y = setup_partial_bnn_mlp['y']
    key = setup_partial_bnn_mlp['key']
    
    # Fit the model with minimal steps
    partial_bnn.fit(
        X=X, 
        y=y, 
        num_warmup=num_warmup, 
        num_samples=num_samples,
        sgd_epochs=sgd_epochs,
        progress_bar=False,
        rng_key=key
    )
    
    # Get samples
    samples = partial_bnn.get_samples()
    
    assert isinstance(samples, dict), "Samples should be returned as a dictionary"
    assert len(samples) > 0, "Samples dictionary should not be empty"
    assert "sig" in samples, "Samples should include observational noise 'sig'"

def test_partial_bnn_set_data(setup_partial_bnn_mlp):
    """Test that set_data properly formats input data"""
    partial_bnn = setup_partial_bnn_mlp['partial_bnn']
    X = setup_partial_bnn_mlp['X']
    y = setup_partial_bnn_mlp['y']
    
    # Test with both X and y
    X_processed, y_processed = partial_bnn.set_data(X, y)
    
    assert X_processed.shape == X.shape, "X shape should be preserved"
    assert y_processed.ndim == 2, "y should be 2D for regression"
    assert y_processed.shape[0] == y.shape[0], "Number of samples should match"
    assert y_processed.shape[1] == 1, "y should have a feature dimension of 1"
    
    # Test with only X
    X_only = partial_bnn.set_data(X)
    assert X_only.shape == X.shape, "X shape should be preserved when y is None"

def test_partial_bnn_sample_from_posterior(setup_partial_bnn_mlp):
    """Test that sample_from_posterior runs without error"""
    partial_bnn = setup_partial_bnn_mlp['partial_bnn']
    X = setup_partial_bnn_mlp['X']
    y = setup_partial_bnn_mlp['y']
    key = setup_partial_bnn_mlp['key']
    
    # Fit with minimal steps
    partial_bnn.fit(
        X=X, 
        y=y, 
        num_warmup=2, 
        num_samples=2,
        sgd_epochs=2,
        progress_bar=False,
        rng_key=key
    )
    
    # Get samples for sampling from posterior
    samples = partial_bnn.get_samples()
    
    # Sample from posterior
    try:
        predictions = partial_bnn.sample_from_posterior(
            rng_key=key,
            X_new=X,
            samples=samples,
            return_sites=["mu", "y"]
        )
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"sample_from_posterior raised an exception: {e}")
    
    assert success
    assert "mu" in predictions, "Predictions should include 'mu' site"
    assert "y" in predictions, "Predictions should include 'y' site"