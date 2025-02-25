import sys
import pytest
import jax.numpy as jnp
import jax
import numpyro.distributions as dist


sys.path.insert(0, "../neurobayes/")

from neurobayes.models.bnn import BNN
from neurobayes.flax_nets import FlaxMLP

@pytest.fixture
def setup_bnn_regression():
    # Set fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create simple dataset
    n_samples = 30
    n_features = 2
    X = jax.random.normal(key, (n_samples, n_features))
    y = jnp.sin(X[:, 0]) + 0.1 * jax.random.normal(key, (n_samples,))
    
    # Create a simple MLP
    mlp = FlaxMLP(
        hidden_dims=[10, 5],
        target_dim=1,
        activation='tanh'
    )
    
    # Create BNN model
    bnn = BNN(
        architecture=mlp,
        num_classes=None,  # Regression task
        noise_prior=dist.HalfNormal(0.1)
    )
    
    return {'X': X, 'y': y, 'bnn': bnn, 'key': key}

@pytest.fixture
def setup_bnn_classification():
    # Set fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create simple dataset
    n_samples = 30
    n_features = 2
    n_classes = 3
    X = jax.random.normal(key, (n_samples, n_features))
    logits = jnp.stack([
        jnp.sin(X[:, 0]),
        jnp.cos(X[:, 1]),
        jnp.sin(X[:, 0] + X[:, 1])
    ], axis=1)
    probs = jax.nn.softmax(logits, axis=1)
    y = jax.random.categorical(key, probs)
    
    # Create a simple MLP
    mlp = FlaxMLP(
        hidden_dims=[10, 5],
        target_dim=n_classes,
        activation='tanh',
        classification=True
    )
    
    # Create BNN model
    bnn = BNN(
        architecture=mlp,
        num_classes=n_classes
    )
    
    return {'X': X, 'y': y, 'bnn': bnn, 'key': key, 'n_classes': n_classes}

def test_bnn_initialization_regression(setup_bnn_regression):
    bnn = setup_bnn_regression['bnn']
    
    assert bnn.is_regression
    assert bnn.nn is not None
    assert bnn.noise_prior is not None
    assert bnn.pretrained_priors is None
    assert bnn.num_classes is None

def test_bnn_initialization_classification(setup_bnn_classification):
    bnn = setup_bnn_classification['bnn']
    n_classes = setup_bnn_classification['n_classes']
    
    assert not bnn.is_regression
    assert bnn.nn is not None
    assert bnn.noise_prior is None
    assert bnn.pretrained_priors is None
    assert bnn.num_classes == n_classes

def test_bnn_set_data_regression(setup_bnn_regression):
    """Test that set_data properly formats input data for regression"""
    bnn = setup_bnn_regression['bnn']
    X = setup_bnn_regression['X']
    y = setup_bnn_regression['y']
    
    # Test with both X and y
    X_processed, y_processed = bnn.set_data(X, y)
    
    assert X_processed.shape == X.shape, "X shape should be preserved"
    assert y_processed.ndim == 2, "y should be 2D for regression"
    assert y_processed.shape[0] == y.shape[0], "Number of samples should match"
    assert y_processed.shape[1] == 1, "y should have a feature dimension of 1"
    
    # Test with only X
    X_only = bnn.set_data(X)
    assert X_only.shape == X.shape, "X shape should be preserved when y is None"

def test_bnn_set_data_classification(setup_bnn_classification):
    """Test that set_data properly formats input data for classification"""
    bnn = setup_bnn_classification['bnn']
    X = setup_bnn_classification['X']
    y = setup_bnn_classification['y']
    
    # Test with both X and y
    X_processed, y_processed = bnn.set_data(X, y)
    
    assert X_processed.shape == X.shape, "X shape should be preserved"
    assert y_processed.ndim == 1, "y should remain 1D for classification"
    assert y_processed.shape[0] == y.shape[0], "Number of samples should match"
    
    # Test with only X
    X_only = bnn.set_data(X)
    assert X_only.shape == X.shape, "X shape should be preserved when y is None"

@pytest.mark.parametrize("num_warmup,num_samples", [(5, 5)])
def test_bnn_fit_regression(setup_bnn_regression, num_warmup, num_samples):
    """Test that fit runs without error with minimal MCMC steps for regression"""
    bnn = setup_bnn_regression['bnn']
    X = setup_bnn_regression['X']
    y = setup_bnn_regression['y']
    key = setup_bnn_regression['key']
    
    try:
        # Run with minimal steps to keep test fast
        bnn.fit(
            X=X, 
            y=y, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            progress_bar=False,
            rng_key=key
        )
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"BNN fit function raised an exception: {e}")
    
    assert success
    assert hasattr(bnn, 'mcmc'), "MCMC object should be created after fitting"

@pytest.mark.parametrize("num_warmup,num_samples", [(5, 5)])
def test_bnn_fit_classification(setup_bnn_classification, num_warmup, num_samples):
    """Test that fit runs without error with minimal MCMC steps for classification"""
    bnn = setup_bnn_classification['bnn']
    X = setup_bnn_classification['X']
    y = setup_bnn_classification['y']
    key = setup_bnn_classification['key']
    
    try:
        # Run with minimal steps to keep test fast
        bnn.fit(
            X=X, 
            y=y, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            progress_bar=False,
            rng_key=key
        )
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"BNN fit function raised an exception: {e}")
    
    assert success
    assert hasattr(bnn, 'mcmc'), "MCMC object should be created after fitting"

@pytest.mark.parametrize("num_warmup,num_samples", [(5, 5)])
def test_bnn_predict_regression(setup_bnn_regression, num_warmup, num_samples):
    """Test that predict returns the correct output shapes for regression"""
    bnn = setup_bnn_regression['bnn']
    X = setup_bnn_regression['X']
    y = setup_bnn_regression['y']
    key = setup_bnn_regression['key']
    
    # Fit the model with minimal steps
    bnn.fit(
        X=X, 
        y=y, 
        num_warmup=num_warmup, 
        num_samples=num_samples,
        progress_bar=False,
        rng_key=key
    )
    
    # Predict on the same data for simplicity
    mean, uncertainty = bnn.predict(X, rng_key=key)
    
    assert mean.shape == (X.shape[0], 1), "Mean predictions should have shape (n_samples, 1)"
    assert uncertainty.shape == (X.shape[0], 1), "Uncertainties should have shape (n_samples, 1)"

@pytest.mark.parametrize("num_warmup,num_samples", [(5, 5)])
def test_bnn_predict_classification(setup_bnn_classification, num_warmup, num_samples):
    """Test that predict returns the correct output shapes for classification"""
    bnn = setup_bnn_classification['bnn']
    X = setup_bnn_classification['X']
    y = setup_bnn_classification['y']
    key = setup_bnn_classification['key']
    n_classes = setup_bnn_classification['n_classes']
    
    # Fit the model with minimal steps
    bnn.fit(
        X=X, 
        y=y, 
        num_warmup=num_warmup, 
        num_samples=num_samples,
        progress_bar=False,
        rng_key=key
    )
    
    # Predict on the same data for simplicity
    mean, uncertainty = bnn.predict(X, rng_key=key)
    
    assert mean.shape == (X.shape[0], n_classes), "Mean predictions should have shape (n_samples, n_classes)"
    assert uncertainty.shape == (X.shape[0],), "Uncertainties should have shape (n_samples,)"
    
    # Test predict_classes
    class_predictions = bnn.predict_classes(X, rng_key=key)
    assert class_predictions.shape == (X.shape[0],), "Class predictions should have shape (n_samples,)"
    assert jnp.all(class_predictions >= 0) and jnp.all(class_predictions < n_classes), "Class predictions should be valid class indices"

@pytest.mark.parametrize("num_warmup,num_samples", [(5, 5)])
def test_bnn_get_samples(setup_bnn_regression, num_warmup, num_samples):
    """Test that get_samples returns the correct samples after fitting"""
    bnn = setup_bnn_regression['bnn']
    X = setup_bnn_regression['X']
    y = setup_bnn_regression['y']
    key = setup_bnn_regression['key']
    
    # Fit the model with minimal steps
    bnn.fit(
        X=X, 
        y=y, 
        num_warmup=num_warmup, 
        num_samples=num_samples,
        progress_bar=False,
        rng_key=key
    )
    
    # Get samples
    samples = bnn.get_samples()
    
    assert isinstance(samples, dict), "Samples should be returned as a dictionary"
    assert len(samples) > 0, "Samples dictionary should not be empty"
    assert "sig" in samples, "Samples should include observational noise 'sig'"
    
    # Check that the first parameter shape is correct
    first_param_name = list(samples.keys())[0]
    first_param = samples[first_param_name]
    assert first_param.shape[0] == num_samples, f"First dimension of samples should be num_samples ({num_samples})"