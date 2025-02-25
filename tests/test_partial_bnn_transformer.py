import sys
import pytest
import jax
import jax.numpy as jnp

sys.path.insert(0, "../neurobayes/")

from neurobayes.models.partial_bnn_transformer import PartialBayesianTransformer
from neurobayes.flax_nets.transformer import FlaxTransformer

@pytest.fixture
def setup_transformer():
    # Set fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define transformer parameters
    vocab_size = 100
    d_model = 16
    nhead = 4
    num_layers = 2
    dim_feedforward = 64
    max_seq_length = 20
    
    # Create a simple transformer model
    transformer = FlaxTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        target_dim=1,  # Regression
        max_seq_length=max_seq_length
    )
    
    # Create sample data
    n_samples = 10
    seq_length = 15
    X = jax.random.randint(key, shape=(n_samples, seq_length), minval=0, maxval=vocab_size)
    y = jnp.sum(X, axis=1).astype(float)  # Simple sum as target
    
    return {
        'transformer': transformer,
        'X': X,
        'y': y,
        'key': key,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'max_seq_length': max_seq_length
    }

@pytest.fixture
def setup_transformer_cls():
    # Set fixed random seed for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Define transformer parameters
    vocab_size = 100
    d_model = 16
    nhead = 4
    num_layers = 2
    dim_feedforward = 64
    max_seq_length = 20
    
    # Create a simple transformer model
    transformer = FlaxTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        classification=True,
        target_dim=3,  # Classification
        max_seq_length=max_seq_length
    )
    
    # Create sample data
    n_samples = 10
    seq_length = 15
    X = jax.random.randint(key, shape=(n_samples, seq_length), minval=0, maxval=vocab_size)
    y = jnp.remainder(jnp.sum(X, axis=1), 3).astype(jnp.int32)
    
    return {
        'transformer': transformer,
        'X': X,
        'y': y,
        'key': key,
        'vocab_size': vocab_size,
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': num_layers,
        'max_seq_length': max_seq_length
    }

def test_transformer_initialization_regression(setup_transformer):
    """Test initialization for regression task"""
    transformer = setup_transformer['transformer']
    
    # Initialize partial Bayesian transformer for regression
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        num_probabilistic_layers=2
    )
    
    assert pb_transformer.is_regression
    assert pb_transformer.deterministic_nn is not None
    assert pb_transformer.deterministic_weights is None
    assert len(pb_transformer.layer_configs) > 0
    
    # Check that the right number of layers are probabilistic
    prob_layers = [config for config in pb_transformer.layer_configs if config['is_probabilistic']]
    assert len(prob_layers) == 2, "Should have exactly 2 probabilistic layers"

def test_transformer_initialization_classification(setup_transformer):
    """Test initialization for classification task"""
    transformer = setup_transformer['transformer']
    
    # Initialize partial Bayesian transformer for classification
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        probabilistic_layer_names=['TokenEmbed', 'FinalDense2'],
        num_classes=3
    )
    
    assert not pb_transformer.is_regression
    assert pb_transformer.num_classes == 3
    
    # Check that only the specified layers are probabilistic
    for config in pb_transformer.layer_configs:
        if config['layer_name'] in ['TokenEmbed', 'FinalDense2']:
            assert config['is_probabilistic']
        else:
            assert not config['is_probabilistic']

def test_transformer_with_probabilistic_neurons(setup_transformer):
    """Test initialization with specific probabilistic neurons"""
    transformer = setup_transformer['transformer']
    
    # Define probabilistic neurons for specific layers
    prob_neurons = {
        'TokenEmbed': [(1, 5), (10, 15)],  # (token_idx, feature_idx)
        'Block0_Attention': {
            'query': [(0, 1, 2)],  # (input_idx, head_idx, feature_idx)
            'key': [(1, 0, 1)],
            'value': [(2, 2, 0)],
            'out': [(0, 3, 1)]
        },
        'FinalDense1': [(5, 10), (15, 20)]
    }
    
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        probabilistic_layer_names=['TokenEmbed', 'Block0_Attention', 'FinalDense1'],
        probabilistic_neurons=prob_neurons
    )
    
    # Check that probabilistic neurons were properly stored
    for config in pb_transformer.layer_configs:
        layer_name = config['layer_name']
        if layer_name in prob_neurons:
            assert config['is_probabilistic']
            assert config['probabilistic_neurons'] == prob_neurons[layer_name]
        else:
            assert config['probabilistic_neurons'] is None


def test_transformer_set_data(setup_transformer):
    """Test data preparation for transformer inputs"""
    transformer = setup_transformer['transformer']
    X = setup_transformer['X']
    y = setup_transformer['y']
    
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        num_probabilistic_layers=1
    )
    
    # Test with both X and y
    X_processed, y_processed = pb_transformer.set_data(X, y)
    
    assert X_processed.shape == X.shape, "X shape should be preserved"
    assert y_processed.ndim == 2, "y should be 2D for regression"
    assert y_processed.shape[0] == y.shape[0], "Number of samples should match"
    assert y_processed.shape[1] == 1, "y should have a feature dimension of 1"
    
    # Test with only X
    X_only = pb_transformer.set_data(X)
    assert X_only.shape == X.shape, "X shape should be preserved when y is None"
    
    # Test with classification
    pb_transformer_cls = PartialBayesianTransformer(
        transformer=transformer,
        num_probabilistic_layers=1,
        num_classes=3
    )
    
    # Create dummy classification targets
    y_cls = jnp.zeros(y.shape[0], dtype=jnp.int32)
    X_processed, y_processed = pb_transformer_cls.set_data(X, y_cls)
    
    assert X_processed.shape == X.shape, "X shape should be preserved"
    assert y_processed.ndim == 1, "y should be 1D for classification"
    assert y_processed.shape[0] == y_cls.shape[0], "Number of samples should match"

@pytest.mark.parametrize("num_warmup,num_samples,sgd_epochs", [(2, 2, 2)])
def test_transformer_minimal_fit(setup_transformer, num_warmup, num_samples, sgd_epochs):
    """Test that fit runs with minimal steps without errors"""
    transformer = setup_transformer['transformer']
    X = setup_transformer['X']
    y = setup_transformer['y']
    key = setup_transformer['key']
    
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        num_probabilistic_layers=1
    )
    
    try:
        # Run with minimal steps to keep test fast
        pb_transformer.fit(
            X=X, 
            y=y, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            sgd_epochs=sgd_epochs,
            progress_bar=False,
            rng_key=key
        )
        success = True
    except Exception as e:
        success = False
        pytest.fail(f"PartialBayesianTransformer fit function raised an exception: {e}")
    
    assert success
    assert hasattr(pb_transformer, 'mcmc'), "MCMC object should be created after fitting"
    assert pb_transformer.deterministic_weights is not None, "Deterministic weights should be set"

@pytest.mark.parametrize("method", ["magnitude"])
def test_transformer_selection_method(setup_transformer, method):
    """Test neuron selection methods for transformer"""
    transformer = setup_transformer['transformer']
    X = setup_transformer['X']
    y = setup_transformer['y']
    key = setup_transformer['key']
    
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        probabilistic_layer_names=['TokenEmbed', 'FinalDense1']
    )
    
    # Configuration for neuron selection
    select_neurons_config = {
        'method': method,
        'layer_names': ['TokenEmbed', 'FinalDense1'],
        'num_pairs_per_layer': 2
    }
    
    try:
        # Fit with neuron selection with minimal steps
        pb_transformer.fit(
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
        pytest.fail(f"PartialBayesianTransformer fit with neuron selection raised an exception: {e}")
    
    assert success
    
    # Check that probabilistic neurons were selected for the specified layers
    for config in pb_transformer.layer_configs:
        if config['layer_name'] in select_neurons_config['layer_names']:
            assert 'probabilistic_neurons' in config
            if config['probabilistic_neurons'] is not None:
                if isinstance(config['probabilistic_neurons'], dict):
                    # For attention layers
                    for component, neurons in config['probabilistic_neurons'].items():
                        assert len(neurons) <= select_neurons_config['num_pairs_per_layer']
                else:
                    # For other layers
                    assert len(config['probabilistic_neurons']) <= select_neurons_config['num_pairs_per_layer']

@pytest.mark.parametrize("num_warmup,num_samples", [(2, 2)])
def test_transformer_predict_regression(setup_transformer, num_warmup, num_samples):
    """Test regression predictions with fitted transformer model"""
    transformer = setup_transformer['transformer']
    X = setup_transformer['X']
    y = setup_transformer['y']
    key = setup_transformer['key']
    
    # Create and fit with minimal steps for regression
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        num_probabilistic_layers=1
    )
    
    pb_transformer.fit(
        X=X, 
        y=y, 
        num_warmup=num_warmup, 
        num_samples=num_samples,
        sgd_epochs=2,
        progress_bar=False,
        rng_key=key
    )
    
    # Predict on the same data
    mean, uncertainty = pb_transformer.predict(X, rng_key=key)
    
    assert mean.shape == (X.shape[0], 1), "Mean predictions should have shape (n_samples, 1)"
    assert uncertainty.shape == (X.shape[0], 1), "Uncertainties should have shape (n_samples, 1)"

@pytest.mark.parametrize("num_warmup,num_samples", [(2, 2)])
def test_transformer_predict_classification(setup_transformer_cls, num_warmup, num_samples):
    """Test classification predictions with fitted transformer model"""
    transformer = setup_transformer_cls['transformer']
    X = setup_transformer_cls['X']
    y = jnp.remainder(jnp.sum(X, axis=1), 3).astype(jnp.int32)  # Create class labels
    key = setup_transformer_cls['key']
    
    # Create and fit with minimal steps for classification
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        num_probabilistic_layers=1,
        num_classes=3
    )
    
    pb_transformer.fit(
        X=X, 
        y=y, 
        num_warmup=num_warmup, 
        num_samples=num_samples,
        sgd_epochs=2,
        progress_bar=False,
        rng_key=key
    )
    
    # Predict classes
    class_probs, uncertainty = pb_transformer.predict(X, rng_key=key)
    assert class_probs.shape == (X.shape[0], 3), "Class probabilities should have shape (n_samples, 3)"
    
    # Test predict_classes
    classes = pb_transformer.predict_classes(X, rng_key=key)
    assert classes.shape == (X.shape[0],), "Class predictions should have shape (n_samples,)"
    assert jnp.all(classes >= 0) and jnp.all(classes < 3), "Classes should be in valid range"

def test_transformer_sample_from_posterior(setup_transformer):
    """Test that sample_from_posterior works with transformer"""
    transformer = setup_transformer['transformer']
    X = setup_transformer['X']
    y = setup_transformer['y']
    key = setup_transformer['key']
    
    # Create and fit with minimal steps
    pb_transformer = PartialBayesianTransformer(
        transformer=transformer,
        num_probabilistic_layers=1
    )
    
    pb_transformer.fit(
        X=X, 
        y=y, 
        num_warmup=2, 
        num_samples=2,
        sgd_epochs=2,
        progress_bar=False,
        rng_key=key
    )
    
    # Get samples for sampling from posterior
    samples = pb_transformer.get_samples()
    
    # Sample from posterior
    try:
        predictions = pb_transformer.sample_from_posterior(
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