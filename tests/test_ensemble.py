import pytest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, "../neurobayes/")

from neurobayes import FlaxMLP, FlaxMLP2Head, EnsembleDeterministicNN

# Set seed for reproducibility
np.random.seed(42)
jax.config.update("jax_enable_x64", False)


@pytest.fixture
def regression_data():
    """Generate simple regression data"""
    n_samples = 100
    X = np.random.rand(n_samples, 2) * 10
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.5
    return X, y


@pytest.fixture
def classification_data():
    """Generate simple binary classification data"""
    n_samples = 100
    X = np.random.rand(n_samples, 2) * 10
    y = (X[:, 0] + X[:, 1] > 10).astype(int)
    return X, y


@pytest.fixture
def regression_ensemble():
    """Create a small ensemble for regression"""
    mlp = FlaxMLP(
        hidden_dims=[10, 5],
        target_dim=1,
        activation='tanh'
    )
    
    ensemble = EnsembleDeterministicNN(
        architecture=mlp,
        input_shape=2,
        loss='homoskedastic',
        num_models=3,
        learning_rate=0.01,
        init_random_seeds=[42, 43, 44]
    )
    
    return ensemble


@pytest.fixture
def classification_ensemble():
    """Create a small ensemble for classification"""
    mlp = FlaxMLP(
        hidden_dims=[10, 5],
        target_dim=2,
        activation='tanh',
        classification=True
    )
    
    ensemble = EnsembleDeterministicNN(
        architecture=mlp,
        input_shape=2,
        loss='classification',
        num_models=3,
        learning_rate=0.01,
        init_random_seeds=[42, 43, 44]
    )
    
    return ensemble


def test_ensemble_initialization():
    """Test that ensemble initializes correctly with the right number of models"""
    mlp = FlaxMLP(
        hidden_dims=[10],
        target_dim=1
    )
    
    # Test with different ensemble sizes
    for num_models in [1, 3, 5]:
        ensemble = EnsembleDeterministicNN(
            architecture=mlp,
            input_shape=2,
            num_models=num_models
        )
        
        assert len(ensemble.models) == num_models
        assert ensemble.trained is False
        
        # Check that models are unique (different random initializations)
        model_params = [model.state.params for model in ensemble.models]
        for i in range(len(model_params)):
            for j in range(i+1, len(model_params)):
                # Check that at least one parameter differs
                def flatten_dict(d, parent_key='', sep='/'):
                    items = []
                    for k, v in d.items():
                        new_key = parent_key + sep + k if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)

                flat_params_i = flatten_dict(model_params[i])
                flat_params_j = flatten_dict(model_params[j])

                for key in flat_params_i:
                    if key in flat_params_j and not np.allclose(flat_params_i[key], flat_params_j[key]):
                        break

                flat_params_i = flatten_dict(model_params[i])
                flat_params_j = flatten_dict(model_params[j])

                for key in flat_params_i:
                    if key in flat_params_j and not np.allclose(flat_params_i[key], flat_params_j[key]):
                        break
                else:
                    assert False, "Models should have different initializations"


def test_regression_training(regression_ensemble, regression_data):
    """Test training on a regression problem"""
    X, y = regression_data
    
    # Train with small number of epochs for testing
    regression_ensemble.train(X, y, epochs=5, batch_size=10)
    
    assert regression_ensemble.trained is True
    
    # Make a prediction
    X_test = np.random.rand(10, 2) * 10
    mean, var = regression_ensemble.predict(X_test)
    
    # Check shapes
    assert mean.shape == (10, 1)
    assert var.shape == (10, 1)
    
    # Test with individual predictions
    mean, var, individual_preds = regression_ensemble.predict(X_test, return_individual=True)
    
    # Check we got the right number of individual predictions
    assert len(individual_preds) == 3  # num_models
    assert all(pred.shape == (10, 1) for pred in individual_preds)


def test_classification_training(classification_ensemble, classification_data):
    """Test training on a classification problem"""
    X, y = classification_data
    
    # Train with small number of epochs for testing
    classification_ensemble.train(X, y, epochs=5, batch_size=10)
    
    assert classification_ensemble.trained is True
    
    # Make a prediction
    X_test = np.random.rand(10, 2) * 10
    probs, uncertainty = classification_ensemble.predict(X_test)
    
    # Check shapes
    assert probs.shape == (10, 2)  # (n_samples, n_classes)
    assert uncertainty.shape == (10, 2)
    
    # Check probabilities sum to 1
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    
    # Test with individual predictions
    probs, uncertainty, individual_preds = classification_ensemble.predict(X_test, return_individual=True)
    
    # Check we got the right number of individual predictions
    assert len(individual_preds) == 3  # num_models


def test_bootstrap_training(regression_ensemble, regression_data):
    """Test training with bootstrap sampling"""
    X, y = regression_data
    
    # Train with bootstrap
    regression_ensemble.train(X, y, epochs=5, batch_size=10, bootstrap=True, bootstrap_fraction=0.8)
    assert regression_ensemble.trained is True
    
    # Train without bootstrap
    regression_ensemble.train(X, y, epochs=5, batch_size=10, bootstrap=False)
    assert regression_ensemble.trained is True


def test_batch_prediction(regression_ensemble, regression_data):
    """Test batch prediction functionality"""
    X, y = regression_data
    
    # Train the ensemble
    regression_ensemble.train(X, y, epochs=5)
    
    # Create larger test set
    X_test = np.random.rand(50, 2) * 10
    
    # Test batch prediction with small batch size
    mean_batch, var_batch = regression_ensemble.predict_in_batches(X_test, batch_size=10)
    
    # Test regular prediction
    mean_full, var_full = regression_ensemble.predict(X_test)
    
    # Results should be identical
    assert np.allclose(mean_batch, mean_full)
    assert np.allclose(var_batch, var_full)


def test_heteroskedastic_ensemble():
    """Test heteroskedastic ensemble model"""
    # Create data
    n_samples = 100
    X = np.random.rand(n_samples, 2) * 10
    # Heteroskedastic noise
    noise_std = 0.1 + 0.2 * X[:, 0]
    y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * noise_std
    
    # Create heteroskedastic model
    mlp = FlaxMLP2Head(
        hidden_dims=[10, 5],
        target_dim=1,
        activation='tanh'
    )
    
    ensemble = EnsembleDeterministicNN(
        architecture=mlp,
        input_shape=2,
        loss='heteroskedastic',
        num_models=2,
        learning_rate=0.01
    )
    
    # Train
    ensemble.train(X, y, epochs=5, batch_size=10)
    
    # Predict
    X_test = np.random.rand(10, 2) * 10
    mean, var = ensemble.predict(X_test)
    
    # Check shapes
    assert mean.shape == (10, 1)
    assert var.shape == (10, 1)
    
    # Test with individual predictions
    mean, var, individual_means, individual_variances = ensemble.predict(X_test, return_individual=True)
    
    # Check shapes of individual predictions
    assert len(individual_means) == 2  # num_models
    assert len(individual_variances) == 2  # num_models
    assert all(m.shape == (10, 1) for m in individual_means)
    assert all(v.shape == (10, 1) for v in individual_variances)


def test_save_load_ensemble(regression_ensemble, regression_data):
    """Test saving and loading ensemble models"""
    X, y = regression_data
    
    # Train the ensemble
    regression_ensemble.train(X, y, epochs=3)
    
    # Make initial predictions
    X_test = np.random.rand(10, 2) * 10
    orig_mean, orig_var = regression_ensemble.predict(X_test)
    
    # Save the ensemble to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as temp_file:
        try:
            temp_path = temp_file.name
            regression_ensemble.save_ensemble(temp_path)
            
            # Load the ensemble
            loaded_ensemble = EnsembleDeterministicNN.load_ensemble(temp_path)
            
            # Check loaded ensemble properties
            assert loaded_ensemble.num_models == regression_ensemble.num_models
            assert loaded_ensemble.loss == regression_ensemble.loss
            assert loaded_ensemble.trained is True
            
            # Make predictions with loaded ensemble
            loaded_mean, loaded_var = loaded_ensemble.predict(X_test)
            
            # Results should be identical
            assert np.allclose(loaded_mean, orig_mean)
            assert np.allclose(loaded_var, orig_var)
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)