import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../neurobayes/")

from neurobayes.models.partial_bnn import PartialBNN
from neurobayes.flax_nets.mlp import FlaxMLP
from neurobayes.flax_nets.deterministic_nn import DeterministicNN


def get_dummy_data(feature_dim=1, target_dim=1, squeezed=False, n_points=8):
    X = onp.random.randn(n_points, feature_dim)
    y = onp.random.randn(X.shape[0], target_dim)
    if squeezed:
        return X.squeeze(), y.squeeze()
    return X, y

@pytest.mark.parametrize("n_targets", [1, 2])
@pytest.mark.parametrize("n_features", [1, 4])
@pytest.mark.parametrize("squeezed", [True, False])
def test_pbnn_fit_all(n_features, n_targets, squeezed):
    X, y = get_dummy_data(n_features, n_targets, squeezed)
    net = FlaxMLP(hidden_dims=[4, 2], target_dim=n_targets)
    pbnn = PartialBNN(net, num_probabilistic_layers=1)
    pbnn.fit(X, y, num_warmup=10, num_samples=10)
    assert pbnn.mcmc is not None

@pytest.mark.parametrize("n_targets", [1, 2])
@pytest.mark.parametrize("n_features", [1, 4])
@pytest.mark.parametrize("squeezed", [True, False])
def test_pbnn_fit_pretrained(n_features, n_targets, squeezed):
    X, y = get_dummy_data(n_features, n_targets, squeezed)
    net = FlaxMLP(hidden_dims=[4, 2], target_dim=n_targets)
    detnn_model = DeterministicNN(net, n_features)
    detnn_model.train(X, y, epochs=10)
    pbnn = PartialBNN(
        detnn_model.model, detnn_model.state.params,
        num_probabilistic_layers=1)
    pbnn.fit(X, y, num_warmup=10, num_samples=10)
    assert pbnn.mcmc is not None


@pytest.mark.parametrize("n_targets", [1, 2])
@pytest.mark.parametrize("n_features", [1, 4])
def test_bnn_fit_predict(n_features, n_targets):
    X, y = get_dummy_data(n_features, n_targets)
    X_test, _ = get_dummy_data(n_features, n_targets, n_points=20)
    net = FlaxMLP(hidden_dims=[4, 2], target_dim=n_targets)
    pbnn = PartialBNN(net, num_probabilistic_layers=1)
    pbnn.fit(X, y, num_warmup=10, num_samples=10)
    pmean, pvar = pbnn.predict(X_test)
    assert_equal(pmean.shape, (len(X_test), n_targets))
    assert_equal(pmean.shape, pvar.shape)