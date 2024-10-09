import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../neurobayes/")

from neurobayes.models.dkl import DKL
from neurobayes.models.kernels import RBFKernel


def get_dummy_data(feature_dim=1, target_dim=1, squeezed=False, n_points=8):
    X = onp.random.randn(n_points, feature_dim)
    y = onp.random.randn(X.shape[0], target_dim)
    if squeezed:
        return X.squeeze(), y.squeeze()
    return X, y

@pytest.mark.parametrize("n_latent", [1, 2])
@pytest.mark.parametrize("n_features", [1, 4])
@pytest.mark.parametrize("squeezed", [True, False])
def test_dkl_fit(n_features, n_latent, squeezed):
    X, y = get_dummy_data(n_features, 1, squeezed)
    dkl = DKL(n_features, n_latent, RBFKernel, hidden_dim=[4, 2])
    dkl.fit(X, y, num_warmup=10, num_samples=10)
    assert dkl.mcmc is not None


@pytest.mark.parametrize("n_features", [1, 4])
def test_dkl_fit_predict(n_features):
    X, y = get_dummy_data(n_features)
    X_test, _ = get_dummy_data(n_features, n_points=20)
    dkl = DKL(n_features, 2, RBFKernel, hidden_dim=[4, 2])
    dkl.fit(X, y, num_warmup=10, num_samples=10)
    pmean, pvar = dkl.predict(X_test)
    assert_equal(pmean.shape, (len(X_test),))
    assert_equal(pmean.shape, pvar.shape)