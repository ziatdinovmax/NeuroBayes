import sys
import pytest
import numpy as onp
import jax.numpy as jnp
import jax
import numpyro
from numpy.testing import assert_equal, assert_array_equal

sys.path.insert(0, "../neurobayes/")

from neurobayes.models.partial_dkl import PartialDKL
from neurobayes.models.kernels import RBFKernel
from neurobayes.flax_nets.mlp import FlaxMLP


def get_dummy_data(feature_dim=1, target_dim=1, squeezed=False, n_points=8):
    X = onp.random.randn(n_points, feature_dim)
    y = onp.random.randn(X.shape[0], target_dim)
    if squeezed:
        return X.squeeze(), y.squeeze()
    return X, y

@pytest.mark.parametrize("n_latent", [1, 2])
@pytest.mark.parametrize("squeezed", [True, False])
def test_dkl_fit_all(n_latent, squeezed):
    net = FlaxMLP(hidden_dims=[4, 2], output_dim=1)
    X, y = get_dummy_data(4, 1, squeezed)
    dkl = PartialDKL(n_latent, RBFKernel, net)
    dkl.fit(X, y, num_warmup=10, num_samples=10)
    assert dkl.mcmc is not None


def test_dkl_fit_predict():
    net = FlaxMLP(hidden_dims=[4, 2], output_dim=1)
    X, y = get_dummy_data(4)
    X_test, _ = get_dummy_data(4, n_points=20)
    dkl = PartialDKL(1, RBFKernel, net)
    dkl.fit(X, y, num_warmup=10, num_samples=10)
    pmean, pvar = dkl.predict(X_test)
    assert_equal(pmean.shape, (len(X_test),))
    assert_equal(pmean.shape, pvar.shape)