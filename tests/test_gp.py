import math

import pytest

import jax.numpy as jnp

import adkf.gp


@pytest.fixture
def xy_train():
    return jnp.array([[1, 2, 3], [4, 5, 6]]), jnp.array([1.0, 2.0])


@pytest.fixture
def xy_test():
    # Just slightly corrupted versions of the training points
    return jnp.array([[2, 1, 3], [4, 6, 5]]), jnp.array([1.2, 1.8])


@pytest.fixture
def gp_params():
    return adkf.gp.GPParams(
        log_amplitude=jnp.array(math.log(2.0)),
        log_noise=jnp.array(math.log(0.1)),
        log_lengthscale=jnp.array(math.log(3.0)),
    )


def test_rbf_kernel(xy_train, gp_params):
    x, _ = xy_train
    k_actual = adkf.gp.rbf_kernel(x, x, gp_params)
    k_expected = jnp.array([[2.0, 0.44626042], [0.44626042, 2.0]])
    assert jnp.allclose(k_actual, k_expected)


def test_train_mll(xy_train, gp_params):
    x, y = xy_train
    mll_output = adkf.gp.train_mll(x, y, gp_params)
    assert jnp.isclose(mll_output, -3.59153)


def test_pred_mll(xy_train, xy_test, gp_params):
    x, y = xy_train
    x_q, y_q = xy_test
    pred_mll = adkf.gp.predictive_mll(x_q, y_q, x, y, gp_params)
    assert jnp.isclose(pred_mll, -1.3878)
