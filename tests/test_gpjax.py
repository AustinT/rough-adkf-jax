import pytest

import jax.numpy as jnp

from adkf import gpjax


@pytest.fixture
def xy_train():
    return jnp.array([[1, 2, 3], [4, 5, 6]]), jnp.array([1.0, 2.0])


@pytest.fixture
def xy_test():
    # Just slightly corrupted versions of the training points
    return jnp.array([[2, 1, 3], [4, 6, 5]]), jnp.array([1.2, 1.8])


@pytest.fixture
def gp_params():
    return {
        "kernel": {
            "lengthscale": jnp.array(gpjax.standard_inverse_transform(3.0)),
            "variance": jnp.array(gpjax.standard_inverse_transform(2.0)),
        },
        "likelihood": {
            "obs_noise": jnp.array(gpjax.standard_inverse_transform(0.1)),
        },
    }


def test_train_mll(xy_train, gp_params):
    x, y = xy_train
    mll_output = gpjax.train_mll(x, y, gp_params)
    assert jnp.isclose(mll_output, -3.59153)


def test_pred_mll(xy_train, xy_test, gp_params):
    x, y = xy_train
    x_q, y_q = xy_test
    pred_mll = gpjax.predictive_mll(x_q, y_q, x, y, gp_params)
    assert jnp.isclose(pred_mll, -1.3878)
