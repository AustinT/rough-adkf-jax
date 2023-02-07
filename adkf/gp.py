"""Minimal code for GPs."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def standard_positive_transform(x: jnp.array) -> jnp.array:
    """Apply softplus transform to x."""
    return jax.nn.softplus(x) + 1e-6  # hard-coded epsilon


def standard_inverse_transform(x: jnp.array) -> jnp.array:
    """Implements inverse softplus transform."""
    return jnp.log(jnp.exp(x) - 1)


class GPParams(NamedTuple):
    raw_amplitude: jnp.array
    raw_noise: jnp.array
    raw_lengthscale: jnp.array


@jax.jit
def rbf_kernel(x1: jnp.array, x2: jnp.array, params: GPParams) -> jnp.array:
    """Forward function for RBF kernel."""

    # Apply lengthscales
    lengthscale = standard_positive_transform(params.raw_lengthscale)
    x1 = x1 / lengthscale
    x2 = x2 / lengthscale

    # Calculate pairwise distance matrix
    # Not sure if there is a nice built-in way to do this, so I just do it myself
    # Maybe this could be vectorized?
    x1_dot_x2 = jnp.matmul(x1, x2.T)
    x1_norm_squared = jnp.sum(x1**2, axis=-1, keepdims=True)
    x2_norm_squared = jnp.sum(x2**2, axis=-1, keepdims=True)
    pairwise_distances_squared = x1_norm_squared + x2_norm_squared.T - 2 * x1_dot_x2

    # Return kernel matrix
    amplitude = standard_positive_transform(params.raw_amplitude)
    return amplitude * jnp.exp(-pairwise_distances_squared / 2)


def _add_noise_to_kernel(k: jnp.array, params: GPParams) -> jnp.array:
    noise = standard_positive_transform(params.raw_noise)
    return k + jnp.eye(k.shape[0]) * noise


@jax.jit
def train_mll(x: jnp.array, y: jnp.array, params: GPParams) -> jnp.array:
    """
    Return marginal log likelihood for labels y at locations x
    under an RBF GP prior with hyperparameters given in `params`
    """

    # Compute kernel
    K_no_noise = rbf_kernel(x1=x, x2=x, params=params)
    K_with_noise = _add_noise_to_kernel(K_no_noise, params)

    return jsp.stats.multivariate_normal.logpdf(y, jnp.zeros_like(y), K_with_noise)


@jax.jit
def gp_predictive_distribution(
    x_query: jnp.array, x_train: jnp.array, y_train: jnp.array, params: GPParams
) -> tuple[jnp.array, jnp.array]:
    """Return mean and covariance of GP predictions."""

    # Form necessary kernel matrices
    K_xx_no_noise = rbf_kernel(x1=x_train, x2=x_train, params=params)
    K_xx = _add_noise_to_kernel(K_xx_no_noise, params)
    L_xx_tuple = jsp.linalg.cho_factor(K_xx)
    K_qx = rbf_kernel(x1=x_query, x2=x_train, params=params)
    K_qq_no_noise = rbf_kernel(x1=x_query, x2=x_query, params=params)
    K_qq = _add_noise_to_kernel(K_qq_no_noise, params)

    # Step 1: mean
    mu = jnp.matmul(K_qx, jsp.linalg.cho_solve(L_xx_tuple, y_train))

    # Step 2: covariance
    covar_reduction = jnp.matmul(K_qx, jsp.linalg.cho_solve(L_xx_tuple, K_qx.T))
    sigma = K_qq - covar_reduction

    return mu, sigma


@jax.jit
def predictive_mll(
    x_query: jnp.array,
    y_query: jnp.array,
    x_train: jnp.array,
    y_train: jnp.array,
    params: GPParams,
) -> jnp.array:
    """Return marginal log likelihood of predictions."""
    mu_pred, sigma_pred = gp_predictive_distribution(
        x_query=x_query, x_train=x_train, y_train=y_train, params=params
    )
    return jsp.stats.multivariate_normal.logpdf(y_query, mu_pred, sigma_pred)
