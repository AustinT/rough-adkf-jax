"""Code for application of implicit function theorem for ADKF."""
from __future__ import annotations

import math
import logging
from typing import Any, Callable, TypedDict

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

from .gp import GPParams, train_mll, predictive_mll

logger = logging.getLogger(__name__)


class DeepKernelGPParams(TypedDict):
    feature_extractor: Any
    gp: GPParams


param_combine_type = Callable[[Any, Any], DeepKernelGPParams]


def train_mll_loss(
    adapt_params: Any,
    meta_params: Any,
    param_combine_fn: param_combine_type,
    x: jnp.ndarray,
    y: jnp.ndarray,
    feature_extractor: nn.Module,
):
    all_params = param_combine_fn(adapt_params, meta_params)
    z = feature_extractor.apply(all_params["feature_extractor"], x)
    return -train_mll(z, y, all_params["gp"])


def pred_mll_loss(
    adapt_params: Any,
    meta_params: Any,
    param_combine_fn: param_combine_type,
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_q: jnp.ndarray,
    y_q: jnp.ndarray,
    feature_extractor: nn.Module,
):
    all_params = param_combine_fn(adapt_params, meta_params)
    z = feature_extractor.apply(all_params["feature_extractor"], x)
    z_q = feature_extractor.apply(all_params["feature_extractor"], x_q)
    return -predictive_mll(
        x_query=z_q, y_query=y_q, x_train=z, y_train=y, params=all_params["gp"]
    )


def optimize_train_loss_adam(
    adapt_params: Any,
    meta_params: Any,
    param_combine_fn: param_combine_type,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    feature_extractor: nn.Module,
    tol: float = 1e-4,
) -> DeepKernelGPParams:

    # Initialize optimizer
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(adapt_params)

    # Minimization loop
    last_change = 10 * tol
    last_val = math.inf
    n_steps = 0
    while last_change > tol:
        # NOTE: train_mll_loss could be made into an adjustable parameter in the future
        val, grads = jax.value_and_grad(train_mll_loss, argnums=0)(
            adapt_params,
            meta_params,
            param_combine_fn,
            x_train,
            y_train,
            feature_extractor,
        )

        updates, opt_state = optimizer.update(grads, opt_state)
        adapt_params = optax.apply_updates(adapt_params, updates)
        n_steps += 1

        # Whether to terminate
        last_change = abs(float(val) - float(last_val))
        last_val = float(val)

        logger.log(
            level=logging.DEBUG - 1,
            msg=f"Step {n_steps}: change={last_change}, grad={grads}, val={val}",
        )

    # Log termination
    logger.debug(
        f"Optimization terminated after {n_steps} steps."
        f" Last change={last_change}, last val={last_val}"
        f" adapt_params={adapt_params}"
        f" grad={grads}"
    )
    return adapt_params


def ift_gradient_update(
    adapt_params: Any,
    meta_params: Any,
    param_combine_fn: param_combine_type,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_pred: jnp.ndarray,
    y_pred: jnp.ndarray,
    feature_extractor: nn.Module,
) -> tuple[jnp.ndarray, Any]:
    """Performs IFT gradient update. NOTE: assumes adapt_params are already optimized."""

    # Form flat versions of parameters
    adapt_params_flat, _ = jax.flatten_util.ravel_pytree(adapt_params)
    meta_params_flat, meta_unflatten = jax.flatten_util.ravel_pytree(meta_params)

    # Form Hessian and cross-derivatives
    L_T = train_mll_loss  # could change this to be a parameter
    hes = jax.jacfwd(jax.jacrev(L_T))(
        adapt_params,
        meta_params,
        param_combine_fn,
        x_train,
        y_train,
        feature_extractor,
    )
    cross_derivs = jax.jacfwd(jax.jacrev(L_T, argnums=0), argnums=1)(
        adapt_params,
        meta_params,
        param_combine_fn,
        x_train,
        y_train,
        feature_extractor,
    )

    # Solve for d_adapt / d_meta
    hes_array, _ = jax.flatten_util.ravel_pytree(hes)
    cross_array, _ = jax.flatten_util.ravel_pytree(cross_derivs)
    d_adapt_by_d_meta = -jnp.linalg.solve(
        hes_array.reshape(len(adapt_params_flat), len(adapt_params_flat)),
        cross_array.reshape(len(adapt_params_flat), len(meta_params_flat)),
    )

    # Find gradients of validation loss
    val, grad = jax.value_and_grad(pred_mll_loss, argnums=(0, 1))(
        adapt_params,
        meta_params,
        param_combine_fn,
        x_train,
        y_train,
        x_pred,
        y_pred,
        feature_extractor,
    )

    # Form update
    adapt_grad_flat, _ = jax.flatten_util.ravel_pytree(grad[0])
    meta_grad_flat, _ = jax.flatten_util.ravel_pytree(grad[1])
    ift_term = adapt_grad_flat.T @ d_adapt_by_d_meta

    # Log things
    logger.debug(f"Val loss: {val}")
    logger.debug(f"\tIFT term: {ift_term}")
    logger.debug(f"\tDirect gradient: {meta_grad_flat}")

    # Return update
    return val, meta_unflatten(meta_grad_flat + ift_term)
