"""Code for application of implicit function theorem for ADKF."""
from __future__ import annotations

import math
import logging
from typing import Any, Callable, TypedDict, Dict

import jax
import jax.numpy as jnp
import jaxopt
from flax import linen as nn
import optax

logger = logging.getLogger(__name__)


class DeepKernelGPParams(TypedDict):
    feature_extractor: Any
    gp: Dict


param_combine_type = Callable[[Any, Any], DeepKernelGPParams]
adapt_loss_type = Callable[
    [Any, Any, param_combine_type, jnp.ndarray, jnp.ndarray, nn.Module], float
]
meta_loss_type = Callable[
    [
        Any,
        Any,
        param_combine_type,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        nn.Module,
    ],
    float,
]


def optimize_train_loss_adam(
    adapt_params: Any,
    meta_params: Any,
    param_combine_fn: param_combine_type,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    feature_extractor: nn.Module,
    adapt_loss: adapt_loss_type,
    tol: float = 1e-4,
) -> Any:

    # Initialize optimizer
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(adapt_params)

    # Minimization loop
    last_change = 10 * tol
    last_val = math.inf
    n_steps = 0
    while last_change > tol:
        val, grads = jax.value_and_grad(adapt_loss, argnums=0)(
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


def optimize_train_loss_lbfgs(
    adapt_params: Any,
    meta_params: Any,
    param_combine_fn: param_combine_type,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    feature_extractor: nn.Module,
    adapt_loss: adapt_loss_type,
) -> Any:

    # Wrapper function just depends just on adapt params
    def _lbfgs_obj(p):
        return adapt_loss(
            p, meta_params, param_combine_fn, x_train, y_train, feature_extractor
        )

    # Optimize training loss
    solver = jaxopt.ScipyMinimize(
        method="L-BFGS-B",
        fun=_lbfgs_obj,
    )
    res = solver.run(adapt_params)
    return res.params


def ift_gradient_update(
    adapt_params: Any,
    meta_params: Any,
    param_combine_fn: param_combine_type,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_pred: jnp.ndarray,
    y_pred: jnp.ndarray,
    feature_extractor: nn.Module,
    adapt_loss: adapt_loss_type,
    meta_loss: meta_loss_type,
    fix_singular_hessian: bool = False,
) -> tuple[jnp.ndarray, Any]:
    """Performs IFT gradient update. NOTE: assumes adapt_params are already optimized."""

    # Form flat versions of parameters
    adapt_params_flat, _ = jax.flatten_util.ravel_pytree(adapt_params)
    meta_params_flat, meta_unflatten = jax.flatten_util.ravel_pytree(meta_params)

    # Form Hessian and cross-derivatives
    L_T = adapt_loss
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

    hes_matrix = hes_array.reshape(len(adapt_params_flat), len(adapt_params_flat))
    cross_matrix = cross_array.reshape(len(adapt_params_flat), len(meta_params_flat))

    if fix_singular_hessian and jnp.linalg.det(hes_matrix) == 0:
        eig_val, eig_vec = jnp.linalg.eigh(hes_matrix)
        new_eig_val = eig_val.at[eig_val == 0].set(1e-6)
        hes_matrix = eig_vec @ jnp.diag(new_eig_val) @ jnp.linalg.inv(eig_vec)

    d_adapt_by_d_meta = -jnp.linalg.solve(
        hes_matrix,
        cross_matrix,
    )

    # Find gradients of validation loss
    L_V = meta_loss
    val, grad = jax.value_and_grad(L_V, argnums=(0, 1))(
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
