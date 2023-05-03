from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jaxutils

import gpjax as gpx
from jax import jit
import jaxkern as jk


def standard_positive_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Apply softplus transform to x."""
    return jax.nn.softplus(x) + 1e-6


def standard_inverse_transform(x: jnp.ndarray) -> jnp.ndarray:
    """Implements inverse softplus transform."""
    return jnp.log(jnp.exp(x - 1e-6) - 1)


def setup_gp(num_datapoints: int):
    """Define the GP model."""
    prior = gpx.Prior(kernel=jk.RBF())
    likelihood = gpx.Gaussian(num_datapoints=num_datapoints)
    return prior, likelihood


def ift_to_gpx(ift_params: Any) -> Any:
    """Applies"""
    return {
        "kernel": {
            "lengthscale": standard_positive_transform(
                ift_params["kernel"]["lengthscale"]
            ),
            "variance": standard_positive_transform(ift_params["kernel"]["variance"]),
        },
        "likelihood": {
            "obs_noise": standard_positive_transform(
                ift_params["likelihood"]["obs_noise"]
            )
        },
        "mean_function": {},
    }


def gpx_to_ift(gpx_params: Any) -> Any:
    return {
        "kernel": {
            "lengthscale": standard_inverse_transform(
                gpx_params["kernel"]["lengthscale"]
            ),
            "variance": standard_inverse_transform(gpx_params["kernel"]["variance"]),
        },
        "likelihood": {
            "obs_noise": standard_inverse_transform(
                gpx_params["likelihood"]["obs_noise"]
            )
        },
    }


def neg_mll(x: jnp.ndarray, y: jnp.ndarray) -> Callable[[Any], float]:
    """Creates a function to calculate the neg marginal likelihood."""
    dataset = jaxutils.Dataset(X=x, y=y.reshape(-1, 1))
    prior, likelihood = setup_gp(num_datapoints=x.shape[0])
    posterior = prior * likelihood
    mll = jit(posterior.marginal_log_likelihood(dataset, negative=True))
    return lambda param: mll(ift_to_gpx(param))


def train_mll(x: jnp.array, y: jnp.array, params: Any) -> jnp.array:
    """
    Return marginal log likelihood for labels y at locations x
    under an RBF GP prior with hyperparameters given in `params`
    """
    # To maintain same interface with original gp module, remove if not needed
    return -neg_mll(x, y)(params)


@jax.jit
def gp_predictive_dist(
    x_train: jnp.ndarray, y_train: jnp.ndarray, x_query: jnp.ndarray, params: Any
):
    """Computes the predictive distribution."""
    gpx_params = ift_to_gpx(params)
    dataset = jaxutils.Dataset(X=x_train, y=y_train.reshape(-1, 1))
    prior, likelihood = setup_gp(num_datapoints=x_train.shape[0])
    posterior = prior * likelihood
    latent_dist = posterior(gpx_params, dataset)(x_query)
    return likelihood(gpx_params, latent_dist)


@jax.jit
def expected_improvement(
    x_query: jnp.ndarray,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    params,
) -> jnp.ndarray:
    """Computes the expected improvement at x_query points, assumes minimization."""
    dist = gp_predictive_dist(
        x_train=x_train, y_train=y_train, x_query=x_query, params=params
    )

    incumbent = y_train.max()
    improvement = incumbent - dist.mean()
    z = improvement / (dist.stddev() + 1e-3)
    ei = improvement * jsp.stats.norm.cdf(z) + dist.stddev() * jsp.stats.norm.pdf(z)
    return ei, dist.mean(), dist.stddev()


@jax.jit
def predictive_mll(
    x_query: jnp.ndarray,
    y_query: jnp.ndarray,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    params,
) -> jnp.ndarray:
    """Return marginal log likelihood of predictions."""
    dist = gp_predictive_dist(
        x_train=x_train, y_train=y_train, x_query=x_query, params=params
    )
    return jsp.stats.multivariate_normal.logpdf(y_query, dist.mean(), dist.covariance())
