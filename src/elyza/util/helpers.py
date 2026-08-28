"""Standalone numerical helper functions used throughout ``elyza``.

Includes covariance/correlation utilities, a robust least-squares wrapper,
shape-normalization helpers, a Gaussian KL-divergence, greedy inducing-point
selection, elementwise activation functions and their inverses, and a
vectorized kernel-matrix evaluator.
"""
from elyza.util.imports import *


def matrix_cov(Y1: jax.Array, Y2: jax.Array):
    """Compute the cross-covariance matrix between two column-aligned datasets.

    Args:
        Y1: Array of shape ``(n_samples, d1)``.
        Y2: Array of shape ``(n_samples, d2)``.

    Returns:
        jax.Array: Covariance matrix of shape ``(d1, d2)``.

    Raises:
        AssertionError: If ``Y1`` and ``Y2`` do not have the same number of samples.
    """
    assert Y1.shape[0] == Y2.shape[0], "Y1 and Y2 must contain the same number of samples"
    Y1c = Y1 - Y1.mean(axis=0).reshape(1,-1)
    Y2c = Y2 - Y2.mean(axis=0).reshape(1,-1)
    return Y1c.T @ Y2c / (Y1c.shape[0]-1)

def matrix_corr(Y1: jax.Array, Y2: jax.Array):
    """Compute the cross-correlation matrix between two column-aligned datasets.

    Args:
        Y1: Array of shape ``(n_samples, d1)``.
        Y2: Array of shape ``(n_samples, d2)``.

    Returns:
        jax.Array: Correlation matrix of shape ``(d1, d2)``, with entries in ``[-1, 1]``.

    Raises:
        AssertionError: If ``Y1`` and ``Y2`` do not have the same number of samples.
    """
    assert Y1.shape[0] == Y2.shape[0], "Y1 and Y2 must contain the same number of samples"
    Y1c = Y1 - Y1.mean(axis=0).reshape(1, -1)
    Y2c = Y2 - Y2.mean(axis=0).reshape(1, -1)
    cov = Y1c.T @ Y2c / (Y1c.shape[0] - 1)
    std1 = jnp.sqrt(jnp.sum(Y1c**2, axis=0) / (Y1c.shape[0] - 1))
    std2 = jnp.sqrt(jnp.sum(Y2c**2, axis=0) / (Y2c.shape[0] - 1))
    return cov / (std1.reshape(-1, 1) @ std2.reshape(1, -1))

def ls(A, B, rcond=None):
    """Solve the linear least-squares problem ``min ||A @ X - B||``.

    Thin wrapper around :func:`jax.numpy.linalg.lstsq` that returns only the
    solution array (not residuals, rank, or singular values).

    Args:
        A: Coefficient matrix of shape ``(m, n)``.
        B: Right-hand side of shape ``(m, k)`` or ``(m,)``.
        rcond: Relative condition number used to truncate small singular
            values; forwarded to ``jnp.linalg.lstsq``.

    Returns:
        jax.Array: Least-squares solution ``X`` of shape ``(n, k)`` or ``(n,)``.
    """
    return jnp.linalg.lstsq(A,B, rcond=rcond)[0]

def ensure_2d(X):
    """Reshape an array to be at least 2-dimensional.

    Scalars become shape ``(1, 1)`` and 1-d arrays become column vectors of
    shape ``(-1, 1)``; arrays that are already 2-d (or higher) pass through
    unchanged.

    Args:
        X: Input array of any rank.

    Returns:
        jax.Array: ``X`` with rank at least 2.
    """
    if len(X.shape) == 0:
        return X.reshape(1,1)
    elif len(X.shape) == 1:
        return X.reshape(-1,1)
    else:
        return X


def KL_div(mu_q, L_q, mu_p, L_p):
    """KL divergence KL(q || p) for Gaussians parameterized by Cholesky factors.

    Args:
        mu_q: Mean vector of ``q``, shape ``(k,)``.
        L_q: Lower-triangular Cholesky factor of ``q``'s covariance, shape ``(k, k)``.
        mu_p: Mean vector of ``p``, shape ``(k,)``.
        L_p: Lower-triangular Cholesky factor of ``p``'s covariance, shape ``(k, k)``.

    Returns:
        jax.Array: Scalar KL divergence KL(q || p).
    """
    k = mu_q.shape[0]

    # Covariance matrices
    Sigma_q = L_q @ L_q.T
    Sigma_p = L_p @ L_p.T

    # Trace term: tr(Sigma_p^{-1} Sigma_q)
    # Solve instead of explicitly inverting
    Sigma_p_inv = jnp.linalg.inv(Sigma_p)
    Tr_q = jnp.trace(Sigma_p_inv @ Sigma_q)

    # Mean term: (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
    diff = mu_q - mu_p
    mean_term = jnp.inner(diff, cho_solve((L_p, True), diff))

    # Log-determinant ratio
    logdet_q = 2.0 * jnp.sum(jnp.log(jnp.diag(L_q)))
    logdet_p = 2.0 * jnp.sum(jnp.log(jnp.diag(L_p)))
    logdet_ratio = logdet_p - logdet_q

    return 0.5 * (Tr_q + mean_term - k + logdet_ratio)

def greedy_k_center(key, X:jax.Array, k:int):
    """Greedy k-centers selection of inducing inputs.

    Starting from a random point, repeatedly selects the point farthest
    (in Euclidean distance) from the already-selected set, giving a set of
    ``k`` points that spreads coverage over ``X``.

    Args:
        X: Candidate points, shape ``(n_points, n_features)``.
        k: Number of centers to select.
        seed: Seed for the initial random point.

    Returns:
        tuple: ``(selected_points, selected_indices)`` where
        ``selected_points`` has shape ``(k, n_features)`` and
        ``selected_indices`` is the list of row indices into ``X``.
    """
    N = X.shape[0]
    selected_indices = []
    idx = jrand.randint(key, shape = (N,))
    selected_indices.append(idx)

    distances = jnp.linalg.norm(X - X[idx], axis=1)

    for _ in range(1, k):
        idx = np.argmax(distances)
        selected_indices.append(idx)
        new_distances = jnp.linalg.norm(X - X[idx], axis=1)
        distances = jnp.minimum(distances, new_distances)

    return X[jnp.array(selected_indices)], selected_indices

def sigmoid(x):
    """Sigmoid activation elementwise: ``1 / (1 + exp(-x))``.

    Args:
        x: Input array.

    Returns:
        jax.Array: Elementwise sigmoid of ``x``.
    """
    return 1.0 / (1.0 + jnp.exp(-x))

def inv_sigmoid(y):
    """Inverse sigmoid (logit): maps ``(0, 1)`` back to the real line.

    Args:
        y: Input array with values in ``(0, 1)``.

    Returns:
        jax.Array: Elementwise logit of ``y``.
    """
    return jnp.log(y/(1-y))

def softplus(x):
    """Softplus activation: ``log(1 + exp(x))``.

    Commonly used to map an unconstrained parameter to a strictly positive
    value.

    Args:
        x: Input array.

    Returns:
        jax.Array: Elementwise softplus of ``x``.
    """
    return jnp.log(1.0 + jnp.exp(x))

def inv_softplus(y):
    """Inverse softplus: maps positive values back to unconstrained space.

    Args:
        y: Input array with strictly positive values.

    Returns:
        jax.Array: Elementwise inverse softplus of ``y``.
    """
    return jnp.log(jnp.exp(y) - 1.0)

def kernel_mat(X1, X2, kernel, kernel_params):
    """Compute the full kernel matrix between two point sets.

    Args:
        X1: First set of points, shape ``(n1, input_dim)``.
        X2: Second set of points, shape ``(n2, input_dim)``.
        kernel: A kernel object exposing ``eval(x, y, kernel_params)`` for a
            single pair of points.
        kernel_params: Parameter array passed through to ``kernel.eval``.

    Returns:
        jax.Array: Kernel matrix of shape ``(n1, n2)`` where entry ``(i, j)``
        is ``kernel.eval(X1[i], X2[j], kernel_params)``.
    """
    return vmap(lambda x: vmap(lambda y: kernel.eval(x, y, kernel_params))(X2))(X1)
