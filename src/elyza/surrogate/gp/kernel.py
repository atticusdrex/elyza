"""Covariance kernels used by :class:`~elyza.surrogate.gp.gp.GaussianProcess`.

Every kernel exposes an ``eval(x1, x2, params)`` method and a ``p_dim``
computed field giving the number of raw parameters it expects; ``params``
are always passed through :func:`~elyza.util.helpers.softplus` inside
``eval`` so that unconstrained (e.g. optimizer-facing) parameters map to
strictly positive amplitude/bandwidth values.
"""
from elyza.util.imports import *
from elyza.util.helpers import softplus

class BaseKernel(BaseModel):
    """Base class for GP covariance kernels.

    Attributes:
        input_dim: Input dimension the kernel operates on.
        epsilon: Small positive jitter used by subclasses to avoid
            division-by-zero.
    """
    input_dim: int
    epsilon: float = 1e-12

class RBF(BaseKernel):
    """Isotropic (single-bandwidth) radial basis function kernel."""
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of raw parameters (amplitude, bandwidth) this kernel expects."""
        return 2

    def eval(self, x1, x2, params):
        """Evaluate the RBF kernel between two points.

        To obtain the exact amplitude and lengthscale for the kernel
        ``l * exp(-||x - x'||^2 / (2*sigma^2))``, convert ``params`` via::

            l = softplus(params[0])
            sigma = sqrt(0.5 * softplus(params[1]))

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length 2;
                softplus'd internally to ``[amplitude, bandwidth]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        params = softplus(params.ravel()) # softplusing params for positivity constraints
        return params[0]*jnp.exp(-jnp.sum(((x1 - x2).ravel())**2 / params[1]))

class ARD(BaseKernel):
    """Automatic relevancy determination kernel (per-dimension bandwidths)."""
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of raw parameters (1 amplitude + ``input_dim`` bandwidths)."""
        return 1 + self.input_dim

    def eval(self, x1, x2, params):
        """Evaluate the ARD kernel between two points.

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length
                ``1 + input_dim``; softplus'd internally to
                ``[amplitude, bandwidth_1, ..., bandwidth_d]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        params = softplus(params.ravel()) # softplusing params for positivity constraints
        return params[0]*jnp.exp(-jnp.sum(((x1 - x2).ravel())**2 / params[1:]))

class Laplace(BaseKernel):
    """Laplace kernel (uses the 1-norm instead of the 2-norm)."""
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of raw parameters (1 amplitude + ``input_dim`` bandwidths)."""
        return 1 + self.input_dim

    def eval(self, x1, x2, params):
        """Evaluate the Laplace kernel between two points.

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length
                ``1 + input_dim``; softplus'd internally to
                ``[amplitude, bandwidth_1, ..., bandwidth_d]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        params = softplus(params.ravel()) # softplusing params for positivity constraints
        return params[0]*jnp.exp(-jnp.sum(jnp.abs((x1 - x2).ravel()) / params[1:]))
