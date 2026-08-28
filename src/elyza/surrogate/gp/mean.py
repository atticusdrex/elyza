"""Mean functions used by :class:`~elyza.surrogate.gp.gp.GaussianProcess`.

Every mean function exposes an ``eval(x, params)`` method and a ``p_dim``
computed field giving the number of parameters it expects.
"""
from elyza.util.imports import *

class BaseMean(BaseModel):
    """Base class for GP mean functions.

    Attributes:
        input_dim: Input dimension the mean function operates on.
        epsilon: Small positive jitter available to subclasses.
    """
    input_dim : int
    epsilon : float = Field(default_factory = 1e-12)

class Zero(BaseMean):
    """A trivial zero-mean function."""
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of parameters this mean function expects (always 0)."""
        return 0

    def eval(self, x, params):
        """Evaluate the mean at ``x``.

        Args:
            x: Input point(s).
            params: Unused (this mean function has no parameters).

        Returns:
            float: Always ``0.0``.
        """
        return jnp.zeros(shape = (x.shape[0], 1))

class Constant(BaseMean):
    """A constant mean function."""
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of parameters this mean function expects (always 1)."""
        return 1

    def eval(self, x, params):
        """Evaluate the mean at ``x``.

        Args:
            x: Input point(s) (unused; the mean is constant).
            params: Parameter array of length 1, ``[constant]``.

        Returns:
            The constant value ``params[0]``.
        """
        return params[0] * jnp.ones(shape = (x.shape[0],1))

class Linear(BaseMean):
    """A linear mean function."""
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of parameters this mean function expects (``1 + input_dim``)."""
        return 1 + self.input_dim

    def eval(self, x, params):
        """Evaluate the mean at ``x``.

        Args:
            x: Input point, shape ``(input_dim,)``.
            params: Parameter array of length ``1 + input_dim``,
                ``[intercept, slope_1, ..., slope_d]``.

        Returns:
            jax.Array: The scalar value ``params[0] + params[1:] . x``.
        """
        return params[0] + jnp.inner(params[1:], x)
