"""A data registry containing all relevant input functionality, sampling, etc.

Defines :class:`Variable`, which describes a named, box-bounded input of an
:class:`~elyza.core.evaluator.Evaluator`. See :mod:`elyza.core.random` for
:class:`~elyza.core.random.RandomVariable` subclasses that know how to draw
samples from themselves given a PRNG key.
"""
from elyza.util.imports import *
from elyza.util.helpers import ensure_2d
from types import MethodType

class Variable(BaseModel):
    """Base class for a named, box-bounded model input.

    Attributes:
        name: Unique input name.
        dim: Dimension of the input.
        dtype: Datatype ``lower``/``upper`` are cast to.
        lower: Lower bound(s), either a single value broadcast across every
            dimension or a per-dimension array of length ``dim``.
        upper: Upper bound(s), either a single value broadcast across every
            dimension or a per-dimension array of length ``dim``.
        sampling_func: Function that takes a single PRNG key and returns one
            sample. Unused by :class:`Variable` itself; available for
            subclasses (see :mod:`elyza.core.random`) that want to draw
            samples one PRNG key at a time.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name : int | str = Field(default = "anonymous", description = "Unique input name")
    dim : int = Field(default = 1, description = "dimension of the input")
    dtype : ScalarMeta = Field(default = jnp.float64, description = "input datatype")

    lower : float | jax.Array | np.ndarray = Field(default = -jnp.inf, description = "variable lower bounds")
    upper : float | jax.Array | np.ndarray = Field(default = jnp.inf, description = "variable upper bounds")

    sampling_func : SkipValidation[callable] | None = Field(default = None, description = "function which takes a PRNG key as an input and returns a single sample")

    def model_post_init(self, __context):
        """Cast ``lower``/``upper`` to ``dtype`` and validate their shapes.

        Raises:
            AssertionError: If ``lower`` or ``upper`` is neither a single
                shared bound nor a per-dimension array of length ``dim``.
        """
        # casting upper and lower bounds into correct datatypes
        self.lower = jnp.asarray(self.lower, dtype = self.dtype).ravel()
        self.upper = jnp.asarray(self.upper, dtype = self.dtype).ravel()

        # ensuring that the upper and lower bounds are the right dimension
        assert len(self.lower) == 1 or self.lower.shape[0] == self.dim, "lower bounds are incorrect dimension"
        assert len(self.upper) == 1 or self.upper.shape[0] == self.dim, "upper bounds are incorrect dimension"

    def print(self):
        """Print the input's name, dimension, and concrete type."""
        print(" * Name: %s, Dimension: %d, Type: %s" % (
            self.name, self.dim, str(type(self))
        ))
        print("   - Lower bounds: %s" % (self.lower,))
        print("   - Upper bounds: %s" % (self.upper,))

    def _clip(self, X:jax.Array) -> jax.Array:
        """Clip ``X`` to lie within ``[lower, upper]``, per dimension.

        Args:
            X: Points to clip, shape ``(n_samples, dim)``.

        Returns:
            jax.Array: ``X`` clipped elementwise to ``[lower, upper]``, same
            shape as ``X``.

        Raises:
            AssertionError: If ``X`` does not have ``dim`` columns.
        """
        assert X.shape[1] == self.dim, "attempting to clip data which is wrong idmension"
        return jnp.maximum(
            self.lower.reshape(1,-1), jnp.minimum(
                self.upper.reshape(1,-1), X
            )
        )
