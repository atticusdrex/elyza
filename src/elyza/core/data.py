"""A data registry containing all relevant input functionality, sampling, etc.

Defines the :class:`Input` base class and its :class:`ScalarInput` and
:class:`VectorInput` specializations, which describe the inputs of an
:class:`~elyza.core.evaluator.Evaluator` and know how to draw samples from
themselves given a PRNG key.
"""
from elyza.util.imports import *
from elyza.util.helpers import ensure_2d
from types import MethodType

class Input(BaseModel):
    """Base class for a named, sampleable model input.

    Attributes:
        name: Unique input name.
        dim: Dimension of the input.
        sampling_func: Function that takes a single PRNG key and returns one
            sample; used by :meth:`sample` to draw a batch of samples.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name : int | str = Field(default = "anonymous", description = "Unique input name")
    dim : int = Field(default = 1, description = "dimension of the input")
    dtype : ScalarMeta = Field(default = jnp.float64, description = "input datatype")

    sampling_func : SkipValidation[callable] | None = Field(default = None, description = "function which takes a PRNG key as an input and returns a single sample")

    def sample(self, key, n_points : int) -> jax.Array:
        """Draw a batch of samples from ``sampling_func``.

        Args:
            key: A JAX PRNG key, split internally into ``n_points`` subkeys
                (one per sample).
            n_points: Number of samples to draw.

        Returns:
            jax.Array: Samples of shape ``(n_points, dim)``.

        Raises:
            AssertionError: If no ``sampling_func`` has been set.
        """
        assert self.sampling_func is not None, "No sampling function provided!"

        # splitting the jrand key into the number of points needed
        keys = jrand.split(key, n_points)

        # using vmap to sample over the keys
        return ensure_2d(vmap(self.sampling_func, in_axes=0)(keys)).astype(self.dtype)

    def print(self):
        """Print the input's name, dimension, and concrete type."""
        print(" * Name: %s, Dimension: %d, Type: %s" % (
            self.name, self.dim, str(type(self))
        ))

class ScalarInput(Input):
    """A one-dimensional input bounded by ``[minval, maxval]``.

    Attributes:
        minval: Minimum input value.
        maxval: Maximum input value.
    """
    minval : float | jax.Array = Field(default = jnp.array(0.0), description = "Minimum input value")
    maxval : float | jax.Array = Field(default = jnp.array(1.0), description = "Maximum output value")

    def model_post_init(self, __context):
        """Force ``dim`` to 1, since a scalar input is always one-dimensional."""
        # enforcing dimension to 1
        self.dim = 1

        # converting minval and maxval to correct datatype
        self.minval = jnp.array(self.minval, dtype = self.dtype)
        self.maxval = jnp.array(self.maxval, dtype = self.dtype)


    def print(self):
        """Print the input's summary along with its min/max bounds."""
        super().print()
        print("   - Min Value: %s, Max Value: %s" % (self.minval, self.maxval))

class VectorInput(Input):
    """A multi-dimensional input with per-dimension lower/upper bounds.

    Attributes:
        minval: 1-d array of lower bounds, length ``dim``.
        maxval: 1-d array of upper bounds, length ``dim``.
    """
    minval : jax.Array | np.ndarray = Field(description = "1d array of lower bounds")
    maxval : jax.Array | np.ndarray = Field(description = "1d array of upper bounds")

    def model_post_init(self, __context):
        """Validate bound shapes against ``dim`` and convert them to JAX arrays.

        Raises:
            AssertionError: If ``minval`` or ``maxval`` does not have length
                ``dim``.
        """
        assert self.minval.shape[0] == self.dim, "minimum values array != input dimension"
        assert self.maxval.shape[0] == self.dim, "maximum values array != input dimension"

        # converting to jax arrays
        self.minval = jnp.array(self.minval).astype(self.dtype) 
        self.maxval = jnp.array(self.maxval).astype(self.dtype) 
