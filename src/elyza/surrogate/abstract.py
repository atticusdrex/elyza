"""Base surrogate modeling class from which all multifidelity functionality is built.

Defines a common callable structure so that multifidelity surrogate models
don't have to be tailor-made to specific types of ML models. It loosely
follows the scikit-learn regressor structure to act as a wrapper for
general types of ML models, plus :class:`SupervisedDataset`, a container for
the input/output arrays a surrogate is trained on.

Example:
    Suppose we train a model using::

        model.train_model(
            X_data=X,
            Y_data=Y,
            initial_params=None,
            momentum=0.9
        )

    We would then define the :class:`Surrogate` subclass using::

        class MySurrogate(Surrogate):
            def fit(self, X, Y, momentum=0.9, initial_params=None):
                self.model.train_model(
                    X_data=X,
                    Y_data=Y,
                    initial_params=initial_params,
                    momentum=momentum
                )

            def predict(self, X, full_cov=False):
                self.model.make_prediction(X_data=X, full_cov=full_cov)

    When a child class does not implement all the methods (e.g. a deep
    learning class may not implement a ``sample()`` or ``update()`` method),
    the surrogate model base class defaults to raising a
    ``NotImplementedError``.
"""
from elyza.util.imports import *
from elyza.core.data import *
from elyza.util.helpers import ensure_2d
from elyza.optim.abstract import Optimizer, OptimizerOptions

class Surrogate(BaseModel):
    """Abstract base class for surrogate (ML) models used across ``elyza``.

    Subclasses implement :meth:`fit`, :meth:`set_optimizer`, :meth:`predict`,
    and optionally :meth:`sample`/:meth:`update`; any method left
    unimplemented raises ``NotImplementedError`` by default.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dtype : ScalarMeta = Field(default = jnp.float64, description = "input datatype")

    @abstractmethod
    def fit(
        self,
        X: np.ndarray | jax.Array,
        Y: np.ndarray | jax.Array,
        **kwargs
    ) -> None:
        """Fit the surrogate model to training data.

        Args:
            X: Training inputs.
            Y: Training outputs.
            **kwargs: Model-specific fitting options.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("This feature is not implemented yet.")

    @abstractmethod
    def set_optimizer(
        self,
        optimizer:Optimizer,
        optimizer_opts:OptimizerOptions
    ):
        """Assign the optimizer (and its options) used by :meth:`fit`.

        Args:
            optimizer: An :class:`~elyza.optim.abstract.Optimizer` class.
            optimizer_opts: An :class:`~elyza.optim.abstract.OptimizerOptions`
                instance configuring that optimizer.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("This feature is not implemented yet.")

    @abstractmethod
    def predict(
        self,
        X: np.ndarray | jax.Array,
        **kwargs
    ) -> tuple[jax.Array]:
        """Return predictive mean (and optionally variance) at new points.

        Args:
            X: Query inputs.
            **kwargs: Model-specific prediction options.

        Returns:
            tuple[jax.Array]: Predictive mean and (model-dependent) variance/covariance.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("This feature is not implemented yet.")

    @abstractmethod
    def sample(self, key, X:jax.Array, n_samples: int = 1, **kwargs) -> jax.Array:
        """Draw samples from the posterior/predictive distribution.

        Args:
            key: A JAX PRNG key.
            X: Query inputs.
            n_samples: Number of samples to draw per query point.

        Returns:
            jax.Array: Sampled outputs.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("This feature is not implemented yet.")

    @abstractmethod
    def update(self, X, Y) -> None:
        """Update the model with new observations (e.g. online/incremental fitting).

        Args:
            X: New inputs.
            Y: New outputs.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("This feature is not implemented yet.")


class SupervisedDataset(BaseModel):
    """Wrapper class for working with supervised learning datasets.

    Attributes:
        input_data: List of in-order inputs and the data associated with
            those inputs.
        output_data: An array of the corresponding model outputs associated
            with these inputs.
        noise_var: Variance of Gaussian white noise in the output data.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_data : list[jax.Array] = Field(description = "list of in-order inputs and the data associated with those inputs")
    output_data : jax.Array = Field(description = "an array of the corresponding model outputs associated with these inputs")
    noise_var : float = Field(default = 0.0, description = "variance of Gaussian white noise in the output data")

    def concatenate_inputs(self):
        """Concatenate the list of input arrays into a single feature matrix.

        Returns:
            jax.Array: Inputs concatenated along axis 1.
        """
        return jnp.concatenate(self.input_data, axis=1)

    def model_post_init(self, __context):
        """Coerce ``output_data`` to a 2-d array."""
        self.output_data = ensure_2d(self.output_data)

    def update(self, *new_inputs : list[jax.Array], new_outputs : jax.Array):
        """Append new observations to the dataset in place.

        Args:
            *new_inputs: One array per existing input, appended row-wise to
                the corresponding entry of ``input_data``.
            new_outputs: New output rows, appended to ``output_data``.
        """
        # adding the new inputs to the existing input
        for i, (existing_input, new_input) in enumerate(zip(self.input_data, new_inputs)):
            self.input_data[i] = jnp.concatenate((existing_input, ensure_2d(new_input)), axis=0)

        # adding the new outputs to the existing outputs
        self.output_data = jnp.concatenate((self.output_data, ensure_2d(new_outputs)), axis=0)
