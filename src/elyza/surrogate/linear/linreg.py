"""Linear regression surrogate models.

Currently provides :class:`Ridge`, a closed-form (normal-equations) ridge
regression :class:`~elyza.surrogate.abstract.Surrogate`.
"""
from elyza.util.imports import *
from elyza.surrogate import Surrogate
from elyza.util.helpers import ensure_2d
from elyza.optim.abstract import Optimizer, OptimizerOptions

class Ridge(Surrogate):
    """Closed-form L2-regularized (ridge) linear regression.

    Attributes:
        l2_reg: L2 regularization strength.
        _beta: Fitted regression coefficients, shape ``(n_features, output_dim)``.
    """
    l2_reg : float = Field(default = 1e-3, description="l2_regularization")

    # private attributes
    _beta : jax.Array | None = PrivateAttr(default = None)

    def fit(self, X:jax.Array, Y:jax.Array):
        """Fit ridge regression coefficients via the normal equations.

        Args:
            X: Training inputs, shape ``(n_samples, n_features)``.
            Y: Training outputs, shape ``(n_samples, output_dim)`` or
                ``(n_samples,)``.
        """
        X, Y = ensure_2d(X), ensure_2d(Y)
        A = X.T @ X + self.l2_reg * jnp.eye(X.shape[1])
        b = X.T @ Y

        self._beta = jnp.linalg.solve(A, b)

    def predict(self, X:jax.Array) -> jax.Array:
        """Predict outputs for new inputs.

        Args:
            X: Query inputs, shape ``(n_samples, n_features)``.

        Returns:
            jax.Array: Predicted outputs, shape ``(n_samples, output_dim)``.

        Raises:
            AssertionError: If :meth:`fit` has not been called yet.
        """
        assert self._beta is not None, "must call Ridge.fit() to make predictions first"
        return X @ self._beta

    def sample(self, key, X:jax.Array, n_samples: int = 1) -> jax.Array:
        """Not supported for this surrogate; Ridge has no posterior to sample from.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("this method is not available for this surrogate")

    def update(self, X:jax.Array, Y:jax.Array):
        """Not supported for this surrogate.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("this method is not available for this surrogate")

    def set_optimizer(self, optimizer:Optimizer, optimizer_opts:OptimizerOptions):
        """Not supported for this surrogate; fitting is closed-form.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("this method is not available for this surrogate")
