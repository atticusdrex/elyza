from elyza.util.imports import * 
from elyza.surrogate import Surrogate 
from elyza.util.helpers import ensure_2d
from elyza.optim.abstract import Optimizer, OptimizerOptions 

class Ridge(Surrogate): 
    l2_reg : float = Field(default = 1e-3, description="l2_regularization") 

    # private attributes 
    _beta : jax.Array | None = PrivateAttr(default = None) 

    def fit(self, X:jax.Array, Y:jax.Array):
        X, Y = ensure_2d(X), ensure_2d(Y) 
        A = X.T @ X + self.l2_reg * jnp.eye(X.shape[1]) 
        b = X.T @ Y 

        self._beta = jnp.linalg.solve(A, b)

    def predict(self, X:jax.Array) -> jax.Array: 
        assert self._beta is not None, "must call Ridge.fit() to make predictions first" 
        return X @ self._beta 

    def update(self, X:jax.Array, Y:jax.Array): 
        raise NotImplementedError("this method is not available for this surrogate")

    def set_optimizer(self, optimizer:Optimizer, optimizer_opts:OptimizerOptions):
        raise NotImplementedError("this method is not available for this surrogate")


    