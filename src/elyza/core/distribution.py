from elyza.util.imports import *
from elyza.util.helpers import ensure_2d
from jax.scipy.linalg import cholesky
'''
Distribution 
------------

abstract class which is inherited by the child classes
'''
class Distribution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod 
    def sample(self, key:jax.Array, n_points:int) -> jax.Array: 
        pass 


class Gaussian(Distribution):
    mean : float | jax.Array = Field(default = 0.0, description = "mean of the Gaussian distribution") 
    variance : float | jax.Array = Field(default = 1.0, description = "variance of the Gaussian distribution")

    # private fields
    _dim = int | None = PrivateAttr(default = None)
    _L : jax.Array | None = PrivateAttr(default = None) 

    def model_post_init(self, __context):
        super().model_post_init(__context) 

        # converting mean and variance to jax arrays 
        self.mean, self.variance = ensure_2d(jnp.array(self.mean)), ensure_2d(jnp.array(self.variance)) 

        # ensure they have the same shape 
        assert self.mean.shape[0] == self.variance.shape[0] == self.variance.shape[1], "mean dimension and variance dimensions mismatch" 
        self._dim = self.mean.shape[0] 
        self._L = cholesky(self.variance, lower=True) 
        assert not jnp.isnan(self._L.ravel()).any(), "variance is not symmetric positive definite"

    def sample(self, key:jax.Array, n_points:int) -> jax.Array: 
        return self.mean + self._L @ jrand.normal(key, shape = (self._dim, n_points))

class GaussianMixture(Distribution):
    means : list[jax.Array] = Field(description = "list of mean vectors")
    
    weights : jax.Array = Field(description = "weighting of each gaussian distribution")

    # private fields 
    _K : int | None = PrivateAttr(default = None) 

    def model_post_init(self, __context):
        super().model_post_init(__context) 

        # raveling weights 
        self.weights = self.weights.ravel()

        # ensuring weights are valid 
        assert self.weights.shape[0] == len(self.gaussians), "weights must match the number of Gaussian distributions passed in" 
        self._K = len(self.gaussians) 

         

