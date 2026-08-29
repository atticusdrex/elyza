from elyza.util.imports import *
from elyza.util.helpers import ensure_2d, softmax
from jax.scipy.linalg import cholesky, cho_solve

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
    # private fields
    _eps : float = PrivateAttr(default = 1e-8)
    _dim : int | None = PrivateAttr(default = None)
    _p : dict | None = PrivateAttr(default = None) 
    _constr : dict | None = PrivateAttr(default = None)

    def __init__(self, mean:jax.Array, cov:jax.Array, **kwargs):
        mean, cov = jnp.array(mean).ravel(), ensure_2d(jnp.array(cov)) 
        assert mean.shape[0] == cov.shape[0] == cov.shape[1], "mean dimension and variance dimensions mismatch" 
        L = cholesky(cov, lower=True) 
        assert not jnp.isnan(L.ravel()).any(), "variance is not symmetric positive definite"
        super().__init__(**kwargs)
        self._dim = mean.shape[0] 
        self._p = {'mean':mean, 'L':L} 
        self._constr = {'L':lambda L: jnp.tril(L, k=-1) + jnp.diag(jnp.maximum(jnp.diag(L), self._eps))}

    def sample(self, key:jax.Array, n_points:int) -> jax.Array: 
        return self._p['mean'] + self._p['L'] @ jrand.normal(key, shape = (self._dim, n_points))

    '''
    function for returning log-likelihood 
    '''
    def log_pdf(self, x:jax.Array, p : dict | None = None) -> float: 
        if p is None:
            mean, L = self._p['mean'], self._p['L']
        else:
            mean, L = p['mean'], p['L'] 

        x = jnp.array(x).ravel() 
        return -0.5*((x - mean).T @ cho_solve((L, True), x - mean) + 2.0*jnp.sum(jnp.diag(L)) - self._dim * jnp.log(2*jnp.pi)).ravel()

class GaussianMixture(Distribution):
    # private fields
    _p : dict | None = PrivateAttr(default = None) 
    _K : int | None = PrivateAttr(default = None) 
    _constr : dict | None = PrivateAttr(default = None) 
    _dim : int | None = PrivateAttr(default = None) 

    def __init__(self, means:list[jax.Array], covs:list[jax.Array], weights:list[float]|jax.Array,**kwargs):
        super().__init__(**kwargs) 
        self._constr = {'L':{}} 
        stored_means, stored_Ls = [], []
        for i, (mean, cov) in enumerate(zip(means, covs)):
            mean, cov = jnp.array(mean).ravel(), ensure_2d(jnp.array(cov))
            assert mean.shape[0] == cov.shape[0] == cov.shape[1], "mean dimension and variance dimensions mismatch" 
            L = cholesky(cov, lower=True) 
            assert not jnp.isnan(L.ravel()).any(), "variance is not symmetric positive definite"
            stored_means.append(mean) 
            stored_Ls.append(L) 
            self._constr['L'][i] = lambda L: jnp.tril(L, k=-1) + jnp.diag(jnp.maximum(jnp.diag(L), self._eps))
        self._p = {'mean':jnp.stack(stored_means),'L':jnp.stack(stored_Ls)}
        self._K = len(self._p['mean']) 
        self._dim = self._p['mean'][0].shape[0]
        weights = jnp.array(weights).ravel()
        assert weights.shape[0] == len(self._p['mean']), "weights must match the number of Gaussian distributions passed in" 
        self._p['w'] = softmax(weights)

    def sample(self, key: jax.Array, n_points: int) -> jax.Array:
        key_idx, key_z = jrand.split(key)
        idx = jrand.choice(key_idx, self._K, shape=(n_points,), p=self._p['w'])
        z = jrand.normal(key_z, shape=(n_points, self._dim))
        L_sel = self._p['L'][idx]       # (n_points, dim, dim)
        mean_sel = self._p['mean'][idx]  # (n_points, dim)
        return (mean_sel + jnp.einsum('nij,nj->ni', L_sel, z)).T

