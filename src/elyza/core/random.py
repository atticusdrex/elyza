"""Sampleable random-variable inputs.

Defines :class:`RandomVariable`, a :class:`~elyza.core.data.Variable` that
adds an abstract ``sample(key, n_points)`` method, plus three concrete
distributions built on top of it: :class:`Uniform`, :class:`Gaussian`, and
:class:`GaussianMixture`.
"""
from elyza.util.imports import *
from elyza.util.helpers import ensure_2d, softmax
from jax.scipy.linalg import cholesky, cho_solve
from elyza.core import Variable

class RandomVariable(Variable):
    """Abstract base class for a sampleable :class:`~elyza.core.data.Variable`.

    Subclasses must implement :meth:`sample`; any subclass that doesn't
    override it cannot be instantiated in a useful way, since
    :meth:`sample` is the only way to draw data from a random variable.
    """
    @abstractmethod
    def sample(self, key:jax.Array, n_points:int) -> jax.Array:
        """Draw a batch of samples.

        Args:
            key: A JAX PRNG key.
            n_points: Number of samples to draw.

        Returns:
            jax.Array: Samples, shape ``(n_points, self.dim)``.
        """
        pass


class Uniform(RandomVariable):
    """Uniform distribution over the box ``[lower, upper]``.

    ``lower``/``upper`` may each be a single shared bound (broadcast across
    every dimension) or a per-dimension array of length ``self.dim``, per
    :meth:`~elyza.core.data.Variable.model_post_init`.
    """
    def model_post_init(self, __context):
        """Validate bounds, on top of :meth:`Variable.model_post_init`.

        Raises:
            AssertionError: If ``lower`` or ``upper`` is not finite --
                sampling uniformly over an unbounded interval is undefined.
        """
        super().model_post_init(__context)
        assert jnp.isfinite(self.lower).all() and jnp.isfinite(self.upper).all(), "uniform distribution requires finite lower/upper bounds"

    def sample(self, key:jax.Array, n_points:int) -> jax.Array:
        """Draw a batch of samples uniformly from ``[lower, upper]``.

        Args:
            key: A JAX PRNG key.
            n_points: Number of samples to draw.

        Returns:
            jax.Array: Samples, shape ``(n_points, self.dim)``.
        """
        u = jrand.uniform(key, shape=(n_points, self.dim), dtype=self.dtype)
        return self.lower.reshape(1,-1) + (self.upper - self.lower).reshape(1,-1) * u

class Gaussian(RandomVariable):
    """A multivariate Gaussian random variable, parameterized by its Cholesky factor.

    Attributes:
        _eps: Small positive jitter added to the diagonal of ``L`` by
            :attr:`_constr`, to keep it a valid (non-singular)
            lower-triangular Cholesky factor under constrained optimization.
        _dim: Dimension of the Gaussian, inferred from ``mean``.
        _p: Parameter dict (``mean``, ``L``), where ``L`` is the
            lower-triangular Cholesky factor of the covariance.
        _constr: Constraint dict enforcing ``L`` stays lower-triangular with
            a positive diagonal, keyed to match ``_p``.
    """
    # private fields
    _eps : float = PrivateAttr(default = 1e-8)
    _dim : int | None = PrivateAttr(default = None)
    _p : dict | None = PrivateAttr(default = None)
    _constr : dict | None = PrivateAttr(default = None)

    def __init__(self, mean:jax.Array, cov:jax.Array, **kwargs):
        """Build the Gaussian from a mean vector and covariance matrix.

        Args:
            mean: Mean vector, shape ``(dim,)``.
            cov: Covariance matrix, shape ``(dim, dim)``; must be symmetric
                positive definite.
            **kwargs: Forwarded to :class:`~elyza.core.data.Variable`
                (e.g. ``name``, ``dim``, ``lower``, ``upper``, ``dtype``).

        Raises:
            AssertionError: If ``mean`` and ``cov`` have mismatched
                dimensions, or if ``cov`` is not symmetric positive
                definite (its Cholesky factor contains ``nan``).
        """
        mean, cov = jnp.array(mean).ravel(), ensure_2d(jnp.array(cov))
        assert mean.shape[0] == cov.shape[0] == cov.shape[1], "mean dimension and variance dimensions mismatch"
        L = cholesky(cov, lower=True)
        assert not jnp.isnan(L.ravel()).any(), "variance is not symmetric positive definite; cholesky factor contains nan"
        super().__init__(**kwargs)
        self._dim = mean.shape[0]
        self._p = {'mean':mean, 'L':L}
        self._constr = {'L':lambda L: jnp.tril(L, k=-1) + jnp.diag(jnp.maximum(jnp.diag(L), self._eps))}

    def sample(self, key:jax.Array, n_points:int) -> jax.Array:
        """Draw a batch of samples via the reparameterization trick.

        Args:
            key: A JAX PRNG key.
            n_points: Number of samples to draw.

        Returns:
            jax.Array: Samples, shape ``(n_points, self._dim)``.
        """
        return (self._p['mean'] + self._p['L'] @ jrand.normal(key, shape = (self._dim, n_points))).T

    def log_pdf(self, x:jax.Array, p : dict | None = None) -> float:
        """Compute the Gaussian log-density at ``x``.

        Args:
            x: Point to evaluate the log-density at, shape ``(dim,)``.
            p: Parameter dict (``mean``, ``L``) to evaluate at; defaults to
                ``self._p`` (the fitted/constructed parameters).

        Returns:
            float: The log-density ``log N(x; mean, L @ L.T)``.
        """
        if p is None:
            mean, L = self._p['mean'], self._p['L']
        else:
            mean, L = p['mean'], p['L']

        x = jnp.array(x).ravel()
        return -0.5*((x - mean).T @ cho_solve((L, True), x - mean) + 2.0*jnp.sum(jnp.diag(L)) - self._dim * jnp.log(2*jnp.pi)).ravel()

class GaussianMixture(RandomVariable):
    """A mixture of multivariate Gaussian random variables.

    Attributes:
        _p: Parameter dict (``mean``, ``L``, ``w``), where ``mean``/``L``
            are stacked across components (leading axis of size ``_K``) and
            ``w`` holds the (softmax-normalized) mixture weights.
        _K: Number of mixture components.
        _constr: Per-component constraint dict enforcing each ``L`` stays
            lower-triangular with a positive diagonal.
        _dim: Dimension of each Gaussian component, inferred from ``means``.
        _eps: Small positive jitter added to the diagonal of each ``L`` by
            :attr:`_constr`, to keep it a valid (non-singular)
            lower-triangular Cholesky factor under constrained optimization.
    """
    # private fields
    _p : dict | None = PrivateAttr(default = None)
    _K : int | None = PrivateAttr(default = None)
    _constr : dict | None = PrivateAttr(default = None)
    _dim : int | None = PrivateAttr(default = None)
    _eps : float = PrivateAttr(default = 1e-8)

    def __init__(self, means:list[jax.Array], covs:list[jax.Array], weights:list[float]|jax.Array,**kwargs):
        """Build the mixture from per-component means, covariances, and weights.

        Args:
            means: One mean vector per component, each shape ``(dim,)``.
            covs: One covariance matrix per component, each shape
                ``(dim, dim)``; must be symmetric positive definite.
            weights: Unnormalized mixture weights, one per component;
                normalized internally via :func:`~elyza.util.helpers.softmax`.
            **kwargs: Forwarded to :class:`~elyza.core.data.Variable`
                (e.g. ``name``, ``dim``, ``lower``, ``upper``, ``dtype``).

        Raises:
            AssertionError: If any component's ``mean``/``cov`` dimensions
                mismatch, if any ``cov`` is not symmetric positive definite,
                or if ``weights`` doesn't have one entry per component.
        """
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
        """Draw a batch of samples: pick a component, then sample from it.

        Args:
            key: A JAX PRNG key.
            n_points: Number of samples to draw.

        Returns:
            jax.Array: Samples, shape ``(n_points, self._dim)``.
        """
        key_idx, key_z = jrand.split(key)
        idx = jrand.choice(key_idx, self._K, shape=(n_points,), p=self._p['w'])
        z = jrand.normal(key_z, shape=(n_points, self._dim))
        L_sel = self._p['L'][idx]       # (n_points, dim, dim)
        mean_sel = self._p['mean'][idx]  # (n_points, dim)
        return mean_sel + jnp.einsum('nij,nj->ni', L_sel, z)

