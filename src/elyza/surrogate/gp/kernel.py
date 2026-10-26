"""Covariance kernels used by :class:`~elyza.surrogate.gp.gp.GaussianProcess`.

Every kernel exposes an ``eval(x1, x2, params)`` method and a ``p_dim``
computed field giving the number of raw parameters it expects; ``params``
are always passed through :func:`~elyza.util.helpers.softplus` inside
``eval`` so that unconstrained (e.g. optimizer-facing) parameters map to
strictly positive amplitude/bandwidth values.

Each kernel also provides ``init_params(input_var, output_var)``, which the
GP models call to get a data-driven starting point for the raw parameters.
"""
from elyza.util.imports import *
from elyza.util.helpers import softplus, inv_softplus

def _safe_sqrt(x):
    """Square root whose gradient is 0 (not NaN) at ``x == 0``.

    Needed by kernels that use the (non-squared) distance ``r``: the plain
    ``jnp.sqrt`` has an infinite derivative at 0, which turns the
    hyperparameter gradients of the diagonal entries ``k(x, x)`` into NaNs.
    """
    positive = x > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)

class BaseKernel(BaseModel):
    """Base class for GP covariance kernels.

    Attributes:
        input_dim: Input dimension the kernel operates on.
        epsilon: Small positive jitter used by subclasses to avoid
            division-by-zero.
        dtype: Datatype ``eval`` casts its inputs/outputs to.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dim: int
    epsilon: float = 1e-12
    dtype: ScalarMeta = Field(default = jnp.float64, description = "datatype eval() casts its inputs/outputs to")

    def init_params(self, input_var, output_var):
        """Data-driven initial raw parameters for a ``[amplitude, bandwidth(s)]`` layout.

        The amplitude is set to the output variance and the bandwidth(s) to
        ``2 * Var(x)``, since ``E[(x_i - x_j)^2] = 2*Var(x)`` for iid
        ``x_i, x_j`` (matching the ``exp(-d^2/bandwidth)`` form). Kernels whose
        parameters have a different layout or meaning override this.

        Args:
            input_var: Per-dimension variance of the (scaled) training inputs,
                shape ``(input_dim,)``.
            output_var: Scalar variance of the training outputs.

        Returns:
            jax.Array: Raw (unconstrained) parameter array of length ``p_dim``.
        """
        n_bandwidth = self.p_dim - 1
        if n_bandwidth == input_var.shape[0]:
            bandwidth = 2.0 * input_var + self.epsilon
        else:
            # kernels with a single shared bandwidth (e.g. RBF) -- use the average
            bandwidth = jnp.full((n_bandwidth,), 2.0 * jnp.mean(input_var) + self.epsilon, dtype=self.dtype)
        return jnp.concatenate([inv_softplus(output_var).reshape(1), inv_softplus(bandwidth)]).astype(self.dtype)

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
        params = softplus(jnp.asarray(params, dtype=self.dtype).ravel()) # softplusing params for positivity constraints
        x1, x2 = jnp.asarray(x1, dtype=self.dtype), jnp.asarray(x2, dtype=self.dtype)
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
        params = softplus(jnp.asarray(params, dtype=self.dtype).ravel()) # softplusing params for positivity constraints
        x1, x2 = jnp.asarray(x1, dtype=self.dtype), jnp.asarray(x2, dtype=self.dtype)
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
        params = softplus(jnp.asarray(params, dtype=self.dtype).ravel()) # softplusing params for positivity constraints
        x1, x2 = jnp.asarray(x1, dtype=self.dtype), jnp.asarray(x2, dtype=self.dtype)
        return params[0]*jnp.exp(-jnp.sum(jnp.abs((x1 - x2).ravel()) / params[1:]))

class _Matern(BaseKernel):
    """Shared parameter layout for the ARD Matern kernels.

    Each subclass evaluates its kernel as a function of the scaled distance
    ``r = sqrt(sum_i (x1_i - x2_i)^2 / bandwidth_i)``, i.e. ``bandwidth_i`` is
    the squared lengthscale of dimension ``i``.
    """
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of raw parameters (1 amplitude + ``input_dim`` bandwidths)."""
        return 1 + self.input_dim

    def _amp_and_r(self, x1, x2, params):
        """Return the (softplus'd) amplitude and scaled distance ``r``."""
        params = softplus(jnp.asarray(params, dtype=self.dtype).ravel()) # softplusing params for positivity constraints
        x1, x2 = jnp.asarray(x1, dtype=self.dtype), jnp.asarray(x2, dtype=self.dtype)
        return params[0], _safe_sqrt(jnp.sum(((x1 - x2).ravel())**2 / params[1:]))

class Matern12(_Matern):
    """ARD Matern-1/2 (exponential) kernel: ``a * exp(-r)``.

    Sample paths are continuous but not differentiable. Unlike
    :class:`Laplace` this uses the 2-norm, so it is rotation-invariant up to
    the per-dimension bandwidths.
    """
    def eval(self, x1, x2, params):
        """Evaluate the Matern-1/2 kernel between two points.

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length
                ``1 + input_dim``; softplus'd internally to
                ``[amplitude, bandwidth_1, ..., bandwidth_d]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        a, r = self._amp_and_r(x1, x2, params)
        return a*jnp.exp(-r)

class Matern32(_Matern):
    """ARD Matern-3/2 kernel: ``a * (1 + sqrt(3) r) * exp(-sqrt(3) r)``.

    Sample paths are once differentiable.
    """
    def eval(self, x1, x2, params):
        """Evaluate the Matern-3/2 kernel between two points.

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length
                ``1 + input_dim``; softplus'd internally to
                ``[amplitude, bandwidth_1, ..., bandwidth_d]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        a, r = self._amp_and_r(x1, x2, params)
        sr = jnp.sqrt(3.0)*r
        return a*(1.0 + sr)*jnp.exp(-sr)

class Matern52(_Matern):
    """ARD Matern-5/2 kernel: ``a * (1 + sqrt(5) r + 5 r^2 / 3) * exp(-sqrt(5) r)``.

    Sample paths are twice differentiable; a common default for modelling
    physical responses that are smooth but not as smooth as the RBF assumes.
    """
    def eval(self, x1, x2, params):
        """Evaluate the Matern-5/2 kernel between two points.

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length
                ``1 + input_dim``; softplus'd internally to
                ``[amplitude, bandwidth_1, ..., bandwidth_d]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        a, r = self._amp_and_r(x1, x2, params)
        sr = jnp.sqrt(5.0)*r
        return a*(1.0 + sr + sr**2/3.0)*jnp.exp(-sr)

class RationalQuadratic(BaseKernel):
    """ARD rational quadratic kernel: ``a * (1 + r^2 / alpha)^(-alpha)``.

    Here ``r^2 = sum_i (x1_i - x2_i)^2 / bandwidth_i``. It is a scale mixture
    of RBF kernels with different lengthscales, and recovers :class:`ARD`
    (``a * exp(-r^2)``) as ``alpha -> inf``.
    """
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of raw parameters (1 amplitude + ``input_dim`` bandwidths + 1 shape ``alpha``)."""
        return 2 + self.input_dim

    def init_params(self, input_var, output_var):
        """Same as :meth:`BaseKernel.init_params`, with the shape ``alpha`` set to 1.

        Args:
            input_var: Per-dimension variance of the (scaled) training inputs,
                shape ``(input_dim,)``.
            output_var: Scalar variance of the training outputs.

        Returns:
            jax.Array: Raw (unconstrained) parameter array of length ``p_dim``.
        """
        return jnp.concatenate([
            inv_softplus(output_var).reshape(1),
            inv_softplus(2.0 * input_var + self.epsilon),
            inv_softplus(jnp.ones(1, dtype=self.dtype)),
        ]).astype(self.dtype)

    def eval(self, x1, x2, params):
        """Evaluate the rational quadratic kernel between two points.

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length
                ``2 + input_dim``; softplus'd internally to
                ``[amplitude, bandwidth_1, ..., bandwidth_d, alpha]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        params = softplus(jnp.asarray(params, dtype=self.dtype).ravel()) # softplusing params for positivity constraints
        x1, x2 = jnp.asarray(x1, dtype=self.dtype), jnp.asarray(x2, dtype=self.dtype)
        r2 = jnp.sum(((x1 - x2).ravel())**2 / params[1:-1])
        alpha = params[-1]
        return params[0]*(1.0 + r2/alpha)**(-alpha)

class Periodic(BaseKernel):
    """ARD periodic (exp-sine-squared) kernel.

    ``k(x1, x2) = a * exp(-sum_i sin^2(pi * (x1_i - x2_i) / period_i) / bandwidth_i)``

    Each input dimension has its own period and bandwidth. Note that the
    marginal likelihood is typically multimodal in the periods, so the fit
    can be sensitive to their initial values.
    """
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of raw parameters (1 amplitude + ``input_dim`` bandwidths + ``input_dim`` periods)."""
        return 1 + 2*self.input_dim

    def init_params(self, input_var, output_var):
        """Data-driven initial raw parameters.

        The amplitude is the output variance, each bandwidth is 1 (since
        ``sin^2`` lies in ``[0, 1]``) and each period is ``4 * std(x)``,
        i.e. roughly one period across the bulk of the data.

        Args:
            input_var: Per-dimension variance of the (scaled) training inputs,
                shape ``(input_dim,)``.
            output_var: Scalar variance of the training outputs.

        Returns:
            jax.Array: Raw (unconstrained) parameter array of length ``p_dim``.
        """
        return jnp.concatenate([
            inv_softplus(output_var).reshape(1),
            inv_softplus(jnp.ones(self.input_dim, dtype=self.dtype)),
            inv_softplus(4.0 * jnp.sqrt(input_var) + self.epsilon),
        ]).astype(self.dtype)

    def eval(self, x1, x2, params):
        """Evaluate the periodic kernel between two points.

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length
                ``1 + 2*input_dim``; softplus'd internally to
                ``[amplitude, bandwidth_1, ..., bandwidth_d, period_1, ..., period_d]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        params = softplus(jnp.asarray(params, dtype=self.dtype).ravel()) # softplusing params for positivity constraints
        x1, x2 = jnp.asarray(x1, dtype=self.dtype), jnp.asarray(x2, dtype=self.dtype)
        bandwidth, period = params[1:1 + self.input_dim], params[1 + self.input_dim:]
        return params[0]*jnp.exp(-jnp.sum(jnp.sin(jnp.pi*(x1 - x2).ravel()/period)**2 / bandwidth))

class Linear(BaseKernel):
    """ARD linear (dot-product) kernel: ``b + sum_i w_i * x1_i * x2_i``.

    Non-stationary; a GP with this kernel is Bayesian linear regression with
    per-dimension weight variances ``w_i`` and bias variance ``b``. The kernel
    matrix has rank at most ``input_dim + 1``, so it relies on the GP's noise
    variance for a stable Cholesky factorization. Note that the name is
    shared with :class:`~elyza.surrogate.gp.mean.Linear`; import it via the
    module (``kernel.Linear``) or alias it if both are needed.
    """
    @computed_field
    @property
    def p_dim(self) -> int:
        """int: Number of raw parameters (1 bias variance + ``input_dim`` weight variances)."""
        return 1 + self.input_dim

    def init_params(self, input_var, output_var):
        """Data-driven initial raw parameters.

        The weight variances are chosen so each dimension explains an equal
        share of the output variance (``w_i * Var(x_i) = Var(y) / input_dim``),
        and the bias variance is set to the output variance.

        Args:
            input_var: Per-dimension variance of the (scaled) training inputs,
                shape ``(input_dim,)``.
            output_var: Scalar variance of the training outputs.

        Returns:
            jax.Array: Raw (unconstrained) parameter array of length ``p_dim``.
        """
        weights = output_var / (self.input_dim * (input_var + self.epsilon))
        return jnp.concatenate([inv_softplus(output_var).reshape(1), inv_softplus(weights)]).astype(self.dtype)

    def eval(self, x1, x2, params):
        """Evaluate the linear kernel between two points.

        Args:
            x1: First point, shape ``(input_dim,)``.
            x2: Second point, shape ``(input_dim,)``.
            params: Raw (unconstrained) parameter array of length
                ``1 + input_dim``; softplus'd internally to
                ``[bias_variance, weight_variance_1, ..., weight_variance_d]``.

        Returns:
            jax.Array: Scalar kernel value ``k(x1, x2)``.
        """
        params = softplus(jnp.asarray(params, dtype=self.dtype).ravel()) # softplusing params for positivity constraints
        x1, x2 = jnp.asarray(x1, dtype=self.dtype), jnp.asarray(x2, dtype=self.dtype)
        return params[0] + jnp.sum(params[1:]*x1.ravel()*x2.ravel())
