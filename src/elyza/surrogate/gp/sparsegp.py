"""Sparse variational Gaussian Process regression.

Defines :class:`SparseGP`, a :class:`~elyza.surrogate.abstract.Surrogate`
that approximates a full GP with a small set of inducing points and a
variational posterior over their function values, fit by maximizing a
Monte Carlo estimate of the evidence lower bound (ELBO). Trades the exact
``O(n^3)`` Cholesky of a full :class:`~elyza.surrogate.gp.gp.GaussianProcess`
for an ``O(n * m^2)`` cost in the number of inducing points ``m``, at the
cost of an approximate (rather than exact) posterior.
"""
from elyza.util.imports import *
from elyza.surrogate import Surrogate
from elyza.surrogate.gp.kernel import BaseKernel
from elyza.surrogate.gp.mean import BaseMean
from elyza.optim.abstract import BatchGradientOptimizer, OptimizerOptions
from elyza.util.helpers import ensure_2d, ls, softplus, inv_softplus, kernel_mat, greedy_k_center, cholesky_KL_div
from jax.scipy.linalg import solve_triangular, cho_solve, cholesky

class SparseGP(Surrogate):
    """Sparse variational Gaussian Process with a pluggable kernel and mean function.

    Attributes:
        input_dim: Input dimension.
        kernel_cls: Kernel class.
        mean_cls: Mean class.
        n_inducing_points: Number of inducing points.
        calibrate_noise: Whether or not to calibrate the noise variance to
            reduce the condition number of the inducing-point kernel matrix.
        noise_var: Variance of Gaussian white noise in the model outputs.
        eps: Small positive jitter value to avoid singular kernel matrices
            and divide-by-zero errors.
        max_cond: Maximum condition number for kernel matrices.
        verbose: Whether or not to print the training and calibration progress.
        p: Model parameters (``kernel``, ``mean``, ``noise``, ``inducing``,
            ``q_mu``, ``q_L``); auto-initialized if not given.
        _kernel: Instantiated kernel object built from ``kernel_cls``.
        _mean: Instantiated mean object built from ``mean_cls``.
        _X: Stored (scaled) training inputs.
        _Y: Stored training outputs.
        _X_mean: Per-dimension input mean used for input standardization.
        _X_std: Per-dimension input scale used for input standardization.
        _calibrated: Whether the model has completed at least one fit/calibration.
        _optimizer: The optimizer instance assigned via :meth:`set_optimizer`.
        _L: Lower-triangular Cholesky factor of the inducing-point kernel matrix.
    """
    # public fields
    input_dim: int = Field(description = "input dimension")
    kernel_cls: type[BaseKernel] = Field(description = "kernel class")
    mean_cls: type[BaseMean] = Field(description = "mean class")
    n_inducing_points : int = Field(description = "number of inducing points")
    calibrate_noise: bool = Field(default = False, description = "whether or not to calibrate the noise variance to reduce the condition number of the kernel matrix")
    noise_var: float = Field(default = 0.0, description = "variance of Gaussian white noise in the model outputs")
    eps: float = Field(default = 1e-12, description = "small positive jitter value to avoid singular kernel matrices and divide-by-zero errors")
    max_cond: float = Field(default = 1e5, description = "maximum condition number for kernel matrices")
    verbose: bool = Field(default = False, description = "whether or not to print the training and calibration progress")
    p: dict | None = Field(default = None, description = "an optional value depending on whether the user wants to instantiate the GP with predefined model parameters")

    # private/internal state
    _kernel: BaseKernel | None = PrivateAttr(default=None)
    _mean: BaseMean | None = PrivateAttr(default=None)
    _X: np.ndarray | jax.Array | None = PrivateAttr(default=None)
    _Y: np.ndarray | jax.Array | None = PrivateAttr(default=None)
    _X_mean: jax.Array | None = PrivateAttr(default=None)
    _X_std: jax.Array | None = PrivateAttr(default=None)
    _calibrated: bool = PrivateAttr(default=False)
    _optimizer: object | None = PrivateAttr(default=None)
    _L: jax.Array | None = PrivateAttr(default=None)
    # post-init class to run once the base variables are run
    def model_post_init(self, __context):
        """Instantiate the kernel/mean objects and the initial (all-ones) parameters."""
        # instantiating mean and kernel classes
        self._kernel = self.kernel_cls(
            input_dim=self.input_dim,
            epsilon=self.eps
        )
        self._mean = self.mean_cls(
            input_dim=self.input_dim,
            epsilon=self.eps
        )

        # instantiating the parameters
        self.p = {
            'kernel': jnp.ones(self._kernel.p_dim),
            'mean': jnp.ones(self._mean.p_dim),
            'noise': inv_softplus(self.noise_var + self.eps),
            'inducing':None, 
            'q_mu':None, 
            'q_L':None 
        }

    def _scale(self, X: jax.Array) -> jax.Array:
        """Standard-scale ``X`` using statistics fit from the first training batch.

        Standard-scales X (zero mean, unit variance per input dimension)
        using statistics fit once from the first training batch and reused
        for every later fit/predict/update call, so the model always sees
        inputs on a consistent, well-conditioned scale.

        Args:
            X: Inputs to scale, shape ``(n_samples, input_dim)``.

        Returns:
            jax.Array: Scaled inputs, same shape as ``X``.
        """
        return (X - self._X_mean) / self._X_std

    def _smart_init(self, X: jax.Array, Y: jax.Array):
        """Data-driven initialization of the kernel/mean hyperparameters.

        Runs once on the training data before the optimizer takes its first
        step (instead of the naive all-ones default from
        ``model_post_init``):

        - kernel bandwidth(s) from the per-dimension input variance
        - kernel amplitude from the output variance
        - mean function from an ordinary least-squares fit of ``Y`` on
          ``[1, X]``, which reduces to the sample mean of ``Y`` when the
          mean has no linear term (e.g. :class:`~elyza.surrogate.gp.mean.Constant`)

        Args:
            X: (Already-scaled) training inputs, shape ``(n_samples, input_dim)``.
            Y: Training outputs, shape ``(n_samples, output_dim)``.
        """
        input_var = jnp.var(X, axis=0)
        output_var = jnp.var(Y) + self.eps

        # kernel: amplitude from output variance, bandwidth(s) from input variance.
        # E[(x_i - x_j)^2] = 2*Var(x) for iid x_i, x_j, matching the exp(-d^2/bandwidth) form.
        n_bandwidth = self._kernel.p_dim - 1
        if n_bandwidth == input_var.shape[0]:
            bandwidth = 2.0 * input_var + self.eps
        else:
            # kernels with a single shared bandwidth (e.g. RBF) -- use the average
            bandwidth = jnp.full((n_bandwidth,), 2.0 * jnp.mean(input_var) + self.eps)
        self.p['kernel'] = jnp.concatenate([inv_softplus(output_var).reshape(1), inv_softplus(bandwidth)])

        # mean: OLS fit of Y on [1, X] (Linear-style p_dim), or just mean(Y) for a
        # constant-only mean; a Zero mean (p_dim == 0) has nothing to initialize
        if self._mean.p_dim == 1 + X.shape[1]:
            design = jnp.concatenate([jnp.ones((X.shape[0], 1)), X], axis=1)
            self.p['mean'] = ls(design, Y).ravel()
        elif self._mean.p_dim == 1:
            self.p['mean'] = jnp.atleast_1d(jnp.mean(Y))

    def _calibrate(self, X: np.ndarray | jax.Array, Y: np.ndarray | jax.Array,
                   max_cond: float, calibrate_noise: bool = False):
        """Store training data and (re)compute the inducing-point Cholesky factor.

        Args:
            X: (Scaled) training inputs.
            Y: Training outputs.
            max_cond: Maximum allowed condition number, forwarded to
                :meth:`_calibrate_noise`.
            calibrate_noise: If ``True``, inflate the noise variance to
                bring the inducing-point kernel matrix's condition number
                under ``max_cond``.
        """
        # storing training data
        if self._X is None:
            self._X = X
        if self._Y is None:
            self._Y = ensure_2d(Y)

        # calibrate the noise to return a specific condition number
        if calibrate_noise:
            self._calibrate_noise(max_cond=max_cond)

        # indicate that the model has been calibrated
        self._calibrated = True

        # store L value for the predict() function
        self._L = self._get_L(self.p['inducing'], self.p['kernel'], self.p['noise'])

    def _calibrate_noise(self, max_cond=1e5) -> None:
        """Increase white noise variance to lower the kernel condition number.

        Args:
            max_cond: Maximum allowed condition number of the training
                kernel matrix.
        """
        L = self._get_L(self.p['inducing'], self.p['kernel'], self.p['noise'])
        Kmat = L @ L.T
        cond_num = jnp.linalg.cond(Kmat)
        lambda_max = jnp.linalg.matrix_norm(Kmat)
        lambda_min = lambda_max / cond_num
        max_cond = min(max_cond, cond_num)
        sigma_opt = (lambda_max - max_cond * lambda_min) / ((max_cond - 1) + self.eps) + self.eps
        self.p['noise'] = inv_softplus(max(softplus(self.p['noise']), sigma_opt))

        self.verbose and print("Calibrated white noise variance: %.4e" % (softplus(self.p['noise'])))

    def _get_L(self, X, k_param, noise_var) -> jax.Array:
        """Return the lower-triangular Cholesky factor of a kernel matrix.

        Called with the inducing-point locations (``p['inducing']``)
        rather than the full training set, so this factors the much
        smaller ``(n_inducing_points, n_inducing_points)`` kernel matrix.

        Args:
            X: Points to build the kernel matrix from, shape ``(n, input_dim)``.
            k_param: Kernel parameter array.
            noise_var: Raw (softplus-inverted) noise variance.

        Returns:
            jax.Array: Lower-triangular Cholesky factor, shape ``(n, n)``.
        """
        Ktrain = kernel_mat(X, X, self._kernel, k_param) + (self.eps + softplus(noise_var)) * jnp.eye(X.shape[0])
        return cholesky(Ktrain, lower=True)

    def predict(self, X, key = jrand.PRNGKey(42), n_samples : int = 100) -> tuple[jax.Array]:
        """Estimate the posterior mean and variance at new points via Monte Carlo.

        Draws ``n_samples`` posterior function samples at ``X`` (see
        :meth:`_samp`) and returns their empirical mean and variance --
        there is no closed-form marginal here since the variational
        inducing-point posterior is sampled rather than solved exactly.

        Args:
            X: Raw (unscaled) query inputs, shape ``(n_queries, input_dim)``.
            key: A JAX PRNG key for the Monte Carlo draws.
            n_samples: Number of posterior samples to draw per query point.

        Returns:
            tuple[jax.Array]: ``(mu, var)``, each shape ``(n_queries,)`` --
            the empirical mean and variance across the Monte Carlo samples.
        """
        ysamp = self._samp(key, X, n_samples, self.p)
        return ysamp.mean(axis=1), ysamp.std(axis=1)**2

    def _samp(self, key, X, n_samples, p):
        """Draw posterior function samples via the inducing-point reparameterization trick.

        Samples the variational posterior ``q(u) ~ N(q_mu, q_L @ q_L.T)``
        at the inducing points, then pushes each draw through the GP
        conditional mean to obtain a posterior function sample at ``X``.

        Args:
            key: A JAX PRNG key.
            X: Raw (unscaled) query inputs, shape ``(n, input_dim)`` --
                scaled internally with :meth:`_scale` before evaluation.
            n_samples: Number of samples to draw.
            p: Parameter pytree (``kernel``, ``mean``, ``noise``,
                ``inducing``, ``q_mu``, ``q_L``) to evaluate at.

        Returns:
            jax.Array: Samples, shape ``(n, n_samples)``.
        """
        X = self._scale(jnp.array(X))

        # Getting cholesky factors and solve linear system
        p_mu = self._mean.eval(p['inducing'], p['mean'])

        L = self._get_L(p['inducing'], p['kernel'], p['noise'])
        q_sample = (p['q_mu'].reshape(-1,1) - p_mu) + p['q_L'] @ jrand.normal(key,shape=(self.n_inducing_points, n_samples))
        alpha_sample = cho_solve((L, True), q_sample) 
        k_train = kernel_mat(X, p['inducing'], self._kernel, p['kernel']) 
        return self._mean.eval(X, p['mean']).reshape(-1,1) + k_train @ alpha_sample

    def sample(self, key, X, n_samples: int = 1) -> jax.Array:
        """Draw samples from the sparse GP's variational posterior at ``X``.

        Delegates directly to :meth:`_samp`, which samples the inducing-point
        variational distribution ``q(u) ~ N(q_mu, q_L @ q_L.T)`` via the
        reparameterization trick and pushes each draw through the GP
        conditional mean to obtain a posterior function sample. ``X`` is
        scaled internally, same as in :meth:`predict`.

        Args:
            key: A JAX PRNG key.
            X: Raw (unscaled) query inputs, shape ``(n, input_dim)``.
            n_samples: Number of independent samples to draw.

        Returns:
            jax.Array: Samples, shape ``(n, n_samples)``.
        """
        return self._samp(key, X, n_samples, self.p)


    def _objective(self,key, X, Y, p, n_mc = 25) -> float:
        """Compute a Monte Carlo estimate of the negative ELBO training objective.

        Draws ``n_mc`` samples from the variational posterior (via the same
        reparameterization as :meth:`_samp`) to Monte-Carlo estimate the
        expected negative data log-likelihood on this batch, rescaled to the
        full training set size, and adds the KL divergence between the
        variational distribution and the GP prior at the inducing points
        (:func:`~elyza.util.helpers.cholesky_KL_div`).

        Args:
            key: A JAX PRNG key for the Monte Carlo draws.
            X: Training inputs (batch), shape ``(n_batch, input_dim)``.
            Y: Training outputs (batch), shape ``(n_batch, output_dim)``.
            p: Parameter pytree (``kernel``, ``mean``, ``noise``,
                ``inducing``, ``q_mu``, ``q_L``) to evaluate at.
            n_mc: Number of Monte Carlo samples used to estimate the
                expected log-likelihood term.

        Returns:
            float: The negative ELBO estimate (expected negative
            log-likelihood plus the KL-divergence term).
        """
        p_L = self._get_L(p['inducing'], p['kernel'], p['noise'])
        p_mu = self._mean.eval(p['inducing'], p['mean'])
        q_L = p['q_L']
        L = self._get_L(p['inducing'], p['kernel'], p['noise'])
        q_sample = (p['q_mu'].reshape(-1,1) - p_mu) + q_L @ jrand.normal(key,shape=(self.n_inducing_points, n_mc))
        alpha_sample = cho_solve((L, True), q_sample) 
        k_train = kernel_mat(X, p['inducing'], self._kernel, p['kernel']) 
        yhat_samp = self._mean.eval(X, p['mean']).reshape(-1,1) + k_train @ alpha_sample
        mean_error = ((yhat_samp - Y)**2).mean(axis=1)

        # compute log-likelihood term 
        N = self._X.shape[0]
        n_batch = mean_error.shape[0]
        log_likelihood = (N / n_batch) * 0.5 * (n_batch * jnp.log(2 * jnp.pi * (softplus(p['noise']) + self.eps)) + jnp.inner(mean_error, mean_error) / (self.eps + softplus(p['noise'])))


        # computingthe kl-divergence between the variational distribution and the prior
        kl_term = cholesky_KL_div(p['q_mu'], q_L, p_mu.ravel(),p_L)                                                                                                    
        return log_likelihood + kl_term 

    def set_optimizer(self, optimizer : BatchGradientOptimizer, optimizer_opts : OptimizerOptions):
        """Assign the optimizer (and its options) used by :meth:`fit`.

        Args:
            optimizer: A :class:`~elyza.optim.abstract.BatchGradientOptimizer` class.
            optimizer_opts: An :class:`~elyza.optim.abstract.OptimizerOptions`
                instance configuring that optimizer.
        """
        self._optimizer = optimizer(opts = optimizer_opts)

    def fit(
        self,
        X: np.ndarray | jax.Array,
        Y: np.ndarray | jax.Array,
        n_monte_carlo : int = 25, 
        random_state : int = 42 
    ):
        """Fit kernel/mean/noise hyperparameters and the variational inducing-point posterior.

        On the first call, this also fits the input standard-scaler,
        data-driven-initializes the hyperparameters (see
        :meth:`_smart_init`), selects ``n_inducing_points`` inducing
        locations via a greedy k-center heuristic
        (:func:`~elyza.util.helpers.greedy_k_center`), initializes the
        variational posterior (``q_mu``, ``q_L``), and (optionally)
        calibrates the noise variance for numerical conditioning.

        Args:
            X: Training inputs, shape ``(n_samples, input_dim)``.
            Y: Training outputs, shape ``(n_samples, output_dim)``.
            n_monte_carlo: Number of Monte Carlo samples used to estimate
                the training objective (see :meth:`_objective`) at every step.
            random_state: Random seed used for inducing-point selection
                (on the first call) and the objective's Monte Carlo draws.

        Raises:
            AssertionError: If :meth:`set_optimizer` has not been called
                yet, or (on the first call) if ``X`` has no more rows than
                ``n_inducing_points``.
        """
        # making sure an optimizer has been declared
        assert self._optimizer is not None, "must declare an optimizer"

        # making sure p_init is specified
        self._optimizer.opts.p_init = deepcopy(self.p)

        # converting training data to jax arrays
        X, Y = ensure_2d(jnp.array(X)), ensure_2d(jnp.array(Y))

        # fit the input scaler once, from the first-ever training batch, and reuse it for
        # every later fit/predict/update call on this model
        if not self._calibrated:
            self._X_mean = jnp.mean(X, axis=0)
            self._X_std = jnp.std(X, axis=0) + self.eps
        X = self._scale(X)

        # calibrate the model if not calibrated
        if not self._calibrated:
            # data-driven starting point instead of the naive all-ones default, before the
            # optimizer takes its first step. This replaces whatever p_init the caller passed
            # in for this first-ever fit -- in practice callers just echo self.p as p_init, so
            # this makes sure that echoed value is actually the smart-initialized one. Runs
            # only this once: _calibrated flips True below and gates out every later fit() call.
            self._smart_init(X, Y)

            # initializing the inducing points 
            assert X.shape[0] > self.n_inducing_points, "you must have more training data than inducing points!"
            self.p['inducing'], inds = greedy_k_center(jrand.PRNGKey(random_state), X, self.n_inducing_points) 

            # initializing the variational output distribution parameters 
            self.p['q_mu'] = (Y[inds] - self._mean.eval(X[inds], self.p['mean'])).ravel()
            self.p['q_L'] = jnp.eye(self.n_inducing_points) * self.eps 

            self._optimizer.opts.p_init = self.p
            self._calibrate(X, Y, max_cond=self.max_cond, calibrate_noise=self.calibrate_noise)

        self._optimizer.loss_grad_fn = jit(value_and_grad(lambda X, Y, p: self._objective(jrand.PRNGKey(random_state), X, Y, p, n_mc = n_monte_carlo), argnums=2))

        # setting the constraints of the variational points 
        # clipping the variational inputs so they stay within bounds
        constraints = {
            'inducing':lambda Z: jnp.maximum(X.min(axis=0).reshape(-1,1), jnp.minimum(X.max(axis=0).reshape(-1,1), Z)), 
            'q_L': lambda L: jnp.tril(L, k=-1) + jnp.diag(jnp.maximum(jnp.diag(L), self.eps))
        }

        self._optimizer.opts.constraints = constraints 

        # run the optimizer
        new_params = self._optimizer.run(X, Y)

        # setting the new params
        self.p = deepcopy(new_params)

        # set the L and alpha values
        self._calibrate(X, Y, self.max_cond)


    def update(self, X: np.ndarray | jax.Array, Y: np.ndarray | jax.Array) -> None:
        """Not implemented for :class:`SparseGP`.

        Unlike the full :class:`~elyza.surrogate.gp.gp.GaussianProcess`,
        there is no dense training kernel matrix here to rank-update --
        incorporating new data would mean refitting (or moving) the
        variational inducing-point posterior instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("this method is not implemented on SparseGP as there is no kernel matrix to update")


    