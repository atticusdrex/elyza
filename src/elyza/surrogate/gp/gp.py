from .kernel import *
from .mean import *
from elyza.util.imports import * 
from elyza.surrogate.abstract import Surrogate
from elyza.optim.abstract import BatchGradientOptimizer, OptimizerOptions
from elyza.util.helpers import ensure_2d, ls, softplus, inv_softplus, kernel_mat
from jax.scipy.linalg import solve_triangular, cho_solve, cholesky

'''
~---------------------------------------------~
|  Vanilla Gaussian Process regression class  |
~---------------------------------------------~
'''
class GaussianProcess(Surrogate):
    # public fields
    input_dim: int = Field(description = "input dimension")
    kernel_cls: type[BaseKernel] = Field(description = "kernel class")
    mean_cls: type[BaseMean] = Field(description = "mean class")
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
    _alpha: jax.Array | None = PrivateAttr(default=None)

    # post-init class to run once the base variables are run
    def model_post_init(self, __context):
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
            'noise': inv_softplus(self.noise_var + self.eps)
        }

    '''
    Standard-scales X (zero mean, unit variance per input dimension) using statistics
    fit once from the first training batch and reused for every later fit/predict/update
    call, so the model always sees inputs on a consistent, well-conditioned scale.
    '''
    def _scale(self, X: jax.Array) -> jax.Array:
        return (X - self._X_mean) / self._X_std

    '''
    Data-driven initialization of the kernel/mean hyperparameters, run once on the
    training data before the optimizer takes its first step (instead of the naive
    all-ones default from model_post_init):
      - kernel bandwidth(s) from the per-dimension input variance
      - kernel amplitude from the output variance
      - mean function from an ordinary least-squares fit of Y on [1, X], which
        reduces to the sample mean of Y when the mean has no linear term (e.g. Constant)
    '''
    def _smart_init(self, X: jax.Array, Y: jax.Array):
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
        self._L = self._get_L(X, self.p['kernel'], self.p['noise'])
        self._alpha = self._get_alpha(self._L, self.p['mean'])

    def _calibrate_noise(self, max_cond=1e5) -> None:
        '''Increase white noise variance to lower the kernel condition number.'''
        L = self._get_L(self._X, self.p['kernel'], self.p['noise'])
        Kmat = L @ L.T
        cond_num = jnp.linalg.cond(Kmat)
        lambda_max = jnp.linalg.matrix_norm(Kmat)
        lambda_min = lambda_max / cond_num
        max_cond = min(max_cond, cond_num)
        sigma_opt = (lambda_max - max_cond * lambda_min) / ((max_cond - 1) + self.eps) + self.eps
        self.p['noise'] = inv_softplus(max(softplus(self.p['noise']), sigma_opt))

        self.verbose and print("Calibrated white noise variance: %.4e" % (softplus(self.p['noise'])))

    def _get_L(self, X, k_param, noise_var) -> jax.Array:
        '''Return lower-triangular Cholesky factor of the training kernel.'''
        Ktrain = kernel_mat(X, X, self._kernel, k_param) + (self.eps + softplus(noise_var)) * jnp.eye(X.shape[0])
        return cholesky(Ktrain, lower=True)

    def _get_alpha(self, L, m_param) -> jax.Array:
        '''Solve for alpha = K^{-1} (Y - m(X)) using Cholesky solve.'''
        mean = ensure_2d(jnp.asarray(self._mean.eval(self._X, m_param)))
        return cho_solve((L, True), self._Y - mean)

    def predict(self, X, full_cov=False) -> tuple[jax.Array]:
        '''
        Compute posterior mean and covariance (full or marginal variances).
        '''
        X = self._scale(jnp.array(X))
        Ktest = kernel_mat(X, self._X, self._kernel, self.p['kernel'])
        mean = ensure_2d(jnp.asarray(self._mean.eval(X, self.p['mean'])))
        mu = (Ktest @ self._alpha + mean).ravel()

        if full_cov:
            cov = kernel_mat(X, X, self._kernel, self.p['kernel']) - Ktest @ cho_solve((self._L, True), Ktest.T)
            return mu, cov
        else:
            Kaux = (jax.vmap(lambda x: self._kernel.eval(x, x, self.p['kernel']))(X)).ravel()
            alpha = cho_solve((self._L, True), Ktest.T)
            cov_diag = Kaux - jnp.sum(Ktest * alpha.T, axis=1)
            return mu, cov_diag

    def sample(self, key, X, n_samples: int = 1) -> jax.Array:
        '''
        Draw samples from the GP posterior at X via the reparameterization trick,
        treating each point independently (using only the marginal variances from
        predict(X, full_cov=False) rather than the full posterior covariance) so
        this stays cheap for large X: z ~ N(0, I), sample = mu + sqrt(var) * z.
        Keeping mu/var in the computation graph (rather than e.g.
        jax.random.multivariate_normal) makes the samples differentiable w.r.t.
        the posterior mean and variance.
        '''
        mu, var = self.predict(X, full_cov=False)
        std = jnp.sqrt(var + self.eps)
        z = jrand.normal(key, shape=(mu.shape[0], n_samples), dtype=mu.dtype)
        return mu.reshape(-1, 1) + std.reshape(-1, 1) * z

    def _objective(self, p, X, Y) -> float:
        # Getting cholesky factors and solve linear system
        L = self._get_L(X, p['kernel'], p['noise'])
        mean = ensure_2d(jnp.asarray(self._mean.eval(X, p['mean'])))
        Ytilde = ensure_2d(Y) - mean
        quad_term = jnp.sum(Ytilde * cho_solve((L, True), Ytilde))
        logdet_term = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        # Return quadratic and log-determinant components
        return quad_term + logdet_term
    '''
    function for setting an optimizer for the estimator 
    '''
    def set_optimizer(self, optimizer : BatchGradientOptimizer, optimizer_opts : OptimizerOptions):
        self._optimizer = optimizer(opts = optimizer_opts)

    def fit(
        self,
        X: np.ndarray | jax.Array,
        Y: np.ndarray | jax.Array,
    ):
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
            self._optimizer.opts.p_init = self.p
            self._calibrate(X, Y, max_cond=self.max_cond, calibrate_noise=self.calibrate_noise)

        self._optimizer.loss_grad_fn = jit(value_and_grad(lambda p, X, Y: self._objective(p, X, Y), argnums=2))

        # run the optimizer
        new_params = self._optimizer.run(X, Y)

        # setting the new params
        self.p = deepcopy(new_params)

        # set the L and alpha values
        self._calibrate(X, Y, self.max_cond)


    def update(self, X: np.ndarray | jax.Array, Y: np.ndarray | jax.Array) -> None:
        if not self._calibrated:
            raise RuntimeError("Model must be fit() at least once before calling update().")

        X = self._scale(jnp.array(X))
        Y = ensure_2d(jnp.array(Y))

        n = self._X.shape[0]
        m = X.shape[0]

        # cross-covariance block kernel_mat(X, X) and new diagonal block kernel_mat(X, X)
        K12 = kernel_mat(self._X, X, self._kernel, self.p['kernel'])                     # (n, m)
        K22 = kernel_mat(X, X, self._kernel, self.p['kernel']) \
            + (self.eps + softplus(self.p['noise'])) * jnp.eye(m)

        # Solve L @ B = K12 so the augmented Cholesky factor remains consistent
        # with the block covariance structure K_aug = [[K11, K12], [K12^T, K22]].
        B_T = solve_triangular(self._L, K12, lower=True)

        # Schur complement, factorized directly (only m x m, cheap)
        schur = K22 - B_T.T @ B_T
        schur = schur + (self.eps + 1e-12) * jnp.eye(K22.shape[0])
        C = cholesky(schur, lower=True)

        # assemble the augmented lower-triangular factor
        L_new = jnp.zeros((n + m, n + m))
        L_new = L_new.at[:n, :n].set(self._L)
        L_new = L_new.at[n:, :n].set(B_T.T)
        L_new = L_new.at[n:, n:].set(C)

        # update stored state
        self._X = jnp.concatenate([self._X, X], axis=0)
        self._Y = jnp.concatenate([self._Y, Y], axis=0)
        self._L = L_new
        self._alpha = self._get_alpha(self._L, self.p['mean'])

