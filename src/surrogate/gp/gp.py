from .util import *
from .kernel import *
from .mean import *
from optim.gradient import ADAM
from pydantic import BaseModel, ConfigDict, PrivateAttr
from surrogate.surrogate import Surrogate 
from jax.scipy.linalg import solve_triangular 

'''
~---------------------------------------------~
|  Vanilla Gaussian Process regression class  |
~---------------------------------------------~
'''
class GaussianProcess(Surrogate):
    # public configuration fields
    input_dim: int
    kernel_cls: type[BaseKernel]
    mean_cls: type[BaseMean]
    calibrate_noise: bool = False
    noise_var: float = 0.0
    eps: float = 1e-12
    max_cond: float = 1e5
    verbose: bool = True
    p: dict | None = None

    # private/internal state
    _kernel: BaseKernel | None = PrivateAttr(default=None)
    _mean: BaseMean | None = PrivateAttr(default=None)
    _X: np.ndarray | jax.Array | None = PrivateAttr(default=None)
    _Y: np.ndarray | jax.Array | None = PrivateAttr(default=None)
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
            'noise': inv_softplus(self.noise_var)
        }

    def _calibrate(self, X: np.ndarray | jax.Array, Y: np.ndarray | jax.Array,
                   max_cond: float, calibrate_noise: bool = False):
        # storing training data
        if self._X is None:
            self._X = X
        if self._Y is None:
            self._Y = Y

        # calibrate the noise to return a specific condition number
        if calibrate_noise:
            self._calibrate_noise(max_cond=max_cond)

        # indicate that the model has been calibrated
        self._calibrated = True

        # store L value for the predict() function
        self._L = self._get_L(self.p['kernel'], self.p['noise'])
        self._alpha = self._get_alpha(self._L, self.p['mean'])

    def _calibrate_noise(self, max_cond=1e5) -> None:
        '''Increase white noise variance to lower the kernel condition number.'''
        L = self._get_L(self.p['kernel'], self.p['noise'])
        Kmat = L @ L.T
        cond_num = jnp.linalg.cond(Kmat)
        lambda_max = jnp.linalg.matrix_norm(Kmat)
        lambda_min = lambda_max / cond_num
        max_cond = min(max_cond, cond_num)
        sigma_opt = (lambda_max - max_cond * lambda_min) / ((max_cond - 1) + self.eps) + self.eps
        self.p['noise'] = inv_softplus(max(softplus(self.p['noise']), sigma_opt))

        self.verbose and print("Calibrated white noise variance: %.4e" % (softplus(self.p['noise'])))

    def _get_L(self, k_param, noise_var) -> jax.Array:
        '''Return lower-triangular Cholesky factor of the training kernel.'''
        Ktrain = K(self._X, self._X, self._kernel, k_param) + (self.eps + softplus(noise_var)) * jnp.eye(self._X.shape[0])
        return cholesky(Ktrain, lower=True)

    def _get_alpha(self, L, m_param) -> jax.Array:
        '''Solve for alpha = K^{-1} (Y - m(X)) using Cholesky solve.'''
        return cho_solve((L, True), self._Y - self._mean.eval(self._X, m_param))

    def predict(self, X, full_cov=False) -> tuple[jax.Array]:
        '''
        Compute posterior mean and covariance (full or marginal variances).
        '''
        Ktest = K(X, self._X, self._kernel, self.p['kernel'])
        mu = (Ktest @ self._alpha + self._mean.eval(X, self.p['mean'])).ravel()

        if full_cov:
            cov = K(X, X, self._kernel, self.p['kernel']) - Ktest @ cho_solve((self._L, True), Ktest.T)
            return mu, cov
        else:
            Kaux = (jax.vmap(lambda x: self._kernel.eval(x, x, self.p['kernel']))(X)).ravel()
            alpha = cho_solve((self._L, True), Ktest.T)
            cov_diag = Kaux - jnp.sum(Ktest * alpha.T, axis=1)
            return mu, cov_diag

    def _score_function(self, X, Y, p) -> float:
        # Getting cholesky factors and solve linear system
        L = self._get_L(p['kernel'], p['noise'])
        Ytilde = Y - self._mean.eval(X, p['mean'])
        quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
        logdet_term = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        # Return quadratic and log-determinant components
        return quad_term + logdet_term

    def fit(
        self,
        X: np.ndarray | jax.Array,
        Y: np.ndarray | jax.Array,
        solver: str = "adam",
        learning_rate: float = 1e-3,
        steps: int = 1000,
        beta1: float = 0.9,
        beta2: float = 0.999,
        active_params: dict = None
    ):
        # converting training data to jax arrays
        X, Y = jnp.array(X), jnp.array(Y)

        # calibrate the model if not calibrated
        if not self._calibrated:
            self._calibrate(X, Y, max_cond=self.max_cond, calibrate_noise=self.calibrate_noise)

        # jit-compiled objective function
        loss_grad_fn = jit(value_and_grad(lambda p: self._score_function(X, Y, p)))

        # running the optimizer
        if solver == "adam":
            self._optimizer = ADAM(
                loss_grad_fn=loss_grad_fn,
                constraints=None,
                beta1=beta1,
                beta2=beta2,
                lr=learning_rate,
                eps=self.eps
            )
            new_params = self._optimizer.run(
                lr=learning_rate,
                steps=steps,
                p_init=self.p,
                active_params=active_params,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # setting the new params
        self.p = deepcopy(new_params)

        # set the L and alpha values
        self._calibrate(X, Y, self.max_cond)


    def update(self, X: np.ndarray | jax.Array, Y: np.ndarray | jax.Array) -> None:
        if not self._calibrated:
            raise RuntimeError("Model must be fit() at least once before calling update().")

        X = jnp.array(X)
        Y = jnp.array(Y)

        n = self._X.shape[0]
        m = X.shape[0]

        # cross-covariance block K(X, X) and new diagonal block K(X, X)
        K12 = K(self._X, X, self._kernel, self.p['kernel'])                     # (n, m)
        K22 = K(X, X, self._kernel, self.p['kernel']) \
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


'''
~-----------------------------------------~
|  Kennedy O'Hagan type Gaussian Process  |
~-----------------------------------------~
Gaussian Process where instead of approximating y(x), we approximate: 
δ(x) = y1(x) - ρ * y2(x). 

The parameter ρ is calibrated via maximum marginal likelihood estimation. 
'''
class DeltaGP(GaussianProcess):
    _Y1: np.ndarray | jax.Array | None = PrivateAttr(default=None)
    _Y2: np.ndarray | jax.Array | None = PrivateAttr(default=None)

    def model_post_init(self, __context):
        # initializing regular GP mean and kernel functions w/ parameters
        super().model_post_init(__context) 

        # initializing rho to 1.0 (identity transform) 
        self.p['rho'] = 1.0 
    
    def _calibrate(self, X: np.ndarray | jax.Array, Y1: np.ndarray | jax.Array, Y2 : np.ndarray | jax.Array, max_cond: float, calibrate_noise: bool = False):
        # storing training data
        if self._X is None:
            self._X = X
        if self._Y is None:
            self._Y1 = Y1
            self._Y2 = Y2 

        # calibrate the noise to return a specific condition number
        if calibrate_noise:
            self._calibrate_noise(max_cond=max_cond)

        # indicate that the model has been calibrated
        self._calibrated = True

        # store L value for the predict() function
        self._L = self._get_L(self.p['kernel'], self.p['noise'])
        self._alpha = self._get_alpha(self._L, self.p['mean'], self.p['rho'])

    def _get_alpha(self, L, m_param, rho) -> jax.Array:
        '''Solve for alpha = K^{-1} (Y1 - ρ Y2 - m(X)) using Cholesky solve.'''
        Ytilde = self._Y1 - rho * self._Y2 # output difference calculation 
        return cho_solve((L, True), Ytilde - self._mean.eval(self._X, m_param))

    def _score_function(self, X, Y1, Y2, p) -> float:
        # Getting cholesky factors and solve linear system
        L = self._get_L(p['kernel'], p['noise'])
        Ytilde = Y1 - p['rho'] * Y2 - self._mean.eval(X, p['mean'])
        quad_term = jnp.inner(Ytilde, cho_solve((L, True), Ytilde))
        logdet_term = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

        # Return quadratic and log-determinant components
        return quad_term + logdet_term
    
    def fit(
        self,
        X: np.ndarray | jax.Array,
        Y1: np.ndarray | jax.Array,
        Y2: np.ndarray | jax.Array, 
        solver: str = "adam",
        learning_rate: float = 1e-3,
        steps: int = 1000,
        beta1: float = 0.9,
        beta2: float = 0.999,
        active_params: dict = {'rho':True, 'mean':True, 'kernel':True, 'noise':False}
    ):
        # converting training data to jax arrays
        X, Y1, Y2 = jnp.array(X), jnp.array(Y1), jnp.array(Y2)

        # calibrate the model if not calibrated
        if not self._calibrated:
            self._calibrate(X, Y1, Y2, max_cond=self.max_cond, calibrate_noise=self.calibrate_noise)

        # jit-compiled objective function
        loss_grad_fn = jit(value_and_grad(lambda p: self._score_function(X, Y1, Y2, p)))

        # running the optimizer
        if solver == "adam":
            self._optimizer = ADAM(
                loss_grad_fn=loss_grad_fn,
                constraints=None,
                beta1=beta1,
                beta2=beta2,
                lr=learning_rate,
                eps=self.eps
            )
            new_params = self._optimizer.run(
                lr=learning_rate,
                steps=steps,
                p_init=self.p,
                active_params=active_params,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"Unknown solver: {solver}")

        # setting the new params
        self.p = deepcopy(new_params)

        # set the L and alpha values
        self._calibrate(X, Y1, Y2, self.max_cond)

    def update(self, X: np.ndarray | jax.Array, Y1_new: np.ndarray | jax.Array, Y2_new: np.ndarray | jax.Array) -> None:
        if not self._calibrated:
            raise RuntimeError("Model must be fit() at least once before calling update().")

        X = jnp.array(X)
        Y1_new, Y2_new = jnp.array(Y1_new), jnp.array(Y2_new)

        n = self._X.shape[0]
        m = X.shape[0]

        # cross-covariance block K(X, X) and new diagonal block K(X, X)
        K12 = K(self._X, X, self._kernel, self.p['kernel'])                     # (n, m)
        K22 = K(X, X, self._kernel, self.p['kernel']) \
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
        self._Y1 = jnp.concatenate([self._Y1, Y1_new], axis=0)
        self._Y2 = jnp.concatenate([self._Y2, Y2_new], axis=0)
        self._L = L_new
        self._alpha = self._get_alpha(self._L, self.p['mean'], self.p['rho'])

