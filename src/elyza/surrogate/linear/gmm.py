from elyza.util.imports import *
from elyza.surrogate import Surrogate
from elyza.util.helpers import ensure_2d, softmax
from elyza.optim.abstract import Optimizer, OptimizerOptions
from elyza.core import GaussianMixture
from jax.scipy.special import logsumexp

class GMMRegression(Surrogate): 
    # public attributes 
    input_dim : int = Field(default = 1, description = "input dimension")
    n_dist : int = Field(default = 3, description = "number of gaussians in the mixture model")
    scale: float = Field(default = 1.0, description = "number to inflate the initial distributions of GMMs")
    random_state : int = Field(default = 42, description = "random state to initialize the GMM parameters")
    
    # private attributes 
    _dist : type[GaussianMixture] | None = PrivateAttr(default = None) 
    _X : jax.Array | None = PrivateAttr(default = None) 
    _Y : jax.Array | None = PrivateAttr(default = None) 
    _optimizer : type[Optimizer] | None = PrivateAttr(default = None) 

    def model_post_init(self, __context):
        # initializing the parent class 
        super().model_post_init(__context) 

        # initializing the gaussian mixture model parameters
        keys = jrand.split(jrand.PRNGKey(self.random_state), self.n_dist) 
        means = [self.scale * jrand.normal(key, shape = (self.input_dim, 1)) for key in keys]
        covs = [self.scale * jnp.eye(self.input_dim) for _ in keys] 

        # declaring the mixture model 
        self._dist = GaussianMixture(means, covs, weights = jnp.ones(self.n_dist) / self.n_dist)

    # computing discrete samples from the GMM
    def _samp(self, key:jax.Array, X : jax.Array, n_samples : int, p:dict) -> jax.Array: 
        # assigning the distribution parameters 
        self._dist._p = p 
        # sampling the distribution 
        beta_samples = self._dist.sample(key, n_samples) # of size (n_samples, self.dim) 
        # computing the online predictions 
        return X @ beta_samples.T # shape (n_observations, n_samples) 

    def _objective(self, p, X, Y) -> float:
        # a linear map of a Gaussian mixture is again a Gaussian mixture,
        # so the predictive density at each row of X is available in
        # closed form -- no Monte Carlo sampling needed. minimizing its
        # negative log-likelihood (rather than plain squared error) keeps
        # a log(variance) term that penalizes the predictive variance
        # collapsing to zero, which squared error does not.
        mean, L, w = p['mean'], p['L'], softmax(p['w'])  # (K, dim), (K, dim, dim), (K,)

        comp_mean = X @ mean.T                                # (n_obs, K)
        XL = jnp.einsum('nd,kde->nke', X, L)                  # x^T L_k per component
        comp_var = jnp.sum(XL**2, axis=-1) + self._dist._eps  # (n_obs, K), x^T Sigma_k x

        log_normal = -0.5 * (jnp.log(2 * jnp.pi * comp_var) + (Y - comp_mean)**2 / comp_var)
        log_mix = logsumexp(jnp.log(w) + log_normal, axis=-1)  # (n_obs,)

        return -jnp.mean(log_mix)

    def fit(
        self,
        X: np.ndarray | jax.Array,
        Y: np.ndarray | jax.Array,
    ) -> None:
        """Fit the surrogate model to training data.

        Args:
            X: Training inputs.
            Y: Training outputs.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        # making sure an optimizer has been declared
        assert self._optimizer is not None, "must declare an optimizer"

        # making sure p_init is specified
        self._optimizer.opts.p_init = deepcopy(self._dist._p)

        # setting the proper constraints to the optimizer
        self._optimizer.opts.constraints = deepcopy(self._dist._constr)

        # converting training data to jax arrays
        X, Y = ensure_2d(jnp.array(X, dtype=self.dtype)), ensure_2d(jnp.array(Y, dtype=self.dtype))

        # setting the loss function -- the exact GMM log-likelihood, so no
        # PRNG key/sampling is needed here
        self._optimizer.loss_grad_fn = jit(value_and_grad(
            lambda X, Y, p: self._objective(p, X, Y), argnums=2
        ))

        # run the optimizer
        new_params = self._optimizer.run(X, Y)

        # setting the new params
        self._dist._p = deepcopy(new_params)


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
        self._optimizer = optimizer(opts = optimizer_opts)

    def predict(
        self,
        X: np.ndarray | jax.Array,
        key : jax.Array = jrand.PRNGKey(42),
        n_samples : int = 100
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
        samples = self._samp(key, X, n_samples, self._dist._p) 

        # returning mean and variance
        return samples.mean(axis=1), samples.std(axis=1)**2

    def sample(self, key, X:jax.Array, n_samples: int = 1) -> jax.Array:
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
        return self._samp(key, X, n_samples, self._dist._p)

    def update(self, X, Y) -> None:
        """Update the model with new observations (e.g. online/incremental fitting).

        Args:
            X: New inputs.
            Y: New outputs.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("This feature is not implemented yet. Just use model.fit(X, Y)")
        
