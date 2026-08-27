from itertools import combinations, combinations_with_replacement

from elyza.util.imports import *
from elyza.util.helpers import ensure_2d, inv_softplus

class StandardScaler(BaseModel):
    _mean : jax.Array | None = PrivateAttr(default = None)
    _stds : jax.Array | None = PrivateAttr(default = None) 
    _eps : float = PrivateAttr(default = 1e-8) 

    def fit(self, X):
        # taking the mean 
        self._mean = X.mean(axis=0).reshape(1,-1) 

        # taking the standard deviations 
        self._stds = X.mean(axis=0).reshape(1,-1) 

    def transform(self, X):
        assert self._mean is not None, "you must call StandardScaler.fit or StandardScaler.fit_transform before transforming new data"
        return (X - self._mean) / (self._stds + self._eps) 

    def fit_transform(self, X):
        self.fit(X) 
        return self.transform(X) 

class OrthonormalScaler(BaseModel): 
    rcond : float | None = Field(default = None, description = "relative condition number to truncate singular values when taking the SVD") 
    eps : float = Field(default = 1e-12, description = "small jitter to prevent division by zero. acts as l2 regularization on the singular value decomposition.")

    _mean : jax.Array | None = PrivateAttr(default = None) 
    _A : jax.Array | None = PrivateAttr(default = None) 

    def fit(self, X):
        assert X.shape[0] >= X.shape[1], "you need at least as many observations as the number of features to compute orthogonal features"
        # taking the row-wise mean 
        self._mean = X.mean(axis=0).reshape(1,-1) 
        # centering the inputs 
        Xc = X - self._mean 
        # computing the svd of the centered matrix
        _, S, Ut = jnp.linalg.svd(Xc, full_matrices = False) 

        # truncating singular values above some relative tolerance 
        if self.rcond is not None: 
            S = S[S / S[0] > self.rcond] 
            Ut = Ut[:len(S),:]

        # computing the scaling matrix 
        self._A = Ut.T @ jnp.diag(jnp.sqrt(X.shape[0] - 1) / (S + self.eps))

    def transform(self, X):
        assert self._mean is not None, "you must call fit() or fit_transform() before transforming new data"
        return (X - self._mean) @ self._A 

    def fit_transform(self, X):
        self.fit(X) 
        return self.transform(X) 

from elyza.surrogate.gp.kernel import BaseKernel, ARD
from elyza.util.helpers import kernel_mat 

class KernelFeatures(BaseModel):
    input_dim : int = Field(description = "the input dimension")
    kernel_cls : type[BaseKernel] = Field(default = ARD, description = "the kernel to use as the features")
    eps : float = Field(default = 1e-12, description = "small jitter to prevent division by zero. acts as l2 regularization on the singular value decomposition.")

    _centers : jax.Array | None = PrivateAttr(default = None)
    _k_params : jax.Array | None = PrivateAttr(default = None) 
    _mapping_func : SkipValidation[callable] | None = PrivateAttr(default = None) 
    _kernel : BaseKernel | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        # defining kernel function
        self._kernel = self.kernel_cls(
            input_dim = self.input_dim,
            epsilon = self.eps
        )

    def fit(self, kernel_params : jax.Array, centers:jax.Array): 
        # storing the centers
        self._centers = ensure_2d(centers) 
        # softplus-inverting the kernel params 
        self._k_params = inv_softplus(kernel_params)

    def transform(self, X:jax.Array):
        return kernel_mat(X, self._centers, self._kernel, self._k_params)

class PolynomialFeatures(BaseModel):
    degree : int = Field(default = 2, description = "the maximum polynomial degree of the generated features")
    interaction_only : bool = Field(default = False, description = "if True, only products of distinct input features are produced (no powers of a single feature above degree 1)")
    include_bias : bool = Field(default = True, description = "if True, include a bias column of all ones")

    _powers : jax.Array | None = PrivateAttr(default = None)

    def fit(self, X):
        n_features = ensure_2d(X).shape[1]
        combo_fn = combinations if self.interaction_only else combinations_with_replacement

        powers_list = []
        for d in range(0 if self.include_bias else 1, self.degree + 1):
            if d == 0:
                powers_list.append([0] * n_features)
                continue
            for combo in combo_fn(range(n_features), d):
                powers = [0] * n_features
                for idx in combo:
                    powers[idx] += 1
                powers_list.append(powers)

        self._powers = jnp.array(powers_list)

    def transform(self, X):
        assert self._powers is not None, "you must call fit() or fit_transform() before transforming new data"
        X = ensure_2d(X)
        # X: (N, n_features), _powers: (n_output_features, n_features)
        return jnp.prod(X[:, None, :] ** self._powers[None, :, :], axis=-1)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def n_output_features_(self):
        assert self._powers is not None, "you must call fit() or fit_transform() first"
        return self._powers.shape[0]
