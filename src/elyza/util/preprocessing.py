"""Feature preprocessing and transformation classes.

Provides scikit-learn-style ``fit``/``transform``/``fit_transform`` classes
for standardizing inputs (:class:`StandardScaler`), decorrelating inputs via
an SVD-based whitening transform (:class:`OrthonormalScaler`), mapping
inputs into a kernel-feature space (:class:`KernelFeatures`), and expanding
inputs into polynomial feature bases (:class:`PolynomialFeatures`).
"""
from itertools import combinations, combinations_with_replacement

from elyza.util.imports import *
from elyza.util.helpers import ensure_2d, inv_softplus

class StandardScaler(BaseModel):
    """Standardizes features by removing the mean and scaling to unit variance.

    Attributes:
        _mean: Per-feature mean computed by :meth:`fit`, shape ``(1, n_features)``.
        _stds: Per-feature scale computed by :meth:`fit`, shape ``(1, n_features)``.
        _eps: Small constant added to the scale to avoid division by zero.
    """
    _mean : jax.Array | None = PrivateAttr(default = None)
    _stds : jax.Array | None = PrivateAttr(default = None)
    _eps : float = PrivateAttr(default = 1e-8)

    def fit(self, X):
        """Compute the per-feature mean and scale from training data.

        Args:
            X: Training data, shape ``(n_samples, n_features)``.
        """
        # taking the mean
        self._mean = X.mean(axis=0).reshape(1,-1)

        # taking the standard deviations
        self._stds = X.std(axis=0).reshape(1,-1)

    def transform(self, X):
        """Standardize ``X`` using previously fit statistics.

        Args:
            X: Data to transform, shape ``(n_samples, n_features)``.

        Returns:
            jax.Array: Standardized data of the same shape as ``X``.

        Raises:
            AssertionError: If :meth:`fit` has not been called yet.
        """
        assert self._mean is not None, "you must call StandardScaler.fit or StandardScaler.fit_transform before transforming new data"
        return (X - self._mean) / (self._stds + self._eps)

    def fit_transform(self, X):
        """Fit to ``X`` and return the standardized result in one call.

        Args:
            X: Training data, shape ``(n_samples, n_features)``.

        Returns:
            jax.Array: Standardized data of the same shape as ``X``.
        """
        self.fit(X)
        return self.transform(X)

class OrthonormalScaler(BaseModel):
    """Whitens features via a truncated SVD so transformed columns are orthonormal.

    Attributes:
        rcond: Relative condition number used to truncate small singular
            values when computing the SVD; ``None`` keeps all of them.
        eps: Small jitter added to singular values to prevent division by
            zero, acting as L2 regularization on the decomposition.
        _mean: Per-feature mean computed by :meth:`fit`, shape ``(1, n_features)``.
        _A: Whitening matrix computed by :meth:`fit`.
    """
    rcond : float | None = Field(default = None, description = "relative condition number to truncate singular values when taking the SVD")
    eps : float = Field(default = 1e-12, description = "small jitter to prevent division by zero. acts as l2 regularization on the singular value decomposition.")

    _mean : jax.Array | None = PrivateAttr(default = None)
    _A : jax.Array | None = PrivateAttr(default = None)

    def fit(self, X):
        """Compute the whitening transform from training data via SVD.

        Args:
            X: Training data, shape ``(n_samples, n_features)``, with at
                least as many samples as features.

        Raises:
            AssertionError: If ``X`` has fewer rows than columns.
        """
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
        """Apply the fitted whitening transform to ``X``.

        Args:
            X: Data to transform, shape ``(n_samples, n_features)``.

        Returns:
            jax.Array: Orthonormalized features.

        Raises:
            AssertionError: If :meth:`fit` has not been called yet.
        """
        assert self._mean is not None, "you must call fit() or fit_transform() before transforming new data"
        return (X - self._mean) @ self._A

    def fit_transform(self, X):
        """Fit to ``X`` and return the whitened result in one call.

        Args:
            X: Training data, shape ``(n_samples, n_features)``.

        Returns:
            jax.Array: Orthonormalized features.
        """
        self.fit(X)
        return self.transform(X)

from elyza.surrogate.gp.kernel import BaseKernel, ARD
from elyza.util.helpers import kernel_mat

class KernelFeatures(BaseModel):
    """Maps inputs to features given by kernel evaluations against fixed centers.

    Attributes:
        input_dim: Input dimension.
        kernel_cls: Kernel class used to compute the feature map.
        eps: Small jitter to prevent division by zero; acts as L2
            regularization on the underlying decomposition.
        dtype: Datatype the underlying kernel casts its inputs/outputs to.
        _centers: Fixed center points supplied to :meth:`fit`.
        _k_params: Softplus-inverted kernel parameters supplied to :meth:`fit`.
        _mapping_func: Unused placeholder for a custom mapping function.
        _kernel: Instantiated kernel object built from ``kernel_cls``.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_dim : int = Field(description = "the input dimension")
    kernel_cls : type[BaseKernel] = Field(default = ARD, description = "the kernel to use as the features")
    eps : float = Field(default = 1e-12, description = "small jitter to prevent division by zero. acts as l2 regularization on the singular value decomposition.")
    dtype : ScalarMeta = Field(default = jnp.float64, description = "datatype the underlying kernel casts its inputs/outputs to")

    _centers : jax.Array | None = PrivateAttr(default = None)
    _k_params : jax.Array | None = PrivateAttr(default = None)
    _mapping_func : SkipValidation[callable] | None = PrivateAttr(default = None)
    _kernel : BaseKernel | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        """Instantiate the kernel object used to compute features."""
        # defining kernel function
        self._kernel = self.kernel_cls(
            input_dim = self.input_dim,
            epsilon = self.eps,
            dtype = self.dtype
        )

    def fit(self, kernel_params : jax.Array, centers:jax.Array):
        """Store the kernel parameters and center points used for transformation.

        Args:
            kernel_params: Raw (already-positive) kernel hyperparameters;
                stored internally in softplus-inverted (unconstrained) form.
            centers: Center points the kernel is evaluated against, coerced
                to 2-d via :func:`ensure_2d`.
        """
        # storing the centers
        self._centers = ensure_2d(centers)
        # softplus-inverting the kernel params
        self._k_params = inv_softplus(kernel_params)

    def transform(self, X:jax.Array):
        """Compute kernel features of ``X`` against the fitted centers.

        Args:
            X: Input points, shape ``(n_samples, input_dim)``.

        Returns:
            jax.Array: Kernel matrix of shape ``(n_samples, n_centers)``.
        """
        return kernel_mat(X, self._centers, self._kernel, self._k_params)

    def fit_transform(self, X: jax.Array, kernel_params:jax.Array, centers:jax.Array):
        self.fit(kernel_params, centers) 
        return self.transform(X) 

class PolynomialFeatures(BaseModel):
    """Expands inputs into polynomial (and optionally interaction-only) features.

    Attributes:
        degree: Maximum polynomial degree of the generated features.
        interaction_only: If ``True``, only products of distinct input
            features are produced (no powers of a single feature above
            degree 1).
        include_bias: If ``True``, include a bias column of all ones.
        _powers: Exponent matrix computed by :meth:`fit`, shape
            ``(n_output_features, n_features)``.
    """
    degree : int = Field(default = 2, description = "the maximum polynomial degree of the generated features")
    interaction_only : bool = Field(default = False, description = "if True, only products of distinct input features are produced (no powers of a single feature above degree 1)")
    include_bias : bool = Field(default = True, description = "if True, include a bias column of all ones")

    _powers : jax.Array | None = PrivateAttr(default = None)

    def fit(self, X):
        """Determine the set of exponent combinations for the polynomial expansion.

        Args:
            X: Reference data used only for its feature count, shape
                ``(n_samples, n_features)``.
        """
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
        """Expand ``X`` into its polynomial features.

        Args:
            X: Input data, shape ``(n_samples, n_features)``.

        Returns:
            jax.Array: Polynomial features, shape ``(n_samples, n_output_features_)``.

        Raises:
            AssertionError: If :meth:`fit` has not been called yet.
        """
        assert self._powers is not None, "you must call fit() or fit_transform() before transforming new data"
        X = ensure_2d(X)
        # X: (N, n_features), _powers: (n_output_features, n_features)
        return jnp.prod(X[:, None, :] ** self._powers[None, :, :], axis=-1)

    def fit_transform(self, X):
        """Fit to ``X`` and return the polynomial features in one call.

        Args:
            X: Input data, shape ``(n_samples, n_features)``.

        Returns:
            jax.Array: Polynomial features, shape ``(n_samples, n_output_features_)``.
        """
        self.fit(X)
        return self.transform(X)

    @property
    def n_output_features_(self):
        """int: Number of polynomial features produced by :meth:`transform`.

        Raises:
            AssertionError: If :meth:`fit` has not been called yet.
        """
        assert self._powers is not None, "you must call fit() or fit_transform() first"
        return self._powers.shape[0]
