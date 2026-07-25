from .imports import * 

class OrthonormalFeatures(BaseModel):
    mean: jax.Array | None = None   # column-wise mean
    A: jax.Array | None = None      # projection / decorrelation matrix
    A_inv: jax.Array | None = None  # pseudoinverse of A

    def fit_transform(self, X: np.ndarray | jax.Array, rcond: float = 1e-12) -> jax.Array:
        X = jnp.array(X)

        # mean-center the data
        self.mean = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean

        # SVD of Xc^T
        U, S, _ = jnp.linalg.svd(Xc.T, full_matrices=False)

        if S.shape[0] == 0:
            raise ValueError("Input data has no singular values.")

        # keep only sufficiently large singular values
        mask = S / S[0] > rcond
        Sk = S[mask]
        Uk = U[:, mask]

        if Sk.size == 0:
            raise ValueError("All singular values were truncated. Lower rcond.")

        # decorrelation / projection matrix
        self.A = Uk @ jnp.diag(1.0 / Sk)

        # pseudoinverse, since A may be rectangular
        self.A_inv = jnp.linalg.pinv(self.A)

        # transformed data
        return Xc @ self.A

    def transform(self, X: np.ndarray | jax.Array) -> jax.Array:
        if self.mean is None or self.A is None:
            raise ValueError("Model has not been fit yet.")
        X = jnp.array(X)
        return (X - self.mean) @ self.A

    def inverse_transform(self, X: np.ndarray | jax.Array) -> jax.Array:
        if self.mean is None or self.A_inv is None:
            raise ValueError("Model has not been fit yet.")
        
        X = jnp.array(X)
        return X @ self.A_inv + self.mean

         