from .imports import * 

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
        return (X - self._mean) @ self._A 

    def fit_transform(self, X):
        self.fit(X) 
        return self.transform(X) 
    

        
            