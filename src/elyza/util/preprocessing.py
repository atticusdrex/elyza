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

         