from elyza.util.imports import * 

# helper function for computing covariances
def matrix_cov(Y1:jax.Array, Y2:jax.Array):
    assert Y1.shape[0] == Y2.shape[0], "Y1 and Y2 must contain the same number of samples"
    Y1c = Y1 - Y1.mean(axis=0).reshape(1,-1) 
    Y2c = Y2 - Y2.mean(axis=0).reshape(1,-1) 
    return Y1c.T @ Y2c / (Y1c.shape[0]-1)

# function for computing correlations 
def matrix_corr(Y1: jax.Array, Y2: jax.Array):
    assert Y1.shape[0] == Y2.shape[0], "Y1 and Y2 must contain the same number of samples"
    Y1c = Y1 - Y1.mean(axis=0).reshape(1, -1)
    Y2c = Y2 - Y2.mean(axis=0).reshape(1, -1)
    cov = Y1c.T @ Y2c / (Y1c.shape[0] - 1)
    std1 = jnp.sqrt(jnp.sum(Y1c**2, axis=0) / (Y1c.shape[0] - 1))
    std2 = jnp.sqrt(jnp.sum(Y2c**2, axis=0) / (Y2c.shape[0] - 1))
    return cov / (std1.reshape(-1, 1) @ std2.reshape(1, -1))

# Defining a better least-squares function 
def ls(A, B, rcond=None):
    return jnp.linalg.lstsq(A,B, rcond=rcond)[0]