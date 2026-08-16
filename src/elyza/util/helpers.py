from elyza.util.imports import * 

# helper function for computing covariances
def matrix_cov(Y1:jax.Array, Y2:jax.Array):
    assert Y1.shape[0] == Y2.shape[0], "Y1 and Y2 must contain the same number of samples"
    Y1c = Y1 - Y1.mean(axis=0).reshape(1,-1) 
    Y2c = Y2 - Y2.mean(axis=0).reshape(1,-1) 
    return Y1c.T @ Y2c / Y1c.shape[0]

# Defining a better least-squares function 
def ls(A, B, rcond=None):
    return jnp.linalg.lstsq(A,B, rcond=rcond)[0]