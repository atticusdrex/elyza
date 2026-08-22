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

# ensures that X is a 2d array better than numpy's stupid built-in function
def ensure_2d(X):
    if len(X.shape) == 0: 
        return X.reshape(1,1) 
    elif len(X.shape) == 1: 
        return X.reshape(-1,1) 
    else: 
        return X


# Special KL-divergence function for two Gaussians distributions
def KL_div(mu_q, L_q, mu_p, L_p):
    """KL divergence KL(q || p) for Gaussians parameterized by Cholesky factors.

    Arguments:
      mu_q, L_q : mean and lower-triangular Cholesky factor for q
      mu_p, L_p : mean and lower-triangular Cholesky factor for p
    """
    k = mu_q.shape[0]

    # Covariance matrices
    Sigma_q = L_q @ L_q.T
    Sigma_p = L_p @ L_p.T

    # Trace term: tr(Sigma_p^{-1} Sigma_q)
    # Solve instead of explicitly inverting
    Sigma_p_inv = jnp.linalg.inv(Sigma_p)
    Tr_q = jnp.trace(Sigma_p_inv @ Sigma_q)

    # Mean term: (mu_p - mu_q)^T Sigma_p^{-1} (mu_p - mu_q)
    diff = mu_q - mu_p
    mean_term = jnp.inner(diff, cho_solve((L_p, True), diff))

    # Log-determinant ratio
    logdet_q = 2.0 * jnp.sum(jnp.log(jnp.diag(L_q)))
    logdet_p = 2.0 * jnp.sum(jnp.log(jnp.diag(L_p)))
    logdet_ratio = logdet_p - logdet_q

    return 0.5 * (Tr_q + mean_term - k + logdet_ratio)

# Function for greedily choosing the number of inducing inputs 
def greedy_k_center(X, k, seed=42):
    """Greedy k-centers selection of inducing inputs.

    Returns a tuple (selected_points, selected_indices).
    """
    np.random.seed(seed)
    N = X.shape[0]
    selected_indices = []
    idx = np.random.randint(N)
    selected_indices.append(idx)

    distances = np.linalg.norm(X - X[idx], axis=1)

    for _ in range(1, k):
        idx = np.argmax(distances)
        selected_indices.append(idx)
        new_distances = np.linalg.norm(X - X[idx], axis=1)
        distances = np.minimum(distances, new_distances)

    return X[np.array(selected_indices)], selected_indices

def sigmoid(x):
    """Sigmoid activation elementwise: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + jnp.exp(-x))

def inv_sigmoid(y):
    """Inverse sigmoid (logit): maps (0,1) -> R."""
    return jnp.log(y/(1-y))

def softplus(x):
    """Softplus activation: log(1 + exp(x))."""
    return jnp.log(1.0 + jnp.exp(x))

def inv_softplus(y):
    """Inverse softplus: maps positive values back to unconstrained space."""
    return jnp.log(jnp.exp(y) - 1.0)

def kernel_mat(X1, X2, kernel, kernel_params):
    """Compute the full kernel matrix between two point sets.

    Returns an array with shape (len(X1), len(X2)) where each entry is
    kernel.eval(x,y,kernel_params).
    """
    return vmap(lambda x: vmap(lambda y: kernel.eval(x, y, kernel_params))(X2))(X1)
