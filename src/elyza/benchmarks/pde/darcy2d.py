from elyza.util.imports import *
from elyza.core.data import VectorInput
from elyza.core.evaluator import Evaluator
from PIL import Image

from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy import ndimage

import functools

def build_image_source(path, grid_dim, inflation = 1.0):
    # loading image 
    image = np.array(Image.open(path))
    height, width = image.shape[:2] 

    # ---------------------------------------------------------------------
    # uncomment for 3-channel RGB image 
    # ---------------------------------------------------------------------
    image = image[:,:,:3].mean(axis=2).astype(np.float64)
    image = image / np.max(image)

    image[image < 1e-1] = -0.5

    image *= inflation 

    # # threshold for images to generate more interesting force fields
    # thresh = 0.5
    # image[image < thresh] = -image[image < thresh]
    # image[image > 0.5] = 0.5

    # # inflating source term 
    # image *= 2.0
    # ---------------------------------------------------------------------

    image = ndimage.zoom(image, (grid_dim/height, grid_dim/width), order=3)

    # zooming to make square 
    return image

def get_field(key, grf_mean, grf_std, scaled_eigenvecs, n_kl):
    xi = jrand.normal(key, shape=(n_kl,))               # i.i.d. N(0,1)
    log_kappa = grf_mean + grf_std * (scaled_eigenvecs @ xi)   # (N,)
    return jnp.exp(log_kappa)    
# ---------------------------------------------------------------------------
# KL expansion helpers  (computed once, on CPU via NumPy/SciPy)
# ---------------------------------------------------------------------------

def _grid_points(grid_n: int) -> np.ndarray:
    """(grid_n*grid_n, 2) cell-centre coordinates over [0,1]^2."""
    x1d = np.linspace(0.0, 1.0, grid_n, endpoint=False) + 0.5 / grid_n
    xv, yv = np.meshgrid(x1d, x1d, indexing="ij")
    return np.column_stack([xv.ravel(), yv.ravel()])


def build_kl_basis(
    grid_n: int,
    length_scale: float,
    n_kl_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the leading KL eigenpairs of a squared-exponential covariance
    kernel on an (n×n) grid with periodic-free domain [0,1]².

    The covariance is discretised as a dense (N×N) matrix
    C_{ij} = exp(-‖x_i - x_j‖² / (2 ℓ²)),  N = grid_n².

    Returns
    -------
    eigvals : (n_kl_terms,) float64
    eigvecs : (N, n_kl_terms) float64   — already scaled by √λ_k
    """
    def fix_signs(V):
        signs = np.sign(V[0, :])
        signs[signs == 0] = 1
        return V * signs
    N = grid_n * grid_n
    pts = _grid_points(grid_n)  # (N, 2)

    # Squared-exponential covariance matrix  (may be large — warn if so)
    if N > 8192:
        import warnings
        warnings.warn(
            f"KL basis: dense covariance matrix is ({N}×{N}). "
            "Consider reducing grid_n or n_kl_terms.",
            stacklevel=4,
        )

    diff = pts[:, None, :] - pts[None, :, :]          # (N, N, 2)
    sq_dist = np.sum(diff ** 2, axis=-1)               # (N, N)
    C = np.exp(-sq_dist / (2.0 * length_scale ** 2))  # (N, N)

    # Compute the *largest* n_kl_terms eigen-pairs
    k = np.min([n_kl_terms, N])
    # eigh returns eigenvalues in ascending order → take the last k
    # eigvals=(lo, hi) selects indices [lo, hi] inclusive (scipy < 1.5 compatible)
    # vals, vecs = eigh(C, eigvals=(N - k, N - 1))
    # vals, vecs = eigh(C, subset_by_index=(N - k, N - 1))
    vals, vecs = eigsh(C, k=k, which='LM', v0=np.ones(N))
    vecs = fix_signs(vecs)

    # Reverse so index 0 is the dominant mode
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    # Scale eigenvectors: φ_k ← √λ_k · φ_k
    # Then a sample is  μ + Σ_k ξ_k · φ_k,  ξ_k ~ N(0,1)
    scaled_vecs = vecs * np.sqrt(np.maximum(vals, 0.0))[None, :]

    return vals.astype(np.float32), scaled_vecs.astype(np.float32)


# Cache of reference KL eigenbases, keyed by (length_scale, n_kl_terms,
# reference_grid_dim). Multiple GRFInputs that share a cache key share the
# exact same underlying eigenfunctions, which is what lets them be coupled
# across resolutions (see build_kl_basis_at below).
_kl_reference_cache: dict = {}


def build_kl_basis_at(
    grid_n: int,
    length_scale: float,
    n_kl_terms: int,
    reference_grid_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    KL eigenbasis evaluated at an (grid_n x grid_n) grid, via a Nystrom
    extension of a basis computed (and cached) on a fixed reference grid.

    build_kl_basis computes an independent eigendecomposition of the
    covariance matrix discretised AT grid_n -- two different grid_n values
    give unrelated eigenvectors, so a shared PRNGKey does not correspond to
    the same underlying field at two different resolutions. Nystrom
    extension instead evaluates ONE fixed set of eigenfunctions (computed
    once on the reference grid) at grid_n's own points, so any two
    grid_n's sharing (length_scale, n_kl_terms, reference_grid_dim) sample
    the same continuous field, just discretised at different resolutions.

    For grid_n == reference_grid_dim this reduces exactly to build_kl_basis.
    """
    cache_key = (length_scale, n_kl_terms, reference_grid_dim)
    if cache_key not in _kl_reference_cache:
        ref_pts = _grid_points(reference_grid_dim)
        ref_vals, ref_scaled_vecs = build_kl_basis(reference_grid_dim, length_scale, n_kl_terms)
        _kl_reference_cache[cache_key] = (ref_pts, ref_vals, ref_scaled_vecs)

    ref_pts, vals, scaled_vecs_ref = _kl_reference_cache[cache_key]

    if grid_n == reference_grid_dim:
        return vals, scaled_vecs_ref

    query_pts = _grid_points(grid_n)                                       # (n_query, 2)
    diff = query_pts[:, None, :] - ref_pts[None, :, :]                     # (n_query, M_ref, 2)
    C_cross = np.exp(-np.sum(diff ** 2, axis=-1) / (2.0 * length_scale ** 2))  # (n_query, M_ref)

    # Nystrom extension: phi_k(x) = (1/lambda_k) sum_j C(x, x_j) phi_k(x_j),
    # applied directly to the already-sqrt(lambda)-scaled reference vectors.
    scaled_vecs_query = (C_cross @ scaled_vecs_ref) / np.maximum(vals, 1e-12)[None, :]

    return vals.astype(np.float32), scaled_vecs_query.astype(np.float32)


# ---------------------------------------------------------------------------
# Source-term construction
# ---------------------------------------------------------------------------

def build_source_field(
    source_3x3: jnp.ndarray,
    grid_n: int,
) -> jnp.ndarray:
    """
    Expand a (3,3) source array into an (grid_n, grid_n) field by tiling
    the domain into a 3×3 tic-tac-toe partition.  Each cell of the partition
    receives the corresponding constant value.

    Returns flattened (grid_n*grid_n,) array.
    """
    # Repeat each value along each axis: grid_n // 3 interior + remainder
    # Use jnp.repeat to keep things traceable
    rows = jnp.repeat(source_3x3, grid_n // 3 + (jnp.arange(3) < grid_n % 3).astype(int), axis=0)
    # Simpler: just use np.kron-style expansion outside jit
    # Build partition boundaries
    # We do this purely in numpy (called once, outside jit)
    src = np.array(source_3x3)
    field = np.zeros((grid_n, grid_n), dtype=np.float32)
    edges = np.array_split(np.arange(grid_n), 3)
    for i in range(3):
        for j in range(3):
            ii = np.ix_(edges[i], edges[j])
            field[ii] = src[i, j]
    return jnp.array(field.ravel())  # (N,)


# ---------------------------------------------------------------------------
# FD stiffness-matrix helpers
# ---------------------------------------------------------------------------

def _fd_laplacian_action(
    u: jnp.ndarray,
    kappa: jnp.ndarray,
    grid_n: int,
    h: float,
) -> jnp.ndarray:
    """
    Compute  A(κ) u  via a cell-centred finite-difference discretisation
    of  -∇·(κ ∇u)  with homogeneous Dirichlet BCs.

    Uses harmonic-mean interface permeabilities (standard for Darcy).

    Parameters
    ----------
    u      : (N,)  current iterate / vector
    kappa  : (N,)  permeability field (strictly positive)
    grid_n : int
    h      : float  grid spacing

    Returns
    -------
    Au : (N,)
    """
    n = grid_n
    U = u.reshape(n, n)
    K = kappa.reshape(n, n)

    # Harmonic-mean interface permeabilities
    # East / West faces
    K_e = 2.0 * K[:, :-1] * K[:, 1:]  / (K[:, :-1] + K[:, 1:] + 1e-30)
    K_w = K_e  # symmetric

    # North / South faces
    K_n = 2.0 * K[:-1, :] * K[1:, :]  / (K[:-1, :] + K[1:, :] + 1e-30)
    K_s = K_n

    # Flux contributions: Au_ij = (1/h²) Σ K_face (u_ij - u_neighbour)
    h2 = h * h
    AU = jnp.zeros_like(U)

    # East contribution: flux from (i,j) → (i,j+1)
    AU = AU.at[:, :-1].add( K_e / h2 * (U[:, :-1] - U[:, 1:]))
    AU = AU.at[:, 1: ].add( K_e / h2 * (U[:, 1:]  - U[:, :-1]))

    # North contribution: flux from (i,j) → (i+1,j)
    AU = AU.at[:-1, :].add( K_n / h2 * (U[:-1, :] - U[1:, :]))
    AU = AU.at[1:,  :].add( K_n / h2 * (U[1:,  :] - U[:-1, :]))

    # Dirichlet BCs: boundary nodes are pinned to zero
    # (ghost-cell approach: fluxes across boundary use u=0 outside)
    # Left boundary (j=0): neighbour to the west is 0
    AU = AU.at[:, 0].add(K[:, 0] / h2 * U[:, 0])
    # Right boundary (j=n-1)
    AU = AU.at[:, -1].add(K[:, -1] / h2 * U[:, -1])
    # Bottom boundary (i=0)
    AU = AU.at[0, :].add(K[0, :] / h2 * U[0, :])
    # Top boundary (i=n-1)
    AU = AU.at[-1, :].add(K[-1, :] / h2 * U[-1, :])

    return AU.ravel()


# ---------------------------------------------------------------------------
# Conjugate-Gradient solver  (pure JAX, jit-able)
# ---------------------------------------------------------------------------

def _cg_solve(
    matvec,
    b: jnp.ndarray,
    tol: float = 1e-6,
    max_iter: int = 2000,
) -> jnp.ndarray:
    """
    Unpreconditioned Conjugate Gradient solver for  A x = b.
    Uses `jax.lax.while_loop` so it is fully jit-compatible.
    """
    N = b.shape[0]
    x0 = jnp.zeros(N)
    r0 = b - matvec(x0)
    p0 = r0
    rr0 = jnp.dot(r0, r0)

    def cond(state):
        _, _, _, rr, i = state
        return (rr > tol ** 2 * jnp.dot(b, b) + 1e-30) & (i < max_iter)

    def body(state):
        x, r, p, rr, i = state
        Ap = matvec(p)
        alpha = rr / (jnp.dot(p, Ap) + 1e-30)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rr_new = jnp.dot(r_new, r_new)
        beta = rr_new / (rr + 1e-30)
        p_new = r_new + beta * p
        return x_new, r_new, p_new, rr_new, i + 1

    x, _, _, _, _ = jax.lax.while_loop(cond, body, (x0, r0, p0, rr0, 0))
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GRFInput(VectorInput):
    """
    A log-normal Gaussian random field input, represented by a truncated KL
    expansion of a squared-exponential covariance kernel on an (grid_dim x
    grid_dim) grid over [0,1]^2. `.sample(key, n_points)` returns n_points
    flattened field realisations as a single (n_points, grid_dim**2) array.

    To couple samples across resolutions (e.g. for multifidelity Monte
    Carlo), set `reference_grid_dim` to the SAME value -- along with
    matching `length_scale`/`n_kl_terms` -- on every GRFInput that should
    share the same underlying field. The same PRNGKey then yields the same
    field, just discretised at each instance's own `grid_dim`. Leave it
    unset for standalone use (each instance gets its own independent basis,
    computed at its own grid_dim).
    """
    grid_dim : int = Field(description = "field grid resolution (grid_dim x grid_dim)")
    length_scale : float = Field(default = 0.25, description = "correlation length of the squared-exponential kernel")
    n_kl_terms : int = Field(default = 64, description = "number of KL expansion terms retained")
    grf_mean : float = Field(default = 0.0, description = "mean of the underlying (log-permeability) Gaussian random field")
    grf_std : float = Field(default = 1.0, description = "standard deviation of the underlying Gaussian random field")
    reference_grid_dim : int | None = Field(default = None, description = "grid resolution the KL eigenbasis is computed on; set the SAME value on multiple GRFInputs (with matching length_scale/n_kl_terms) to couple their sampled fields across resolutions. Defaults to grid_dim (no coupling)")

    # the field is unconstrained, so the box-bound fields inherited from
    # VectorInput aren't meaningful here
    minval : jax.Array | np.ndarray | None = Field(default = None, description = "unused for GRF inputs")
    maxval : jax.Array | np.ndarray | None = Field(default = None, description = "unused for GRF inputs")

    _scaled_eigvecs : jax.Array | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        # each flattened field realisation has grid_dim**2 entries
        self.dim = self.grid_dim ** 2
        if self.minval is None:
            self.minval = -jnp.inf * jnp.ones(self.dim)
        if self.maxval is None:
            self.maxval = jnp.inf * jnp.ones(self.dim)

        # computing the KL basis once, at construction time
        reference_grid_dim = self.grid_dim if self.reference_grid_dim is None else self.reference_grid_dim
        _, scaled_eigvecs = build_kl_basis_at(self.grid_dim, self.length_scale, self.n_kl_terms, reference_grid_dim)
        self._scaled_eigvecs = jnp.array(scaled_eigvecs)

        # wiring up the sampling function used by Input.sample()
        def sampling_func(key):
            return get_field(key, self.grf_mean, self.grf_std, self._scaled_eigvecs, self.n_kl_terms)
        self.sampling_func = sampling_func

        super().model_post_init(__context)


class DarcyFlowEvaluator(Evaluator):
    """
    Solves 2-D Darcy flow, -div(kappa grad(u)) = f on [0,1]^2 with u|dOmega = 0,
    for a given (flattened) permeability field kappa via cell-centred FD + CG.
    `.evaluate(kappa_fields)` takes the (n_points, grid_dim**2) array returned
    by a GRFInput's `.sample()` and returns the matching flow fields.
    """
    model_config = ConfigDict(arbitrary_types_allowed = True)

    grid_dim : int = Field(description = "solver grid resolution (grid_dim x grid_dim)")
    source_term : str | jax.Array | np.ndarray | None = Field(default = None, description = "(3,3) piecewise-constant source values, a path to an image to use as a full-resolution source field, or None for a uniform unit source")
    source_inflation : float = Field(default = 1.0, description = "scale factor applied to an image-derived source field (only used when source_term is a path)")
    cg_tol : float = Field(default = 1e-6, description = "relative residual tolerance for the CG solver")
    cg_max_iter : int = Field(default = 3000, description = "maximum CG iterations")

    _f : jax.Array | np.ndarray | None = PrivateAttr(default = None)
    _h : float = PrivateAttr(default = 0.0)
    _batched_solve : Callable | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        self.output_dim = self.grid_dim ** 2

        self._h = 1.0 / self.grid_dim

        if isinstance(self.source_term, str):
            # building a full-resolution source field directly from an image
            self._f = jnp.array(build_image_source(self.source_term, self.grid_dim, inflation = self.source_inflation).ravel())
        else:
            source = np.ones((3, 3), dtype = np.float32) if self.source_term is None else np.asarray(self.source_term, dtype = np.float32)
            assert source.shape == (3, 3), f"source_term must be an image path, or an array of shape (3, 3), got {source.shape}"
            self._f = build_source_field(source, self.grid_dim)

        def evaluation_func(kappa):
            matvec = functools.partial(_fd_laplacian_action, kappa = kappa, grid_n = self.grid_dim, h = self._h)
            return _cg_solve(matvec, self._f, tol = self.cg_tol, max_iter = self.cg_max_iter)
        self.evaluation_func = evaluation_func

        # jit-compiling the vmapped solve once, up front, so repeated calls to
        # evaluate() (e.g. across pilot samples / MC iterations) reuse the
        # compiled executable instead of re-tracing the CG while_loop each time
        self._batched_solve = jit(vmap(evaluation_func, in_axes = 0))

        super().model_post_init(__context)

    def evaluate(self, *input_vals : list[jax.Array]) -> jax.Array:
        return self._batched_solve(input_vals[0]).reshape(-1, self.output_dim)