<p align="center">
  <img src="https://raw.githubusercontent.com/atticusdrex/elyza/main/misc/elyza_logo.PNG" width="400" />
</p>

<!-- summary-start -->
`elyza` is a [Jax](https://github.com/jax-ml/jax)-based library for general purpose scientific computing. It provides a set of composable, [Pydantic](https://docs.pydantic.dev/)-validated building blocks for defining simulation inputs and evaluators, fitting surrogate models, optimizing expensive differentiable functions, and performing uncertainty quantification.

## Installation

`elyza` requires Python 3.10+.

```bash
pip install elyza
```

Core dependencies (installed automatically): `numpy`, `jax`/`jaxlib`, `pydantic`, `tqdm`, `pillow`, `scipy`. NOTE: if you want `jax` to run on the GPU, you must install a CUDA-compatible version of jax using `pip install "jax[cudaXX]"` where `XX` is the version of CUDA installed on your NVIDIA-GPU-powered machine. 

To work on `elyza` itself, clone the repository and install it in editable mode with the development dependencies (`build`, `twine`, `pytest`):

```bash
git clone https://github.com/atticusdrex/elyza.git
cd elyza
pip install -e ".[dev]"
```

## Overview

The library is organized into a few main packages under `src/elyza/`:

### `core` — inputs and evaluators
- **`data`**: `Input` classes (`ScalarInput`, `VectorInput`) that wrap a sampling function and know how to draw batches of samples via a JAX PRNG key.
- **`evaluator`**: The `Evaluator` class wraps a callable simulation or model (e.g. an expensive PDE solver or a cheap analytical approximation) together with its inputs, output dimension, and evaluation cost, and vectorizes evaluation over batches with `vmap`.

### `surrogate` — surrogate models
- **`abstract`**: `Surrogate`, the base interface (`fit`, `predict`, `sample`, `update`, `set_optimizer`) that every surrogate model implements, plus `SupervisedDataset`, a container for a model's training inputs/outputs.
- **`gp`**: `GaussianProcess`, a Gaussian process regressor with pluggable kernels (`RBF`, `ARD`, `Laplace` in `kernel.py`) and mean functions (`Zero`, `Constant`, `Linear` in `mean.py`), fit via gradient-based hyperparameter calibration and supporting incremental (rank-update) fitting.
- **`dnn`**: `MLPRegressor`, a feedforward neural network surrogate.
- **`linear`**: `Ridge`, closed-form L2-regularized linear regression.

### `multifidelity` — combining models across fidelity levels
- **`surrogate`**: `HierarchicalSurrogate` and `MAGPI` (Multifidelity-Augmented GP Inputs), which fit a chain of level-specific surrogates where each level's features are augmented with the predictions of every lower-fidelity level.
- **`montecarlo`**: A family of multifidelity Monte Carlo estimators built on top of `MultifidelityMonteCarlo`:
  - `HFMC` — plain high-fidelity-only Monte Carlo (baseline).
  - `MLMC` — multilevel Monte Carlo, telescoping sums across fidelity levels.
  - `MFMC` — multifidelity Monte Carlo with optimal per-level control-variate coefficients.
  - `RMFMC` — regression-based multifidelity Monte Carlo, a more general variant of MFMC using least-squares control-variate coefficients over the full joint covariance of the lower-fidelity levels.

  These estimators use pilot samples to estimate cross-fidelity covariances, then allocate sampling budgets across fidelity levels to minimize estimator variance for a fixed computational cost.

### `optim` — gradient-based optimizers
- **`abstract`**: `Optimizer`/`BatchGradientOptimizer`, the base interfaces used to calibrate surrogate hyperparameters.
- **`adam`**: `ADAM`, a batched Adam optimizer built on `jax.lax.scan`.
- **`lbfgs`**: `LBFGS`, a batched limited-memory BFGS optimizer with a two-loop recursion and Armijo backtracking line search.

### `benchmarks` — reference problems for testing/demos
- **`multifidelity/magpi_analytical`**: A cheap analytical three-fidelity benchmark (`hf_evaluator`, `mf_evaluator`, `lf_evaluator`) sharing a single scalar input.
- **`pde/darcy2d`**: A 2D Darcy flow benchmark (`DarcyFlowEvaluator`) that solves `-div(kappa * grad(u)) = f` on the unit square via finite differences and conjugate gradient, with permeability fields sampled from a Gaussian random field (`GRFInput`) via a KL expansion — usable at different grid resolutions to form a fidelity hierarchy.

### `util` — shared helpers
- **`imports`**: Centralized `numpy`/`jax`/`pydantic`/`tqdm` imports used throughout the package.
- **`helpers`**: Covariance/correlation utilities (`matrix_cov`, `matrix_corr`), a regularized least-squares solver (`ls`), activation functions, and other small numerical helpers.
- **`preprocessing`**: `StandardScaler`, `OrthonormalScaler`, `KernelFeatures`, and `PolynomialFeatures` for feature preprocessing.

## Examples

See `experiments/` for end-to-end scripts demonstrating the library, including:
- `multifidelity/monte-carlo/legendre_benchmark.py` — multifidelity Monte Carlo estimation on an analytical benchmark.
- `multifidelity/monte-carlo/darcy_benchmark.py` — multifidelity Monte Carlo estimation on the 2D Darcy flow PDE benchmark.
- `multifidelity/surrogate/magpi_analytical.py` — fitting a `MAGPI` hierarchical surrogate on the analytical benchmark.
- `surrogate/gp_testing.py` and `surrogate/dnn_testing.py` — fitting `GaussianProcess` and `MLPRegressor` surrogates to a single-fidelity benchmark.



## Citation

If you use `elyza` in your research, please cite it as:

<!-- citation-start -->
```bibtex
@software{rex2026elyza,
  author  = {Rex, Atticus},
  title   = {elyza: A jax-based library for efficient multifidelity surrogate modeling and uncertainty quantification},
  year    = {2026},
  url     = {https://github.com/atticusdrex/elyza},
  version = {0.1.1}
}
```
<!-- citation-end -->

<!-- summary-end -->

## Documentation

Full API documentation (built with Sphinx from the docstrings in `src/`) lives under `docs/`, including a [surrogate modeling quickstart](https://github.com/atticusdrex/elyza/blob/main/docs/source/surrogate.rst). To build it locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
```

Then open `docs/build/html/index.html`.

## License

`elyza` is licensed under the [BSD 3-Clause License](LICENSE).


