<p align="center">
  <img src="./misc/elyza_logo.PNG" width="400" />
</p>
`elyza` is a [Jax](https://github.com/jax-ml/jax)-backed library for multifidelity surrogate modeling, inference, and uncertainty quantification. It provides a set of composable, [Pydantic](https://docs.pydantic.dev/)-validated building blocks for defining simulation inputs and evaluators, fitting Gaussian process surrogates (including multifidelity/autoregressive models), and combining models of varying cost and accuracy with multifidelity Monte Carlo estimators.

## Installation

`elyza` requires Python 3.10+. Clone the repository and install it in editable mode:

```bash
git clone https://github.com/atticusdrex/elyza.git
cd elyza
pip install -e .
```

To include the development dependencies (`build`, `twine`, `pytest`):

```bash
pip install -e ".[dev]"
```

Core dependencies (installed automatically): `numpy`, `jax`/`jaxlib`, `pydantic`, `tqdm`, `pillow`, `scipy`.

## Overview

The library is organized into a few main packages under `src/elyza/`:

### `core` — inputs and evaluators
- **`data`**: `Input` classes (`ScalarInput`, `VectorInput`) that wrap a sampling function and know how to draw batches of samples via a JAX PRNG key.
- **`evaluator`**: The `Evaluator` class wraps a callable simulation or model (e.g. an expensive PDE solver or a cheap analytical approximation) together with its inputs, output dimension, and evaluation cost, and vectorizes evaluation over batches with `vmap`.

### `surrogate` — surrogate models
- **`surrogate`**: `Surrogate`, the base interface (`fit`, `predict`, `sample`, `update`) that all surrogate models implement.
- **`gp`**: A Gaussian process implementation (`GaussianProcess`, `DeltaGP`) with pluggable kernels (`RBF`, `ARD`, `Laplace` in `kernel.py`) and mean functions (`Zero`, `Constant`, `Linear` in `mean.py`), fit via gradient-based hyperparameter calibration.

### `multifidelity` — combining models across fidelity levels
- **`surrogate`**: Hierarchical surrogate models (e.g. `GPKennedyOHagan`) that fuse training data across multiple levels of fidelity.
- **`montecarlo`**: A family of multifidelity Monte Carlo estimators built on top of `MultifidelityMonteCarlo`:
  - `HFMC` — plain high-fidelity-only Monte Carlo (baseline).
  - `MLMC` — multilevel Monte Carlo, telescoping sums across fidelity levels.
  - `MFMC` — multifidelity Monte Carlo with optimal control-variate coefficients.
  - `RMFMC` — a regularized, more general variant of MFMC using least-squares control-variate coefficients.

  These estimators use pilot samples to estimate cross-fidelity covariances, then allocate sampling budgets across fidelity levels to minimize estimator variance for a fixed computational cost.

### `optim` — gradient-based optimizers
- **`gradient`**: Lightweight `ADAM`, `BatchADAM`, and `BatchSGD` optimizers used internally to calibrate surrogate hyperparameters, and usable standalone.

### `benchmarks` — reference problems for testing/demos
- **`multifidelity/magpi_analytical`**: Analytical multifidelity benchmark functions.
- **`pde/darcy2d`**: A 2D Darcy flow benchmark (`DarcyFlowEvaluator`) that solves `-div(kappa * grad(u)) = f` on the unit square via finite differences and conjugate gradient, with permeability fields sampled from a Gaussian random field (`GRFInput`) via a KL expansion.

### `util` — shared helpers
- **`helpers`**: Covariance/correlation utilities (`matrix_cov`, `matrix_corr`) and a regularized least-squares solver (`ls`).
- **`preprocessing`**: `OrthonormalFeatures` for feature preprocessing.
- **`imports`**: Centralized `numpy`/`jax`/`tqdm` imports used throughout the package.

## Examples

See `experiments/multifidelity/monte-carlo/` for end-to-end scripts demonstrating multifidelity Monte Carlo estimation, including:
- `legendre_benchmark.py` — an analytical multifidelity benchmark.
- `darcy_benchmark.py` — multifidelity estimation on the 2D Darcy flow PDE benchmark.

## Testing

Tests live under `tests/` and use `pytest`:

```bash
pip install -e ".[dev]"
pytest
```

## License

`elyza` is licensed under the [BSD 3-Clause License](./LICENSE).
