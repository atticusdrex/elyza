"""Multifidelity Monte Carlo estimators.

Defines the :class:`MultifidelityMonteCarlo` base class and four concrete
estimators built on top of it: :class:`RMFMC` (regression-based
multifidelity MC, with matrix/vector/scalar coefficient variants),
:class:`MFMC` (a single-coefficient-per-level special case of RMFMC),
:class:`MLMC` (telescoping multilevel MC), and :class:`HFMC` (plain
high-fidelity-only MC, used as a baseline). Each estimator combines
:class:`~elyza.core.evaluator.Evaluator` samples across fidelity levels into
a low-variance estimate of the high-fidelity expectation, and computes an
optimal sample allocation for a given evaluation budget.
"""
from elyza.util.imports import *
from elyza.core.evaluator import Evaluator
from elyza.util.helpers import matrix_cov, matrix_corr, ls

class MultifidelityMonteCarlo(BaseModel):
    """Base class for multifidelity Monte Carlo estimators.

    Attributes:
        evaluators: A list of multifidelity evaluators (index 0 is
            lowest-fidelity, the last entry is high-fidelity).
        covs: A 2-d nested list of covariance matrices relating the levels
            of fidelity such that ``covs[level1][level2] = Cov(level1, level2)``.
        corrs: Matching nested list of correlation matrices.
        _K: Number of levels of fidelity.
        _hf_dim: Output dimension of the high-fidelity evaluator.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    evaluators : list[Evaluator] = Field(default = [], description = "A list of multifidelity evaluators (0 is high-fidelity)")
    covs : list[list[jax.Array]] | None = Field(default = None, description = "A 2d nested list of covariance matrices relating the levels of fidelity such that covs[level1][level2] = Cov{level1}{level2}")
    corrs : list[list[jax.Array]] | None = Field(default = None, description = "correlation matrix")

    _K : int = PrivateAttr(default = 0)
    _hf_dim : int = PrivateAttr(default = 1)

    def model_post_init(self, __context):
        """Validate ``evaluators`` and cache the fidelity count/high-fidelity dim.

        Raises:
            AssertionError: If ``evaluators`` is empty.
        """
        assert len(self.evaluators) != 0, "passed empty list of fidelities"
        self._K = len(self.evaluators) # setting the number of levels of fidelity
        self._hf_dim = self.evaluators[-1].output_dim # setting the high-fidelity dimension

    def get_pilots(self, key, n_pilots:int, set_costs = False, noise_std = 0.0, compute_correlation = False):
        """Draw pilot samples from every evaluator and compute cross-level covariances.

        Populates ``self.covs`` (and ``self.corrs``, if requested) from
        ``n_pilots`` shared-key samples of each evaluator's inputs.

        Args:
            key: A JAX PRNG key, reused across evaluators/levels so their
                pilot samples correspond to the same underlying draws.
            n_pilots: Number of pilot samples per evaluator.
            set_costs: If ``True``, calibrate each evaluator's ``cost`` from
                the timed pilot evaluation.
            noise_std: Standard deviation of optional Gaussian noise added
                to each evaluator's pilot outputs (for experimentation).
            compute_correlation: If ``True``, also populate ``self.corrs``.
        """
        pilot_samples = []

        # iterating through and computing random pilot samples of each evaluator
        for evaluator in self.evaluators:
            # sampling the inputs to each evaluator with the same key
            input_vals = []
            for input in evaluator.inputs:
                input_vals.append(input.sample(key, n_pilots))

            # timing each evaluator on the input samples
            pilot_samples.append(evaluator.evaluate_timed(*input_vals, set_cost = set_costs))

            # adding random noise if desired (more for experimentation)
            if noise_std != 0.0:
                pilot_samples[-1] += noise_std * jrand.normal(key, shape = pilot_samples[-1].shape)

        # computing the covariance for each fidelity pair
        covs, corrs = [], []
        for level1 in range(self._K):
            row_covs = []
            if compute_correlation:
                row_corrs = []

            for level2 in range(self._K):
                row_covs.append(matrix_cov(pilot_samples[level1],pilot_samples[level2]))
                if compute_correlation:
                    row_corrs.append(matrix_corr(pilot_samples[level1],pilot_samples[level2]))


            covs.append(row_covs)
            if compute_correlation:
                corrs.append(row_corrs)
        # setting the global covariance object
        self.covs = covs

        # computing the correlation
        if compute_correlation:
            self.corrs = corrs

    def level_mean(self, key, level, n_points):
        """Estimate the mean output of a given fidelity level from fresh samples.

        Args:
            key: A JAX PRNG key used to sample this level's inputs.
            level: Fidelity level index.
            n_points: Number of samples to draw.

        Returns:
            jax.Array: Sample mean of the level's outputs, shape ``(output_dim,)``.
        """
        # sampling the input with the same key
        input_vals = []
        for input in self.evaluators[level].inputs:
            input_vals.append(input.sample(key, n_points))

        # computing the mean for a certain level
        return self.evaluators[level].evaluate(*input_vals).mean(axis=0)

    def level_sum(self, key, level, n_points):
        """Estimate the summed output of a given fidelity level from fresh samples.

        Args:
            key: A JAX PRNG key used to sample this level's inputs.
            level: Fidelity level index.
            n_points: Number of samples to draw.

        Returns:
            jax.Array: Sample sum of the level's outputs, shape ``(output_dim,)``.
        """
        # sampling the input with the same key
        input_vals = []
        for input in self.evaluators[level].inputs:
            input_vals.append(input.sample(key, n_points))

        # computing the mean for a certain level
        return self.evaluators[level].evaluate(*input_vals).sum(axis=0)

    def print(self):
        """Print every evaluator's summary."""
        for evaluator in self.evaluators:
            evaluator.print()

class RMFMC(MultifidelityMonteCarlo):
    """Regression-based multifidelity Monte Carlo (RMFMC) estimator.

    Combines nested samples across fidelity levels using regression
    coefficients computed from the pilot covariances (see
    :meth:`get_matrix_coefs`, :meth:`get_vector_coefs`,
    :meth:`get_scalar_coefs`) to form an unbiased, reduced-variance
    estimate of the high-fidelity mean.

    Attributes:
        l2_reg: Regularization parameter for the least-squares coefficient solve.
        rcond: Relative condition number for the least-squares coefficient solve.
        _betas: Nested per-(sample level, fidelity level) coefficients used
            by :meth:`evaluate`, derived from ``_coefs`` and the sample sizes.
        _info_coefs: Per-level variance-reduction "information" coefficients,
            used for sample allocation.
        _coefs: Nested per-level regression coefficients computed by
            ``get_*_coefs``.
        _costs: Per-evaluator cost array.
    """
    l2_reg : float = Field(default = 0.0, description = "regularization parameter for least-squares solve")
    rcond : float = Field(default = 1e-12, description = "relative condition number for least squares solve")

    _betas : list[list[jax.Array]] | None = PrivateAttr(default = None)
    _info_coefs : list[float] | None = PrivateAttr(default = None)
    _coefs : list[list[jax.Array]] | None = PrivateAttr(default = None)
    _costs : jax.Array | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        """Initialize the per-level beta placeholders and evaluator cost array."""
        super().model_post_init(__context)

        # initializing the betas
        self._betas = []
        for level in range(self._K):
            self._betas.append([None for _ in range(level + 1)])

        # initializing the costs
        self._costs = jnp.array([eval.cost for eval in self.evaluators])

    def evaluate(self, key, sample_sizes : list[int]) -> jax.Array:
        """Compute the RMFMC estimate of the high-fidelity mean.

        Args:
            key: A JAX PRNG key, split across fidelity levels for sampling.
            sample_sizes: Number of (nested) samples to draw at each level,
                ordered from lowest to highest fidelity.

        Returns:
            jax.Array: The RMFMC estimate, shape ``(hf_dim,)``.

        Raises:
            AssertionError: If coefficients have not been computed yet (see
                ``get_matrix_coefs``/``get_vector_coefs``/``get_scalar_coefs``).
        """
        assert self._coefs is not None, "must compute coefficients before evaluating!"

        # convert from the estimator-specific coefficients to the nested coefficients
        self._get_nested_coefs(sample_sizes)

        # breaking the rng key into the number of levels of fidelity
        level_keys = jrand.split(key, self._K)

        # computing the lowest-fidelity estimate first
        estimate = self._betas[0][0] @ self.level_sum(level_keys[0], 0, sample_sizes[0])

        # iterating through the iid input samples and computing the means
        for sample_level in range(1, self._K):
            for fidelity_level in range(0, sample_level+1):
                # evaluate this specific level of fidelity on this specific set of inputs with this specific parameter
                estimate += self._betas[sample_level][fidelity_level] @ self.level_sum(level_keys[sample_level], fidelity_level, sample_sizes[sample_level])

        return estimate

    def _get_level_trace(self, level):
        """Compute the trace term used for the information coefficient at ``level``.

        Args:
            level: Fidelity level index.

        Returns:
            jax.Array: Scalar trace term for this level.
        """
        if level < self._K - 1:
            # define the block matrix coefficients
            A_level  = jnp.block([self._coefs[level]])

            # obtain the covariance matrix
            V_level = jnp.block([row[0:level+1] for row in self.covs[0:level+1]])
            C_level = jnp.block([self.covs[-1][i] for i in range(level+1)])

            # compute the trace term
            return jnp.trace(A_level @ V_level @ A_level.T - 2 * C_level @ A_level.T)
        else:
            return jnp.trace(self.covs[-1][-1])

    def _get_variance(self, ms : jax.Array):
        """Compute the estimator variance for given per-level sample sizes.

        Args:
            ms: Per-level sample sizes.

        Returns:
            jax.Array: Scalar estimator variance.
        """
        return jnp.sum(jnp.array(self._info_coefs) / jnp.array(ms))

    def get_matrix_coefs(self):
        """Compute optimal full-matrix regression coefficients at every level.

        This function gets overwritten for the other estimator
        implementations (e.g. :class:`MFMC`) depending on how the
        coefficients are formulated. Populates ``self._coefs``.
        """
        # reset the current _coefs list
        self._coefs = []

        # computing the rmfmc coefficients at each level of fidelity
        for level in range(self._K-1):
            # obtain the covariance matrix
            V_level = jnp.block([row[0:level+1] for row in self.covs[0:level+1]])
            C_level = jnp.block([self.covs[-1][i] for i in range(level+1)])

            # add l2 regularization for stability
            V_level += self.l2_reg * jnp.eye(V_level.shape[0])

            # compute the optimal block matrix coefficients
            A_level = ls(V_level, C_level.T, rcond = self.rcond).T

            # divide into the sample-model coefficients
            dim_counter = 0

            level_coefs = []

            for l in range(level + 1):
                dim = self.evaluators[l].output_dim
                level_coefs.append(
                    A_level[:,dim_counter:dim_counter + dim]
                )
                dim_counter += dim

            self._coefs.append(level_coefs)

        # setting the high-fidelity coefficients for unbiasedness
        self._coefs.append([jnp.zeros((self._hf_dim, dim)) for dim in [eval.output_dim for eval in self.evaluators]])
        self._coefs[-1][-1] = jnp.eye(self._hf_dim)

    def get_vector_coefs(self):
        """Compute optimal diagonal-matrix (vector) regression coefficients at every level.

        Restricts each level's coefficient block to a diagonal matrix,
        requiring all evaluators to share the same output dimension.
        Populates ``self._coefs``.

        Raises:
            AssertionError: If evaluators don't all share the same output dimension.
        """
        assert len(set([eval.output_dim for eval in self.evaluators])) == 1, "all evaluators must have the same output dimension"

        # a function to return the matrix of only the diagonal
        def ddiag(A): return jnp.diag(jnp.diag(A))

        # reset the current _coefs list
        self._coefs = []

        # computing the rmfmc coefficients at each level of fidelity
        for level in range(self._K-1):
            # obtain the covariance matrix
            V_level = []

            for i in range(level+1):
                row = []
                for j in range(level+1):
                    row.append(ddiag(self.covs[i][j]))
                V_level.append(row)

            V_level = jnp.block(V_level)

            # V_level = jnp.block([ddiag(row[0:level+1]) for row in self.covs[0:level+1]])
            C_level = jnp.concatenate([jnp.diag(self.covs[-1][i]) for i in range(level+1)], axis=0)

            # add l2 regularization for stability
            V_level += self.l2_reg * jnp.eye(V_level.shape[0])

            # compute the optimal block matrix coefficients
            vecs = ls(V_level, C_level, rcond = self.rcond).T

            # divide into the sample-model coefficients
            dim_counter = 0

            level_coefs = []

            for l in range(level + 1):
                dim = self.evaluators[l].output_dim
                level_coefs.append(
                    jnp.diag(vecs[dim_counter:dim_counter + dim])
                )
                dim_counter += dim

            self._coefs.append(level_coefs)

        # setting the high-fidelity coefficients for unbiasedness
        self._coefs.append([jnp.zeros((self._hf_dim, dim)) for dim in [eval.output_dim for eval in self.evaluators]])
        self._coefs[-1][-1] = jnp.eye(self._hf_dim)

    def get_scalar_coefs(self):
        """Compute optimal scalar (isotropic) regression coefficients at every level.

        Restricts each level's coefficient block to a scalar multiple of
        the identity, requiring all evaluators to share the same output
        dimension. Populates ``self._coefs``.

        Raises:
            AssertionError: If evaluators don't all share the same output dimension.
        """
        assert len(set([eval.output_dim for eval in self.evaluators])) == 1, "all evaluators must have the same output dimension"

        # reset the current _coefs list
        self._coefs = []

        # computing the rmfmc coefficients at each level of fidelity
        for level in range(self._K-1):
            # obtain the covariance matrix
            V_level = []

            for i in range(level+1):
                row = []
                for j in range(level+1):
                    row.append(jnp.trace(self.covs[i][j]))
                V_level.append(row)

            V_level = jnp.array(V_level)

            # V_level = jnp.block([ddiag(row[0:level+1]) for row in self.covs[0:level+1]])
            C_level = jnp.array([jnp.trace(self.covs[-1][i]) for i in range(level+1)])

            # add l2 regularization for stability
            V_level += self.l2_reg * jnp.eye(V_level.shape[0])

            # compute the optimal block matrix coefficients
            scalars = ls(V_level, C_level, rcond = self.rcond).T

            level_coefs = []

            for l in range(level + 1):
                dim = self.evaluators[l].output_dim
                level_coefs.append(
                    scalars[l] * jnp.eye(dim)
                )

            self._coefs.append(level_coefs)

        # setting the high-fidelity coefficients for unbiasedness
        self._coefs.append([jnp.zeros((self._hf_dim, dim)) for dim in [eval.output_dim for eval in self.evaluators]])
        self._coefs[-1][-1] = jnp.eye(self._hf_dim)

    def _get_nested_coefs(self, sample_sizes : list[int]):
        """Convert estimator-specific ``_coefs`` into nested-sample ``_betas``.

        Args:
            sample_sizes: Per-level sample sizes, ordered from lowest to
                highest fidelity, describing a nested sampling scheme.
        """
        # computing the nested sample sizes
        ms = (jnp.array(sample_sizes[::-1]).cumsum())[::-1]

        # iterating through the nested coefficients
        for sample_level in range(self._K):
            for fidelity_level in range(sample_level + 1):
                self._betas[sample_level][fidelity_level] = 1 / ms[sample_level] * self._coefs[sample_level][fidelity_level]

                for l in range(fidelity_level, sample_level):
                    self._betas[sample_level][fidelity_level] += (1/ms[l] - 1/ms[l+1]) * self._coefs[l][fidelity_level]

    def _get_info_coefs(self):
        """Compute the per-level variance-reduction "information" coefficients.

        Populates ``self._info_coefs``, used by :meth:`budget_alloc` and
        :meth:`_budget_fractional_alloc`.
        """
        # compute for lowest-fidelity level
        self._info_coefs = [-self._get_level_trace(0)]

        # compute for the intermediate fidelity-levels
        for level in range(1, self._K-1):
            self._info_coefs.append(
                self._get_level_trace(level-1) - self._get_level_trace(level)
            )

        # compute for the highest-fidelity
        self._info_coefs.append(
            self._get_level_trace(self._K - 2) + self._get_level_trace(self._K - 1)
        )

    def _budget_fractional_alloc(self, budget : float) -> list[float]:
        """Solve the (real-valued) Lagrangian-relaxed budget allocation problem.

        Args:
            budget: Total evaluation budget.

        Returns:
            list[float]: Fractional (non-integer) per-level sample allocations.
        """
        self._get_info_coefs() # computing the information coefficients

        # computing the denominator first
        denom = jnp.array(
            [jnp.sqrt(a_l * c_l) for a_l, c_l in zip(self._info_coefs, self._costs)]
        ).sum()

        # computing the sample allocations
        ms = [budget * jnp.sqrt(a_l / c_l) / denom for a_l, c_l in zip(self._info_coefs, self._costs)]

        return ms

    def _check_order(self):
        """Check whether the fidelity ordering is valid for RMFMC.

        Raises:
            AssertionError: If ``info_coef / cost`` is not strictly
                decreasing across levels (from lowest to highest fidelity).
        """
        # computing the ai / ci ratios
        ratios = jnp.array(self._info_coefs) / self._costs

        # assessing that they're all positive
        assert (jnp.diff(ratios) < 0).all(), "levels of fidelities are out of order; ai/ci must be strictly decreasing: \n" + str(ratios)

    def budget_alloc(self, budget : float, warm_start : bool = True) -> list[float]:
        """Compute the (integer, nested) per-level sample allocation for a budget.

        Greedily increments the sample size of whichever level offers the
        best marginal variance reduction per unit cost, subject to the
        nesting constraint (``ms[level] >= ms[level + 1]``) and the budget.

        Args:
            budget: Total evaluation budget.
            warm_start: If ``True``, start from the (feasible, rounded-down)
                fractional-relaxation solution; otherwise start from the
                smallest valid nested allocation.

        Returns:
            list[float]: Per-level integer sample allocations.

        Raises:
            AssertionError: If coefficients haven't been computed yet, the
                fidelity ordering is invalid, or ``budget`` is too small for
                even the minimal allocation.
        """
        assert self._coefs is not None, "must compute coefficients first!"

        # compute information coefficients if they don't already exist
        if self._info_coefs is None:
            self._get_info_coefs()

        # check the ordering
        self._check_order()

        # warm start by rounding down the fractional allocation
        if warm_start:
            # solving the lagrangian relaxation problem
            relaxed_ms = self._budget_fractional_alloc(budget)

            # initializing the sample allocs and last sample alloc
            last_m, ms = 0, []
            # flooring and ensuring feasibility
            for m in relaxed_ms[::-1]:
                ms.append(int(jnp.maximum(last_m + 1, jnp.floor(m))))
                last_m = ms[-1]

            # reversing the list
            ms.reverse()
        else:
            # just starting at the smallest possible sample allocation
            ms = [i + 1 for i in range(self._K)][::-1]

        # compute initial budget
        current_budget = jnp.inner(jnp.array(ms), self._costs)

        # checking that the budget is large enough
        assert current_budget <= budget, "budget is too small! try setting warm_start = False"

        # initialize deltas
        deltas = [ai / mi - ai / (mi + 1) for ai, mi in zip(self._info_coefs, ms)]

        # loop through and increment sample sizes
        while any(d > 0 for d in deltas):
            # finding the maximum ratio of variance reduction to cost
            level = jnp.argmax(jnp.array(deltas) / self._costs)

            # checking if feasible
            budget_valid = current_budget + self._costs[level] <= budget
            order_valid = level == self._K-1 or (ms[level] > ms[level+1])

            # increment sample size if valid
            if  budget_valid:
                # updating the sample size, delta, and budget used
                ms[level] += 1
                deltas[level] = self._info_coefs[level] * (1/ms[level] - 1/(ms[level] + 1))
                current_budget += self._costs[level]
            elif not budget_valid:
                # if we can't afford to increment this level anymore we take its candidacy away
                deltas[level] = -1

        return ms

    def _get_level_variance(self, level : int):
        """Compute the per-output-dimension trace term at ``level``.

        Args:
            level: Fidelity level index.

        Returns:
            jax.Array: Per-dimension trace-term vector, shape ``(hf_dim,)``.
        """
        if level < self._K - 1:
            # define the block matrix coefficients
            A_level  = jnp.block([self._coefs[level]])

            # obtain the covariance matrix
            V_level = jnp.block([row[0:level+1] for row in self.covs[0:level+1]])
            C_level = jnp.block([self.covs[-1][i] for i in range(level+1)])

            # compute the trace term
            return jnp.diag(A_level @ V_level @ A_level.T - 2 * C_level @ A_level.T)
        else:
            return jnp.diag(self.covs[-1][-1])

    def _get_info_vars(self):
        """Compute the per-dimension, per-level information coefficients.

        Returns:
            jax.Array: Array of shape ``(_K, hf_dim)`` with the per-level,
            per-output-dimension variance-reduction coefficients.
        """
        # compute for lowest-fidelity level
        info_vars = [-self._get_level_variance(0)]

        # compute for the intermediate fidelity-levels
        for level in range(1, self._K-1):
            info_vars.append(
                self._get_level_variance(level-1) - self._get_level_variance(level)
            )

        # compute for the highest-fidelity
        info_vars.append(
            self._get_level_variance(self._K - 2) + self._get_level_variance(self._K - 1)
        )

        return jnp.array(info_vars)

    def get_entry_variance(self, ms:list[int]):
        """Compute the per-output-dimension estimator variance for given sample sizes.

        Args:
            ms: Per-level sample sizes.

        Returns:
            jax.Array: Per-dimension variance, shape ``(hf_dim,)``.
        """
        info_vars = self._get_info_vars()
        result = np.zeros_like(info_vars[0])

        for info_var, m in zip(info_vars, ms):
            result += info_var / m

        return result



class MFMC(RMFMC):
    """Multifidelity Monte Carlo (MFMC) estimator.

    A special case of :class:`RMFMC` in which each level's regression
    coefficient is derived only from its own covariance/cross-covariance
    with the high-fidelity level, rather than the full joint covariance of
    all lower levels.
    """
    def _get_level_trace(self, level):
        """Compute the trace term used for the information coefficient at ``level``.

        Args:
            level: Fidelity level index.

        Returns:
            jax.Array: Scalar trace term for this level.
        """
        if level < self._K - 1:
            # define the block matrix coefficients
            A_level  = self._coefs[level]

            # obtain the covariance matrix
            V_level = self.covs[level][level]
            C_level = self.covs[-1][level]

            # compute the trace term
            return jnp.trace(A_level @ V_level @ A_level.T - 2 * C_level @ A_level.T)
        else:
            return jnp.trace(self.covs[-1][-1])

    def get_matrix_coefs(self):
            """Compute the per-level MFMC regression coefficients.

            Unlike :meth:`RMFMC.get_matrix_coefs`, each level's coefficient
            depends only on its own covariance/cross-covariance with the
            high-fidelity level. Populates ``self._coefs``.
            """
            # reset the current _coefs list
            self._coefs = []

            # computing the rmfmc coefficients at each level of fidelity
            for level in range(self._K-1):
                # only level-specific changes for MFMC
                V_level = self.covs[level][level]
                C_level = self.covs[-1][level]

                # add l2 regularization for stability
                V_level += self.l2_reg * jnp.eye(V_level.shape[0])

                # compute the optimal block matrix coefficients
                self._coefs.append(ls(V_level, C_level.T, rcond = self.rcond).T)

            # setting the high-fidelity coefficients for unbiasedness
            self._coefs.append(jnp.eye(self._hf_dim))


class MLMC(MultifidelityMonteCarlo):
    """Multilevel Monte Carlo (MLMC) estimator.

    Combines independently-sampled telescoping corrections
    ``E[f_l] - E[f_{l-1}]`` across levels, rather than nested regression
    coefficients.

    Attributes:
        _costs: Per-evaluator cost array.
        _level_costs: Per-level cost of one telescoping-correction sample
            (level 0 evaluates only evaluator 0; level ``l > 0`` evaluates
            both evaluators ``l - 1`` and ``l``).
        _info_coefs: Per-level variance-reduction coefficients used for
            sample allocation.
    """
    _costs : jax.Array | None = PrivateAttr(default = None)
    _level_costs : jax.Array | None = PrivateAttr(default = None)
    _info_coefs : list[float] | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        """Compute per-evaluator and per-level (telescoping-pair) costs."""
        super().model_post_init(__context)

        # cost of a single evaluation of each evaluator
        self._costs = jnp.array([eval.cost for eval in self.evaluators])

        # cost per sample at each telescoping level: level 0 evaluates only
        # evaluator 0, while level l (l>0) evaluates both evaluators l-1 and
        # l on the same shared batch
        self._level_costs = jnp.array(
            [self._costs[0]] + [self._costs[l-1] + self._costs[l] for l in range(1, self._K)]
        )

    def evaluate(self, key, sample_sizes : list[int]) -> jax.Array:
        """Compute the MLMC telescoping-sum estimate of the high-fidelity mean.

        Args:
            key: A JAX PRNG key, split across fidelity levels for sampling.
            sample_sizes: Number of samples to draw at each level.

        Returns:
            jax.Array: The MLMC estimate, shape ``(hf_dim,)``.
        """
        # breaking the rng key into the number of levels of fidelity
        level_keys = jrand.split(key, self._K)

        # base estimate using the lowest-fidelity model's own batch
        estimate = self.level_mean(level_keys[0], 0, sample_sizes[0])

        # telescoping corrections: each level shares one batch between the
        # current and previous fidelity levels
        for level in range(1, self._K):
            estimate += (
                self.level_mean(level_keys[level], level, sample_sizes[level])
                - self.level_mean(level_keys[level], level - 1, sample_sizes[level])
            )

        return estimate

    def _get_level_trace(self, level):
        """Compute the trace of the (differenced, for level > 0) covariance at ``level``.

        Args:
            level: Fidelity level index.

        Returns:
            jax.Array: Scalar trace term for this level.
        """
        if level == 0:
            return jnp.trace(self.covs[0][0])
        else:
            diff_cov = (
                self.covs[level][level] + self.covs[level-1][level-1]
                - self.covs[level][level-1] - self.covs[level-1][level]
            )
            return jnp.trace(diff_cov)

    def _get_info_coefs(self):
        """Compute and store the per-level variance-reduction coefficients."""
        self._info_coefs = [self._get_level_trace(level) for level in range(self._K)]

    def _get_variance(self, ms : jax.Array):
        """Compute the estimator variance for given per-level sample sizes.

        Args:
            ms: Per-level sample sizes.

        Returns:
            jax.Array: Scalar estimator variance.
        """
        if self._info_coefs is None:
            self._get_info_coefs()

        return jnp.sum(jnp.array(self._info_coefs) / jnp.array(ms))

    def _get_level_variance(self, level):
        """Compute the per-output-dimension (differenced, for level > 0) variance at ``level``.

        Args:
            level: Fidelity level index.

        Returns:
            jax.Array: Per-dimension variance vector, shape ``(hf_dim,)``.
        """
        if level == 0:
            return jnp.diag(self.covs[0][0])
        else:
            diff_cov = (
                self.covs[level][level] + self.covs[level-1][level-1]
                - self.covs[level][level-1] - self.covs[level-1][level]
            )
            return jnp.diag(diff_cov)

    def get_entry_variance(self, ms : list[int]):
        """Compute the per-output-dimension estimator variance for given sample sizes.

        Args:
            ms: Per-level sample sizes.

        Returns:
            jax.Array: Per-dimension variance, shape ``(hf_dim,)``.
        """
        result = jnp.zeros(self._hf_dim)

        for level in range(self._K):
            result += self._get_level_variance(level) / ms[level]

        return result

    def _budget_fractional_alloc(self, budget:float):
        """Solve the (real-valued) Lagrangian-relaxed budget allocation problem.

        Args:
            budget: Total evaluation budget.

        Returns:
            list[int]: Per-level integer (floored, feasibility-clamped)
            sample allocations.
        """
        # solving the lagrangian relaxation problem
        denom = jnp.array(
            [jnp.sqrt(a_l * c_l) for a_l, c_l in zip(self._info_coefs, self._level_costs)]
        ).sum()
        relaxed_ns = [budget * jnp.sqrt(a_l / c_l) / denom for a_l, c_l in zip(self._info_coefs, self._level_costs)]

        # flooring and ensuring feasibility
        return [int(jnp.maximum(1, jnp.floor(n))) for n in relaxed_ns]

    def budget_alloc(self, budget : float, warm_start : bool = True) -> list[float]:
        """Compute the per-level sample allocation for a given budget.

        MLMC levels use independent batches, so (unlike RMFMC/MFMC) there
        is no nesting/ordering constraint between sample sizes.

        Args:
            budget: Total evaluation budget.
            warm_start: If ``True``, start from the (feasible) fractional
                relaxation solution; otherwise start from one sample per
                level.

        Returns:
            list[float]: Per-level integer sample allocations.

        Raises:
            AssertionError: If ``budget`` is too small for even the minimal
                allocation.
        """
        if self._info_coefs is None:
            self._get_info_coefs()

        # MLMC levels use independent batches, so there is no nesting/ordering
        # constraint between sample sizes (unlike RMFMC/MFMC)
        if warm_start:
            ns = self._budget_fractional_alloc(budget)
        else:
            # starting at the smallest possible sample allocation
            ns = [1 for _ in range(self._K)]

        # compute initial budget
        current_budget = jnp.inner(jnp.array(ns), self._level_costs)

        # checking that the budget is large enough
        assert current_budget <= budget, "budget is too small! try setting warm_start = False"

        # initialize deltas
        deltas = [a_l / n - a_l / (n + 1) for a_l, n in zip(self._info_coefs, ns)]

        # loop through and increment sample sizes
        while any(d > 0 for d in deltas):
            # finding the maximum ratio of variance reduction to cost
            level = int(jnp.argmax(jnp.array(deltas) / self._level_costs))

            # increment sample size if affordable
            if current_budget + self._level_costs[level] <= budget:
                ns[level] += 1
                deltas[level] = self._info_coefs[level] * (1 / ns[level] - 1 / (ns[level] + 1))
                current_budget += self._level_costs[level]
            else:
                # if we can't afford to increment this level anymore we take its candidacy away
                deltas[level] = -1

        return ns




class HFMC(MultifidelityMonteCarlo):
    """Plain high-fidelity-only Monte Carlo estimator (a baseline).

    Ignores every lower-fidelity level and spends the entire budget on the
    high-fidelity evaluator.

    Attributes:
        _costs: Per-evaluator cost array.
    """
    _costs : jax.Array | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        """Compute per-evaluator costs."""
        super().model_post_init(__context)

        self._costs = jnp.array([eval.cost for eval in self.evaluators])

    def evaluate(self, key, sample_sizes : list[int]) -> jax.Array:
        """Compute the plain Monte Carlo estimate using only the high-fidelity level.

        Args:
            key: A JAX PRNG key.
            sample_sizes: Per-level sample sizes; only the last entry
                (high-fidelity) is used.

        Returns:
            jax.Array: The Monte Carlo estimate, shape ``(hf_dim,)``.
        """
        return self.level_mean(key, self._K - 1, sample_sizes[-1])

    def _get_variance(self, ms : list[int]):
        """Compute the estimator variance for a given high-fidelity sample size.

        Args:
            ms: Per-level sample sizes; only the last entry is used.

        Returns:
            jax.Array: Scalar estimator variance.
        """
        return jnp.trace(self.covs[-1][-1]) / ms[-1]

    def get_entry_variance(self, ms : list[int]):
        """Compute the per-output-dimension estimator variance.

        Args:
            ms: Per-level sample sizes; only the last entry is used.

        Returns:
            jax.Array: Per-dimension variance, shape ``(hf_dim,)``.
        """
        return jnp.diag(self.covs[-1][-1]) / ms[-1]

    def _budget_fractional_alloc(self, budget : float):
        """Allocate the entire budget to the high-fidelity level.

        Args:
            budget: Total evaluation budget.

        Returns:
            list[float]: Zero for every lower-fidelity level, with the
            (real-valued) high-fidelity sample count last.
        """
        n_hf = budget / self._costs[-1]

        return [0 for _ in range(self._K - 1)] + [n_hf]

    def budget_alloc(self, budget : float, warm_start : bool = True) -> list[float]:
        """Allocate the entire budget to the high-fidelity level.

        Args:
            budget: Total evaluation budget.
            warm_start: Unused; present for interface compatibility with the
                other estimators.

        Returns:
            list[float]: Zero for every lower-fidelity level, with the
            (integer) high-fidelity sample count last.

        Raises:
            AssertionError: If ``budget`` cannot afford even a single
                high-fidelity evaluation.
        """
        n_hf = int(budget // self._costs[-1])

        assert n_hf >= 1, "budget is too small!"

        return [0 for _ in range(self._K - 1)] + [n_hf]
