from elyza.util.imports import * 
from elyza.core.evaluator import Evaluator
from elyza.util.helpers import matrix_cov, matrix_corr, ls

'''
This is the multifidelity monte carlo base model. 
'''
class MultifidelityMonteCarlo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) 

    evaluators : list[Evaluator] = Field(default = [], description = "A list of multifidelity evaluators (0 is high-fidelity)")
    covs : list[list[jax.Array]] | None = Field(default = None, description = "A 2d nested list of covariance matrices relating the levels of fidelity such that covs[level1][level2] = Cov{level1}{level2}")
    corrs : list[list[jax.Array]] | None = Field(default = None, description = "correlation matrix") 

    _K : int = PrivateAttr(default = 0)
    _hf_dim : int = PrivateAttr(default = 1)

    def model_post_init(self, __context):
        assert len(self.evaluators) != 0, "passed empty list of fidelities"
        self._K = len(self.evaluators) # setting the number of levels of fidelity
        self._hf_dim = self.evaluators[-1].output_dim # setting the high-fidelity dimension 

    '''
    a function for storing and computing pilot 
    '''
    def get_pilots(self, key, n_pilots:int, set_costs = False): 
        pilot_samples = []

        # iterating through and computing random pilot samples of each evaluator
        for evaluator in self.evaluators:
            # sampling the inputs to each evaluator with the same key 
            input_vals = [] 
            for input in evaluator.inputs:
                input_vals.append(input.sample(key, n_pilots))

            # timing each evaluator on the input samples
            pilot_samples.append(evaluator.evaluate_timed(*input_vals, set_cost = set_costs))

        # computing the covariance for each fidelity pair
        covs, corrs = [], []  
        for level1 in range(self._K):
            row_covs = [] 
            row_corrs = [] 

            for level2 in range(self._K): 
                row_covs.append(matrix_cov(pilot_samples[level1],pilot_samples[level2]))
                row_corrs.append(matrix_corr(pilot_samples[level1],pilot_samples[level2]))


            covs.append(row_covs)
            corrs.append(row_corrs) 
        # setting the global covariance object
        self.covs = covs
        self.corrs = corrs 

    def level_mean(self, key, level, n_points):
        # sampling the input with the same key
        input_vals = [] 
        for input in self.evaluators[level].inputs:
            input_vals.append(input.sample(key, n_points))

        # computing the mean for a certain level
        return self.evaluators[level].evaluate(*input_vals).mean(axis=0)

    def level_sum(self, key, level, n_points):
        # sampling the input with the same key
        input_vals = [] 
        for input in self.evaluators[level].inputs:
            input_vals.append(input.sample(key, n_points))

        # computing the mean for a certain level
        return self.evaluators[level].evaluate(*input_vals).sum(axis=0)

    def print(self):
        for evaluator in self.evaluators:
            evaluator.print()

class RMFMC(MultifidelityMonteCarlo):
    l2_reg : float = Field(default = 0.0, description = "regularization parameter for least-squares solve")
    rcond : float = Field(default = 1e-12, description = "relative condition number for least squares solve")

    _betas : list[list[jax.Array]] | None = PrivateAttr(default = None)
    _info_coefs : list[float] | None = PrivateAttr(default = None)
    _coefs : list[list[jax.Array]] | None = PrivateAttr(default = None) 
    _costs : jax.Array | None = PrivateAttr(default = None) 

    def model_post_init(self, __context):
        super().model_post_init(__context)

        # initializing the betas 
        self._betas = [] 
        for level in range(self._K):
            self._betas.append([None for _ in range(level + 1)])

        # initializing the costs 
        self._costs = jnp.array([eval.cost for eval in self.evaluators])

    def evaluate(self, key, sample_sizes : list[int]) -> jax.Array:
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
        return jnp.sum(jnp.array(self._info_coefs) / jnp.array(ms))

    '''
    this function gets overwritten for the other estimator implementations depending on how the coefficients are formulated
    '''
    def get_matrix_coefs(self):
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
        # computing the nested sample sizes 
        ms = (jnp.array(sample_sizes[::-1]).cumsum())[::-1]

        # iterating through the nested coefficients
        for sample_level in range(self._K):
            for fidelity_level in range(sample_level + 1):
                self._betas[sample_level][fidelity_level] = 1 / ms[sample_level] * self._coefs[sample_level][fidelity_level] 

                for l in range(fidelity_level, sample_level):
                    self._betas[sample_level][fidelity_level] += (1/ms[l] - 1/ms[l+1]) * self._coefs[l][fidelity_level]

    def _get_info_coefs(self):
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
        self._get_info_coefs() # computing the information coefficients 

        # computing the denominator first
        denom = jnp.array(
            [jnp.sqrt(a_l * c_l) for a_l, c_l in zip(self._info_coefs, self._costs)]
        ).sum() 

        # computing the sample allocations 
        ms = [budget * jnp.sqrt(a_l / c_l) / denom for a_l, c_l in zip(self._info_coefs, self._costs)]

        return ms
    '''
    function to check whether the ordering is valid 
    '''
    def _check_order(self):
        # computing the ai / ci ratios 
        ratios = jnp.array(self._info_coefs) / self._costs

        # assessing that they're all positive 
        assert (jnp.diff(ratios) < 0).all(), "levels of fidelities are out of order; ai/ci must be strictly decreasing: \n" + str(ratios)

    def budget_alloc(self, budget : float, warm_start : bool = True) -> list[float]:
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
        info_vars = self._get_info_vars() 
        result = np.zeros_like(info_vars[0]) 

        for info_var, m in zip(info_vars, ms):
            result += info_var / m

        return result 
        


class MFMC(RMFMC):
    def _get_level_trace(self, level):
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
    _costs : jax.Array | None = PrivateAttr(default = None)
    _level_costs : jax.Array | None = PrivateAttr(default = None)
    _info_coefs : list[float] | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
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
        if level == 0:
            return jnp.trace(self.covs[0][0])
        else:
            diff_cov = (
                self.covs[level][level] + self.covs[level-1][level-1]
                - self.covs[level][level-1] - self.covs[level-1][level]
            )
            return jnp.trace(diff_cov)

    def _get_info_coefs(self):
        self._info_coefs = [self._get_level_trace(level) for level in range(self._K)]

    def _get_variance(self, ms : jax.Array):
        if self._info_coefs is None:
            self._get_info_coefs()

        return jnp.sum(jnp.array(self._info_coefs) / jnp.array(ms))

    def _budget_fractional_alloc(self, budget:float):
        # solving the lagrangian relaxation problem
        denom = jnp.array(
            [jnp.sqrt(a_l * c_l) for a_l, c_l in zip(self._info_coefs, self._level_costs)]
        ).sum()
        relaxed_ns = [budget * jnp.sqrt(a_l / c_l) / denom for a_l, c_l in zip(self._info_coefs, self._level_costs)]

        # flooring and ensuring feasibility
        return [int(jnp.maximum(1, jnp.floor(n))) for n in relaxed_ns]

    def budget_alloc(self, budget : float, warm_start : bool = True) -> list[float]:
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
    _costs : jax.Array | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        super().model_post_init(__context)

        self._costs = jnp.array([eval.cost for eval in self.evaluators])

    def evaluate(self, key, sample_sizes : list[int]) -> jax.Array:
        return self.level_mean(key, self._K - 1, sample_sizes[-1])

    def _get_variance(self, ms : list[int]):
        return jnp.trace(self.covs[-1][-1]) / ms[-1]

    def get_entry_variance(self, ms : list[int]):
        return jnp.diag(self.covs[-1][-1]) / ms[-1]

    def _budget_fractional_alloc(self, budget : float):
        n_hf = budget / self._costs[-1]
                
        return [0 for _ in range(self._K - 1)] + [n_hf]

    def budget_alloc(self, budget : float, warm_start : bool = True) -> list[float]:
        n_hf = int(budget // self._costs[-1])

        assert n_hf >= 1, "budget is too small!"

        return [0 for _ in range(self._K - 1)] + [n_hf]

