from elyza.util.imports import * 
from elyza.core.evaluator import Evaluator
from elyza.util.helpers import matrix_cov, ls

'''
This is the multifidelity monte carlo base model. 
'''
class MultifidelityMonteCarlo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) 

    evaluators : list[Evaluator] = Field(default = [], description = "A list of multifidelity evaluators (0 is high-fidelity)")
    covs : list[list[jax.Array]] | None = Field(default = None, description = "A 2d nested list of covariance matrices relating the levels of fidelity such that covs[level1][level2] = Cov{level1}{level2}")

    _K : int = PrivateAttr(default = 0)
    _hf_dim : int = PrivateAttr(default = 1)

    def model_post_init(self, __context):
        assert len(self.evaluators) != 0, "passed empty list of fidelities"
        self._K = len(self.evaluators) # setting the number of levels of fidelity
        self._hf_dim = self.evaluators[-1].output_dim # setting the high-fidelity dimension 

    '''
    a function for storing and computing pilot 
    '''
    def get_pilots(self, key, n_pilots:int, set_costs = True): 
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
        covs = [] 
        for level1 in range(self._K):
            row_covs = [] 

            for level2 in range(self._K): 
                row_covs.append(matrix_cov(pilot_samples[level1],pilot_samples[level2]))

            covs.append(row_covs)
        # setting the global covariance object
        self.covs = covs

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


'''
The Multilevel Monte Carlo algorithm proposed by Giles et al. in 2015 (and earlier)
'''
class MultilevelMonteCarlo(MultifidelityMonteCarlo):
    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert len(set([evaluator.output_dim for evaluator in self.evaluators])) == 1, "output dimensions must match for MLMC!"

    def evaluate(self, key, sample_sizes : list[int]):
        # breaking the rng key into the number of levels of fidelity 
        level_keys = jrand.split(key, self._K)

        # iterating through and computing the estimate 
        estimate = self.level_mean(level_keys[0], 0, sample_sizes[0]) 

        # iteratively computing the estimator
        for level in range(1, self._K + 1):
            # correcting the lower-fidelity 
            estimate -= self.level_mean(level_keys[level], level - 1, sample_sizes[level]) 

            # performing the higher-fidelity monte carlo evaluation
            estimate += self.level_mean(level_keys[level], level, sample_sizes[level])

        return estimate

class NestedEstimator(MultifidelityMonteCarlo):
    l2_reg : float = Field(default = 0.0, description = "regularization parameter for least-squares solve")
    rcond : float = Field(default = 1e-12, description = "relative condition number for least squares solve")

    _eval_mask : list[list[bool]] | None = PrivateAttr(default = None)
    _betas : list[list[jax.Array]] | None = PrivateAttr(default = None)
    _equiv_costs : list[float] | None = PrivateAttr(default = None)
    _alphas : list | None = PrivateAttr(default = None)
    _info_coefs : list[float] | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        super().model_post_init(__context)

        # initializing the betas 
        self._betas = [] 
        for level in range(self._K):
            self._betas.append([None for _ in range(level + 1)])

        # initializing the equivalent costs 
        self._equiv_costs = [self.evaluators[level].cost for level in range(self._K)]


    def evaluate(self, key, sample_sizes : list[int]) -> jax.Array:
        # breaking the rng key into the number of levels of fidelity 
        level_keys = jrand.split(key, self._K)

        # computing the lowest-fidelity estimate first 
        estimate = self._betas[0][0] @ self.level_sum(level_keys[0], 0, sample_sizes[0])

        # iterating through the iid input samples and computing the means
        for sample_level in range(1, self._K):
            for fidelity_level in range(0, sample_level+1):
                # checking if this level-sample combo is active in the estimator 
                if self._eval_mask[sample_level][fidelity_level]:
                    # evaluate this specific level of fidelity on this specific set of inputs with this specific parameter
                    estimate += self._betas[sample_level][fidelity_level] @ self.level_sum(level_keys[sample_level], fidelity_level, sample_sizes[sample_level])

        return estimate

    '''
    function for the generalized method of sample allocation
    '''
    def _get_equiv_costs(self):
        # compute lowest-fidelity equivalent cost 
        self._equiv_costs[0] = self.evaluators[0].cost 

        # compute high-fidelity equivalent cost 
        self._equiv_costs[-1] = float(self.evaluators[-1].cost + jnp.sum(
            jnp.array([self.evaluators[level].cost * (self._eval_mask[-1][level] - self._eval_mask[-2][level]) for level in range(self._K - 1)])
        ))

        # compute intermediate-fidelity equivalent costs 
        for sample_level in range(1, self._K-1):
            self._equiv_costs[sample_level] = 0.0 
            for fidelity_level in range(sample_level + 1):
                if fidelity_level == sample_level: 
                    self._equiv_costs[sample_level] += int(self._eval_mask[sample_level][fidelity_level]) * self.evaluators[fidelity_level].cost
                else: 
                    self._equiv_costs[sample_level] += (int(self._eval_mask[sample_level][fidelity_level]) - int(self._eval_mask[sample_level - 1][fidelity_level])) * self.evaluators[fidelity_level].cost 

    '''
    function to define the information coefficients 
    '''
    def _get_info_coefs(self):
        self._info_coefs = [
            float(jnp.trace(
                self.covs[-1][0] @ ls(self.covs[0][0], self.covs[0][-1] + self.l2_reg * jnp.eye(self.evaluators[0].output_dim), rcond = self.rcond)
            ))
        ]

        for level in range(1, self._K-1):
            Vl = jnp.block([row[0:level+1] for row in self.covs[0:level+1]])
            Cl = jnp.block([self.covs[-1][i] for i in range(level+1)])
            Vl += self.l2_reg * jnp.eye(Vl.shape[0])

            Vl1 = jnp.block([row[0:level] for row in self.covs[0:level]])
            Cl1 = jnp.block([self.covs[-1][i] for i in range(level)])

            Vl1 += self.l2_reg * jnp.eye(Vl1.shape[0])

            self._info_coefs.append(
                float(jnp.trace(
                    Cl @ ls(Vl, Cl.T, rcond = self.rcond) - Cl1 @ ls(Vl1, Cl1.T, rcond = self.rcond) 
                ))
            )

        # computing the higest-fidelity info coefficient
        Vl1 = jnp.block([row[0:-1] for row in self.covs[0:-1]])
        Cl1 = jnp.block([self.covs[-1][i] for i in range(self._K-1)])
        Vl1 += self.l2_reg * jnp.eye(Vl1.shape[0])

        self._info_coefs.append(float(jnp.trace(
            self.covs[-1][-1] + self.l2_reg * jnp.eye(self._hf_dim) - Cl1 @ ls(Vl1, Cl1.T, rcond = self.rcond)
        )))


    
    def budget_sample_allocation(self, budget : float) -> list[float]:
        # getting the costs and information coefficients 
        self._get_info_coefs() 
        self._get_equiv_costs() 

        # computing the denominator first
        denom = jnp.array(
            [jnp.sqrt(a_l * c_l) for a_l, c_l in zip(self._info_coefs, self._equiv_costs)]
        ).sum() 

        # computing the sample allocations 
        ms = [jnp.sqrt(a_l / c_l) / budget for a_l, c_l in zip(self._info_coefs, self._equiv_costs)]

        return jnp.array(ms)
        


class MFMC(NestedEstimator):
    def model_post_init(self, __context):
        super().model_post_init(__context) 

        # forming the MFMC evaluation mask to determine which levels are active 
        self._eval_mask = [] 
        for sample_level in range(self._K):
            sample_mask = [] 

            for _ in range(sample_level + 1):
                sample_mask.append(True) 

            self._eval_mask.append(sample_mask) 

    def get_alphas(self):
        # computing the alphas first
        self._alphas = []
        for level in range(0, self._K -1):
            self._alphas.append(
                ls(self.covs[level][level] + self.l2_reg * jnp.eye(self.evaluators[level].output_dim), self.covs[level][-1],  
                rcond = self.rcond).T
            )

        # appending the high-fidelity "coefficient" for consistency
        self._alphas.append(jnp.eye(self._hf_dim))

    def evaluate(self, key, sample_sizes : list[int]) -> jax.Array:
        assert (jnp.array(sample_sizes) >= 1).all(), "sample sizes must be at least 1"
        assert self._alphas is not None, "must initialize the alphas before evaluating"
        # converting the fidelity-specific sample sizes to nested sample sizes 
        ms = (jnp.array(sample_sizes[::-1]).cumsum())[::-1]

        # converting the alphas to nested parameters 
        for sample_level in range(self._K):
            for fidelity_level in range(sample_level + 1):
                if sample_level == fidelity_level: 
                    self._betas[sample_level][fidelity_level] = self._alphas[fidelity_level] / ms[sample_level] 
                else:
                    self._betas[sample_level][fidelity_level] = (1.0 / ms[fidelity_level] - 1.0 / ms[fidelity_level + 1]) * self._alphas[fidelity_level]

        # using the parent class' evaluate function once the betas are defined
        return super().evaluate(key, sample_sizes)


        

