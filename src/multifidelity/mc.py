from util.imports import * 
from core.evaluator import Evaluator
from util.helpers import matrix_cov 

'''
This is the multifidelity monte carlo base model. 
'''
class MultifidelityMC(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True) 

    evaluators : list[Evaluator] = Field(default = [], description = "A list of multifidelity evaluators (0 is high-fidelity)")
    covs : list[list[jax.Array]] | None = Field(default = None, description = "A 2d nested list of covariance matrices relating the levels of fidelity such that covs[level1][level2] = Cov{level1}{level2}")

    _K : int = PrivateAttr(default = 0)

    def model_post_init(self, __context):
        assert len(self.evaluators) != 0, "Passed empty list of fidelities"
        self._K = len(self.evaluators)

    def pilot_covariances(self, n_pilots:int, seed = 42, set_costs = True): 
        pilot_samples = [] 
        key = jrand.PRNGKey(seed) 

        for evaluator in self.evaluators: 
            pilot_samples.append(evaluator.random_evaluate_timed(key, n_pilots, check_inputs = False, set_cost = set_costs))

        covs = [] 
        for level1 in range(self._K):
            row_covs = [] 
            for level2 in range(self._K): 
                row_covs.append(
                    matrix_cov(
                        pilot_samples[level1],
                        pilot_samples[level2]))

        self.covs = covs

    def single_level_eval(self, key, level, n_points, check_inputs = False):
        input_samples = [] 
        for input in self.evaluators[level].inputs:
            input_samples.append(
                input.sample(key, (n_points, input.dim))
            )

        return self.evaluators[level].evaluate_timed(*input_samples).mean(axis=0)

    def print(self):
        for evaluator in self.evaluators:
            evaluator.print()


'''
The Multilevel Monte Carlo algorithm proposed by Giles et al. in 2015 (and earlier)
'''
class MultilevelMonteCarlo(MultifidelityMC):
    def model_post_init(self, __context):
        super().model_post_init(__context)
        assert len(set([evaluator.output_dim for evaluator in self.evaluators])) == 1, "output dimensions must match for MLMC!"

    def evaluate(self, sample_sizes : list[int], seed = 42, check_inputs = True) -> jax.Array: 
        assert len(sample_sizes) == self._K, "sample sizes must be the same number of model evaluations"
        for i in range(1, len(sample_sizes)):
            assert sample_sizes[i] > sample_sizes[i-1], "sample sizes must be monotonically increasing"

        keys = jrand.split(jrand.PRNGKey(seed), self._K)

        pass 

        # TO BE IMPLEMENTED 



