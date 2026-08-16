from elyza.surrogate.gp.gp import GaussianProcess, DeltaGP 
from elyza.surrogate.gp.kernel import ARD
from elyza.surrogate.gp.mean import Linear 
from elyza.surrogate.surrogate import Surrogate

from elyza.core.data import Input
from elyza.core.evaluator import Evaluator 

from elyza.util.imports import * 

from matplotlib.pyplot import * 

'''
Hierarchical Surrogate Model base class 

'''
def HierarchicalSurrogate(BaseModel):
    mode_config = ConfigDict(arbitrary_types_allowed=True)

    # public fields 
    X : list[np.ndarray | jax.Array]
    """list of 2d numpy/jax arrays of input data """
    Y : list[np.ndarray | jax.Array] 
    """list of 1d or 2d numpy/jax arrays of corresponding output data"""
    evaluators : list[Evaluator] | None = Field(
        default = None, 
        description = "list of evaluators which determine each level of fidelity. if active learning is enabled, these can be called but do not have to be called"
    )
    noise_vars : list[float] | None = Field(default = None)
    """list of noise variances at each level of fidelity"""
    eps : float = Field(default = 1e-12) 

    # private fields 
    _K : int = Field(default = 1)

    def model_post_init(self, __context):
        # checking that the number of levels of fidelity is consistent 
        assert len(self.X) == len(self.Y), "X and Y have different numbers of levels of fidelities"
        if self.evaluators is not None: 
            assert len(self.X) == self.evaluators, "inconsistent number of fidelity levels"

        # asserting that n_obs for each level of fidelity is the same
        for level, X_level, Y_level in enumerate(zip(self.X, self.Y)):
            assert X_level.shape[0] == Y_level.shape[0], "level %d has inconsistent number of samples" % level 
            assert len(X_level.shape) == 2, "X must be a 2d array" 

        # storing the X and Y arrays as jax arrays 
        self.X, self.Y = jnp.array(self.X), jnp.array(self.Y) 

        # storing number of levels of fidelity 
        self._K = len(self.X)

from elyza.surrogate.gp.kernel import BaseKernel, ARD
from elyza.surrogate.gp.mean import BaseMean, Constant


def GPKennedyOHagan(HierarchicalSurrogate):
    kernel_cls : BaseKernel = Field(default = ARD)
    mean_cls : BaseMean = Field(default = Constant)
    verbose : bool = Field(default = True)
    max_cond : float = Field(default = 1e5)
    verbose : bool = Field(default = True)

    _models : list[Surrogate] | None = PrivateAttr(default = None)
    '''A list of the surrogate models used'''

    def model_post_init(self, __context):
        super().model_post_init(__context)

        # initializing the lowest-fidelity GP 
        self._models = [
            GaussianProcess(
                input_dim = self.X[-1].shape[1], 
                kernel_cls = self.kernel_cls, 
                mean_cls = self.mean_cls, 
                calibrate_noise = True, 
                noise_var = self.noise_vars[-1], 
                eps = self.eps, 
                max_cond = self.max_cond, 
                verbose = self.verbose
            )
        ]

        # saving a delta-GP for until high-fidelity is reached
        for level in range(0, self._K-1)[::-1]:
            self._models.append(
                DeltaGP(
                    input_dim = self.X[level].shape[1], 
                    kernel_cls = self.kernel_cls, 
                    mean_cls = self.mean_cls, 
                    calibrate_noise = True, 
                    noise_var = self.noise_vars[level], 
                    eps = self.eps, 
                    max_cond = self.max_cond, 
                    verbose = self.verbose
                )
            )

        # reversing the order so it goes from highest to lowest-fidelity
        self._models = self._models[::-1]
