from elyza.surrogate.gp import GaussianProcess, ARD, Linear
from elyza.surrogate.abstract import Surrogate, SupervisedDataset

from elyza.core.data import Input
from elyza.core.evaluator import Evaluator 

from elyza.util.imports import * 
from elyza.util.helpers import ensure_2d
from elyza.util.preprocessing import StandardScaler

from elyza.optim.abstract import Optimizer, OptimizerOptions 

'''
Hierarchical Surrogate Model base class 
'''
class HierarchicalSurrogate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # public fields 
    data : list[SupervisedDataset] = Field(default = None, description = "list of individual supervised datasets ")
    evaluators : list[Evaluator] | None = Field(default = None, description = "list of evaluators in case we want to generate data on the fly")

    # private fields 
    _K : int | None = PrivateAttr(default = None)
    _surrogates : list[Surrogate] | None = PrivateAttr(default = None) 
    _pred_kwargs : list[list] | None = PrivateAttr(default = None)

    def model_post_init(self, __context):
        assert len(self.evaluators) == len(self.data), "number of datasets doesn't match number of evaluators"

        # setting the number of levels of fidelity 
        self._K = len(self.evaluators)  

        # initializing the list of surrogates 
        self._surrogates = [None] * self._K 

        # initializing prediction keyword arguments 
        self._pred_kwargs = [[]] * self._K

'''
Multifidelity-augmented Gaussian Process inputs class
'''
class MAGPI(HierarchicalSurrogate):
    def model_post_init(self, __context):
        super().model_post_init(__context) 

    '''
    this is for setting a level-specific surrogate model. the surrogate must be declared ahead of time. pred_kwargs is the set of prediction keyword arguments 
    '''
    def set_surrogate(self, level : int, surrogate : Surrogate, **pred_kwargs):
        # declaring the surrogate using the keyword arguments 
        self._surrogates[level] = surrogate
        # setting the prediction keyword arguments 
        self._pred_kwargs[level] = pred_kwargs

    def set_optimizer(self, level:int, optimizer:Optimizer, optimizer_opts:OptimizerOptions):
        assert self._surrogates[level] is not None, "you must use set_surrogate() to assign a surrogate model to this level of fidelity" 

        self._surrogates[level].set_optimizer()


    '''
    fitting level-specific surrogate models to data. requires making predictions at the lower-fidelity surrogate models so you have to make sure those surrogate models have been fit first. 
    '''
    def fit(self, level:int):
        # compute the level inputs 
        features = self.data[level].concatenate_inputs() 
        level_outputs = self.data[level].output_data 

        # lower-fidelity outputs 
        for lower_level in range(level):
            # obtain the level-specific outputs
            outputs = self._surrogates[lower_level].predict(
                features,  
                **self._pred_kwargs[lower_level] 
            )

            # if the model returns multiple outputs always take the first arguments
            if type(outputs) is tuple: 
                outputs = outputs[0] 

            # append the model output to the lf_outputs 
            features= jnp.concatenate(
                (features, ensure_2d(outputs)), axis=1
            )

        # fitting the level-specific surrogate models 
        self._surrogates[level].fit(
            features, 
            level_outputs
        )

    def update(self, new_data : SupervisedDataset, level : int, **kwargs):
        # updating the data with new data 
        self.data[level].update(*new_data.input_data, new_data.output_data)

        # updating the surrogate model with the new data 
        self._surrogates[level].update(
            new_data.concatenate_inputs(), 
            new_data.output_data, 
            **kwargs
        )

    def predict(self, *new_inputs : jax.Array, level : int, **pred_kwargs) -> jax.Array | tuple[jax.Array]:
        # compute the level inputs 
        features = jnp.concatenate(new_inputs) 

        # lower-fidelity outputs 
        for lower_level in range(level):
            outputs = self._surrogates[lower_level].predict(
                features, **self._pred_kwargs[lower_level]
            )

            # if the model returns multiple outputs always take the first arguments
            if type(outputs) is tuple: 
                outputs = outputs[0] 

            # append the model output to the lf_outputs 
            features = jnp.concatenate((features, ensure_2d(outputs)), axis=1)

        # making the prediction at this level 
        return self._surrogates[level].predict(
            features, **pred_kwargs
        )

    def sample(self, *new_inputs : jax.Array, level:int):
        raise NotImplementedError("this method has not been implemented yet")



