from elyza.util.imports import *

'''
OptimizerOptions 
-----------------
a simple abstract class to 
'''
class OptimizerOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

'''
BatchGradientOptimizer
----------------------
an abstract class for gradient-based optimization in batches
'''
class BatchGradientOptimizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    loss_grad_fn : SkipValidation[callable] | None = Field(default = None, description = "a function in the form def func(p, *args) -> float")
    opts : OptimizerOptions = Field(default = OptimizerOptions(), description = "optimizer options")

    '''Function to break up data into batches'''
    def _get_batches(self, key, batch_size: int, *data) -> list[tuple[jax.Array]]:
        n = data[0].shape[0] 
        perm = jax.random.permutation(key, n)

        # shuffling the data by the same indices
        data_shuffled = [datum[perm] for datum in data] 

        n_batches = n // batch_size  # drop last incomplete batch
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batches.append(tuple(
                [datum[start:end] for datum in data_shuffled]
            ))

        return batches

    '''
    each batch gradient optimizer must have a run function in the following style
    '''
    def run(*args):
        raise NotImplementedError("this method is only a placeholder and hasn't been implemented")


