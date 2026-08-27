from elyza.util.imports import *
from jax.tree_util import tree_map

'''
fill_pytree_spec
----------------
builds a pytree with the same nested-dict structure as `template` (e.g. a parameter
pytree like p_init), so that callers can specify `active_params`/`constraints` etc.
for only the leaves/subtrees they care about instead of the entire pytree.

- any leaf/subtree left unspecified in `partial` is filled in with `default`
- giving a value for an intermediate dict key in `partial` broadcasts that value to
  every leaf beneath it (e.g. {'weights': False} turns off every weight layer)
- an unknown key in `partial` (typo, wrong nesting) raises rather than being silently
  dropped
'''
def fill_pytree_spec(template, partial, default):
    def _fill(node, spec, path):
        if isinstance(node, dict):
            if spec is not None and not isinstance(spec, dict):
                return tree_map(lambda _leaf: spec, node)

            spec = spec or {}
            unknown = set(spec) - set(node)
            if unknown:
                location = ".".join(map(str, path)) or "<root>"
                raise ValueError(f"unknown key(s) {unknown} at '{location}'; expected one of {set(node)}")

            return {key: _fill(subnode, spec.get(key), path + [key]) for key, subnode in node.items()}

        return default if spec is None else spec

    return _fill(template, partial, [])

'''
OptimizerOptions
-----------------
a simple abstract class to 
'''
class OptimizerOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

'''
Optimizer 
---------
abstract class for general optimizers 
'''
class Optimizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    '''
    each batch gradient optimizer must have a run function in the following style
    '''
    @abstractmethod
    def run(*args):
        raise NotImplementedError("this method is only a placeholder and hasn't been implemented")

'''
BatchGradientOptimizer
----------------------
an abstract class for gradient-based optimization in batches
'''
class BatchGradientOptimizer(Optimizer):
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

    


