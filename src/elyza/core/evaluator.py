from elyza.util.imports import *
from elyza.core.data import * 
import time 

'''
# %% The evaluator class is a way to evaluate external computer models and also store data 
'''
class Evaluator(BaseModel):
    name : int | str = Field(
        description = "the unique name for this evaluator"
    )
    inputs : list[Input] = Field(
        description = "the list of inputs associated for this evaluator"
    )
    output_dim : int = Field(default = 1, 
        description = "the output dimension of the evaluator"
    )

    evaluation_func : Callable | None = Field(
        default = None, 
        description = "a function which takes in arguments for each of the specified inputs and returns an array of the appropriate output dimension."
    )

    cost : float | None = Field(
        default = None, 
        description = "The cost to evaluate."
    )

    @computed_field # function for getting the set of input names
    def _input_names(self) -> set: 
        return set([input.name for input in self.inputs])

    def model_post_init(self, __context):
        assert len(self.inputs) == len(self._input_names), "duplicate input names detected"

    def single_eval(self, *input_vals : list[Input]):
        '''
        This is the function to evaluate on a single set of inputs
        '''
        
        # compute the evaluation function on this specific set of inputs
        return self.evaluation_func(*input_vals)

    '''
    input_vals is a list of arrays of inputs assumed to be valid for the evaluator, where the shape of each list item is (# of data points, input dimension).
    '''
    def evaluate(self, *input_vals : list[jax.Array]):
        return vmap(self.evaluation_func, in_axes=[0]*len(input_vals))(*input_vals).reshape(-1,self.output_dim)

    def print(self):
        print("\n------------------------------------------------")
        print("Evaluator Name: %s" % self.name)
        print("Output Dimension: %d" % self.output_dim)
        if self.cost is not None: 
            print("Evaluation Cost: %.4e" % self.cost)
        print("Inputs:")
        for this_input in self.inputs:
            this_input.print()
        print("------------------------------------------------\n")

    def set_cost(self, cost: float):
        assert cost > 0.0, "cost must be positive"
        self.cost = cost 

    def evaluate_timed(self, *input_vals : list[jax.Array], set_cost = True):
        n_points = input_vals[0].shape[0] 
        start_time = time.time()
        result = self.evaluate(*input_vals)
        end_time = time.time() 
        print("Total time: %.4e (s)" % (end_time - start_time))
        print("Per-evaluation time: %.4e (s)" % ((end_time - start_time) / n_points))
        if set_cost: 
            self.cost = (end_time - start_time) / n_points
        return result 
