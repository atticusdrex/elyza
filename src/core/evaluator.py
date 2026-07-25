from util.imports import *
from core.data import * 
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
        description = "The cost to evaluate. "
    )

    @computed_field # function for getting the set of input names
    def _input_names(self) -> set: 
        return set([input.name for input in self.inputs])

    def model_post_init(self, __context):
        assert len(self.inputs) == len(self._input_names), "duplicate input names detected"

    def single_eval(self, *input_vals : list[Input], check_inputs : bool = False):
        '''
        This is the function to evaluate on a single set of inputs
        '''

        # checking the validity of the input can be slow when vmapping so sometimes we opt to skip 
        if check_inputs: 
            # making sure the evaluation function exists 
            assert self.evaluation_func is not None, "no evaluation function specified"
            # making sure the inputs are the same length 
            assert len(input_vals) == len(self.inputs), "mismatched input variables"

            # making the inputs are valid 
            for i, input in enumerate(input_vals): 
                self.inputs[i].check_valid(input)
        
        # compute the evaluation function on this specific set of inputs
        return self.evaluation_func(*input_vals)
    
    def evaluate(self, *input_vals : list[jax.Array], check_inputs:bool=False):
        # checking the validity of the inputs 
        if check_inputs: 
            # checking that the number of observations of each input is the same
            input_lengths = set() 
            for i, input_val in enumerate(input_vals): 
                input_lengths.add(input_val.shape[0]) 
                self.inputs[i].check_valid(input_val)
            assert len(input_lengths) == 1, "passing different numbers of values per input"

            
        
        if self.output_dim == 1: 
            return vmap(self.evaluation_func, in_axes=[0]*len(input_vals))(*input_vals).reshape(-1,1)
        else: 
            return vmap(self.evaluation_func, in_axes=[0]*len(input_vals))(*input_vals)

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
        assert cost > 0.0, "Cost must be positive"
        self.cost = cost 

    def evaluate_timed(self, *input_vals : list[jax.Array], set_cost = True):
        n_points = input_vals[0].shape[0] 
        start_time = time.time()
        result = self.evaluate(*input_vals, check_inputs = False)
        end_time = time.time() 
        print("Total time: %.4e (s)" % (end_time - start_time))
        print("Per-evaluation time: %.4e (s)" % ((end_time - start_time) / n_points))
        if set_cost: 
            self.cost = (end_time - start_time) / n_points
        return result 

    def random_evaluate(self, key, n_points, check_inputs = False):
        input_vals = []
        for this_input in self.inputs: 
            shape = (n_points, this_input.dim)
            input_vals.append(this_input.sample(key, shape))
        return self.evaluate(*input_vals, check_inputs=check_inputs)

    def random_evaluate_timed(self, key, n_points, check_inputs = False, set_cost = True): 
        assert n_points >= 1, "Must have >= 1 evaluation to time"
        start_time = time.time()
        result = self.random_evaluate(key, n_points, check_inputs = check_inputs)
        end_time = time.time() 
        print("Total time: %.4e (s)" % (end_time - start_time))
        print("Per-evaluation time: %.4e (s)" % ((end_time - start_time) / n_points))
        if set_cost: 
            self.cost = (end_time - start_time) / n_points
        return result 
