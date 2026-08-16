from util.imports import * 
from types import MethodType 

'''
A data registry class containing all the relevant input functionality, sampling, etc.
'''

'''
# %% 
A set of classes defining model inputs 
'''
class Input(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name : int | str = Field(description = "Unique input name")
    dim : int = Field(default = 1, description = "dimension of the input")

    sampling_func : SkipValidation[callable] | None = Field(default = None, description = "function which takes a PRNG key and an integer as an input") 

    @abstractmethod 
    def check_valid(self, vals : jax.Array):
        raise NotImplementedError("This feature hasn't been implemented yet")

    '''
    Function to sample from some distribution 
    '''
    def sample(self, key, n_points : int) -> jax.Array: 
        assert self.sampling_func is not None, "No sampling function provided!"
        return self.sampling_func(key, n_points)

    def print(self):
        print(" * Name: %s, Dimension: %d, Type: %s" % (
            self.name, self.dim, str(type(self))
        ))
    
class ContinuousInput(Input):
    minval : float = Field(default = 0.0, description = "Minimum input value")
    maxval : float = Field(default = 1.0, description = "Maximum output value")

    def check_valid(self, vals : jax.Array):
        assert (vals >= self.minval).all() and (vals <= self.maxval).all(), "At least one input is out of bounds!"

    def print(self):
        super().print() 
        print("   - Min Value: %s, Max Value: %s" % (self.minval, self.maxval))

'''
Class for uniform random variables 
'''
class UniformRandomInput(ContinuousInput):
    def sample(self, key, n_points: int) -> jax.Array: 
        return jrand.uniform(key, shape = (n_points, self.dim), minval = self.minval, maxval = self.maxval)


class ContinuousVectorInput(ContinuousInput):
    minval : jax.Array = Field(description = "Vector of minimum entry-wise values of inputs")
    maxval : jax.Array = Field(description = "Vector of maximum entry-wise values of inputs")

    # check for dimension/data consistency 
    def model_post_init(self, __context):
        assert len(self.minval.shape) == 2, "minval must be 1d array"
        assert len(self.maxval.shape) == 2, "maxval must be 1d array"
        assert self.dim > 1, "if passing a 1-d input use the ContinuousInput class" 
        assert self.minval.shape[0] == self.dim, "minval is wrong dimension"
        assert self.maxval.shape[0] == self.dim, "maxval is wrong dimension"

    def check_valid(self, val):
        assert type(val) == jax.Array, "input value is not a valid jax array"
        assert val.shape[0] == self.dim, "input value is of wrong dimension"
        assert (val >= self.minval).all(), "input value exceeds minimum values" 
        assert (val <= self.maxval).all(), "input value exceeds maximum values"

class DiscreteInput(Input):
    possible_values : set[int | float | str] = Field(description = "A set of possible values")

    pmf : dict[int | str, float] | None = Field (
        default = None, 
        description = "A probability mass function in the form of a dictionary mapping a specific item of the input set to the probability of being sampled"
    )

    '''
    A function for sampling discrete input values
    '''
    def sample(self, key, n_points : int) -> jax.Array:
        possible_inputs = jnp.array(list(self.pmf.keys()))
        probabilities = jnp.array(list(self.pmf.values()))
        return jrand.choice(key, possible_inputs, shape=(n_points,), p=probabilities)
    
    def check_valid(self, val):
        assert val in self.possible_values, "input value is not in the set of possible values"

class BinaryInput(DiscreteInput):
    possible_values: set[int] = Field(default_factory=set([0,1]))

    pmf : dict[int, float] | None = Field(
        default = {0:0.5, 1:0.5}, 
        description = "A probability mass function in the form of a dictionary describing the probabilities of each integer value getting chosen"
    )

class OrdinalInput(DiscreteInput):
    possible_values : set[int] = Field(description = "A set of possible integer values the inputs can take")

''' 
A placeholder class for any general categorical inputs
'''
class CategoricalInput(DiscreteInput):
    pass 
        