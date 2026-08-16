from elyza.util.imports import * 
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

    sampling_func : SkipValidation[callable] | None = Field(default = None, description = "function which takes a PRNG key as an input and returns a single sample") 

    '''
    Function to sample from some distribution 
    '''
    def sample(self, key, n_points : int) -> jax.Array:        
        assert self.sampling_func is not None, "No sampling function provided!"
        
        # splitting the jrand key into the number of points needed
        keys = jrand.split(key, n_points) 

        # using vmap to sample over the keys 
        return vmap(self.sampling_func, in_axes=0)(keys)

    '''
    Function to print the contents of the input
    '''
    def print(self):
        print(" * Name: %s, Dimension: %d, Type: %s" % (
            self.name, self.dim, str(type(self))
        ))

class ScalarInput(Input):
    minval : float = Field(description = "Minimum input value")
    maxval : float = Field(description = "Maximum output value")

    def model_post_init(self, __context):
        # enforcing dimension to 1 
        self.dim = 1 

    def print(self):
        super().print() 
        print("   - Min Value: %s, Max Value: %s" % (self.minval, self.maxval))

class VectorInput(Input):
    minval : jax.Array | np.ndarray = Field(description = "1d array of lower bounds")
    maxval : jax.Array | np.ndarray = Field(description = "1d array of upper bounds")

    def model_post_init(self, __context):
        assert self.minval.shape[0] == self.dim, "minimum values array != input dimension"
        assert self.maxval.shape[0] == self.dim, "maximum values array != input dimension"

        # converting to jax arrays 
        self.minval = jnp.array(self.minval)
        self.maxval = jnp.array(self.maxval)
        