from .util import * 

class BaseMean(BaseModel):
    input_dim : int 
    epsilon : float = Field(default_factory = 1e-12) 

# a trivial zero-mean function
class Zero(BaseMean):
    @computed_field
    @property
    def p_dim(self) -> int:
        return 0
    
    def eval(self, x, params):
        return 0.0 
    
# a constant mean function
class Constant(BaseMean):
    @computed_field
    @property
    def p_dim(self) -> int:
        return 1
    
    def eval(self, x, params):
        return params[0] 
    
# a linear mean function 
class Linear(BaseMean):
    @computed_field
    @property
    def p_dim(self) -> int:
        return 1 + self.input_dim 
    
    def eval(self, x, params):
        return params[0] + jnp.inner(params[1:], x)
    
