from .util import * 

class BaseKernel(BaseModel):
    input_dim: int
    epsilon: float = 1e-12

class RBF(BaseKernel):
    @computed_field 
    @property 
    def p_dim(self) -> int: 
        return 2 
    '''
    To obtain the exact bandwidth and lengthscale parameters for kernels expressed by: 
    l * exp(-||x-x'||/(2σ^2)), 
    
    you must convert the params array to: 

    l = softplus(params[0])
    σ = sqrt(0.5*softplus(params[1]))
    '''
    def eval(self, x1, x2, params):
        params = softplus(params.ravel()) # softplusing params for positivity constraints 
        return params[0]*jnp.exp(-jnp.sum(((x1 - x2).ravel())**2 / params[1])) 

'''
Automatic relevancy determination kernel
'''
class ARD(BaseKernel):
    @computed_field
    @property
    def p_dim(self) -> int:
        return 1 + self.input_dim

    def eval(self, x1, x2, params):
        params = softplus(params.ravel()) # softplusing params for positivity constraints 
        return params[0]*jnp.exp(-jnp.sum(((x1 - x2).ravel())**2 / params[1:])) 

'''
Laplace kernel (uses 1-norm instead of 2-norm)
'''
class Laplace(BaseKernel):
    @computed_field
    @property
    def p_dim(self) -> int:
        return 1 + self.input_dim

    def eval(self, x1, x2, params):
        params = softplus(params.ravel()) # softplusing params for positivity constraints 
        return params[0]*jnp.exp(-jnp.sum(jnp.abs((x1 - x2).ravel()) / params[1:])) 


