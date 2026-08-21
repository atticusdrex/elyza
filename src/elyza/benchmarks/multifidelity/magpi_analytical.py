from elyza.util.imports import * 
from elyza.core.data import ScalarInput 
from elyza.core.evaluator import Evaluator

x = ScalarInput(
        name = "x", 
        dim = 1, 
        sampling_func = lambda key: jrand.uniform(key, minval=0, maxval=5), 
        minval = 0.0, 
        maxval = 5.0 
    )

hf_evaluator = Evaluator(
    name = "High-Fidelity", 
    inputs = [x], 
    output_dim = 1, 
    evaluation_func = jit(
        lambda x: jnp.exp(-x) * jnp.sin(2*pi*x)
    )
)
mf_evaluator = Evaluator(
    name = "Medium-Fidelity", 
    inputs = [x], 
    output_dim = 1, 
    evaluation_func = jit(
        lambda x: jnp.sin(2*pi*x)
    )
)
lf_evaluator = Evaluator(
    name = "Low-Fidelity", 
    inputs = [x], 
    output_dim = 1, 
    evaluation_func = jit(
        lambda x: jnp.exp(-x)
    )
)