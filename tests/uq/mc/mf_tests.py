from util.imports import * 
from multifidelity.mc import MultifidelityMC, MultilevelMonteCarlo
from core.data import ContinuousInput 
from core.evaluator import Evaluator

def test_1d():
    x = ContinuousInput(
        name = "x", 
        dim = 1, 
        sampling_func = lambda key, shape: jrand.uniform(key, shape = shape, minval=0, maxval=5), 
        minval = 0.0, 
        maxval = 5.0 
    )

    hf = Evaluator(name = "High-Fidelity", inputs = [x], output_dim = 1, evaluation_func = jit(lambda x: jnp.exp(-x) * jnp.sin(2*pi*x)))
    mf = Evaluator(name = "Medium-Fidelity", inputs = [x], output_dim = 1, evaluation_func = lambda x: jnp.sin(2*pi*x))
    lf = Evaluator(name = "Low-Fidelity", inputs = [x], output_dim = 1, evaluation_func = lambda x: jnp.exp(-x))

    mfmc = MultifidelityMC(
        evaluators = [hf, mf, lf]
    )

    mfmc.pilot_covariances(n_pilots = 100, seed = 42)

    key = jrand.PRNGKey(42) 
    return mfmc.single_level_eval(key, 0, int(1e7))

def mlmc_test1():
    x = ContinuousInput(
        name = "x", 
        dim = 1, 
        sampling_func = lambda key, shape: jrand.uniform(key, shape = shape, minval=0, maxval=5), 
        minval = 0.0, 
        maxval = 5.0 
    )
    
    hf = Evaluator(name = "High-Fidelity", inputs = [x], output_dim = 1, evaluation_func = jit(lambda x: jnp.exp(-x) * jnp.sin(2*pi*x)))
    mf = Evaluator(name = "Medium-Fidelity", inputs = [x], output_dim = 1, evaluation_func = lambda x: jnp.sin(2*pi*x))
    lf = Evaluator(name = "Low-Fidelity", inputs = [x], output_dim = 1, evaluation_func = lambda x: jnp.exp(-x))

    mlmc = MultilevelMonteCarlo(
        evaluators = [hf, mf, lf]
    )

    mlmc.pilot_covariances(
        n_pilots = 10000, seed = 42, set_costs = True
    )

    mlmc.print()

    return mlmc.evaluate([5, 10, 15], seed = 42, check_inputs = True)

    return 

if __name__ == "__main__":
    # print(test_1d().mean())
    print(mlmc_test1())