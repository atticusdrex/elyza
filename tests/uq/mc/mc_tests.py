from core.evaluator import Evaluator 
from core.data import ContinuousInput
from util.imports import * 

def quad_1d():
    x = ContinuousInput(name = "x", dim  = 1, minval = 0, maxval = 1)
    y = Evaluator(name = "y", inputs = [x], output_dim = 1, evaluation_func = lambda x: x**2)
    y.print()
    y_mean = y.evaluate(jrand.uniform(jrand.PRNGKey(42), shape = (int(1e6),)), check_inputs = True).mean(axis=0) 
    return jnp.abs(y_mean - 1.0 / 3) <= 1e-2


if __name__ == "__main__":
    print("Quadratic 1d: ", quad_1d())