"""Analytical three-fidelity benchmark used to exercise MAGPI/multifidelity code.

Defines a single scalar input ``x`` on ``[0, 5]`` and three
:class:`~elyza.core.evaluator.Evaluator` instances of increasing fidelity
that share that input: a low-fidelity decay ``exp(-x)``, a medium-fidelity
oscillation ``sin(2*pi*x)``, and a high-fidelity model combining both,
``exp(-x) * sin(2*pi*x)``.
"""
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
    ),
    jit_compile = True
)
mf_evaluator = Evaluator(
    name = "Medium-Fidelity",
    inputs = [x],
    output_dim = 1,
    evaluation_func = jit(
        lambda x: jnp.sin(2*pi*x)
    ),
    jit_compile = True
)
lf_evaluator = Evaluator(
    name = "Low-Fidelity",
    inputs = [x],
    output_dim = 1,
    evaluation_func = jit(
        lambda x: jnp.exp(-x)
    ),
    jit_compile = True
)
