"""The evaluator class: a way to evaluate external computer models and store data.

Defines :class:`Evaluator`, which wraps an arbitrary evaluation function
together with its :class:`~elyza.core.data.Variable` specification, output
dimension, and (optionally) a calibrated per-evaluation cost -- the unit
that :mod:`elyza.multifidelity` estimators operate on.
"""
from elyza.util.imports import *
from elyza.core.data import Variable
import time

class Evaluator(BaseModel):
    """Wraps a computer model as a batched, vmapped evaluation function.

    Attributes:
        name: The unique name for this evaluator.
        inputs: The list of inputs associated with this evaluator.
        output_dim: The output dimension of the evaluator.
        evaluation_func: A function which takes in arguments for each of the
            specified inputs and returns an array of the appropriate output
            dimension.
        cost: The cost to evaluate, if known/calibrated.
        jit_compile: Whether or not to jit-compile the evaluation function.
    """
    name : int | str = Field(
        description = "the unique name for this evaluator"
    )
    inputs : list[Variable] = Field(
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

    jit_compile : bool = Field(default = False, description = "whether or not to jit-compile the evaluation func")

    @computed_field # function for getting the set of input names
    def _input_names(self) -> set:
        """set: The set of names of this evaluator's inputs."""
        return set([input.name for input in self.inputs])

    def model_post_init(self, __context):
        """Validate that all input names are unique.

        Raises:
            AssertionError: If two or more inputs share the same name.
        """
        assert len(self.inputs) == len(self._input_names), "duplicate input names detected"

    def single_eval(self, *input_vals : list[Variable]):
        """Evaluate the model on a single set of (unbatched) input values.

        Args:
            *input_vals: One value per input, in the same order as ``inputs``.

        Returns:
            The raw output of ``evaluation_func`` for this single evaluation.
        """
        # compute the evaluation function on this specific set of inputs
        return self.evaluation_func(*input_vals)

    def evaluate(self, *input_vals):
        """Evaluate the model over a batch of inputs via ``vmap``.

        Args:
            *input_vals: One array per input, assumed valid for the
                evaluator, where each array has shape
                ``(n_points, input_dim)``.

        Returns:
            jax.Array: Outputs of shape ``(n_points, output_dim)``.
        """
        if self.jit_compile:
            eval_vmap = jit(vmap(self.evaluation_func, in_axes = [0]*len(input_vals)))
        else:
            eval_vmap = vmap(self.evaluation_func, in_axes = [0]*len(input_vals))

        return eval_vmap(*input_vals).block_until_ready().reshape(-1,self.output_dim)

    def print(self):
        """Print the evaluator's name, output dimension, cost, and inputs."""
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
        """Set the per-evaluation cost.

        Args:
            cost: A positive evaluation cost.

        Raises:
            AssertionError: If ``cost`` is not positive.
        """
        assert cost > 0.0, "cost must be positive"
        self.cost = cost

    def evaluate_timed(self, *input_vals : list[jax.Array], set_cost = False):
        """Evaluate a batch of inputs while timing the wall-clock cost.

        Args:
            *input_vals: One array per input, each of shape
                ``(n_points, input_dim)``.
            set_cost: If ``True``, store the measured per-evaluation time as
                this evaluator's ``cost``.

        Returns:
            jax.Array: Outputs of shape ``(n_points, output_dim)``.
        """
        n_points = input_vals[0].shape[0]
        start_time = time.time()
        result = self.evaluate(*input_vals)
        result.block_until_ready() # JAX dispatches asynchronously, so wait for the actual computation to finish before stopping the clock
        end_time = time.time()
        print(self.name, ":")
        print("Total time: %.4e (s)" % (end_time - start_time))
        print("Per-evaluation time: %.4e (s)\n" % ((end_time - start_time) / n_points))
        if set_cost:
            self.cost = (end_time - start_time) / n_points
        return result
