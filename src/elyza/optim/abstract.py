"""Abstract base classes and shared helpers for gradient-based optimizers.

Defines :class:`OptimizerOptions`/:class:`Optimizer` (the interfaces
concrete optimizers such as :class:`~elyza.optim.adam.ADAM` and
:class:`~elyza.optim.lbfgs.LBFGS` implement) and
:class:`BatchGradientOptimizer`, plus :func:`fill_pytree_spec`, used to
expand a partially-specified pytree (e.g. ``active_params``/``constraints``)
into one matching the full parameter pytree.
"""
from elyza.util.imports import *
from jax.tree_util import tree_map

def fill_pytree_spec(template, partial, default):
    """Build a pytree with the same nested-dict structure as ``template``.

    Lets callers specify ``active_params``/``constraints`` etc. for only the
    leaves/subtrees they care about instead of the entire pytree (e.g. a
    parameter pytree like ``p_init``).

    - Any leaf/subtree left unspecified in ``partial`` is filled in with
      ``default``.
    - Giving a value for an intermediate dict key in ``partial`` broadcasts
      that value to every leaf beneath it (e.g. ``{'weights': False}`` turns
      off every weight layer).
    - An unknown key in ``partial`` (typo, wrong nesting) raises rather than
      being silently dropped.

    Args:
        template: The full pytree whose nested-dict structure is mirrored,
            e.g. a parameter pytree like ``p_init``.
        partial: A possibly-partial pytree of the same (or coarser) nested
            structure as ``template``, or ``None``.
        default: Value used to fill in any leaf/subtree not specified in
            ``partial``.

    Returns:
        A pytree matching ``template``'s structure, with every leaf set
        either from ``partial`` (broadcast down from the nearest specified
        ancestor) or from ``default``.

    Raises:
        ValueError: If ``partial`` contains a dict key not present in the
            corresponding level of ``template``.
    """
    def _fill(node, spec, path):
        if isinstance(node, dict):
            if spec is not None and not isinstance(spec, dict):
                return tree_map(lambda _leaf: spec, node)

            spec = spec or {}
            unknown = set(spec) - set(node)
            if unknown:
                location = ".".join(map(str, path)) or "<root>"
                raise ValueError(f"unknown key(s) {unknown} at '{location}'; expected one of {set(node)}")

            return {key: _fill(subnode, spec.get(key), path + [key]) for key, subnode in node.items()}

        return default if spec is None else spec

    return _fill(template, partial, [])

class OptimizerOptions(BaseModel):
    """Base class for the options that parameterize a concrete optimizer.

    Concrete optimizers subclass this (e.g. :class:`~elyza.optim.adam.ADAMOptions`,
    :class:`~elyza.optim.lbfgs.LBFGSOptions`) to declare their own hyperparameters.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Optimizer(BaseModel):
    """Abstract base class for general optimizers."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def run(*args):
        """Run the optimizer and return the optimized parameters.

        Each batch gradient optimizer must implement a ``run`` method in
        this style.

        Raises:
            NotImplementedError: Always, in the base class; subclasses must
                override this method.
        """
        raise NotImplementedError("this method is only a placeholder and hasn't been implemented")

class BatchGradientOptimizer(Optimizer):
    """Abstract base class for gradient-based optimization in mini-batches.

    Attributes:
        loss_grad_fn: A function in the form ``def func(p, *args) -> float``
            (paired with its gradient) used to evaluate the training
            objective on a batch.
        opts: Optimizer options controlling the run.
    """
    loss_grad_fn : SkipValidation[callable] | None = Field(default = None, description = "a function in the form def func(p, *args) -> float")
    opts : OptimizerOptions = Field(default = OptimizerOptions(), description = "optimizer options")

    def _get_batches(self, key, batch_size: int, *data) -> list[tuple[jax.Array]]:
        """Shuffle and split data arrays into equally-sized mini-batches.

        Args:
            key: A JAX PRNG key used to permute the data.
            batch_size: Number of points per batch; any remainder after
                dividing evenly is dropped.
            *data: One or more arrays sharing the same leading (sample)
                dimension, e.g. ``(X, Y)``.

        Returns:
            list[tuple[jax.Array]]: A list of batches, each a tuple with one
            array per element of ``data``, aligned by shuffled index.
        """
        n = data[0].shape[0]
        perm = jax.random.permutation(key, n)

        # shuffling the data by the same indices
        data_shuffled = [datum[perm] for datum in data]

        n_batches = n // batch_size  # drop last incomplete batch
        batches = []
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batches.append(tuple(
                [datum[start:end] for datum in data_shuffled]
            ))

        return batches
