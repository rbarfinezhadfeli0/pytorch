# Documentation: `docs/torch/distributions/constraint_registry.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/constraint_registry.py_docs.md`
- **Size**: 15,832 bytes (15.46 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/constraint_registry.py`

## File Metadata

- **Path**: `torch/distributions/constraint_registry.py`
- **Size**: 10,306 bytes (10.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
r"""
PyTorch provides two global :class:`ConstraintRegistry` objects that link
:class:`~torch.distributions.constraints.Constraint` objects to
:class:`~torch.distributions.transforms.Transform` objects. These objects both
input constraints and return transforms, but they have different guarantees on
bijectivity.

1. ``biject_to(constraint)`` looks up a bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is guaranteed to have
   ``.bijective = True`` and should implement ``.log_abs_det_jacobian()``.
2. ``transform_to(constraint)`` looks up a not-necessarily bijective
   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``
   to the given ``constraint``. The returned transform is not guaranteed to
   implement ``.log_abs_det_jacobian()``.

The ``transform_to()`` registry is useful for performing unconstrained
optimization on constrained parameters of probability distributions, which are
indicated by each distribution's ``.arg_constraints`` dict. These transforms often
overparameterize a space in order to avoid rotation; they are thus more
suitable for coordinate-wise optimization algorithms like Adam::

    loc = torch.zeros(100, requires_grad=True)
    unconstrained = torch.zeros(100, requires_grad=True)
    scale = transform_to(Normal.arg_constraints["scale"])(unconstrained)
    loss = -Normal(loc, scale).log_prob(data).sum()

The ``biject_to()`` registry is useful for Hamiltonian Monte Carlo, where
samples from a probability distribution with constrained ``.support`` are
propagated in an unconstrained space, and algorithms are typically rotation
invariant.::

    dist = Exponential(rate)
    unconstrained = torch.zeros(100, requires_grad=True)
    sample = biject_to(dist.support)(unconstrained)
    potential_energy = -dist.log_prob(sample).sum()

.. note::

    An example where ``transform_to`` and ``biject_to`` differ is
    ``constraints.simplex``: ``transform_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.SoftmaxTransform` that simply
    exponentiates and normalizes its inputs; this is a cheap and mostly
    coordinate-wise operation appropriate for algorithms like SVI. In
    contrast, ``biject_to(constraints.simplex)`` returns a
    :class:`~torch.distributions.transforms.StickBreakingTransform` that
    bijects its input down to a one-fewer-dimensional space; this a more
    expensive less numerically stable transform but is needed for algorithms
    like HMC.

The ``biject_to`` and ``transform_to`` objects can be extended by user-defined
constraints and transforms using their ``.register()`` method either as a
function on singleton constraints::

    transform_to.register(my_constraint, my_transform)

or as a decorator on parameterized constraints::

    @transform_to.register(MyConstraintClass)
    def my_factory(constraint):
        assert isinstance(constraint, MyConstraintClass)
        return MyTransform(constraint.param1, constraint.param2)

You can create your own registry by creating a new :class:`ConstraintRegistry`
object.
"""

from torch.distributions import constraints, transforms
from torch.types import _Number


__all__ = [
    "ConstraintRegistry",
    "biject_to",
    "transform_to",
]


class ConstraintRegistry:
    """
    Registry to link constraints to transforms.
    """

    def __init__(self):
        self._registry = {}
        super().__init__()

    def register(self, constraint, factory=None):
        """
        Registers a :class:`~torch.distributions.constraints.Constraint`
        subclass in this registry. Usage::

            @my_registry.register(MyConstraintClass)
            def construct_transform(constraint):
                assert isinstance(constraint, MyConstraint)
                return MyTransform(constraint.arg_constraints)

        Args:
            constraint (subclass of :class:`~torch.distributions.constraints.Constraint`):
                A subclass of :class:`~torch.distributions.constraints.Constraint`, or
                a singleton object of the desired class.
            factory (Callable): A callable that inputs a constraint object and returns
                a  :class:`~torch.distributions.transforms.Transform` object.
        """
        # Support use as decorator.
        if factory is None:
            return lambda factory: self.register(constraint, factory)

        # Support calling on singleton instances.
        if isinstance(constraint, constraints.Constraint):
            constraint = type(constraint)

        if not isinstance(constraint, type) or not issubclass(
            constraint, constraints.Constraint
        ):
            raise TypeError(
                f"Expected constraint to be either a Constraint subclass or instance, but got {constraint}"
            )

        self._registry[constraint] = factory
        return factory

    def __call__(self, constraint):
        """
        Looks up a transform to constrained space, given a constraint object.
        Usage::

            constraint = Normal.arg_constraints["scale"]
            scale = transform_to(constraint)(torch.zeros(1))  # constrained
            u = transform_to(constraint).inv(scale)  # unconstrained

        Args:
            constraint (:class:`~torch.distributions.constraints.Constraint`):
                A constraint object.

        Returns:
            A :class:`~torch.distributions.transforms.Transform` object.

        Raises:
            `NotImplementedError` if no transform has been registered.
        """
        # Look up by Constraint subclass.
        try:
            factory = self._registry[type(constraint)]
        except KeyError:
            raise NotImplementedError(
                f"Cannot transform {type(constraint).__name__} constraints"
            ) from None
        return factory(constraint)


biject_to = ConstraintRegistry()
transform_to = ConstraintRegistry()


################################################################################
# Registration Table
################################################################################


@biject_to.register(constraints.real)
@transform_to.register(constraints.real)
def _transform_to_real(constraint):
    return transforms.identity_transform


@biject_to.register(constraints.independent)
def _biject_to_independent(constraint):
    base_transform = biject_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )


@transform_to.register(constraints.independent)
def _transform_to_independent(constraint):
    base_transform = transform_to(constraint.base_constraint)
    return transforms.IndependentTransform(
        base_transform, constraint.reinterpreted_batch_ndims
    )


@biject_to.register(constraints.positive)
@biject_to.register(constraints.nonnegative)
@transform_to.register(constraints.positive)
@transform_to.register(constraints.nonnegative)
def _transform_to_positive(constraint):
    return transforms.ExpTransform()


@biject_to.register(constraints.greater_than)
@biject_to.register(constraints.greater_than_eq)
@transform_to.register(constraints.greater_than)
@transform_to.register(constraints.greater_than_eq)
def _transform_to_greater_than(constraint):
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.lower_bound, 1),
        ]
    )


@biject_to.register(constraints.less_than)
@transform_to.register(constraints.less_than)
def _transform_to_less_than(constraint):
    return transforms.ComposeTransform(
        [
            transforms.ExpTransform(),
            transforms.AffineTransform(constraint.upper_bound, -1),
        ]
    )


@biject_to.register(constraints.interval)
@biject_to.register(constraints.half_open_interval)
@transform_to.register(constraints.interval)
@transform_to.register(constraints.half_open_interval)
def _transform_to_interval(constraint):
    # Handle the special case of the unit interval.
    lower_is_0 = (
        isinstance(constraint.lower_bound, _Number) and constraint.lower_bound == 0
    )
    upper_is_1 = (
        isinstance(constraint.upper_bound, _Number) and constraint.upper_bound == 1
    )
    if lower_is_0 and upper_is_1:
        return transforms.SigmoidTransform()

    loc = constraint.lower_bound
    scale = constraint.upper_bound - constraint.lower_bound
    return transforms.ComposeTransform(
        [transforms.SigmoidTransform(), transforms.AffineTransform(loc, scale)]
    )


@biject_to.register(constraints.simplex)
def _biject_to_simplex(constraint):
    return transforms.StickBreakingTransform()


@transform_to.register(constraints.simplex)
def _transform_to_simplex(constraint):
    return transforms.SoftmaxTransform()


# TODO define a bijection for LowerCholeskyTransform
@transform_to.register(constraints.lower_cholesky)
def _transform_to_lower_cholesky(constraint):
    return transforms.LowerCholeskyTransform()


@transform_to.register(constraints.positive_definite)
@transform_to.register(constraints.positive_semidefinite)
def _transform_to_positive_definite(constraint):
    return transforms.PositiveDefiniteTransform()


@biject_to.register(constraints.corr_cholesky)
@transform_to.register(constraints.corr_cholesky)
def _transform_to_corr_cholesky(constraint):
    return transforms.CorrCholeskyTransform()


@biject_to.register(constraints.cat)
def _biject_to_cat(constraint):
    return transforms.CatTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


@transform_to.register(constraints.cat)
def _transform_to_cat(constraint):
    return transforms.CatTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim, constraint.lengths
    )


@biject_to.register(constraints.stack)
def _biject_to_stack(constraint):
    return transforms.StackTransform(
        [biject_to(c) for c in constraint.cseq], constraint.dim
    )


@transform_to.register(constraints.stack)
def _transform_to_stack(constraint):
    return transforms.StackTransform(
        [transform_to(c) for c in constraint.cseq], constraint.dim
    )

```



## High-Level Overview

r"""PyTorch provides two global :class:`ConstraintRegistry` objects that link:class:`~torch.distributions.constraints.Constraint` objects to:class:`~torch.distributions.transforms.Transform` objects. These objects bothinput constraints and return transforms, but they have different guarantees onbijectivity.1. ``biject_to(constraint)`` looks up a bijective   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``   to the given ``constraint``. The returned transform is guaranteed to have   ``.bijective = True`` and should implement ``.log_abs_det_jacobian()``.2. ``transform_to(constraint)`` looks up a not-necessarily bijective   :class:`~torch.distributions.transforms.Transform` from ``constraints.real``   to the given ``constraint``. The returned transform is not guaranteed to   implement ``.log_abs_det_jacobian()``.The ``transform_to()`` registry is useful for performing unconstrainedoptimization on constrained parameters of probability distributions, which areindicated by each distribution's ``.arg_constraints`` dict. These transforms oftenoverparameterize a space in order to avoid rotation; they are thus moresuitable for coordinate-wise optimization algorithms like Adam::    loc = torch.zeros(100, requires_grad=True)    unconstrained = torch.zeros(100, requires_grad=True)    scale = transform_to(Normal.arg_constraints["scale"])(unconstrained)    loss = -Normal(loc, scale).log_prob(data).sum()The ``biject_to()`` registry is useful for Hamiltonian Monte Carlo, wheresamples from a probability distribution with constrained ``.support`` arepropagated in an unconstrained space, and algorithms are typically rotationinvariant.::    dist = Exponential(rate)    unconstrained = torch.zeros(100, requires_grad=True)    sample = biject_to(dist.support)(unconstrained)    potential_energy = -dist.log_prob(sample).sum().. note::    An example where ``transform_to`` and ``biject_to`` differ is    ``constraints.simplex``: ``transform_to(constraints.simplex)`` returns a    :class:`~torch.distributions.transforms.SoftmaxTransform` that simply    exponentiates and normalizes its inputs; this is a cheap and mostly    coordinate-wise operation appropriate for algorithms like SVI. In    contrast, ``biject_to(constraints.simplex)`` returns a    :class:`~torch.distributions.transforms.StickBreakingTransform` that    bijects its input down to a one-fewer-dimensional space; this a more    expensive less numerically stable transform but is needed for algorithms    like HMC.

This Python file contains 5 class(es) and 21 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConstraintRegistry`

**Functions defined**: `my_factory`, `__init__`, `register`, `construct_transform`, `__call__`, `_transform_to_real`, `_biject_to_independent`, `_transform_to_independent`, `_transform_to_positive`, `_transform_to_greater_than`, `_transform_to_less_than`, `_transform_to_interval`, `_biject_to_simplex`, `_transform_to_simplex`, `_transform_to_lower_cholesky`, `_transform_to_positive_definite`, `_transform_to_corr_cholesky`, `_biject_to_cat`, `_transform_to_cat`, `_biject_to_stack`

**Key imports**: constraints, transforms, _Number


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.distributions`: constraints, transforms
- `torch.types`: _Number


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributions`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mixture_same_family.py_docs.md`](./mixture_same_family.py_docs.md)
- [`normal.py_docs.md`](./normal.py_docs.md)
- [`relaxed_categorical.py_docs.md`](./relaxed_categorical.py_docs.md)
- [`laplace.py_docs.md`](./laplace.py_docs.md)
- [`bernoulli.py_docs.md`](./bernoulli.py_docs.md)
- [`distribution.py_docs.md`](./distribution.py_docs.md)
- [`negative_binomial.py_docs.md`](./negative_binomial.py_docs.md)
- [`continuous_bernoulli.py_docs.md`](./continuous_bernoulli.py_docs.md)
- [`half_normal.py_docs.md`](./half_normal.py_docs.md)


## Cross-References

- **File Documentation**: `constraint_registry.py_docs.md`
- **Keyword Index**: `constraint_registry.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributions`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributions`):

- [`wishart.py_docs.md_docs.md`](./wishart.py_docs.md_docs.md)
- [`pareto.py_docs.md_docs.md`](./pareto.py_docs.md_docs.md)
- [`binomial.py_docs.md_docs.md`](./binomial.py_docs.md_docs.md)
- [`half_cauchy.py_docs.md_docs.md`](./half_cauchy.py_docs.md_docs.md)
- [`one_hot_categorical.py_docs.md_docs.md`](./one_hot_categorical.py_docs.md_docs.md)
- [`geometric.py_kw.md_docs.md`](./geometric.py_kw.md_docs.md)
- [`kumaraswamy.py_kw.md_docs.md`](./kumaraswamy.py_kw.md_docs.md)
- [`transformed_distribution.py_kw.md_docs.md`](./transformed_distribution.py_kw.md_docs.md)
- [`log_normal.py_docs.md_docs.md`](./log_normal.py_docs.md_docs.md)
- [`kumaraswamy.py_docs.md_docs.md`](./kumaraswamy.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `constraint_registry.py_docs.md_docs.md`
- **Keyword Index**: `constraint_registry.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
