# Documentation: `torch/distributions/logistic_normal.py`

## File Metadata

- **Path**: `torch/distributions/logistic_normal.py`
- **Size**: 2,290 bytes (2.24 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional, Union

from torch import Tensor
from torch.distributions import constraints, Independent
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import StickBreakingTransform


__all__ = ["LogisticNormal"]


class LogisticNormal(TransformedDistribution):
    r"""
    Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale`
    that define the base `Normal` distribution transformed with the
    `StickBreakingTransform` such that::

        X ~ LogisticNormal(loc, scale)
        Y = log(X / (1 - X.cumsum(-1)))[..., :-1] ~ Normal(loc, scale)

    Args:
        loc (float or Tensor): mean of the base distribution
        scale (float or Tensor): standard deviation of the base distribution

    Example::

        >>> # logistic-normal distributed with mean=(0, 0, 0) and stddev=(1, 1, 1)
        >>> # of the base Normal distribution
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogisticNormal(torch.tensor([0.0] * 3), torch.tensor([1.0] * 3))
        >>> m.sample()
        tensor([ 0.7653,  0.0341,  0.0579,  0.1427])

    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.simplex
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Independent[Normal]

    def __init__(
        self,
        loc: Union[Tensor, float],
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        base_dist = Normal(loc, scale, validate_args=validate_args)
        if not base_dist.batch_shape:
            base_dist = base_dist.expand([1])
        super().__init__(
            base_dist, StickBreakingTransform(), validate_args=validate_args
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogisticNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self) -> Tensor:
        return self.base_dist.base_dist.loc

    @property
    def scale(self) -> Tensor:
        return self.base_dist.base_dist.scale

```



## High-Level Overview

r"""    Creates a logistic-normal distribution parameterized by :attr:`loc` and :attr:`scale`    that define the base `Normal` distribution transformed with the    `StickBreakingTransform` such that::        X ~ LogisticNormal(loc, scale)        Y = log(X / (1 - X.cumsum(-1)))[..., :-1] ~ Normal(loc, scale)    Args:        loc (float or Tensor): mean of the base distribution        scale (float or Tensor): standard deviation of the base distribution    Example::        >>> # logistic-normal distributed with mean=(0, 0, 0) and stddev=(1, 1, 1)        >>> # of the base Normal distribution        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = LogisticNormal(torch.tensor([0.0] * 3), torch.tensor([1.0] * 3))        >>> m.sample()        tensor([ 0.7653,  0.0341,  0.0579,  0.1427])

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LogisticNormal`

**Functions defined**: `__init__`, `expand`, `loc`, `scale`

**Key imports**: Optional, Union, Tensor, constraints, Independent, Normal, TransformedDistribution, StickBreakingTransform


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional, Union
- `torch`: Tensor
- `torch.distributions`: constraints, Independent
- `torch.distributions.normal`: Normal
- `torch.distributions.transformed_distribution`: TransformedDistribution
- `torch.distributions.transforms`: StickBreakingTransform


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

- **File Documentation**: `logistic_normal.py_docs.md`
- **Keyword Index**: `logistic_normal.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
