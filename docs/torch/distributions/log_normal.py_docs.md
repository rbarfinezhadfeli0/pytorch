# Documentation: `torch/distributions/log_normal.py`

## File Metadata

- **Path**: `torch/distributions/log_normal.py`
- **Size**: 2,274 bytes (2.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional, Union

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform


__all__ = ["LogNormal"]


class LogNormal(TransformedDistribution):
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.positive
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Normal

    def __init__(
        self,
        loc: Union[Tensor, float],
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        base_dist = Normal(loc, scale, validate_args=validate_args)
        super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LogNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def loc(self) -> Tensor:
        return self.base_dist.loc

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        return (self.loc + self.scale.pow(2) / 2).exp()

    @property
    def mode(self) -> Tensor:
        return (self.loc - self.scale.square()).exp()

    @property
    def variance(self) -> Tensor:
        scale_sq = self.scale.pow(2)
        return scale_sq.expm1() * (2 * self.loc + scale_sq).exp()

    def entropy(self):
        return self.base_dist.entropy() + self.loc

```



## High-Level Overview

r"""    Creates a log-normal distribution parameterized by    :attr:`loc` and :attr:`scale` where::        X ~ Normal(loc, scale)        Y = exp(X) ~ LogNormal(loc, scale)    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1        tensor([ 0.1046])    Args:        loc (float or Tensor): mean of log of distribution        scale (float or Tensor): standard deviation of log of the distribution

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LogNormal`

**Functions defined**: `__init__`, `expand`, `loc`, `scale`, `mean`, `mode`, `variance`, `entropy`

**Key imports**: Optional, Union, Tensor, constraints, Normal, TransformedDistribution, ExpTransform


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional, Union
- `torch`: Tensor
- `torch.distributions`: constraints
- `torch.distributions.normal`: Normal
- `torch.distributions.transformed_distribution`: TransformedDistribution
- `torch.distributions.transforms`: ExpTransform


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

- **File Documentation**: `log_normal.py_docs.md`
- **Keyword Index**: `log_normal.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
