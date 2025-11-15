# Documentation: `torch/distributions/uniform.py`

## File Metadata

- **Path**: `torch/distributions/uniform.py`
- **Size**: 3,398 bytes (3.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Uniform"]


class Uniform(Distribution):
    r"""
    Generates uniformly distributed random samples from the half-open interval
    ``[low, high)``.

    Example::

        >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))
        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)
        >>> # xdoctest: +SKIP
        tensor([ 2.3418])

    Args:
        low (float or Tensor): lower range (inclusive).
        high (float or Tensor): upper range (exclusive).
    """

    has_rsample = True

    @property
    def arg_constraints(self):
        # TODO allow (loc,scale) parameterization to allow independent constraints.
        return {
            "low": constraints.less_than(self.high),
            "high": constraints.greater_than(self.low),
        }

    @property
    def mean(self) -> Tensor:
        return (self.high + self.low) / 2

    @property
    def mode(self) -> Tensor:
        return nan * self.high

    @property
    def stddev(self) -> Tensor:
        return (self.high - self.low) / 12**0.5

    @property
    def variance(self) -> Tensor:
        return (self.high - self.low).pow(2) / 12

    def __init__(
        self,
        low: Union[Tensor, float],
        high: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        self.low, self.high = broadcast_all(low, high)

        if isinstance(low, _Number) and isinstance(high, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.low.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Uniform, _instance)
        batch_shape = torch.Size(batch_shape)
        new.low = self.low.expand(batch_shape)
        new.high = self.high.expand(batch_shape)
        super(Uniform, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        return constraints.interval(self.low, self.high)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.low.dtype, device=self.low.device)
        return self.low + rand * (self.high - self.low)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        lb = self.low.le(value).type_as(self.low)
        ub = self.high.gt(value).type_as(self.low)
        return torch.log(lb.mul(ub)) - torch.log(self.high - self.low)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        result = (value - self.low) / (self.high - self.low)
        return result.clamp(min=0, max=1)

    def icdf(self, value):
        result = value * (self.high - self.low) + self.low
        return result

    def entropy(self):
        return torch.log(self.high - self.low)

```



## High-Level Overview

r"""    Generates uniformly distributed random samples from the half-open interval    ``[low, high)``.    Example::        >>> m = Uniform(torch.tensor([0.0]), torch.tensor([5.0]))        >>> m.sample()  # uniformly distributed in the range [0.0, 5.0)        >>> # xdoctest: +SKIP        tensor([ 2.3418])    Args:        low (float or Tensor): lower range (inclusive).        high (float or Tensor): upper range (exclusive).

This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Uniform`

**Functions defined**: `arg_constraints`, `mean`, `mode`, `stddev`, `variance`, `__init__`, `expand`, `support`, `rsample`, `log_prob`, `cdf`, `icdf`, `entropy`

**Key imports**: Optional, Union, torch, nan, Tensor, constraints, Distribution, broadcast_all, _Number, _size


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional, Union
- `torch`
- `torch.distributions`: constraints
- `torch.distributions.distribution`: Distribution
- `torch.distributions.utils`: broadcast_all
- `torch.types`: _Number, _size


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

- **File Documentation**: `uniform.py_docs.md`
- **Keyword Index**: `uniform.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
