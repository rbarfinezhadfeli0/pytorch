# Documentation: `torch/distributions/exponential.py`

## File Metadata

- **Path**: `torch/distributions/exponential.py`
- **Size**: 2,769 bytes (2.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Exponential"]


class Exponential(ExponentialFamily):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"rate": constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.rate.reciprocal()

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.rate)

    @property
    def stddev(self) -> Tensor:
        return self.rate.reciprocal()

    @property
    def variance(self) -> Tensor:
        return self.rate.pow(-2)

    def __init__(
        self,
        rate: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        (self.rate,) = broadcast_all(rate)
        batch_shape = torch.Size() if isinstance(rate, _Number) else self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Exponential, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Exponential, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        return self.rate.new(shape).exponential_() / self.rate

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.rate.log() - self.rate * value

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 1 - torch.exp(-self.rate * value)

    def icdf(self, value):
        return -torch.log1p(-value) / self.rate

    def entropy(self):
        return 1.0 - torch.log(self.rate)

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (-self.rate,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        return -torch.log(-x)

```



## High-Level Overview

r"""    Creates a Exponential distribution parameterized by :attr:`rate`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Exponential(torch.tensor([1.0]))        >>> m.sample()  # Exponential distributed with rate=1        tensor([ 0.1046])    Args:        rate (float or Tensor): rate = 1 / scale of the distribution

This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Exponential`

**Functions defined**: `mean`, `mode`, `stddev`, `variance`, `__init__`, `expand`, `rsample`, `log_prob`, `cdf`, `icdf`, `entropy`, `_natural_params`, `_log_normalizer`

**Key imports**: Optional, Union, torch, Tensor, constraints, ExponentialFamily, broadcast_all, _Number, _size


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
- `torch.distributions.exp_family`: ExponentialFamily
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

- **File Documentation**: `exponential.py_docs.md`
- **Keyword Index**: `exponential.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
