# Documentation: `torch/distributions/poisson.py`

## File Metadata

- **Path**: `torch/distributions/poisson.py`
- **Size**: 2,521 bytes (2.46 KB)
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
from torch.types import _Number, Number


__all__ = ["Poisson"]


class Poisson(ExponentialFamily):
    r"""
    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.

    Samples are nonnegative integers, with a pmf given by

    .. math::
      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}

    Example::

        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")
        >>> m = Poisson(torch.tensor([4]))
        >>> m.sample()
        tensor([ 3.])

    Args:
        rate (Number, Tensor): the rate parameter
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"rate": constraints.nonnegative}
    support = constraints.nonnegative_integer

    @property
    def mean(self) -> Tensor:
        return self.rate

    @property
    def mode(self) -> Tensor:
        return self.rate.floor()

    @property
    def variance(self) -> Tensor:
        return self.rate

    def __init__(
        self,
        rate: Union[Tensor, Number],
        validate_args: Optional[bool] = None,
    ) -> None:
        (self.rate,) = broadcast_all(rate)
        if isinstance(rate, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.rate.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Poisson, _instance)
        batch_shape = torch.Size(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(Poisson, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.poisson(self.rate.expand(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        rate, value = broadcast_all(self.rate, value)
        return value.xlogy(rate) - rate - (value + 1).lgamma()

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (torch.log(self.rate),)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        return torch.exp(x)

```



## High-Level Overview

r"""    Creates a Poisson distribution parameterized by :attr:`rate`, the rate parameter.    Samples are nonnegative integers, with a pmf given by    .. math::      \mathrm{rate}^k \frac{e^{-\mathrm{rate}}}{k!}    Example::        >>> # xdoctest: +SKIP("poisson_cpu not implemented for 'Long'")        >>> m = Poisson(torch.tensor([4]))        >>> m.sample()        tensor([ 3.])    Args:        rate (Number, Tensor): the rate parameter

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Poisson`

**Functions defined**: `mean`, `mode`, `variance`, `__init__`, `expand`, `sample`, `log_prob`, `_natural_params`, `_log_normalizer`

**Key imports**: Optional, Union, torch, Tensor, constraints, ExponentialFamily, broadcast_all, _Number, Number


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
- `torch.types`: _Number, Number


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

- **File Documentation**: `poisson.py_docs.md`
- **Keyword Index**: `poisson.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
