# Documentation: `torch/distributions/normal.py`

## File Metadata

- **Path**: `torch/distributions/normal.py`
- **Size**: 3,990 bytes (3.90 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import math
from typing import Optional, Union

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import _standard_normal, broadcast_all
from torch.types import _Number, _size


__all__ = ["Normal"]


class Normal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def stddev(self) -> Tensor:
        return self.scale

    @property
    def variance(self) -> Tensor:
        return self.stddev.pow(2)

    def __init__(
        self,
        loc: Union[Tensor, float],
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Normal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Normal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + eps * self.scale

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        # pyrefly: ignore [unsupported-operation]
        var = self.scale**2
        log_scale = (
            math.log(self.scale)
            if isinstance(self.scale, _Number)
            else self.scale.log()
        )
        return (
            -((value - self.loc) ** 2) / (2 * var)
            - log_scale
            - math.log(math.sqrt(2 * math.pi))
        )

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (
            1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2))
        )

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)

```



## High-Level Overview

r"""    Creates a normal (also called Gaussian) distribution parameterized by    :attr:`loc` and :attr:`scale`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))        >>> m.sample()  # normally distributed with loc=0 and scale=1        tensor([ 0.1046])    Args:        loc (float or Tensor): mean of the distribution (often referred to as mu)        scale (float or Tensor): standard deviation of the distribution            (often referred to as sigma)

This Python file contains 1 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Normal`

**Functions defined**: `mean`, `mode`, `stddev`, `variance`, `__init__`, `expand`, `sample`, `rsample`, `log_prob`, `cdf`, `icdf`, `entropy`, `_natural_params`, `_log_normalizer`

**Key imports**: math, Optional, Union, torch, Tensor, constraints, ExponentialFamily, _standard_normal, broadcast_all, _Number, _size


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `typing`: Optional, Union
- `torch`
- `torch.distributions`: constraints
- `torch.distributions.exp_family`: ExponentialFamily
- `torch.distributions.utils`: _standard_normal, broadcast_all
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
- [`relaxed_categorical.py_docs.md`](./relaxed_categorical.py_docs.md)
- [`laplace.py_docs.md`](./laplace.py_docs.md)
- [`bernoulli.py_docs.md`](./bernoulli.py_docs.md)
- [`distribution.py_docs.md`](./distribution.py_docs.md)
- [`negative_binomial.py_docs.md`](./negative_binomial.py_docs.md)
- [`continuous_bernoulli.py_docs.md`](./continuous_bernoulli.py_docs.md)
- [`half_normal.py_docs.md`](./half_normal.py_docs.md)


## Cross-References

- **File Documentation**: `normal.py_docs.md`
- **Keyword Index**: `normal.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
