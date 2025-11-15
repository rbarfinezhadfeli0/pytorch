# Documentation: `torch/distributions/laplace.py`

## File Metadata

- **Path**: `torch/distributions/laplace.py`
- **Size**: 3,540 bytes (3.46 KB)
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
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Laplace"]


class Laplace(Distribution):
    r"""
    Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # Laplace distributed with loc=0, scale=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution
        scale (float or Tensor): scale of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self) -> Tensor:
        return self.loc

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def variance(self) -> Tensor:
        return 2 * self.scale.pow(2)

    @property
    def stddev(self) -> Tensor:
        return (2**0.5) * self.scale

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
        new = self._get_checked_instance(Laplace, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(Laplace, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        finfo = torch.finfo(self.loc.dtype)
        if torch._C._get_tracing_state():
            # [JIT WORKAROUND] lack of support for .uniform_()
            u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device) * 2 - 1
            return self.loc - self.scale * u.sign() * torch.log1p(
                -u.abs().clamp(min=finfo.tiny)
            )
        u = self.loc.new(shape).uniform_(finfo.eps - 1, 1)
        # TODO: If we ever implement tensor.nextafter, below is what we want ideally.
        # u = self.loc.new(shape).uniform_(self.loc.nextafter(-.5, 0), .5)
        return self.loc - self.scale * u.sign() * torch.log1p(-u.abs())

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -torch.log(2 * self.scale) - torch.abs(value - self.loc) / self.scale

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 - 0.5 * (value - self.loc).sign() * torch.expm1(
            -(value - self.loc).abs() / self.scale
        )

    def icdf(self, value):
        term = value - 0.5
        return self.loc - self.scale * (term).sign() * torch.log1p(-2 * term.abs())

    def entropy(self):
        return 1 + torch.log(2 * self.scale)

```



## High-Level Overview

r"""    Creates a Laplace distribution parameterized by :attr:`loc` and :attr:`scale`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))        >>> m.sample()  # Laplace distributed with loc=0, scale=1        tensor([ 0.1046])    Args:        loc (float or Tensor): mean of the distribution        scale (float or Tensor): scale of the distribution

This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Laplace`

**Functions defined**: `mean`, `mode`, `variance`, `stddev`, `__init__`, `expand`, `rsample`, `log_prob`, `cdf`, `icdf`, `entropy`

**Key imports**: Optional, Union, torch, Tensor, constraints, Distribution, broadcast_all, _Number, _size


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

- May involve **JIT compilation** or compilation optimizations.

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
- [`bernoulli.py_docs.md`](./bernoulli.py_docs.md)
- [`distribution.py_docs.md`](./distribution.py_docs.md)
- [`negative_binomial.py_docs.md`](./negative_binomial.py_docs.md)
- [`continuous_bernoulli.py_docs.md`](./continuous_bernoulli.py_docs.md)
- [`half_normal.py_docs.md`](./half_normal.py_docs.md)


## Cross-References

- **File Documentation**: `laplace.py_docs.md`
- **Keyword Index**: `laplace.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
