# Documentation: `torch/distributions/beta.py`

## File Metadata

- **Path**: `torch/distributions/beta.py`
- **Size**: 4,050 bytes (3.96 KB)
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
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["Beta"]


class Beta(ExponentialFamily):
    r"""
    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0
        tensor([ 0.1046])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Union[Tensor, float],
        concentration0: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        if isinstance(concentration1, _Number) and isinstance(concentration0, _Number):
            concentration1_concentration0 = torch.tensor(
                [float(concentration1), float(concentration0)]
            )
        else:
            concentration1, concentration0 = broadcast_all(
                concentration1, concentration0
            )
            concentration1_concentration0 = torch.stack(
                [concentration1, concentration0], -1
            )
        self._dirichlet = Dirichlet(
            concentration1_concentration0, validate_args=validate_args
        )
        super().__init__(self._dirichlet._batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        new._dirichlet = self._dirichlet.expand(batch_shape)
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def mode(self) -> Tensor:
        return self._dirichlet.mode[..., 0]

    @property
    def variance(self) -> Tensor:
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))

    def rsample(self, sample_shape: _size = ()) -> Tensor:
        return self._dirichlet.rsample(sample_shape).select(-1, 0)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self):
        return self._dirichlet.entropy()

    @property
    def concentration1(self) -> Tensor:
        result = self._dirichlet.concentration[..., 0]
        if isinstance(result, _Number):
            return torch.tensor([result])
        else:
            return result

    @property
    def concentration0(self) -> Tensor:
        result = self._dirichlet.concentration[..., 1]
        if isinstance(result, _Number):
            return torch.tensor([result])
        else:
            return result

    @property
    def _natural_params(self) -> tuple[Tensor, Tensor]:
        return (self.concentration1, self.concentration0)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

```



## High-Level Overview

r"""    Beta distribution parameterized by :attr:`concentration1` and :attr:`concentration0`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Beta(torch.tensor([0.5]), torch.tensor([0.5]))        >>> m.sample()  # Beta distributed with concentration concentration1 and concentration0        tensor([ 0.1046])    Args:        concentration1 (float or Tensor): 1st concentration parameter of the distribution            (often referred to as alpha)        concentration0 (float or Tensor): 2nd concentration parameter of the distribution            (often referred to as beta)

This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Beta`

**Functions defined**: `__init__`, `expand`, `mean`, `mode`, `variance`, `rsample`, `log_prob`, `entropy`, `concentration1`, `concentration0`, `_natural_params`, `_log_normalizer`

**Key imports**: Optional, Union, torch, Tensor, constraints, Dirichlet, ExponentialFamily, broadcast_all, _Number, _size


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
- `torch.distributions.dirichlet`: Dirichlet
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

- **File Documentation**: `beta.py_docs.md`
- **Keyword Index**: `beta.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
