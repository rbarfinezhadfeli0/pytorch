# Documentation: beta.py

## File Metadata
- **Path**: `torch/distributions/beta.py`
- **Size**: 4050 bytes
- **Lines**: 119
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): Beta

### Functions
This file defines 12 function(s): __init__, expand, mean, mode, variance, rsample, log_prob, entropy, concentration1, concentration0, _natural_params, _log_normalizer


## Key Components

The file contains 321 words across 119 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4050 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
