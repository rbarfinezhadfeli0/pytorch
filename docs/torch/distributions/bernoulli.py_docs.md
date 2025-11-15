# Documentation: `torch/distributions/bernoulli.py`

## File Metadata

- **Path**: `torch/distributions/bernoulli.py`
- **Size**: 4,771 bytes (4.66 KB)
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
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.types import _Number, Number


__all__ = ["Bernoulli"]


class Bernoulli(ExponentialFamily):
    r"""
    Creates a Bernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both).

    Samples are binary (0 or 1). They take the value `1` with probability `p`
    and `0` with probability `1 - p`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Bernoulli(torch.tensor([0.3]))
        >>> m.sample()  # 30% chance 1; 70% chance 0
        tensor([ 0.])

    Args:
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
        validate_args (bool, optional): whether to validate arguments, None by default
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.boolean
    has_enumerate_support = True
    _mean_carrier_measure = 0

    def __init__(
        self,
        probs: Optional[Union[Tensor, Number]] = None,
        logits: Optional[Union[Tensor, Number]] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            is_scalar = isinstance(probs, _Number)
            # pyrefly: ignore [read-only]
            (self.probs,) = broadcast_all(probs)
        else:
            assert logits is not None  # helps mypy
            is_scalar = isinstance(logits, _Number)
            # pyrefly: ignore [read-only]
            (self.logits,) = broadcast_all(logits)
        self._param = self.probs if probs is not None else self.logits
        if is_scalar:
            batch_shape = torch.Size()
        else:
            batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Bernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Bernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @property
    def mean(self) -> Tensor:
        return self.probs

    @property
    def mode(self) -> Tensor:
        mode = (self.probs >= 0.5).to(self.probs)
        mode[self.probs == 0.5] = nan
        return mode

    @property
    def variance(self) -> Tensor:
        return self.probs * (1 - self.probs)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        return self._param.size()

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.bernoulli(self.probs.expand(shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return -binary_cross_entropy_with_logits(logits, value, reduction="none")

    def entropy(self):
        return binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction="none"
        )

    def enumerate_support(self, expand=True):
        values = torch.arange(2, dtype=self._param.dtype, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (torch.logit(self.probs),)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        return torch.log1p(torch.exp(x))

```



## High-Level Overview

r"""    Creates a Bernoulli distribution parameterized by :attr:`probs`    or :attr:`logits` (but not both).    Samples are binary (0 or 1). They take the value `1` with probability `p`    and `0` with probability `1 - p`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Bernoulli(torch.tensor([0.3]))        >>> m.sample()  # 30% chance 1; 70% chance 0        tensor([ 0.])    Args:        probs (Number, Tensor): the probability of sampling `1`        logits (Number, Tensor): the log-odds of sampling `1`        validate_args (bool, optional): whether to validate arguments, None by default

This Python file contains 1 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Bernoulli`

**Functions defined**: `__init__`, `expand`, `_new`, `mean`, `mode`, `variance`, `logits`, `probs`, `param_shape`, `sample`, `log_prob`, `entropy`, `enumerate_support`, `_natural_params`, `_log_normalizer`

**Key imports**: Optional, Union, torch, nan, Tensor, constraints, ExponentialFamily, binary_cross_entropy_with_logits, _Number, Number


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
- `torch.nn.functional`: binary_cross_entropy_with_logits
- `torch.types`: _Number, Number


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
- [`distribution.py_docs.md`](./distribution.py_docs.md)
- [`negative_binomial.py_docs.md`](./negative_binomial.py_docs.md)
- [`continuous_bernoulli.py_docs.md`](./continuous_bernoulli.py_docs.md)
- [`half_normal.py_docs.md`](./half_normal.py_docs.md)


## Cross-References

- **File Documentation**: `bernoulli.py_docs.md`
- **Keyword Index**: `bernoulli.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
