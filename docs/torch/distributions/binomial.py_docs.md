# Documentation: `torch/distributions/binomial.py`

## File Metadata

- **Path**: `torch/distributions/binomial.py`
- **Size**: 6,484 bytes (6.33 KB)
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
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)


__all__ = ["Binomial"]


def _clamp_by_zero(x):
    # works like clamp(x, min=0) but has grad at 0 is 0.5
    return (x.clamp(min=0) + x - x.clamp(max=0)) / 2


class Binomial(Distribution):
    r"""
    Creates a Binomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be
    broadcastable with :attr:`probs`/:attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))
        >>> x = m.sample()
        tensor([   0.,   22.,   71.,  100.])

        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))
        >>> x = m.sample()
        tensor([[ 4.,  5.],
                [ 7.,  6.]])

    Args:
        total_count (int or Tensor): number of Bernoulli trials
        probs (Tensor): Event probabilities
        logits (Tensor): Event log-odds
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "total_count": constraints.nonnegative_integer,
        "probs": constraints.unit_interval,
        "logits": constraints.real,
    }
    has_enumerate_support = True

    def __init__(
        self,
        total_count: Union[Tensor, int] = 1,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            (
                self.total_count,
                # pyrefly: ignore [read-only]
                self.probs,
            ) = broadcast_all(total_count, probs)
            self.total_count = self.total_count.type_as(self.probs)
        else:
            assert logits is not None  # helps mypy
            (
                self.total_count,
                # pyrefly: ignore [read-only]
                self.logits,
            ) = broadcast_all(total_count, logits)
            self.total_count = self.total_count.type_as(self.logits)

        self._param = self.probs if probs is not None else self.logits
        batch_shape = self._param.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Binomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(Binomial, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        return constraints.integer_interval(0, self.total_count)

    @property
    def mean(self) -> Tensor:
        return self.total_count * self.probs

    @property
    def mode(self) -> Tensor:
        return ((self.total_count + 1) * self.probs).floor().clamp(max=self.total_count)

    @property
    def variance(self) -> Tensor:
        return self.total_count * self.probs * (1 - self.probs)

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
            return torch.binomial(
                self.total_count.expand(shape), self.probs.expand(shape)
            )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_factorial_n = torch.lgamma(self.total_count + 1)
        log_factorial_k = torch.lgamma(value + 1)
        log_factorial_nmk = torch.lgamma(self.total_count - value + 1)
        # k * log(p) + (n - k) * log(1 - p) = k * (log(p) - log(1 - p)) + n * log(1 - p)
        #     (case logit < 0)              = k * logit - n * log1p(e^logit)
        #     (case logit > 0)              = k * logit - n * (log(p) - log(1 - p)) + n * log(p)
        #                                   = k * logit - n * logit - n * log1p(e^-logit)
        #     (merge two cases)             = k * logit - n * max(logit, 0) - n * log1p(e^-|logit|)
        normalize_term = (
            self.total_count * _clamp_by_zero(self.logits)
            + self.total_count * torch.log1p(torch.exp(-torch.abs(self.logits)))
            - log_factorial_n
        )
        return (
            value * self.logits - log_factorial_k - log_factorial_nmk - normalize_term
        )

    def entropy(self):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `entropy`."
            )

        log_prob = self.log_prob(self.enumerate_support(False))
        return -(torch.exp(log_prob) * log_prob).sum(0)

    def enumerate_support(self, expand=True):
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `enumerate_support`."
            )
        values = torch.arange(
            1 + total_count, dtype=self._param.dtype, device=self._param.device
        )
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

```



## High-Level Overview

r"""    Creates a Binomial distribution parameterized by :attr:`total_count` and    either :attr:`probs` or :attr:`logits` (but not both). :attr:`total_count` must be    broadcastable with :attr:`probs`/:attr:`logits`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Binomial(100, torch.tensor([0 , .2, .8, 1]))        >>> x = m.sample()        tensor([   0.,   22.,   71.,  100.])        >>> m = Binomial(torch.tensor([[5.], [10.]]), torch.tensor([0.5, 0.8]))        >>> x = m.sample()        tensor([[ 4.,  5.],                [ 7.,  6.]])    Args:        total_count (int or Tensor): number of Bernoulli trials        probs (Tensor): Event probabilities        logits (Tensor): Event log-odds

This Python file contains 1 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Binomial`

**Functions defined**: `_clamp_by_zero`, `__init__`, `expand`, `_new`, `support`, `mean`, `mode`, `variance`, `logits`, `probs`, `param_shape`, `sample`, `log_prob`, `entropy`, `enumerate_support`

**Key imports**: Optional, Union, torch, Tensor, constraints, Distribution


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

- **File Documentation**: `binomial.py_docs.md`
- **Keyword Index**: `binomial.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
