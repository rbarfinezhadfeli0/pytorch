# Documentation: `torch/distributions/relaxed_bernoulli.py`

## File Metadata

- **Path**: `torch/distributions/relaxed_bernoulli.py`
- **Size**: 6,151 bytes (6.01 KB)
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
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.utils import (
    broadcast_all,
    clamp_probs,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.types import _Number, _size, Number


__all__ = ["LogitRelaxedBernoulli", "RelaxedBernoulli"]


class LogitRelaxedBernoulli(Distribution):
    r"""
    Creates a LogitRelaxedBernoulli distribution parameterized by :attr:`probs`
    or :attr:`logits` (but not both), which is the logit of a RelaxedBernoulli
    distribution.

    Samples are logits of values in (0, 1). See [1] for more details.

    Args:
        temperature (Tensor): relaxation temperature
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`

    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random
    Variables (Maddison et al., 2017)

    [2] Categorical Reparametrization with Gumbel-Softmax
    (Jang et al., 2017)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    support = constraints.real

    def __init__(
        self,
        temperature: Tensor,
        probs: Optional[Union[Tensor, Number]] = None,
        logits: Optional[Union[Tensor, Number]] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        self.temperature = temperature
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
        new = self._get_checked_instance(LogitRelaxedBernoulli, _instance)
        batch_shape = torch.Size(batch_shape)
        new.temperature = self.temperature
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(batch_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(batch_shape)
            new._param = new.logits
        super(LogitRelaxedBernoulli, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs, is_binary=True)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits, is_binary=True)

    @property
    def param_shape(self) -> torch.Size:
        return self._param.size()

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        probs = clamp_probs(self.probs.expand(shape))
        uniforms = clamp_probs(
            torch.rand(shape, dtype=probs.dtype, device=probs.device)
        )
        return (
            uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()
        ) / self.temperature

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        diff = logits - value.mul(self.temperature)
        return self.temperature.log() + diff - 2 * diff.exp().log1p()


class RelaxedBernoulli(TransformedDistribution):
    r"""
    Creates a RelaxedBernoulli distribution, parametrized by
    :attr:`temperature`, and either :attr:`probs` or :attr:`logits`
    (but not both). This is a relaxed version of the `Bernoulli` distribution,
    so the values are in (0, 1), and has reparametrizable samples.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = RelaxedBernoulli(torch.tensor([2.2]),
        ...                      torch.tensor([0.1, 0.2, 0.3, 0.99]))
        >>> m.sample()
        tensor([ 0.2951,  0.3442,  0.8918,  0.9021])

    Args:
        temperature (Tensor): relaxation temperature
        probs (Number, Tensor): the probability of sampling `1`
        logits (Number, Tensor): the log-odds of sampling `1`
    """

    arg_constraints = {"probs": constraints.unit_interval, "logits": constraints.real}
    # pyrefly: ignore [bad-override]
    support = constraints.unit_interval
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: LogitRelaxedBernoulli

    def __init__(
        self,
        temperature: Tensor,
        probs: Optional[Union[Tensor, Number]] = None,
        logits: Optional[Union[Tensor, Number]] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        base_dist = LogitRelaxedBernoulli(temperature, probs, logits)
        super().__init__(base_dist, SigmoidTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(RelaxedBernoulli, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def temperature(self) -> Tensor:
        return self.base_dist.temperature

    @property
    def logits(self) -> Tensor:
        return self.base_dist.logits

    @property
    def probs(self) -> Tensor:
        return self.base_dist.probs

```



## High-Level Overview

r"""    Creates a LogitRelaxedBernoulli distribution parameterized by :attr:`probs`    or :attr:`logits` (but not both), which is the logit of a RelaxedBernoulli    distribution.    Samples are logits of values in (0, 1). See [1] for more details.    Args:        temperature (Tensor): relaxation temperature        probs (Number, Tensor): the probability of sampling `1`        logits (Number, Tensor): the log-odds of sampling `1`    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random    Variables (Maddison et al., 2017)    [2] Categorical Reparametrization with Gumbel-Softmax    (Jang et al., 2017)

This Python file contains 2 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LogitRelaxedBernoulli`, `RelaxedBernoulli`

**Functions defined**: `__init__`, `expand`, `_new`, `logits`, `probs`, `param_shape`, `rsample`, `log_prob`, `__init__`, `expand`, `temperature`, `logits`, `probs`

**Key imports**: Optional, Union, torch, Tensor, constraints, Distribution, TransformedDistribution, SigmoidTransform, _Number, _size, Number


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
- `torch.distributions.transformed_distribution`: TransformedDistribution
- `torch.distributions.transforms`: SigmoidTransform
- `torch.types`: _Number, _size, Number


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

- **File Documentation**: `relaxed_bernoulli.py_docs.md`
- **Keyword Index**: `relaxed_bernoulli.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
