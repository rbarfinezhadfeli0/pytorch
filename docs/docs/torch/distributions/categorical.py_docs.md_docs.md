# Documentation: `docs/torch/distributions/categorical.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/categorical.py_docs.md`
- **Size**: 10,506 bytes (10.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/categorical.py`

## File Metadata

- **Path**: `torch/distributions/categorical.py`
- **Size**: 6,221 bytes (6.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import lazy_property, logits_to_probs, probs_to_logits


__all__ = ["Categorical"]


class Categorical(Distribution):
    r"""
    Creates a categorical distribution parameterized by either :attr:`probs` or
    :attr:`logits` (but not both).

    .. note::
        It is equivalent to the distribution that :func:`torch.multinomial`
        samples from.

    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.

    If `probs` is 1-dimensional with length-`K`, each element is the relative probability
    of sampling the class at that index.

    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of
    relative probability vectors.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.multinomial`

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor(3)

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    has_enumerate_support = True

    def __init__(
        self,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        if (probs is None) == (logits is None):
            raise ValueError(
                "Either `probs` or `logits` must be specified, but not both."
            )
        if probs is not None:
            if probs.dim() < 1:
                raise ValueError("`probs` parameter must be at least one-dimensional.")
            # pyrefly: ignore [read-only]
            self.probs = probs / probs.sum(-1, keepdim=True)
        else:
            assert logits is not None  # helps mypy
            if logits.dim() < 1:
                raise ValueError("`logits` parameter must be at least one-dimensional.")
            # Normalize
            # pyrefly: ignore [read-only]
            self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        self._param = self.probs if probs is not None else self.logits
        self._num_events = self._param.size()[-1]
        batch_shape = (
            self._param.size()[:-1] if self._param.ndimension() > 1 else torch.Size()
        )
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Categorical, _instance)
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        if "probs" in self.__dict__:
            new.probs = self.probs.expand(param_shape)
            new._param = new.probs
        if "logits" in self.__dict__:
            new.logits = self.logits.expand(param_shape)
            new._param = new.logits
        new._num_events = self._num_events
        super(Categorical, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._param.new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        return constraints.integer_interval(0, self._num_events - 1)

    @lazy_property
    def logits(self) -> Tensor:
        return probs_to_logits(self.probs)

    @lazy_property
    def probs(self) -> Tensor:
        return logits_to_probs(self.logits)

    @property
    def param_shape(self) -> torch.Size:
        return self._param.size()

    @property
    def mean(self) -> Tensor:
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )

    @property
    def mode(self) -> Tensor:
        return self.probs.argmax(dim=-1)

    @property
    def variance(self) -> Tensor:
        return torch.full(
            self._extended_shape(),
            nan,
            dtype=self.probs.dtype,
            device=self.probs.device,
        )

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    def enumerate_support(self, expand=True):
        num_events = self._num_events
        values = torch.arange(num_events, dtype=torch.long, device=self._param.device)
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        if expand:
            values = values.expand((-1,) + self._batch_shape)
        return values

```



## High-Level Overview

r"""    Creates a categorical distribution parameterized by either :attr:`probs` or    :attr:`logits` (but not both).    .. note::        It is equivalent to the distribution that :func:`torch.multinomial`        samples from.    Samples are integers from :math:`\{0, \ldots, K-1\}` where `K` is ``probs.size(-1)``.    If `probs` is 1-dimensional with length-`K`, each element is the relative probability    of sampling the class at that index.    If `probs` is N-dimensional, the first N-1 dimensions are treated as a batch of    relative probability vectors.    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`              will return this normalized value.              The `logits` argument will be interpreted as unnormalized log probabilities              and can therefore be any real number. It will likewise be normalized so that              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`              will return this normalized value.    See also: :func:`torch.multinomial`    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Categorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))        >>> m.sample()  # equal probability of 0, 1, 2, 3        tensor(3)    Args:        probs (Tensor): event probabilities        logits (Tensor): event log probabilities (unnormalized)

This Python file contains 2 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Categorical`

**Functions defined**: `__init__`, `expand`, `_new`, `support`, `logits`, `probs`, `param_shape`, `mean`, `mode`, `variance`, `sample`, `log_prob`, `entropy`, `enumerate_support`

**Key imports**: Optional, torch, nan, Tensor, constraints, Distribution, lazy_property, logits_to_probs, probs_to_logits


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch.distributions`: constraints
- `torch.distributions.distribution`: Distribution
- `torch.distributions.utils`: lazy_property, logits_to_probs, probs_to_logits


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

- **File Documentation**: `categorical.py_docs.md`
- **Keyword Index**: `categorical.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributions`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/torch/distributions`):

- [`wishart.py_docs.md_docs.md`](./wishart.py_docs.md_docs.md)
- [`pareto.py_docs.md_docs.md`](./pareto.py_docs.md_docs.md)
- [`binomial.py_docs.md_docs.md`](./binomial.py_docs.md_docs.md)
- [`half_cauchy.py_docs.md_docs.md`](./half_cauchy.py_docs.md_docs.md)
- [`one_hot_categorical.py_docs.md_docs.md`](./one_hot_categorical.py_docs.md_docs.md)
- [`geometric.py_kw.md_docs.md`](./geometric.py_kw.md_docs.md)
- [`kumaraswamy.py_kw.md_docs.md`](./kumaraswamy.py_kw.md_docs.md)
- [`transformed_distribution.py_kw.md_docs.md`](./transformed_distribution.py_kw.md_docs.md)
- [`log_normal.py_docs.md_docs.md`](./log_normal.py_docs.md_docs.md)
- [`kumaraswamy.py_docs.md_docs.md`](./kumaraswamy.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `categorical.py_docs.md_docs.md`
- **Keyword Index**: `categorical.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
