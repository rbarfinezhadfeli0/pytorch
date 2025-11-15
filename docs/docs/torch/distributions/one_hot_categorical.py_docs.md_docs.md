# Documentation: `docs/torch/distributions/one_hot_categorical.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/one_hot_categorical.py_docs.md`
- **Size**: 9,179 bytes (8.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/one_hot_categorical.py`

## File Metadata

- **Path**: `torch/distributions/one_hot_categorical.py`
- **Size**: 5,028 bytes (4.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.categorical import Categorical
from torch.distributions.distribution import Distribution
from torch.types import _size


__all__ = ["OneHotCategorical", "OneHotCategoricalStraightThrough"]


class OneHotCategorical(Distribution):
    r"""
    Creates a one-hot categorical distribution parameterized by :attr:`probs` or
    :attr:`logits`.

    Samples are one-hot coded vectors of size ``probs.size(-1)``.

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    See also: :func:`torch.distributions.Categorical` for specifications of
    :attr:`probs` and :attr:`logits`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
        >>> m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 0.,  0.,  0.,  1.])

    Args:
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    support = constraints.one_hot
    has_enumerate_support = True

    def __init__(
        self,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        self._categorical = Categorical(probs, logits)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(OneHotCategorical, _instance)
        batch_shape = torch.Size(batch_shape)
        new._categorical = self._categorical.expand(batch_shape)
        super(OneHotCategorical, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @property
    def _param(self) -> Tensor:
        return self._categorical._param

    @property
    def probs(self) -> Tensor:
        return self._categorical.probs

    @property
    def logits(self) -> Tensor:
        return self._categorical.logits

    @property
    def mean(self) -> Tensor:
        return self._categorical.probs

    @property
    def mode(self) -> Tensor:
        probs = self._categorical.probs
        mode = probs.argmax(dim=-1)
        return torch.nn.functional.one_hot(mode, num_classes=probs.shape[-1]).to(probs)

    @property
    def variance(self) -> Tensor:
        return self._categorical.probs * (1 - self._categorical.probs)

    @property
    def param_shape(self) -> torch.Size:
        return self._categorical.param_shape

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        probs = self._categorical.probs
        num_events = self._categorical._num_events
        indices = self._categorical.sample(sample_shape)
        return torch.nn.functional.one_hot(indices, num_events).to(probs)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        indices = value.max(-1)[1]
        return self._categorical.log_prob(indices)

    def entropy(self):
        return self._categorical.entropy()

    def enumerate_support(self, expand=True):
        n = self.event_shape[0]
        values = torch.eye(n, dtype=self._param.dtype, device=self._param.device)
        values = values.view((n,) + (1,) * len(self.batch_shape) + (n,))
        if expand:
            values = values.expand((n,) + self.batch_shape + (n,))
        return values


class OneHotCategoricalStraightThrough(OneHotCategorical):
    r"""
    Creates a reparameterizable :class:`OneHotCategorical` distribution based on the straight-
    through gradient estimator from [1].

    [1] Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation
    (Bengio et al., 2013)
    """

    has_rsample = True

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        samples = self.sample(sample_shape)
        probs = self._categorical.probs  # cached via @lazy_property
        return samples + (probs - probs.detach())

```



## High-Level Overview

r"""    Creates a one-hot categorical distribution parameterized by :attr:`probs` or    :attr:`logits`.    Samples are one-hot coded vectors of size ``probs.size(-1)``.    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`              will return this normalized value.              The `logits` argument will be interpreted as unnormalized log probabilities              and can therefore be any real number. It will likewise be normalized so that              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`              will return this normalized value.    See also: :func:`torch.distributions.Categorical` for specifications of    :attr:`probs` and :attr:`logits`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = OneHotCategorical(torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))        >>> m.sample()  # equal probability of 0, 1, 2, 3        tensor([ 0.,  0.,  0.,  1.])    Args:        probs (Tensor): event probabilities        logits (Tensor): event log probabilities (unnormalized)

This Python file contains 2 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OneHotCategorical`, `OneHotCategoricalStraightThrough`

**Functions defined**: `__init__`, `expand`, `_new`, `_param`, `probs`, `logits`, `mean`, `mode`, `variance`, `param_shape`, `sample`, `log_prob`, `entropy`, `enumerate_support`, `rsample`

**Key imports**: Optional, torch, Tensor, constraints, Categorical, Distribution, _size


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
- `torch.distributions.categorical`: Categorical
- `torch.distributions.distribution`: Distribution
- `torch.types`: _size


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `one_hot_categorical.py_docs.md`
- **Keyword Index**: `one_hot_categorical.py_kw.md`
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
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
- [`geometric.py_kw.md_docs.md`](./geometric.py_kw.md_docs.md)
- [`kumaraswamy.py_kw.md_docs.md`](./kumaraswamy.py_kw.md_docs.md)
- [`transformed_distribution.py_kw.md_docs.md`](./transformed_distribution.py_kw.md_docs.md)
- [`log_normal.py_docs.md_docs.md`](./log_normal.py_docs.md_docs.md)
- [`kumaraswamy.py_docs.md_docs.md`](./kumaraswamy.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `one_hot_categorical.py_docs.md_docs.md`
- **Keyword Index**: `one_hot_categorical.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
