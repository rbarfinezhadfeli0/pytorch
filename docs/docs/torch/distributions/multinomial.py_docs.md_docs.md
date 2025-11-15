# Documentation: `docs/torch/distributions/multinomial.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/multinomial.py_docs.md`
- **Size**: 10,106 bytes (9.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/multinomial.py`

## File Metadata

- **Path**: `torch/distributions/multinomial.py`
- **Size**: 5,714 bytes (5.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch import inf, Tensor
from torch.distributions import Categorical, constraints
from torch.distributions.binomial import Binomial
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["Multinomial"]


class Multinomial(Distribution):
    r"""
    Creates a Multinomial distribution parameterized by :attr:`total_count` and
    either :attr:`probs` or :attr:`logits` (but not both). The innermost dimension of
    :attr:`probs` indexes over categories. All other dimensions index over batches.

    Note that :attr:`total_count` need not be specified if only :meth:`log_prob` is
    called (see example below)

    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,
              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`
              will return this normalized value.
              The `logits` argument will be interpreted as unnormalized log probabilities
              and can therefore be any real number. It will likewise be normalized so that
              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`
              will return this normalized value.

    -   :meth:`sample` requires a single shared `total_count` for all
        parameters and samples.
    -   :meth:`log_prob` allows different `total_count` for each parameter and
        sample.

    Example::

        >>> # xdoctest: +SKIP("FIXME: found invalid values")
        >>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))
        >>> x = m.sample()  # equal probability of 0, 1, 2, 3
        tensor([ 21.,  24.,  30.,  25.])

        >>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)
        tensor([-4.1338])

    Args:
        total_count (int): number of trials
        probs (Tensor): event probabilities
        logits (Tensor): event log probabilities (unnormalized)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    total_count: int

    @property
    def mean(self) -> Tensor:
        return self.probs * self.total_count

    @property
    def variance(self) -> Tensor:
        return self.total_count * self.probs * (1 - self.probs)

    def __init__(
        self,
        total_count: int = 1,
        probs: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
        validate_args: Optional[bool] = None,
    ) -> None:
        if not isinstance(total_count, int):
            raise NotImplementedError("inhomogeneous total_count is not supported")
        self.total_count = total_count
        self._categorical = Categorical(probs=probs, logits=logits)
        self._binomial = Binomial(total_count=total_count, probs=self.probs)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Multinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new.total_count = self.total_count
        new._categorical = self._categorical.expand(batch_shape)
        super(Multinomial, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def _new(self, *args, **kwargs):
        return self._categorical._new(*args, **kwargs)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    # pyrefly: ignore [bad-override]
    def support(self):
        return constraints.multinomial(self.total_count)

    @property
    def logits(self) -> Tensor:
        return self._categorical.logits

    @property
    def probs(self) -> Tensor:
        return self._categorical.probs

    @property
    def param_shape(self) -> torch.Size:
        return self._categorical.param_shape

    def sample(self, sample_shape=torch.Size()):
        sample_shape = torch.Size(sample_shape)
        samples = self._categorical.sample(
            torch.Size((self.total_count,)) + sample_shape
        )
        # samples.shape is (total_count, sample_shape, batch_shape), need to change it to
        # (sample_shape, batch_shape, total_count)
        shifted_idx = list(range(samples.dim()))
        shifted_idx.append(shifted_idx.pop(0))
        samples = samples.permute(*shifted_idx)
        counts = samples.new(self._extended_shape(sample_shape)).zero_()
        counts.scatter_add_(-1, samples, torch.ones_like(samples))
        return counts.type_as(self.probs)

    def entropy(self):
        n = torch.tensor(self.total_count)

        cat_entropy = self._categorical.entropy()
        term1 = n * cat_entropy - torch.lgamma(n + 1)

        support = self._binomial.enumerate_support(expand=False)[1:]
        binomial_probs = torch.exp(self._binomial.log_prob(support))
        weights = torch.lgamma(support + 1)
        term2 = (binomial_probs * weights).sum([0, -1])

        return term1 + term2

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        logits = logits.clone(memory_format=torch.contiguous_format)
        log_factorial_n = torch.lgamma(value.sum(-1) + 1)
        log_factorial_xs = torch.lgamma(value + 1).sum(-1)
        logits[(value == 0) & (logits == -inf)] = 0
        log_powers = (logits * value).sum(-1)
        return log_factorial_n - log_factorial_xs + log_powers

```



## High-Level Overview

r"""    Creates a Multinomial distribution parameterized by :attr:`total_count` and    either :attr:`probs` or :attr:`logits` (but not both). The innermost dimension of    :attr:`probs` indexes over categories. All other dimensions index over batches.    Note that :attr:`total_count` need not be specified if only :meth:`log_prob` is    called (see example below)    .. note:: The `probs` argument must be non-negative, finite and have a non-zero sum,              and it will be normalized to sum to 1 along the last dimension. :attr:`probs`              will return this normalized value.              The `logits` argument will be interpreted as unnormalized log probabilities              and can therefore be any real number. It will likewise be normalized so that              the resulting probabilities sum to 1 along the last dimension. :attr:`logits`              will return this normalized value.    -   :meth:`sample` requires a single shared `total_count` for all        parameters and samples.    -   :meth:`log_prob` allows different `total_count` for each parameter and        sample.    Example::        >>> # xdoctest: +SKIP("FIXME: found invalid values")        >>> m = Multinomial(100, torch.tensor([ 1., 1., 1., 1.]))        >>> x = m.sample()  # equal probability of 0, 1, 2, 3        tensor([ 21.,  24.,  30.,  25.])        >>> Multinomial(probs=torch.tensor([1., 1., 1., 1.])).log_prob(x)        tensor([-4.1338])    Args:        total_count (int): number of trials        probs (Tensor): event probabilities        logits (Tensor): event log probabilities (unnormalized)

This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Multinomial`

**Functions defined**: `mean`, `variance`, `__init__`, `expand`, `_new`, `support`, `logits`, `probs`, `param_shape`, `sample`, `entropy`, `log_prob`

**Key imports**: Optional, torch, inf, Tensor, Categorical, constraints, Binomial, Distribution, broadcast_all


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch.distributions`: Categorical, constraints
- `torch.distributions.binomial`: Binomial
- `torch.distributions.distribution`: Distribution
- `torch.distributions.utils`: broadcast_all


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

- **File Documentation**: `multinomial.py_docs.md`
- **Keyword Index**: `multinomial.py_kw.md`
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

- **File Documentation**: `multinomial.py_docs.md_docs.md`
- **Keyword Index**: `multinomial.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
