# Documentation: `docs/torch/distributions/independent.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/independent.py_docs.md`
- **Size**: 9,151 bytes (8.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/independent.py`

## File Metadata

- **Path**: `torch/distributions/independent.py`
- **Size**: 5,019 bytes (4.90 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Generic, Optional, TypeVar

import torch
from torch import Size, Tensor
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _sum_rightmost
from torch.types import _size


__all__ = ["Independent"]


D = TypeVar("D", bound=Distribution)


class Independent(Distribution, Generic[D]):
    r"""
    Reinterprets some of the batch dims of a distribution as event dims.

    This is mainly useful for changing the shape of the result of
    :meth:`log_prob`. For example to create a diagonal Normal distribution with
    the same shape as a Multivariate Normal distribution (so they are
    interchangeable), you can::

        >>> from torch.distributions.multivariate_normal import MultivariateNormal
        >>> from torch.distributions.normal import Normal
        >>> loc = torch.zeros(3)
        >>> scale = torch.ones(3)
        >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))
        >>> [mvn.batch_shape, mvn.event_shape]
        [torch.Size([]), torch.Size([3])]
        >>> normal = Normal(loc, scale)
        >>> [normal.batch_shape, normal.event_shape]
        [torch.Size([3]), torch.Size([])]
        >>> diagn = Independent(normal, 1)
        >>> [diagn.batch_shape, diagn.event_shape]
        [torch.Size([]), torch.Size([3])]

    Args:
        base_distribution (torch.distributions.distribution.Distribution): a
            base distribution
        reinterpreted_batch_ndims (int): the number of batch dims to
            reinterpret as event dims
    """

    arg_constraints: dict[str, constraints.Constraint] = {}
    base_dist: D

    def __init__(
        self,
        base_distribution: D,
        reinterpreted_batch_ndims: int,
        validate_args: Optional[bool] = None,
    ) -> None:
        if reinterpreted_batch_ndims > len(base_distribution.batch_shape):
            raise ValueError(
                "Expected reinterpreted_batch_ndims <= len(base_distribution.batch_shape), "
                f"actual {reinterpreted_batch_ndims} vs {len(base_distribution.batch_shape)}"
            )
        shape: Size = base_distribution.batch_shape + base_distribution.event_shape
        event_dim: int = reinterpreted_batch_ndims + len(base_distribution.event_shape)
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]
        self.base_dist = base_distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Independent, _instance)
        batch_shape = torch.Size(batch_shape)
        new.base_dist = self.base_dist.expand(
            batch_shape + self.event_shape[: self.reinterpreted_batch_ndims]
        )
        new.reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        super(Independent, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self) -> bool:  # type: ignore[override]
        if self.reinterpreted_batch_ndims > 0:
            return False
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    # pyrefly: ignore [bad-override]
    def support(self):
        result = self.base_dist.support
        if self.reinterpreted_batch_ndims:
            result = constraints.independent(result, self.reinterpreted_batch_ndims)
        return result

    @property
    def mean(self) -> Tensor:
        return self.base_dist.mean

    @property
    def mode(self) -> Tensor:
        return self.base_dist.mode

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape: _size = torch.Size()) -> Tensor:
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        log_prob = self.base_dist.log_prob(value)
        return _sum_rightmost(log_prob, self.reinterpreted_batch_ndims)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return _sum_rightmost(entropy, self.reinterpreted_batch_ndims)

    def enumerate_support(self, expand=True):
        if self.reinterpreted_batch_ndims > 0:
            raise NotImplementedError(
                "Enumeration over cartesian product is not implemented"
            )
        return self.base_dist.enumerate_support(expand=expand)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"({self.base_dist}, {self.reinterpreted_batch_ndims})"
        )

```



## High-Level Overview

r"""    Reinterprets some of the batch dims of a distribution as event dims.    This is mainly useful for changing the shape of the result of    :meth:`log_prob`. For example to create a diagonal Normal distribution with    the same shape as a Multivariate Normal distribution (so they are    interchangeable), you can::        >>> from torch.distributions.multivariate_normal import MultivariateNormal        >>> from torch.distributions.normal import Normal        >>> loc = torch.zeros(3)        >>> scale = torch.ones(3)        >>> mvn = MultivariateNormal(loc, scale_tril=torch.diag(scale))        >>> [mvn.batch_shape, mvn.event_shape]        [torch.Size([]), torch.Size([3])]        >>> normal = Normal(loc, scale)        >>> [normal.batch_shape, normal.event_shape]        [torch.Size([3]), torch.Size([])]        >>> diagn = Independent(normal, 1)        >>> [diagn.batch_shape, diagn.event_shape]        [torch.Size([]), torch.Size([3])]    Args:        base_distribution (torch.distributions.distribution.Distribution): a            base distribution        reinterpreted_batch_ndims (int): the number of batch dims to            reinterpret as event dims

This Python file contains 1 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Independent`

**Functions defined**: `__init__`, `expand`, `has_rsample`, `has_enumerate_support`, `support`, `mean`, `mode`, `variance`, `sample`, `rsample`, `log_prob`, `entropy`, `enumerate_support`, `__repr__`

**Key imports**: Generic, Optional, TypeVar, torch, Size, Tensor, constraints, Distribution, _sum_rightmost, _size, MultivariateNormal, Normal


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Generic, Optional, TypeVar
- `torch`
- `torch.distributions`: constraints
- `torch.distributions.distribution`: Distribution
- `torch.distributions.utils`: _sum_rightmost
- `torch.types`: _size
- `torch.distributions.multivariate_normal`: MultivariateNormal
- `torch.distributions.normal`: Normal


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

- **File Documentation**: `independent.py_docs.md`
- **Keyword Index**: `independent.py_kw.md`
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

- **File Documentation**: `independent.py_docs.md_docs.md`
- **Keyword Index**: `independent.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
