# Documentation: `docs/torch/distributions/half_normal.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/half_normal.py_docs.md`
- **Size**: 5,610 bytes (5.48 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/half_normal.py`

## File Metadata

- **Path**: `torch/distributions/half_normal.py`
- **Size**: 2,447 bytes (2.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import math
from typing import Optional, Union

import torch
from torch import inf, Tensor
from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform


__all__ = ["HalfNormal"]


class HalfNormal(TransformedDistribution):
    r"""
    Creates a half-normal distribution parameterized by `scale` where::

        X ~ Normal(0, scale)
        Y = |X| ~ HalfNormal(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfNormal(torch.tensor([1.0]))
        >>> m.sample()  # half-normal distributed with scale=1
        tensor([ 0.1046])

    Args:
        scale (float or Tensor): scale of the full Normal distribution
    """

    arg_constraints = {"scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.nonnegative
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Normal

    def __init__(
        self,
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        base_dist = Normal(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfNormal, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        return self.scale * math.sqrt(2 / math.pi)

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.scale)

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) * (1 - 2 / math.pi)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_prob = self.base_dist.log_prob(value) + math.log(2)
        log_prob = torch.where(value >= 0, log_prob, -inf)
        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 2 * self.base_dist.cdf(value) - 1

    def icdf(self, prob):
        return self.base_dist.icdf((prob + 1) / 2)

    def entropy(self):
        return self.base_dist.entropy() - math.log(2)

```



## High-Level Overview

r"""    Creates a half-normal distribution parameterized by `scale` where::        X ~ Normal(0, scale)        Y = |X| ~ HalfNormal(scale)    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = HalfNormal(torch.tensor([1.0]))        >>> m.sample()  # half-normal distributed with scale=1        tensor([ 0.1046])    Args:        scale (float or Tensor): scale of the full Normal distribution

This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `HalfNormal`

**Functions defined**: `__init__`, `expand`, `scale`, `mean`, `mode`, `variance`, `log_prob`, `cdf`, `icdf`, `entropy`

**Key imports**: math, Optional, Union, torch, inf, Tensor, constraints, Normal, TransformedDistribution, AbsTransform


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `typing`: Optional, Union
- `torch`
- `torch.distributions`: constraints
- `torch.distributions.normal`: Normal
- `torch.distributions.transformed_distribution`: TransformedDistribution
- `torch.distributions.transforms`: AbsTransform


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


## Cross-References

- **File Documentation**: `half_normal.py_docs.md`
- **Keyword Index**: `half_normal.py_kw.md`
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

- **File Documentation**: `half_normal.py_docs.md_docs.md`
- **Keyword Index**: `half_normal.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
