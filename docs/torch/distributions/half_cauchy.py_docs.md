# Documentation: `torch/distributions/half_cauchy.py`

## File Metadata

- **Path**: `torch/distributions/half_cauchy.py`
- **Size**: 2,682 bytes (2.62 KB)
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
from torch.distributions.cauchy import Cauchy
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AbsTransform


__all__ = ["HalfCauchy"]


class HalfCauchy(TransformedDistribution):
    r"""
    Creates a half-Cauchy distribution parameterized by `scale` where::

        X ~ Cauchy(0, scale)
        Y = |X| ~ HalfCauchy(scale)

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = HalfCauchy(torch.tensor([1.0]))
        >>> m.sample()  # half-cauchy distributed with scale=1
        tensor([ 2.3214])

    Args:
        scale (float or Tensor): scale of the full Cauchy distribution
    """

    arg_constraints = {"scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.nonnegative
    has_rsample = True
    # pyrefly: ignore [bad-override]
    base_dist: Cauchy

    def __init__(
        self,
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        base_dist = Cauchy(0, scale, validate_args=False)
        super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(HalfCauchy, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def scale(self) -> Tensor:
        return self.base_dist.scale

    @property
    def mean(self) -> Tensor:
        return torch.full(
            self._extended_shape(),
            math.inf,
            dtype=self.scale.dtype,
            device=self.scale.device,
        )

    @property
    def mode(self) -> Tensor:
        return torch.zeros_like(self.scale)

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = torch.as_tensor(
            value, dtype=self.base_dist.scale.dtype, device=self.base_dist.scale.device
        )
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

r"""    Creates a half-Cauchy distribution parameterized by `scale` where::        X ~ Cauchy(0, scale)        Y = |X| ~ HalfCauchy(scale)    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = HalfCauchy(torch.tensor([1.0]))        >>> m.sample()  # half-cauchy distributed with scale=1        tensor([ 2.3214])    Args:        scale (float or Tensor): scale of the full Cauchy distribution

This Python file contains 1 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `HalfCauchy`

**Functions defined**: `__init__`, `expand`, `scale`, `mean`, `mode`, `variance`, `log_prob`, `cdf`, `icdf`, `entropy`

**Key imports**: math, Optional, Union, torch, inf, Tensor, constraints, Cauchy, TransformedDistribution, AbsTransform


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
- `torch.distributions.cauchy`: Cauchy
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
- [`half_normal.py_docs.md`](./half_normal.py_docs.md)


## Cross-References

- **File Documentation**: `half_cauchy.py_docs.md`
- **Keyword Index**: `half_cauchy.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
