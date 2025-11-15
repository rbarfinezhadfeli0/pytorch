# Documentation: `docs/torch/distributions/gumbel.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/gumbel.py_docs.md`
- **Size**: 6,394 bytes (6.24 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/gumbel.py`

## File Metadata

- **Path**: `torch/distributions/gumbel.py`
- **Size**: 3,052 bytes (2.98 KB)
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
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all, euler_constant
from torch.types import _Number


__all__ = ["Gumbel"]


class Gumbel(TransformedDistribution):
    r"""
    Samples from a Gumbel Distribution.

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2
        tensor([ 1.0124])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # pyrefly: ignore [bad-override]
    support = constraints.real

    def __init__(
        self,
        loc: Union[Tensor, float],
        scale: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        self.loc, self.scale = broadcast_all(loc, scale)
        finfo = torch.finfo(self.loc.dtype)
        if isinstance(loc, _Number) and isinstance(scale, _Number):
            base_dist = Uniform(finfo.tiny, 1 - finfo.eps, validate_args=validate_args)
        else:
            base_dist = Uniform(
                torch.full_like(self.loc, finfo.tiny),
                torch.full_like(self.loc, 1 - finfo.eps),
                validate_args=validate_args,
            )
        transforms = [
            ExpTransform().inv,
            AffineTransform(loc=0, scale=-torch.ones_like(self.scale)),
            ExpTransform().inv,
            AffineTransform(loc=loc, scale=-self.scale),
        ]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Gumbel, _instance)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    # Explicitly defining the log probability function for Gumbel due to precision issues
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        y = (self.loc - value) / self.scale
        return (y - y.exp()) - self.scale.log()

    @property
    def mean(self) -> Tensor:
        return self.loc + self.scale * euler_constant

    @property
    def mode(self) -> Tensor:
        return self.loc

    @property
    def stddev(self) -> Tensor:
        return (math.pi / math.sqrt(6)) * self.scale

    @property
    def variance(self) -> Tensor:
        return self.stddev.pow(2)

    def entropy(self):
        return self.scale.log() + (1 + euler_constant)

```



## High-Level Overview

r"""    Samples from a Gumbel Distribution.    Examples::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Gumbel(torch.tensor([1.0]), torch.tensor([2.0]))        >>> m.sample()  # sample from Gumbel distribution with loc=1, scale=2        tensor([ 1.0124])    Args:        loc (float or Tensor): Location parameter of the distribution        scale (float or Tensor): Scale parameter of the distribution

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Gumbel`

**Functions defined**: `__init__`, `expand`, `log_prob`, `mean`, `mode`, `stddev`, `variance`, `entropy`

**Key imports**: math, Optional, Union, torch, Tensor, constraints, TransformedDistribution, AffineTransform, ExpTransform, Uniform, broadcast_all, euler_constant, _Number


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
- `torch.distributions.transformed_distribution`: TransformedDistribution
- `torch.distributions.transforms`: AffineTransform, ExpTransform
- `torch.distributions.uniform`: Uniform
- `torch.distributions.utils`: broadcast_all, euler_constant
- `torch.types`: _Number


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

- **File Documentation**: `gumbel.py_docs.md`
- **Keyword Index**: `gumbel.py_kw.md`
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

- **File Documentation**: `gumbel.py_docs.md_docs.md`
- **Keyword Index**: `gumbel.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
