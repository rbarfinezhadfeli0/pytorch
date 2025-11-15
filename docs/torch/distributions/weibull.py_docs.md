# Documentation: `torch/distributions/weibull.py`

## File Metadata

- **Path**: `torch/distributions/weibull.py`
- **Size**: 3,467 bytes (3.39 KB)
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
from torch.distributions.exponential import Exponential
from torch.distributions.gumbel import euler_constant
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.utils import broadcast_all


__all__ = ["Weibull"]


class Weibull(TransformedDistribution):
    r"""
    Samples from a two-parameter Weibull distribution.

    Example:

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1
        tensor([ 0.4784])

    Args:
        scale (float or Tensor): Scale parameter of distribution (lambda).
        concentration (float or Tensor): Concentration parameter of distribution (k/shape).
        validate_args (bool, optional): Whether to validate arguments. Default: None.
    """

    arg_constraints = {
        "scale": constraints.positive,
        "concentration": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.positive

    def __init__(
        self,
        scale: Union[Tensor, float],
        concentration: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        self.scale, self.concentration = broadcast_all(scale, concentration)
        self.concentration_reciprocal = self.concentration.reciprocal()
        base_dist = Exponential(
            torch.ones_like(self.scale), validate_args=validate_args
        )
        transforms = [
            PowerTransform(exponent=self.concentration_reciprocal),
            AffineTransform(loc=0, scale=self.scale),
        ]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Weibull, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.concentration_reciprocal = new.concentration.reciprocal()
        base_dist = self.base_dist.expand(batch_shape)
        transforms = [
            PowerTransform(exponent=new.concentration_reciprocal),
            AffineTransform(loc=0, scale=new.scale),
        ]
        super(Weibull, new).__init__(base_dist, transforms, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        return self.scale * torch.exp(torch.lgamma(1 + self.concentration_reciprocal))

    @property
    def mode(self) -> Tensor:
        return (
            self.scale
            * ((self.concentration - 1) / self.concentration)
            ** self.concentration.reciprocal()
        )

    @property
    def variance(self) -> Tensor:
        return self.scale.pow(2) * (
            torch.exp(torch.lgamma(1 + 2 * self.concentration_reciprocal))
            - torch.exp(2 * torch.lgamma(1 + self.concentration_reciprocal))
        )

    def entropy(self):
        return (
            euler_constant * (1 - self.concentration_reciprocal)
            + torch.log(self.scale * self.concentration_reciprocal)
            + 1
        )

```



## High-Level Overview

r"""    Samples from a two-parameter Weibull distribution.    Example:        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))        >>> m.sample()  # sample from a Weibull distribution with scale=1, concentration=1        tensor([ 0.4784])    Args:        scale (float or Tensor): Scale parameter of distribution (lambda).        concentration (float or Tensor): Concentration parameter of distribution (k/shape).        validate_args (bool, optional): Whether to validate arguments. Default: None.

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Weibull`

**Functions defined**: `__init__`, `expand`, `mean`, `mode`, `variance`, `entropy`

**Key imports**: Optional, Union, torch, Tensor, constraints, Exponential, euler_constant, TransformedDistribution, AffineTransform, PowerTransform, broadcast_all


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
- `torch.distributions.exponential`: Exponential
- `torch.distributions.gumbel`: euler_constant
- `torch.distributions.transformed_distribution`: TransformedDistribution
- `torch.distributions.transforms`: AffineTransform, PowerTransform
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

- **File Documentation**: `weibull.py_docs.md`
- **Keyword Index**: `weibull.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
