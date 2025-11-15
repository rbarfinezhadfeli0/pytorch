# Documentation: `docs/torch/distributions/kumaraswamy.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/kumaraswamy.py_docs.md`
- **Size**: 6,692 bytes (6.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/kumaraswamy.py`

## File Metadata

- **Path**: `torch/distributions/kumaraswamy.py`
- **Size**: 3,735 bytes (3.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional, Union

import torch
from torch import nan, Tensor
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, PowerTransform
from torch.distributions.uniform import Uniform
from torch.distributions.utils import broadcast_all, euler_constant


__all__ = ["Kumaraswamy"]


def _moments(a, b, n):
    """
    Computes nth moment of Kumaraswamy using using torch.lgamma
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)


class Kumaraswamy(TransformedDistribution):
    r"""
    Samples from a Kumaraswamy distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Kumaraswamy(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Kumaraswamy distribution with concentration alpha=1 and beta=1
        tensor([ 0.1729])

    Args:
        concentration1 (float or Tensor): 1st concentration parameter of the distribution
            (often referred to as alpha)
        concentration0 (float or Tensor): 2nd concentration parameter of the distribution
            (often referred to as beta)
    """

    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    # pyrefly: ignore [bad-override]
    support = constraints.unit_interval
    has_rsample = True

    def __init__(
        self,
        concentration1: Union[Tensor, float],
        concentration0: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        self.concentration1, self.concentration0 = broadcast_all(
            concentration1, concentration0
        )
        base_dist = Uniform(
            torch.full_like(self.concentration0, 0),
            torch.full_like(self.concentration0, 1),
            validate_args=validate_args,
        )
        transforms = [
            PowerTransform(exponent=self.concentration0.reciprocal()),
            AffineTransform(loc=1.0, scale=-1.0),
            PowerTransform(exponent=self.concentration1.reciprocal()),
        ]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Kumaraswamy, _instance)
        new.concentration1 = self.concentration1.expand(batch_shape)
        new.concentration0 = self.concentration0.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self) -> Tensor:
        return _moments(self.concentration1, self.concentration0, 1)

    @property
    def mode(self) -> Tensor:
        # Evaluate in log-space for numerical stability.
        log_mode = (
            self.concentration0.reciprocal() * (-self.concentration0).log1p()
            - (-self.concentration0 * self.concentration1).log1p()
        )
        log_mode[(self.concentration0 < 1) | (self.concentration1 < 1)] = nan
        return log_mode.exp()

    @property
    def variance(self) -> Tensor:
        return _moments(self.concentration1, self.concentration0, 2) - torch.pow(
            self.mean, 2
        )

    def entropy(self):
        t1 = 1 - self.concentration1.reciprocal()
        t0 = 1 - self.concentration0.reciprocal()
        H0 = torch.digamma(self.concentration0 + 1) + euler_constant
        return (
            t0
            + t1 * H0
            - torch.log(self.concentration1)
            - torch.log(self.concentration0)
        )

```



## High-Level Overview

"""    Computes nth moment of Kumaraswamy using using torch.lgamma

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Kumaraswamy`

**Functions defined**: `_moments`, `__init__`, `expand`, `mean`, `mode`, `variance`, `entropy`

**Key imports**: Optional, Union, torch, nan, Tensor, constraints, TransformedDistribution, AffineTransform, PowerTransform, Uniform, broadcast_all, euler_constant


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
- `torch.distributions.transformed_distribution`: TransformedDistribution
- `torch.distributions.transforms`: AffineTransform, PowerTransform
- `torch.distributions.uniform`: Uniform
- `torch.distributions.utils`: broadcast_all, euler_constant


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

- **File Documentation**: `kumaraswamy.py_docs.md`
- **Keyword Index**: `kumaraswamy.py_kw.md`
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


## Cross-References

- **File Documentation**: `kumaraswamy.py_docs.md_docs.md`
- **Keyword Index**: `kumaraswamy.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
