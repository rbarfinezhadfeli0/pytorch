# Documentation: `docs/torch/distributions/pareto.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/pareto.py_docs.md`
- **Size**: 5,849 bytes (5.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/pareto.py`

## File Metadata

- **Path**: `torch/distributions/pareto.py`
- **Size**: 2,544 bytes (2.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from typing import Optional, Union

from torch import Tensor
from torch.distributions import constraints
from torch.distributions.exponential import Exponential
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import AffineTransform, ExpTransform
from torch.distributions.utils import broadcast_all
from torch.types import _size


__all__ = ["Pareto"]


class Pareto(TransformedDistribution):
    r"""
    Samples from a Pareto Type 1 distribution.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1
        tensor([ 1.5623])

    Args:
        scale (float or Tensor): Scale parameter of the distribution
        alpha (float or Tensor): Shape parameter of the distribution
    """

    arg_constraints = {"alpha": constraints.positive, "scale": constraints.positive}

    def __init__(
        self,
        scale: Union[Tensor, float],
        alpha: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        self.scale, self.alpha = broadcast_all(scale, alpha)
        base_dist = Exponential(self.alpha, validate_args=validate_args)
        transforms = [ExpTransform(), AffineTransform(loc=0, scale=self.scale)]
        # pyrefly: ignore [bad-argument-type]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    def expand(
        self, batch_shape: _size, _instance: Optional["Pareto"] = None
    ) -> "Pareto":
        new = self._get_checked_instance(Pareto, _instance)
        new.scale = self.scale.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        return super().expand(batch_shape, _instance=new)

    @property
    def mean(self) -> Tensor:
        # mean is inf for alpha <= 1
        a = self.alpha.clamp(min=1)
        return a * self.scale / (a - 1)

    @property
    def mode(self) -> Tensor:
        return self.scale

    @property
    def variance(self) -> Tensor:
        # var is inf for alpha <= 2
        a = self.alpha.clamp(min=2)
        return self.scale.pow(2) * a / ((a - 1).pow(2) * (a - 2))

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self) -> constraints.Constraint:
        return constraints.greater_than_eq(self.scale)

    def entropy(self) -> Tensor:
        return (self.scale / self.alpha).log() + (1 + self.alpha.reciprocal())

```



## High-Level Overview

r"""    Samples from a Pareto Type 1 distribution.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Pareto(torch.tensor([1.0]), torch.tensor([1.0]))        >>> m.sample()  # sample from a Pareto distribution with scale=1 and alpha=1        tensor([ 1.5623])    Args:        scale (float or Tensor): Scale parameter of the distribution        alpha (float or Tensor): Shape parameter of the distribution

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Pareto`

**Functions defined**: `__init__`, `expand`, `mean`, `mode`, `variance`, `support`, `entropy`

**Key imports**: Optional, Union, Tensor, constraints, Exponential, TransformedDistribution, AffineTransform, ExpTransform, broadcast_all, _size


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional, Union
- `torch`: Tensor
- `torch.distributions`: constraints
- `torch.distributions.exponential`: Exponential
- `torch.distributions.transformed_distribution`: TransformedDistribution
- `torch.distributions.transforms`: AffineTransform, ExpTransform
- `torch.distributions.utils`: broadcast_all
- `torch.types`: _size


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

- **File Documentation**: `pareto.py_docs.md`
- **Keyword Index**: `pareto.py_kw.md`
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
- [`binomial.py_docs.md_docs.md`](./binomial.py_docs.md_docs.md)
- [`half_cauchy.py_docs.md_docs.md`](./half_cauchy.py_docs.md_docs.md)
- [`one_hot_categorical.py_docs.md_docs.md`](./one_hot_categorical.py_docs.md_docs.md)
- [`geometric.py_kw.md_docs.md`](./geometric.py_kw.md_docs.md)
- [`kumaraswamy.py_kw.md_docs.md`](./kumaraswamy.py_kw.md_docs.md)
- [`transformed_distribution.py_kw.md_docs.md`](./transformed_distribution.py_kw.md_docs.md)
- [`log_normal.py_docs.md_docs.md`](./log_normal.py_docs.md_docs.md)
- [`kumaraswamy.py_docs.md_docs.md`](./kumaraswamy.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `pareto.py_docs.md_docs.md`
- **Keyword Index**: `pareto.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
