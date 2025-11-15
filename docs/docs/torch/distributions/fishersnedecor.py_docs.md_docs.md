# Documentation: `docs/torch/distributions/fishersnedecor.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/fishersnedecor.py_docs.md`
- **Size**: 6,960 bytes (6.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/fishersnedecor.py`

## File Metadata

- **Path**: `torch/distributions/fishersnedecor.py`
- **Size**: 3,715 bytes (3.63 KB)
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
from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.utils import broadcast_all
from torch.types import _Number, _size


__all__ = ["FisherSnedecor"]


class FisherSnedecor(Distribution):
    r"""
    Creates a Fisher-Snedecor distribution parameterized by :attr:`df1` and :attr:`df2`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))
        >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2
        tensor([ 0.2453])

    Args:
        df1 (float or Tensor): degrees of freedom parameter 1
        df2 (float or Tensor): degrees of freedom parameter 2
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {"df1": constraints.positive, "df2": constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        df1: Union[Tensor, float],
        df2: Union[Tensor, float],
        validate_args: Optional[bool] = None,
    ) -> None:
        self.df1, self.df2 = broadcast_all(df1, df2)
        self._gamma1 = Gamma(self.df1 * 0.5, self.df1)
        self._gamma2 = Gamma(self.df2 * 0.5, self.df2)

        if isinstance(df1, _Number) and isinstance(df2, _Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.df1.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(FisherSnedecor, _instance)
        batch_shape = torch.Size(batch_shape)
        new.df1 = self.df1.expand(batch_shape)
        new.df2 = self.df2.expand(batch_shape)
        new._gamma1 = self._gamma1.expand(batch_shape)
        new._gamma2 = self._gamma2.expand(batch_shape)
        super(FisherSnedecor, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self) -> Tensor:
        df2 = self.df2.clone(memory_format=torch.contiguous_format)
        df2[df2 <= 2] = nan
        return df2 / (df2 - 2)

    @property
    def mode(self) -> Tensor:
        mode = (self.df1 - 2) / self.df1 * self.df2 / (self.df2 + 2)
        mode[self.df1 <= 2] = nan
        return mode

    @property
    def variance(self) -> Tensor:
        df2 = self.df2.clone(memory_format=torch.contiguous_format)
        df2[df2 <= 4] = nan
        return (
            2
            * df2.pow(2)
            * (self.df1 + df2 - 2)
            / (self.df1 * (df2 - 2).pow(2) * (df2 - 4))
        )

    def rsample(self, sample_shape: _size = torch.Size(())) -> Tensor:
        shape = self._extended_shape(sample_shape)
        #   X1 ~ Gamma(df1 / 2, 1 / df1), X2 ~ Gamma(df2 / 2, 1 / df2)
        #   Y = df2 * df1 * X1 / (df1 * df2 * X2) = X1 / X2 ~ F(df1, df2)
        X1 = self._gamma1.rsample(sample_shape).view(shape)
        X2 = self._gamma2.rsample(sample_shape).view(shape)
        tiny = torch.finfo(X2.dtype).tiny
        X2.clamp_(min=tiny)
        Y = X1 / X2
        Y.clamp_(min=tiny)
        return Y

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        ct1 = self.df1 * 0.5
        ct2 = self.df2 * 0.5
        ct3 = self.df1 / self.df2
        t1 = (ct1 + ct2).lgamma() - ct1.lgamma() - ct2.lgamma()
        t2 = ct1 * ct3.log() + (ct1 - 1) * torch.log(value)
        t3 = (ct1 + ct2) * torch.log1p(ct3 * value)
        return t1 + t2 - t3

```



## High-Level Overview

r"""    Creates a Fisher-Snedecor distribution parameterized by :attr:`df1` and :attr:`df2`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = FisherSnedecor(torch.tensor([1.0]), torch.tensor([2.0]))        >>> m.sample()  # Fisher-Snedecor-distributed with df1=1 and df2=2        tensor([ 0.2453])    Args:        df1 (float or Tensor): degrees of freedom parameter 1        df2 (float or Tensor): degrees of freedom parameter 2

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FisherSnedecor`

**Functions defined**: `__init__`, `expand`, `mean`, `mode`, `variance`, `rsample`, `log_prob`

**Key imports**: Optional, Union, torch, nan, Tensor, constraints, Distribution, Gamma, broadcast_all, _Number, _size


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
- `torch.distributions.distribution`: Distribution
- `torch.distributions.gamma`: Gamma
- `torch.distributions.utils`: broadcast_all
- `torch.types`: _Number, _size


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

- **File Documentation**: `fishersnedecor.py_docs.md`
- **Keyword Index**: `fishersnedecor.py_kw.md`
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

- **File Documentation**: `fishersnedecor.py_docs.md_docs.md`
- **Keyword Index**: `fishersnedecor.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
