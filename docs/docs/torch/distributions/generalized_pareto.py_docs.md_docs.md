# Documentation: `docs/torch/distributions/generalized_pareto.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/generalized_pareto.py_docs.md`
- **Size**: 9,799 bytes (9.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/generalized_pareto.py`

## File Metadata

- **Path**: `torch/distributions/generalized_pareto.py`
- **Size**: 5,881 bytes (5.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import math
from numbers import Number, Real

import torch
from torch import inf, nan
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all


__all__ = ["GeneralizedPareto"]


class GeneralizedPareto(Distribution):
    r"""
    Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.

    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.
    Special cases include Exponential (when :attr:`loc` = 0, :attr:`concentration` = 0), Pareto (when :attr:`concentration` > 0,
    :attr:`loc` = :attr:`scale` / :attr:`concentration`), and Uniform (when :attr:`concentration` = -1).

    This distribution is often used to model the tails of other distributions. This implementation is based on the
    implementation in TensorFlow Probability.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = GeneralizedPareto(torch.tensor([0.1]), torch.tensor([2.0]), torch.tensor([0.4]))
        >>> m.sample()  # sample from a Generalized Pareto distribution with loc=0.1, scale=2.0, and concentration=0.4
        tensor([ 1.5623])

    Args:
        loc (float or Tensor): Location parameter of the distribution
        scale (float or Tensor): Scale parameter of the distribution
        concentration (float or Tensor): Concentration parameter of the distribution
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.positive,
        "concentration": constraints.real,
    }
    has_rsample = True

    def __init__(self, loc, scale, concentration, validate_args=None):
        self.loc, self.scale, self.concentration = broadcast_all(
            loc, scale, concentration
        )
        if (
            isinstance(loc, Number)
            and isinstance(scale, Number)
            and isinstance(concentration, Number)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GeneralizedPareto, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        super(GeneralizedPareto, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = torch.rand(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.icdf(u)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        y = 1 / safe_conc + torch.ones_like(z)
        where_nonzero = torch.where(y == 0, y, y * torch.log1p(safe_conc * z))
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        return -log_scale - torch.where(eq_zero, z, where_nonzero)

    def log_survival_function(self, value):
        if self._validate_args:
            self._validate_sample(value)
        z = self._z(value)
        eq_zero = torch.isclose(self.concentration, torch.tensor(0.0))
        safe_conc = torch.where(
            eq_zero, torch.ones_like(self.concentration), self.concentration
        )
        where_nonzero = -torch.log1p(safe_conc * z) / safe_conc
        return torch.where(eq_zero, -z, where_nonzero)

    def log_cdf(self, value):
        return torch.log1p(-torch.exp(self.log_survival_function(value)))

    def cdf(self, value):
        return torch.exp(self.log_cdf(value))

    def icdf(self, value):
        loc = self.loc
        scale = self.scale
        concentration = self.concentration
        eq_zero = torch.isclose(concentration, torch.zeros_like(concentration))
        safe_conc = torch.where(eq_zero, torch.ones_like(concentration), concentration)
        logu = torch.log1p(-value)
        where_nonzero = loc + scale / safe_conc * torch.expm1(-safe_conc * logu)
        where_zero = loc - scale * logu
        return torch.where(eq_zero, where_zero, where_nonzero)

    def _z(self, x):
        return (x - self.loc) / self.scale

    @property
    def mean(self):
        concentration = self.concentration
        valid = concentration < 1
        safe_conc = torch.where(valid, concentration, 0.5)
        result = self.loc + self.scale / (1 - safe_conc)
        return torch.where(valid, result, nan)

    @property
    def variance(self):
        concentration = self.concentration
        valid = concentration < 0.5
        safe_conc = torch.where(valid, concentration, 0.25)
        # pyrefly: ignore [unsupported-operation]
        result = self.scale**2 / ((1 - safe_conc) ** 2 * (1 - 2 * safe_conc))
        return torch.where(valid, result, nan)

    def entropy(self):
        ans = torch.log(self.scale) + self.concentration + 1
        return torch.broadcast_to(ans, self._batch_shape)

    @property
    def mode(self):
        return self.loc

    @constraints.dependent_property(is_discrete=False, event_dim=0)
    # pyrefly: ignore [bad-override]
    def support(self):
        lower = self.loc
        upper = torch.where(
            self.concentration < 0, lower - self.scale / self.concentration, inf
        )
        return constraints.interval(lower, upper)

```



## High-Level Overview

r"""    Creates a Generalized Pareto distribution parameterized by :attr:`loc`, :attr:`scale`, and :attr:`concentration`.    The Generalized Pareto distribution is a family of continuous probability distributions on the real line.    Special cases include Exponential (when :attr:`loc` = 0, :attr:`concentration` = 0), Pareto (when :attr:`concentration` > 0,    :attr:`loc` = :attr:`scale` / :attr:`concentration`), and Uniform (when :attr:`concentration` = -1).    This distribution is often used to model the tails of other distributions. This implementation is based on the    implementation in TensorFlow Probability.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = GeneralizedPareto(torch.tensor([0.1]), torch.tensor([2.0]), torch.tensor([0.4]))        >>> m.sample()  # sample from a Generalized Pareto distribution with loc=0.1, scale=2.0, and concentration=0.4        tensor([ 1.5623])    Args:        loc (float or Tensor): Location parameter of the distribution        scale (float or Tensor): Scale parameter of the distribution        concentration (float or Tensor): Concentration parameter of the distribution

This Python file contains 1 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `GeneralizedPareto`

**Functions defined**: `__init__`, `expand`, `rsample`, `log_prob`, `log_survival_function`, `log_cdf`, `cdf`, `icdf`, `_z`, `mean`, `variance`, `entropy`, `mode`, `support`

**Key imports**: math, Number, Real, torch, inf, nan, constraints, Distribution, broadcast_all


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `numbers`: Number, Real
- `torch`
- `torch.distributions`: constraints, Distribution
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

- **File Documentation**: `generalized_pareto.py_docs.md`
- **Keyword Index**: `generalized_pareto.py_kw.md`
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

- **File Documentation**: `generalized_pareto.py_docs.md_docs.md`
- **Keyword Index**: `generalized_pareto.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
