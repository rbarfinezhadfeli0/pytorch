# Documentation: `torch/distributions/dirichlet.py`

## File Metadata

- **Path**: `torch/distributions/dirichlet.py`
- **Size**: 4,561 bytes (4.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from typing import Optional

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily
from torch.types import _size


__all__ = ["Dirichlet"]


# This helper is exposed for testing.
def _Dirichlet_backward(x, concentration, grad_output):
    total = concentration.sum(-1, True).expand_as(concentration)
    grad = torch._dirichlet_grad(x, concentration, total)
    return grad * (grad_output - (x * grad_output).sum(-1, True))


class _Dirichlet(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, concentration):
        x = torch._sample_dirichlet(concentration)
        ctx.save_for_backward(x, concentration)
        return x

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        x, concentration = ctx.saved_tensors
        return _Dirichlet_backward(x, concentration, grad_output)


class Dirichlet(ExponentialFamily):
    r"""
    Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))
        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]
        tensor([ 0.1046,  0.8954])

    Args:
        concentration (Tensor): concentration parameter of the distribution
            (often referred to as alpha)
    """

    # pyrefly: ignore [bad-override]
    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1)
    }
    support = constraints.simplex
    has_rsample = True

    def __init__(
        self,
        concentration: Tensor,
        validate_args: Optional[bool] = None,
    ) -> None:
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` parameter must be at least one-dimensional."
            )
        self.concentration = concentration
        batch_shape, event_shape = concentration.shape[:-1], concentration.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Dirichlet, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape + self.event_shape)
        super(Dirichlet, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape: _size = ()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        concentration = self.concentration.expand(shape)
        return _Dirichlet.apply(concentration)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return (
            torch.xlogy(self.concentration - 1.0, value).sum(-1)
            + torch.lgamma(self.concentration.sum(-1))
            - torch.lgamma(self.concentration).sum(-1)
        )

    @property
    def mean(self) -> Tensor:
        return self.concentration / self.concentration.sum(-1, True)

    @property
    def mode(self) -> Tensor:
        concentrationm1 = (self.concentration - 1).clamp(min=0.0)
        mode = concentrationm1 / concentrationm1.sum(-1, True)
        mask = (self.concentration < 1).all(dim=-1)
        mode[mask] = torch.nn.functional.one_hot(
            mode[mask].argmax(dim=-1), concentrationm1.shape[-1]
        ).to(mode)
        return mode

    @property
    def variance(self) -> Tensor:
        con0 = self.concentration.sum(-1, True)
        return (
            self.concentration
            * (con0 - self.concentration)
            / (con0.pow(2) * (con0 + 1))
        )

    def entropy(self):
        k = self.concentration.size(-1)
        a0 = self.concentration.sum(-1)
        return (
            torch.lgamma(self.concentration).sum(-1)
            - torch.lgamma(a0)
            - (k - a0) * torch.digamma(a0)
            - ((self.concentration - 1.0) * torch.digamma(self.concentration)).sum(-1)
        )

    @property
    def _natural_params(self) -> tuple[Tensor]:
        return (self.concentration,)

    # pyrefly: ignore [bad-override]
    def _log_normalizer(self, x):
        return x.lgamma().sum(-1) - torch.lgamma(x.sum(-1))

```



## High-Level Overview

r"""    Creates a Dirichlet distribution parameterized by concentration :attr:`concentration`.    Example::        >>> # xdoctest: +IGNORE_WANT("non-deterministic")        >>> m = Dirichlet(torch.tensor([0.5, 0.5]))        >>> m.sample()  # Dirichlet distributed with concentration [0.5, 0.5]        tensor([ 0.1046,  0.8954])    Args:

This Python file contains 2 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_Dirichlet`, `Dirichlet`

**Functions defined**: `_Dirichlet_backward`, `forward`, `backward`, `__init__`, `expand`, `rsample`, `log_prob`, `mean`, `mode`, `variance`, `entropy`, `_natural_params`, `_log_normalizer`

**Key imports**: Optional, torch, Tensor, Function, once_differentiable, constraints, ExponentialFamily, _size


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`
- `torch.autograd`: Function
- `torch.autograd.function`: once_differentiable
- `torch.distributions`: constraints
- `torch.distributions.exp_family`: ExponentialFamily
- `torch.types`: _size


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


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

- **File Documentation**: `dirichlet.py_docs.md`
- **Keyword Index**: `dirichlet.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
