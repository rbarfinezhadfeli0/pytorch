# Documentation: `docs/torch/distributions/utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributions/utils.py_docs.md`
- **Size**: 11,896 bytes (11.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/distributions/utils.py`

## File Metadata

- **Path**: `torch/distributions/utils.py`
- **Size**: 8,027 bytes (7.84 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Callable, Sequence
from functools import update_wrapper
from typing import Any, Final, Generic, Optional, overload, TypeVar, Union

import torch
import torch.nn.functional as F
from torch import SymInt, Tensor
from torch.overrides import is_tensor_like
from torch.types import _dtype, _Number, Device, Number


euler_constant: Final[float] = 0.57721566490153286060  # Euler Mascheroni Constant

__all__ = [
    "broadcast_all",
    "logits_to_probs",
    "clamp_probs",
    "probs_to_logits",
    "lazy_property",
    "tril_matrix_to_vec",
    "vec_to_tril_matrix",
]


# FIXME: Use (*values: *Ts) -> tuple[Tensor for T in Ts] if Mapping-Type is ever added.
#   See https://github.com/python/typing/issues/1216#issuecomment-2126153831
def broadcast_all(*values: Union[Tensor, Number]) -> tuple[Tensor, ...]:
    r"""
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `Number`, `torch.*Tensor` or objects implementing __torch_function__)

    Raises:
        ValueError: if any of the values is not a `Number` instance,
            a `torch.*Tensor` instance, or an instance implementing __torch_function__
    """
    if not all(is_tensor_like(v) or isinstance(v, _Number) for v in values):
        raise ValueError(
            "Input arguments must all be instances of Number, "
            "torch.Tensor or objects implementing __torch_function__."
        )
    if not all(is_tensor_like(v) for v in values):
        options: dict[str, Any] = dict(dtype=torch.get_default_dtype())
        for value in values:
            if isinstance(value, torch.Tensor):
                options = dict(dtype=value.dtype, device=value.device)
                break
        new_values = [
            v if is_tensor_like(v) else torch.tensor(v, **options) for v in values
        ]
        return torch.broadcast_tensors(*new_values)
    return torch.broadcast_tensors(*values)


def _standard_normal(
    shape: Sequence[Union[int, SymInt]],
    dtype: Optional[_dtype],
    device: Optional[Device],
) -> Tensor:
    if torch._C._get_tracing_state():
        # [JIT WORKAROUND] lack of support for .normal_()
        return torch.normal(
            torch.zeros(shape, dtype=dtype, device=device),
            torch.ones(shape, dtype=dtype, device=device),
        )
    return torch.empty(shape, dtype=dtype, device=device).normal_()


def _sum_rightmost(value: Tensor, dim: int) -> Tensor:
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


def logits_to_probs(logits: Tensor, is_binary: bool = False) -> Tensor:
    r"""
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)


def clamp_probs(probs: Tensor) -> Tensor:
    """Clamps the probabilities to be in the open interval `(0, 1)`.

    The probabilities would be clamped between `eps` and `1 - eps`,
    and `eps` would be the smallest representable positive number for the input data type.

    Args:
        probs (Tensor): A tensor of probabilities.

    Returns:
        Tensor: The clamped probabilities.

    Examples:
        >>> probs = torch.tensor([0.0, 0.5, 1.0])
        >>> clamp_probs(probs)
        tensor([1.1921e-07, 5.0000e-01, 1.0000e+00])

        >>> probs = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        >>> clamp_probs(probs)
        tensor([2.2204e-16, 5.0000e-01, 1.0000e+00], dtype=torch.float64)

    """
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)


def probs_to_logits(probs: Tensor, is_binary: bool = False) -> Tensor:
    r"""
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return torch.log(ps_clamped) - torch.log1p(-ps_clamped)
    return torch.log(ps_clamped)


T = TypeVar("T", contravariant=True)
R = TypeVar("R", covariant=True)


class lazy_property(Generic[T, R]):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """

    def __init__(self, wrapped: Callable[[T], R]) -> None:
        self.wrapped: Callable[[T], R] = wrapped
        update_wrapper(self, wrapped)  # type:ignore[arg-type]

    @overload
    def __get__(
        self, instance: None, obj_type: Any = None
    ) -> "_lazy_property_and_property[T, R]": ...

    @overload
    def __get__(self, instance: T, obj_type: Any = None) -> R: ...

    def __get__(
        self, instance: Union[T, None], obj_type: Any = None
    ) -> "R | _lazy_property_and_property[T, R]":
        if instance is None:
            return _lazy_property_and_property(self.wrapped)
        with torch.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


class _lazy_property_and_property(lazy_property[T, R], property):
    """We want lazy properties to look like multiple things.

    * property when Sphinx autodoc looks
    * lazy_property when Distribution validate_args looks
    """

    def __init__(self, wrapped: Callable[[T], R]) -> None:
        property.__init__(self, wrapped)


def tril_matrix_to_vec(mat: Tensor, diag: int = 0) -> Tensor:
    r"""
    Convert a `D x D` matrix or a batch of matrices into a (batched) vector
    which comprises of lower triangular elements from the matrix in row order.
    """
    n = mat.shape[-1]
    if not torch._C._get_tracing_state() and (diag < -n or diag >= n):
        raise ValueError(f"diag ({diag}) provided is outside [{-n}, {n - 1}].")
    arange = torch.arange(n, device=mat.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    vec = mat[..., tril_mask]
    return vec


def vec_to_tril_matrix(vec: Tensor, diag: int = 0) -> Tensor:
    r"""
    Convert a vector or a batch of vectors into a batched `D x D`
    lower triangular matrix containing elements from the vector in row order.
    """
    # +ve root of D**2 + (1+2*diag)*D - |diag| * (diag+1) - 2*vec.shape[-1] = 0
    n = (
        -(1 + 2 * diag)
        + ((1 + 2 * diag) ** 2 + 8 * vec.shape[-1] + 4 * abs(diag) * (diag + 1)) ** 0.5
    ) / 2
    eps = torch.finfo(vec.dtype).eps
    if not torch._C._get_tracing_state() and (round(n) - n > eps):
        raise ValueError(
            f"The size of last dimension is {vec.shape[-1]} which cannot be expressed as "
            + "the lower triangular part of a square D x D matrix."
        )
    n = round(n.item()) if isinstance(n, torch.Tensor) else round(n)
    mat = vec.new_zeros(vec.shape[:-1] + torch.Size((n, n)))
    arange = torch.arange(n, device=vec.device)
    tril_mask = arange < arange.view(-1, 1) + (diag + 1)
    mat[..., tril_mask] = vec
    return mat

```



## High-Level Overview

r"""    Given a list of values (possibly containing numbers), returns a list where each    value is broadcasted based on the following rules:      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.      - Number instances (scalars) are upcast to tensors having        the same size and type as the first tensor passed to `values`.  If all the        values are scalars, then they are upcasted to scalar Tensors.    Args:        values (list of `Number`, `torch.*Tensor` or objects implementing __torch_function__)    Raises:        ValueError: if any of the values is not a `Number` instance,            a `torch.*Tensor` instance, or an instance implementing __torch_function__

This Python file contains 3 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `lazy_property`, `_lazy_property_and_property`

**Functions defined**: `broadcast_all`, `_standard_normal`, `_sum_rightmost`, `logits_to_probs`, `clamp_probs`, `probs_to_logits`, `__init__`, `__get__`, `__get__`, `__get__`, `__init__`, `tril_matrix_to_vec`, `vec_to_tril_matrix`

**Key imports**: Callable, Sequence, update_wrapper, Any, Final, Generic, Optional, overload, TypeVar, Union, torch, torch.nn.functional as F, SymInt, Tensor, is_tensor_like, _dtype, _Number, Device, Number


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable, Sequence
- `functools`: update_wrapper
- `typing`: Any, Final, Generic, Optional, overload, TypeVar, Union
- `torch`
- `torch.nn.functional as F`
- `torch.overrides`: is_tensor_like
- `torch.types`: _dtype, _Number, Device, Number


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `utils.py_docs.md`
- **Keyword Index**: `utils.py_kw.md`
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
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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

- **File Documentation**: `utils.py_docs.md_docs.md`
- **Keyword Index**: `utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
