# Documentation: `docs/torch/_refs/special/__init__.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_refs/special/__init__.py_docs.md`
- **Size**: 9,228 bytes (9.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This is a **Python package initialization file**.

## Original Source

```markdown
# Documentation: `torch/_refs/special/__init__.py`

## File Metadata

- **Path**: `torch/_refs/special/__init__.py`
- **Size**: 6,916 bytes (6.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a **Python package initialization file**.

## Original Source

```python
# mypy: allow-untyped-defs
import math
from typing import Optional, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number,
    NumberType,
    TensorLike,
    TensorLikeType,
)
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper
from torch._refs import (
    _make_alias,
    _make_elementwise_binary_reference,
    _make_elementwise_unary_reference,
)


__all__ = [
    "bessel_j0",
    "bessel_j1",
    "entr",
    "erfcx",
    "expit",
    "i0e",
    "i1",
    "i1e",
    "log_ndtr",
    "logit",
    "log_softmax",
    "multigammaln",
    "ndtr",
    "ndtri",
    "softmax",
    "spherical_bessel_j0",
    "xlog1py",
    "zeta",
]
aten = torch._ops.ops.aten


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def bessel_j0(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_j0(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def bessel_j1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_j1(a)


@register_decomposition(aten.special_entr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def entr(a: TensorLikeType) -> TensorLikeType:
    return torch.where(
        torch.isnan(a),
        a,
        torch.where(a > 0, -a * torch.log(a), torch.where(a == 0, 0, -torch.inf)),
    )


@register_decomposition(aten.special_erfcx)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def erfcx(a: TensorLikeType) -> TensorLikeType:
    return prims.erfcx(a)


# alias for sigmoid
expit = _make_alias(torch.sigmoid, "expit")


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i0e(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i0e(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1(a)


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i1e(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1e(a)


@register_decomposition(aten.special_log_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def log_ndtr(a: TensorLikeType) -> TensorLikeType:
    # Note: M_SQRT1_2 is the value of 1 / sqrt(2)
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    t = a * M_SQRT1_2
    return torch.where(
        a < 1.0,
        torch.log(torch.special.erfcx(-t) / 2) - t * t,
        torch.log1p(-torch.erfc(t) / 2),
    )


@register_decomposition(aten.logit)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def logit(self: TensorLikeType, eps: Optional[float] = None) -> TensorLikeType:
    if eps is None:
        eps = -1.0
    lo = eps
    hi = 1 - eps
    self = torch.where(self < lo, lo, torch.where(self > hi, hi, self))
    return torch.log(torch.true_divide(self, torch.sub(1, self)))


@register_decomposition(aten.special_xlog1py)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def xlog1py(a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]):
    torch._check(
        isinstance(a, TensorLike) or isinstance(b, TensorLike),
        lambda: 'Expected either argument a or b to be a Tensor"',
    )

    # Operations like eq and log do not handle scalar values, so we convert them to scalar_tensors.
    if isinstance(a, TensorLike) and isinstance(b, Number):
        # pyrefly: ignore [bad-argument-type]
        b = refs.scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        # pyrefly: ignore [bad-argument-type]
        a = refs.scalar_tensor(a, dtype=b.dtype, device=b.device)

    # mypy: expected "Tensor"
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)
    rhs = torch.where(torch.eq(a, 0), 0, torch.mul(a, torch.log1p(b)))
    return torch.where(torch.isnan(b), float("nan"), rhs)


@register_decomposition(aten.mvlgamma)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def multigammaln(a: TensorLikeType, p: int) -> TensorLikeType:
    c = 0.25 * p * (p - 1) * math.log(math.pi)
    b = 0.5 * torch.arange(start=(1 - p), end=1, step=1, dtype=a.dtype, device=a.device)
    return torch.sum(torch.lgamma(a.unsqueeze(-1) + b), dim=-1) + c


@register_decomposition(aten.special_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtr(a: TensorLikeType) -> TensorLikeType:
    # Note: M_SQRT1_2 is the value of 1 / sqrt(2)
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    a_sqrt_2 = a * M_SQRT1_2
    return (1 + torch.erf(a_sqrt_2)) * 0.5


@register_decomposition(aten.special_ndtri)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def ndtri(a: TensorLikeType) -> TensorLikeType:
    return prims.ndtri(a)


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def log_softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    return torch.log_softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
def softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    return torch.softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def spherical_bessel_j0(a: TensorLikeType) -> TensorLikeType:
    return prims.spherical_bessel_j0(a)


# TODO: add docstring
@_make_elementwise_binary_reference(
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def zeta(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.zeta(a, b)

```



## High-Level Overview


This Python file contains 0 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `bessel_j0`, `bessel_j1`, `entr`, `erfcx`, `i0e`, `i1`, `i1e`, `log_ndtr`, `logit`, `xlog1py`, `multigammaln`, `ndtr`, `ndtri`, `log_softmax`, `softmax`, `spherical_bessel_j0`, `zeta`

**Key imports**: math, Optional, Union, torch, torch._prims as prims, torch._prims_common as utils, torch._refs as refs, Tensor, register_decomposition, elementwise_type_promotion_wrapper, out_wrapper


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_refs/special`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `typing`: Optional, Union
- `torch`
- `torch._prims as prims`
- `torch._prims_common as utils`
- `torch._refs as refs`
- `torch._decomp`: register_decomposition
- `torch._prims_common.wrappers`: elementwise_type_promotion_wrapper, out_wrapper


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/_refs/special`):



## Cross-References

- **File Documentation**: `__init__.py_docs.md`
- **Keyword Index**: `__init__.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_refs/special`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_refs/special`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/torch/_refs/special`):

- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `__init__.py_docs.md_docs.md`
- **Keyword Index**: `__init__.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
