# Documentation: `docs/torch/masked/maskedtensor/unary.py_docs.md`

## File Metadata

- **Path**: `docs/torch/masked/maskedtensor/unary.py_docs.md`
- **Size**: 6,476 bytes (6.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/masked/maskedtensor/unary.py`

## File Metadata

- **Path**: `torch/masked/maskedtensor/unary.py`
- **Size**: 4,197 bytes (4.10 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

from .core import _map_mt_args_kwargs, _wrap_result


__all__ = []  # type: ignore[var-annotated]


UNARY_NAMES = [
    "abs",
    "absolute",
    "acos",
    "arccos",
    "acosh",
    "arccosh",
    "angle",
    "asin",
    "arcsin",
    "asinh",
    "arcsinh",
    "atan",
    "arctan",
    "atanh",
    "arctanh",
    "bitwise_not",
    "ceil",
    "clamp",
    "clip",
    "conj_physical",
    "cos",
    "cosh",
    "deg2rad",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "fix",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logit",
    "i0",
    "isnan",
    "nan_to_num",
    "neg",
    "negative",
    "positive",
    "pow",
    "rad2deg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sgn",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]

INPLACE_UNARY_NAMES = [
    n + "_"
    for n in (list(set(UNARY_NAMES) - {"angle", "positive", "signbit", "isnan"}))
]

# Explicitly tracking functions we know are currently not supported
# This might be due to missing code gen or because of complex semantics
UNARY_NAMES_UNSUPPORTED = [
    "atan2",
    "arctan2",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "copysign",
    "float_power",
    "fmod",
    "frexp",
    "gradient",
    "imag",
    "ldexp",
    "lerp",
    "logical_not",
    "hypot",
    "igamma",
    "igammac",
    "mvlgamma",
    "nextafter",
    "polygamma",
    "real",
    "remainder",
    "true_divide",
    "xlogy",
]


def _unary_helper(fn, args, kwargs, inplace):
    if len(kwargs) != 0:
        raise ValueError(
            "MaskedTensor unary ops require that len(kwargs) == 0. "
            "If you need support for this, please open an issue on Github."
        )
    for a in args[1:]:
        if torch.is_tensor(a):
            raise TypeError(
                "MaskedTensor unary ops do not support additional Tensor arguments"
            )

    mask_args, _mask_kwargs = _map_mt_args_kwargs(
        args, kwargs, lambda x: x._masked_mask
    )
    data_args, _data_kwargs = _map_mt_args_kwargs(
        args, kwargs, lambda x: x._masked_data
    )

    if args[0].layout == torch.sparse_coo:
        data_args[0] = data_args[0].coalesce()
        s = data_args[0].size()
        i = data_args[0].indices()
        data_args[0] = data_args[0].coalesce().values()
        v = fn(*data_args)
        result_data = torch.sparse_coo_tensor(i, v, size=s)

    elif args[0].layout == torch.sparse_csr:
        crow = data_args[0].crow_indices()
        col = data_args[0].col_indices()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_csr_tensor(crow, col, v)

    else:
        result_data = fn(*data_args)

    if inplace:
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    else:
        return _wrap_result(result_data, mask_args[0])


def _torch_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def unary_fn(*args, **kwargs):
        return _unary_helper(fn, args, kwargs, inplace=False)

    return unary_fn


def _torch_inplace_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    def unary_fn(*args, **kwargs):
        return _unary_helper(fn, args, kwargs, inplace=True)

    return unary_fn


NATIVE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_unary(name) for name in UNARY_NAMES
}
NATIVE_INPLACE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_inplace_unary(name)
    for name in INPLACE_UNARY_NAMES
}

NATIVE_UNARY_FNS = list(NATIVE_UNARY_MAP.keys())
NATIVE_INPLACE_UNARY_FNS = list(NATIVE_INPLACE_UNARY_MAP.keys())


def _is_native_unary(fn):
    return fn in NATIVE_UNARY_FNS or fn in NATIVE_INPLACE_UNARY_FNS


def _apply_native_unary(fn, *args, **kwargs):
    if fn in NATIVE_UNARY_FNS:
        return NATIVE_UNARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_UNARY_FNS:
        return NATIVE_INPLACE_UNARY_MAP[fn](*args, **kwargs)
    return NotImplemented

```



## High-Level Overview


This Python file contains 0 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_unary_helper`, `_torch_unary`, `unary_fn`, `_torch_inplace_unary`, `unary_fn`, `_is_native_unary`, `_apply_native_unary`

**Key imports**: torch, _map_mt_args_kwargs, _wrap_result


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/masked/maskedtensor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `.core`: _map_mt_args_kwargs, _wrap_result


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

Files in the same folder (`torch/masked/maskedtensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`core.py_docs.md`](./core.py_docs.md)
- [`passthrough.py_docs.md`](./passthrough.py_docs.md)
- [`creation.py_docs.md`](./creation.py_docs.md)
- [`binary.py_docs.md`](./binary.py_docs.md)
- [`_ops_refs.py_docs.md`](./_ops_refs.py_docs.md)
- [`reductions.py_docs.md`](./reductions.py_docs.md)


## Cross-References

- **File Documentation**: `unary.py_docs.md`
- **Keyword Index**: `unary.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/masked/maskedtensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/masked/maskedtensor`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/masked/maskedtensor`):

- [`passthrough.py_kw.md_docs.md`](./passthrough.py_kw.md_docs.md)
- [`binary.py_docs.md_docs.md`](./binary.py_docs.md_docs.md)
- [`core.py_docs.md_docs.md`](./core.py_docs.md_docs.md)
- [`reductions.py_kw.md_docs.md`](./reductions.py_kw.md_docs.md)
- [`_ops_refs.py_kw.md_docs.md`](./_ops_refs.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`_ops_refs.py_docs.md_docs.md`](./_ops_refs.py_docs.md_docs.md)
- [`core.py_kw.md_docs.md`](./core.py_kw.md_docs.md)
- [`unary.py_kw.md_docs.md`](./unary.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `unary.py_docs.md_docs.md`
- **Keyword Index**: `unary.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
