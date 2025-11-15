# Documentation: `docs/torch/_numpy/fft.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_numpy/fft.py_docs.md`
- **Size**: 6,215 bytes (6.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_numpy/fft.py`

## File Metadata

- **Path**: `torch/_numpy/fft.py`
- **Size**: 2,805 bytes (2.74 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

from __future__ import annotations

import functools

import torch

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, normalizer


def upcast(func):
    """NumPy fft casts inputs to 64 bit and *returns 64-bit results*."""

    @functools.wraps(func)
    def wrapped(tensor, *args, **kwds):
        target_dtype = (
            _dtypes_impl.default_dtypes().complex_dtype
            if tensor.is_complex()
            else _dtypes_impl.default_dtypes().float_dtype
        )
        tensor = _util.cast_if_needed(tensor, target_dtype)
        return func(tensor, *args, **kwds)

    return wrapped


@normalizer
@upcast
def fft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.fft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ifft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.ifft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def rfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.rfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def irfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.irfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def fftn(a: ArrayLike, s=None, axes=None, norm=None):
    return torch.fft.fftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifftn(a: ArrayLike, s=None, axes=None, norm=None):
    return torch.fft.ifftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfftn(a: ArrayLike, s=None, axes=None, norm=None):
    return torch.fft.rfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfftn(a: ArrayLike, s=None, axes=None, norm=None):
    return torch.fft.irfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def fft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return torch.fft.fft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return torch.fft.ifft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return torch.fft.rfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfft2(a: ArrayLike, s=None, axes=(-2, -1), norm=None):
    return torch.fft.irfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def hfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.hfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ihfft(a: ArrayLike, n=None, axis=-1, norm=None):
    return torch.fft.ihfft(a, n, dim=axis, norm=norm)


@normalizer
def fftfreq(n, d=1.0):
    return torch.fft.fftfreq(n, d)


@normalizer
def rfftfreq(n, d=1.0):
    return torch.fft.rfftfreq(n, d)


@normalizer
def fftshift(x: ArrayLike, axes=None):
    return torch.fft.fftshift(x, axes)


@normalizer
def ifftshift(x: ArrayLike, axes=None):
    return torch.fft.ifftshift(x, axes)

```



## High-Level Overview

"""NumPy fft casts inputs to 64 bit and *returns 64-bit results*."""    @functools.wraps(func)    def wrapped(tensor, *args, **kwds):        target_dtype = (            _dtypes_impl.default_dtypes().complex_dtype            if tensor.is_complex()            else _dtypes_impl.default_dtypes().float_dtype        )        tensor = _util.cast_if_needed(tensor, target_dtype)        return func(tensor, *args, **kwds)    return wrapped@normalizer@upcastdef fft(a: ArrayLike, n=None, axis=-1, norm=None):    return torch.fft.fft(a, n, dim=axis, norm=norm)@normalizer@upcastdef ifft(a: ArrayLike, n=None, axis=-1, norm=None):    return torch.fft.ifft(a, n, dim=axis, norm=norm)@normalizer@upcastdef rfft(a: ArrayLike, n=None, axis=-1, norm=None):    return torch.fft.rfft(a, n, dim=axis, norm=norm)@normalizer@upcastdef irfft(a: ArrayLike, n=None, axis=-1, norm=None):    return torch.fft.irfft(a, n, dim=axis, norm=norm)

This Python file contains 0 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `upcast`, `wrapped`, `fft`, `ifft`, `rfft`, `irfft`, `fftn`, `ifftn`, `rfftn`, `irfftn`, `fft2`, `ifft2`, `rfft2`, `irfft2`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`

**Key imports**: annotations, functools, torch, _dtypes_impl, _util, ArrayLike, normalizer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_numpy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `functools`
- `torch`
- `.`: _dtypes_impl, _util
- `._normalizations`: ArrayLike, normalizer


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

Files in the same folder (`torch/_numpy`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_util.py_docs.md`](./_util.py_docs.md)
- [`_funcs_impl.py_docs.md`](./_funcs_impl.py_docs.md)
- [`_ufuncs.py_docs.md`](./_ufuncs.py_docs.md)
- [`linalg.py_docs.md`](./linalg.py_docs.md)
- [`_binary_ufuncs_impl.py_docs.md`](./_binary_ufuncs_impl.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_ndarray.py_docs.md`](./_ndarray.py_docs.md)
- [`random.py_docs.md`](./random.py_docs.md)


## Cross-References

- **File Documentation**: `fft.py_docs.md`
- **Keyword Index**: `fft.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_numpy`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_numpy`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_numpy`):

- [`_funcs_impl.py_kw.md_docs.md`](./_funcs_impl.py_kw.md_docs.md)
- [`_ndarray.py_docs.md_docs.md`](./_ndarray.py_docs.md_docs.md)
- [`_dtypes.py_kw.md_docs.md`](./_dtypes.py_kw.md_docs.md)
- [`_reductions_impl.py_kw.md_docs.md`](./_reductions_impl.py_kw.md_docs.md)
- [`_ufuncs.py_docs.md_docs.md`](./_ufuncs.py_docs.md_docs.md)
- [`fft.py_kw.md_docs.md`](./fft.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`random.py_docs.md_docs.md`](./random.py_docs.md_docs.md)
- [`_dtypes_impl.py_docs.md_docs.md`](./_dtypes_impl.py_docs.md_docs.md)
- [`_unary_ufuncs_impl.py_docs.md_docs.md`](./_unary_ufuncs_impl.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `fft.py_docs.md_docs.md`
- **Keyword Index**: `fft.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
