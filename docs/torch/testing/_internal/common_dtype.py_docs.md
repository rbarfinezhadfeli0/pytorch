# Documentation: common_dtype.py

## File Metadata
- **Path**: `torch/testing/_internal/common_dtype.py`
- **Size**: 5106 bytes
- **Lines**: 227
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: ignore-errors


import torch


# Functions and classes for describing the dtypes a function supports
# NOTE: these helpers should correspond to PyTorch's C++ dispatch macros


# Verifies each given dtype is a torch.dtype
def _validate_dtypes(*dtypes):
    for dtype in dtypes:
        assert isinstance(dtype, torch.dtype)
    return dtypes


# class for tuples corresponding to a PyTorch dispatch macro
class _dispatch_dtypes(tuple):
    __slots__ = ()

    def __add__(self, other):
        assert isinstance(other, tuple)
        return _dispatch_dtypes(tuple.__add__(self, other))


_empty_types = _dispatch_dtypes(())


def empty_types():
    return _empty_types


_floating_types = _dispatch_dtypes((torch.float32, torch.float64))


def floating_types():
    return _floating_types


_floating_types_and_half = _floating_types + (torch.half,)


def floating_types_and_half():
    return _floating_types_and_half


def floating_types_and(*dtypes):
    return _floating_types + _validate_dtypes(*dtypes)


_floating_and_complex_types = _floating_types + (torch.cfloat, torch.cdouble)


def floating_and_complex_types():
    return _floating_and_complex_types


def floating_and_complex_types_and(*dtypes):
    return _floating_and_complex_types + _validate_dtypes(*dtypes)


_double_types = _dispatch_dtypes((torch.float64, torch.complex128))


def double_types():
    return _double_types


# NB: Does not contain uint16/uint32/uint64 for BC reasons
_integral_types = _dispatch_dtypes(
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
)


def integral_types():
    return _integral_types


def integral_types_and(*dtypes):
    return _integral_types + _validate_dtypes(*dtypes)


_all_types = _floating_types + _integral_types


def all_types():
    return _all_types


def all_types_and(*dtypes):
    return _all_types + _validate_dtypes(*dtypes)


_complex_types = _dispatch_dtypes((torch.cfloat, torch.cdouble))


def complex_types():
    return _complex_types


def complex_types_and(*dtypes):
    return _complex_types + _validate_dtypes(*dtypes)


_all_types_and_complex = _all_types + _complex_types


def all_types_and_complex():
    return _all_types_and_complex


def all_types_and_complex_and(*dtypes):
    return _all_types_and_complex + _validate_dtypes(*dtypes)


_all_types_and_half = _all_types + (torch.half,)


def all_types_and_half():
    return _all_types_and_half


_all_mps_types = (
    _dispatch_dtypes({torch.float, torch.half, torch.bfloat16}) + _integral_types
)


def all_mps_types():
    return _all_mps_types


def all_mps_types_and(*dtypes):
    return _all_mps_types + _validate_dtypes(*dtypes)


_float8_types = _dispatch_dtypes(
    (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    )
)


def float8_types():
    return _float8_types


def float8_types_and(*dtypes):
    return _float8_types + _validate_dtypes(*dtypes)


def all_types_complex_float8_and(*dtypes):
    return _all_types + _complex_types + _float8_types + _validate_dtypes(*dtypes)


def custom_types(*dtypes):
    """Create a list of arbitrary dtypes"""
    return _empty_types + _validate_dtypes(*dtypes)


# The functions below are used for convenience in our test suite and thus have no corresponding C++ dispatch macro


# See AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS.
def get_all_dtypes(
    include_half=True,
    include_bfloat16=True,
    include_bool=True,
    include_complex=True,
    include_complex32=False,
    include_qint=False,
) -> list[torch.dtype]:
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(
        include_half=include_half, include_bfloat16=include_bfloat16
    )
    if include_bool:
        dtypes.append(torch.bool)
    if include_complex:
        dtypes += get_all_complex_dtypes(include_complex32)
    if include_qint:
        dtypes += get_all_qint_dtypes()
    return dtypes


def get_all_math_dtypes(device) -> list[torch.dtype]:
    return (
        get_all_int_dtypes()
        + get_all_fp_dtypes(
            include_half=device.startswith("cuda"), include_bfloat16=False
        )
        + get_all_complex_dtypes()
    )


def get_all_complex_dtypes(include_complex32=False) -> list[torch.dtype]:
    return (
        [torch.complex32, torch.complex64, torch.complex128]
        if include_complex32
        else [torch.complex64, torch.complex128]
    )


def get_all_int_dtypes() -> list[torch.dtype]:
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


def get_all_fp_dtypes(include_half=True, include_bfloat16=True) -> list[torch.dtype]:
    dtypes = [torch.float32, torch.float64]
    if include_half:
        dtypes.append(torch.float16)
    if include_bfloat16:
        dtypes.append(torch.bfloat16)
    return dtypes


def get_all_qint_dtypes() -> list[torch.dtype]:
    return [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]


float_to_corresponding_complex_type_map = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): _dispatch_dtypes

### Functions
This file defines 30 function(s): _validate_dtypes, __add__, empty_types, floating_types, floating_types_and_half, floating_types_and, floating_and_complex_types, floating_and_complex_types_and, double_types, integral_types, integral_types_and, all_types, all_types_and, complex_types, complex_types_and, all_types_and_complex, all_types_and_complex_and, all_types_and_half, all_mps_types, all_mps_types_and, float8_types, float8_types_and, all_types_complex_float8_and, custom_types, get_all_dtypes, get_all_math_dtypes, get_all_complex_dtypes, get_all_int_dtypes, get_all_fp_dtypes, get_all_qint_dtypes


## Key Components

The file contains 400 words across 227 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5106 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
