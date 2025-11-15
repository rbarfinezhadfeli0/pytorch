# Documentation: `docs/torch/_numpy/_util.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_numpy/_util.py_docs.md`
- **Size**: 11,390 bytes (11.12 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_numpy/_util.py`

## File Metadata

- **Path**: `torch/_numpy/_util.py`
- **Size**: 7,725 bytes (7.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

"""Assorted utilities, which do not need anything other then torch and stdlib."""

import operator

import torch

from . import _dtypes_impl


# https://github.com/numpy/numpy/blob/v1.23.0/numpy/distutils/misc_util.py#L497-L504
def is_sequence(seq):
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


class AxisError(ValueError, IndexError):
    pass


class UFuncTypeError(TypeError, RuntimeError):
    pass


def cast_if_needed(tensor, dtype):
    # NB: no casting if dtype=None
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


def cast_int_to_float(x):
    # cast integers and bools to the default float dtype
    if _dtypes_impl._category(x.dtype) < 2:
        x = x.to(_dtypes_impl.default_dtypes().float_dtype)
    return x


# a replica of the version in ./numpy/numpy/core/src/multiarray/common.h
def normalize_axis_index(ax, ndim, argname=None):
    if not (-ndim <= ax < ndim):
        raise AxisError(f"axis {ax} is out of bounds for array of dimension {ndim}")
    if ax < 0:
        ax += ndim
    return ax


# from https://github.com/numpy/numpy/blob/main/numpy/core/numeric.py#L1378
def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.
    Used internally by multi-axis-checking logic.

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`
    """
    # Optimization to speed-up the most common cases.
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    # Going via an iterator directly is slower than via list comprehension.
    axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
    if not allow_duplicate and len(set(map(int, axis))) != len(axis):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        else:
            raise ValueError("repeated axis")
    return axis


def allow_only_single_axis(axis):
    if axis is None:
        return axis
    if len(axis) != 1:
        raise NotImplementedError("does not handle tuple axis")
    return axis[0]


def expand_shape(arr_shape, axis):
    # taken from numpy 1.23.x, expand_dims function
    if type(axis) not in (list, tuple):
        axis = (axis,)
    out_ndim = len(axis) + len(arr_shape)
    axis = normalize_axis_tuple(axis, out_ndim)
    shape_it = iter(arr_shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return shape


def apply_keepdims(tensor, axis, ndim):
    if axis is None:
        # tensor was a scalar
        shape = (1,) * ndim
        tensor = tensor.expand(shape).contiguous()
    else:
        shape = expand_shape(tensor.shape, axis)
        tensor = tensor.reshape(shape)
    return tensor


def axis_none_flatten(*tensors, axis=None):
    """Flatten the arrays if axis is None."""
    if axis is None:
        tensors = tuple(ar.flatten() for ar in tensors)
        return tensors, 0
    else:
        return tensors, axis


def typecast_tensor(t, target_dtype, casting):
    """Dtype-cast tensor to target_dtype.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to cast
    target_dtype : torch dtype object
        The array dtype to cast all tensors to
    casting : str
        The casting mode, see `np.can_cast`

     Returns
     -------
    `torch.Tensor` of the `target_dtype` dtype

     Raises
     ------
     ValueError
        if the argument cannot be cast according to the `casting` rule

    """
    can_cast = _dtypes_impl.can_cast_impl

    if not can_cast(t.dtype, target_dtype, casting=casting):
        raise TypeError(
            f"Cannot cast array data from {t.dtype} to"
            f" {target_dtype} according to the rule '{casting}'"
        )
    return cast_if_needed(t, target_dtype)


def typecast_tensors(tensors, target_dtype, casting):
    return tuple(typecast_tensor(t, target_dtype, casting) for t in tensors)


def _try_convert_to_tensor(obj):
    try:
        tensor = torch.as_tensor(obj)
    except Exception as e:
        mesg = f"failed to convert {obj} to ndarray. \nInternal error is: {str(e)}."
        raise NotImplementedError(mesg)  # noqa: B904
    return tensor


def _coerce_to_tensor(obj, dtype=None, copy=False, ndmin=0):
    """The core logic of the array(...) function.

    Parameters
    ----------
    obj : tensor_like
        The thing to coerce
    dtype : torch.dtype object or None
        Coerce to this torch dtype
    copy : bool
        Copy or not
    ndmin : int
        The results as least this many dimensions
    is_weak : bool
        Whether obj is a weakly typed python scalar.

    Returns
    -------
    tensor : torch.Tensor
        a tensor object with requested dtype, ndim and copy semantics.

    Notes
    -----
    This is almost a "tensor_like" coercive function. Does not handle wrapper
    ndarrays (those should be handled in the ndarray-aware layer prior to
    invoking this function).
    """
    if isinstance(obj, torch.Tensor):
        tensor = obj
    else:
        # tensor.dtype is the pytorch default, typically float32. If obj's elements
        # are not exactly representable in float32, we've lost precision:
        # >>> torch.as_tensor(1e12).item() - 1e12
        # -4096.0
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_dtypes_impl.get_default_dtype_for(torch.float32))
        try:
            tensor = _try_convert_to_tensor(obj)
        finally:
            torch.set_default_dtype(default_dtype)

    # type cast if requested
    tensor = cast_if_needed(tensor, dtype)

    # adjust ndim if needed
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,) * ndim_extra + tensor.shape)

    # special handling for np._CopyMode
    try:
        copy = bool(copy)
    except ValueError:
        # TODO handle _CopyMode.IF_NEEDED correctly
        copy = False
    # copy if requested
    if copy:
        tensor = tensor.clone()

    return tensor


def ndarrays_to_tensors(*inputs):
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)"""
    from ._ndarray import ndarray

    if len(inputs) == 0:
        return ValueError()
    elif len(inputs) == 1:
        input_ = inputs[0]
        if isinstance(input_, ndarray):
            return input_.tensor
        elif isinstance(input_, tuple):
            result = []
            for sub_input in input_:
                sub_result = ndarrays_to_tensors(sub_input)
                result.append(sub_result)
            return tuple(result)
        else:
            return input_
    else:
        assert isinstance(inputs, tuple)  # sanity check
        return ndarrays_to_tensors(inputs)

```



## High-Level Overview

"""Assorted utilities, which do not need anything other then torch and stdlib."""import operatorimport torchfrom . import _dtypes_impl# https://github.com/numpy/numpy/blob/v1.23.0/numpy/distutils/misc_util.py#L497-L504def is_sequence(seq):    if isinstance(seq, str):        return False    try:        len(seq)    except Exception:        return False    return Trueclass AxisError(ValueError, IndexError):    passclass UFuncTypeError(TypeError, RuntimeError):    passdef cast_if_needed(tensor, dtype):    # NB: no casting if dtype=None    if dtype is not None and tensor.dtype != dtype:        tensor = tensor.to(dtype)    return tensordef cast_int_to_float(x):    # cast integers and bools to the default float dtype    if _dtypes_impl._category(x.dtype) < 2:        x = x.to(_dtypes_impl.default_dtypes().float_dtype)    return x# a replica of the version in ./numpy/numpy/core/src/multiarray/common.hdef normalize_axis_index(ax, ndim, argname=None):    if not (-ndim <= ax < ndim):        raise AxisError(f"axis {ax} is out of bounds for array of dimension {ndim}")    if ax < 0:        ax += ndim

This Python file contains 2 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AxisError`, `UFuncTypeError`

**Functions defined**: `is_sequence`, `cast_if_needed`, `cast_int_to_float`, `normalize_axis_index`, `normalize_axis_tuple`, `allow_only_single_axis`, `expand_shape`, `apply_keepdims`, `axis_none_flatten`, `typecast_tensor`, `typecast_tensors`, `_try_convert_to_tensor`, `_coerce_to_tensor`, `ndarrays_to_tensors`

**Key imports**: operator, torch, _dtypes_impl, ndarray


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_numpy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `torch`
- `.`: _dtypes_impl
- `._ndarray`: ndarray


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
- [`_funcs_impl.py_docs.md`](./_funcs_impl.py_docs.md)
- [`_ufuncs.py_docs.md`](./_ufuncs.py_docs.md)
- [`linalg.py_docs.md`](./linalg.py_docs.md)
- [`fft.py_docs.md`](./fft.py_docs.md)
- [`_binary_ufuncs_impl.py_docs.md`](./_binary_ufuncs_impl.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_ndarray.py_docs.md`](./_ndarray.py_docs.md)
- [`random.py_docs.md`](./random.py_docs.md)


## Cross-References

- **File Documentation**: `_util.py_docs.md`
- **Keyword Index**: `_util.py_kw.md`
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

- **Error Handling**: Includes exception handling


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

- **File Documentation**: `_util.py_docs.md_docs.md`
- **Keyword Index**: `_util.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
