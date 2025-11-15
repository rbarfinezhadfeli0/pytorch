# Documentation: `torch/_numpy/random.py`

## File Metadata

- **Path**: `torch/_numpy/random.py`
- **Size**: 4,651 bytes (4.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

"""Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy

"""

from __future__ import annotations

import functools
from math import sqrt
from typing import Optional

import torch

from . import _dtypes_impl, _util
from ._normalizations import array_or_scalar, ArrayLike, normalizer


__all__ = [
    "seed",
    "random_sample",
    "sample",
    "random",
    "rand",
    "randn",
    "normal",
    "choice",
    "randint",
    "shuffle",
    "uniform",
]


def use_numpy_random():
    # local import to avoid ref cycles
    import torch._dynamo.config as config

    return config.use_numpy_random_stream


def deco_stream(func):
    @functools.wraps(func)
    def inner(*args, **kwds):
        if not use_numpy_random():
            return func(*args, **kwds)
        else:
            import numpy

            from ._ndarray import ndarray

            f = getattr(numpy.random, func.__name__)

            # numpy funcs accept numpy ndarrays, unwrap
            args = tuple(
                arg.tensor.numpy() if isinstance(arg, ndarray) else arg for arg in args
            )
            kwds = {
                key: val.tensor.numpy() if isinstance(val, ndarray) else val
                for key, val in kwds.items()
            }

            value = f(*args, **kwds)

            # `value` can be either numpy.ndarray or python scalar (or None)
            if isinstance(value, numpy.ndarray):
                value = ndarray(torch.as_tensor(value))

            return value

    return inner


@deco_stream
def seed(seed=None):
    if seed is not None:
        torch.random.manual_seed(seed)


@deco_stream
def random_sample(size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_()
    return array_or_scalar(values, return_scalar=size == ())


def rand(*size):
    if size == ():
        size = None
    return random_sample(size)


sample = random_sample
random = random_sample


@deco_stream
def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_(low, high)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def randn(*size):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.randn(size, dtype=dtype)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).normal_(loc, scale)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def shuffle(x):
    # no @normalizer because we do not cast e.g. lists to tensors
    from ._ndarray import ndarray

    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, ndarray):
        tensor = x.tensor
    else:
        raise NotImplementedError("We do not random.shuffle lists in-place")

    perm = torch.randperm(tensor.shape[0])
    xp = tensor[perm]
    tensor.copy_(xp)


@deco_stream
def randint(low, high=None, size=None):
    if size is None:
        size = ()
    if not isinstance(size, (tuple, list)):
        size = (size,)
    if high is None:
        low, high = 0, low
    values = torch.randint(low, high, size=size)
    return array_or_scalar(values, int, return_scalar=size == ())


@deco_stream
@normalizer
def choice(a: ArrayLike, size=None, replace=True, p: Optional[ArrayLike] = None):
    # https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    if a.numel() == 1:
        a = torch.arange(a)

    # TODO: check a.dtype is integer -- cf np.random.choice(3.4) which raises

    # number of draws
    if size is None:
        num_el = 1
    elif _util.is_sequence(size):
        num_el = 1
        for el in size:
            num_el *= el
    else:
        num_el = size

    # prepare the probabilities
    if p is None:
        p = torch.ones_like(a) / a.shape[0]

    # cf https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx#L973
    atol = sqrt(torch.finfo(p.dtype).eps)
    if abs(p.sum() - 1.0) > atol:
        raise ValueError("probabilities do not sum to 1.")

    # actually sample
    indices = torch.multinomial(p, num_el, replacement=replace)

    if _util.is_sequence(size):
        indices = indices.reshape(size)

    samples = a[indices]

    return samples

```



## High-Level Overview

"""Wrapper to mimic (parts of) np.random API surface.NumPy has strict guarantees on reproducibility etc; here we don't give any.Q: default dtype is float64 in numpy

This Python file contains 0 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `use_numpy_random`, `deco_stream`, `inner`, `seed`, `random_sample`, `rand`, `uniform`, `randn`, `normal`, `shuffle`, `randint`, `choice`

**Key imports**: annotations, functools, sqrt, Optional, torch, _dtypes_impl, _util, array_or_scalar, ArrayLike, normalizer, to avoid ref cycles, torch._dynamo.config as config, numpy


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_numpy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `functools`
- `math`: sqrt
- `typing`: Optional
- `torch`
- `.`: _dtypes_impl, _util
- `._normalizations`: array_or_scalar, ArrayLike, normalizer
- `to avoid ref cycles`
- `torch._dynamo.config as config`
- `numpy`
- `._ndarray`: ndarray


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
- [`fft.py_docs.md`](./fft.py_docs.md)
- [`_binary_ufuncs_impl.py_docs.md`](./_binary_ufuncs_impl.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`_ndarray.py_docs.md`](./_ndarray.py_docs.md)


## Cross-References

- **File Documentation**: `random.py_docs.md`
- **Keyword Index**: `random.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
