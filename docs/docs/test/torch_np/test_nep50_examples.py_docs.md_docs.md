# Documentation: `docs/test/torch_np/test_nep50_examples.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/test_nep50_examples.py_docs.md`
- **Size**: 10,677 bytes (10.43 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/test_nep50_examples.py`

## File Metadata

- **Path**: `test/torch_np/test_nep50_examples.py`
- **Size**: 6,755 bytes (6.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

"""Test examples for NEP 50."""

import itertools
from unittest import skipIf as skipif, SkipTest


try:
    import numpy as _np

    v = _np.__version__.split(".")
    HAVE_NUMPY = int(v[0]) >= 1 and int(v[1]) >= 24
except ImportError:
    HAVE_NUMPY = False

import torch._numpy as tnp
from torch._numpy import (  # noqa: F401
    array,
    bool_,
    complex128,
    complex64,
    float32,
    float64,
    inf,
    int16,
    int32,
    int64,
    uint8,
)
from torch._numpy.testing import assert_allclose
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


uint16 = uint8  # can be anything here, see below


# from numpy import array, uint8, uint16, int64, float32, float64, inf
# from numpy.testing import assert_allclose
# import numpy as np
# np._set_promotion_state('weak')

from pytest import raises as assert_raises


unchanged = None

# expression    old result   new_result
examples = {
    "uint8(1) + 2": (int64(3), uint8(3)),
    "array([1], uint8) + int64(1)": (array([2], uint8), array([2], int64)),
    "array([1], uint8) + array(1, int64)": (array([2], uint8), array([2], int64)),
    "array([1.], float32) + float64(1.)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1.], float32) + array(1., float64)": (
        array([2.0], float32),
        array([2.0], float64),
    ),
    "array([1], uint8) + 1": (array([2], uint8), unchanged),
    "array([1], uint8) + 200": (array([201], uint8), unchanged),
    "array([100], uint8) + 200": (array([44], uint8), unchanged),
    "array([1], uint8) + 300": (array([301], uint16), Exception),
    "uint8(1) + 300": (int64(301), Exception),
    "uint8(100) + 200": (int64(301), uint8(44)),  # and RuntimeWarning
    "float32(1) + 3e100": (float64(3e100), float32(inf)),  # and RuntimeWarning [T7]
    "array([1.0], float32) + 1e-14 == 1.0": (array([True]), unchanged),
    "array([0.1], float32) == float64(0.1)": (array([True]), array([False])),
    "array(1.0, float32) + 1e-14 == 1.0": (array(False), array(True)),
    "array([1.], float32) + 3": (array([4.0], float32), unchanged),
    "array([1.], float32) + int64(3)": (array([4.0], float32), array([4.0], float64)),
    "3j + array(3, complex64)": (array(3 + 3j, complex128), array(3 + 3j, complex64)),
    "float32(1) + 1j": (array(1 + 1j, complex128), array(1 + 1j, complex64)),
    "int32(1) + 5j": (array(1 + 5j, complex128), unchanged),
    # additional examples from the NEP text
    "int16(2) + 2": (int64(4), int16(4)),
    "int16(4) + 4j": (complex128(4 + 4j), unchanged),
    "float32(5) + 5j": (complex128(5 + 5j), complex64(5 + 5j)),
    "bool_(True) + 1": (int64(2), unchanged),
    "True + uint8(2)": (uint8(3), unchanged),
}


@skipif(not HAVE_NUMPY, reason="NumPy not found")
@instantiate_parametrized_tests
class TestNEP50Table(TestCase):
    @parametrize("example", examples)
    def test_nep50_exceptions(self, example):
        old, new = examples[example]

        if new is Exception:
            with assert_raises(OverflowError):
                eval(example)

        else:
            result = eval(example)

            if new is unchanged:
                new = old

            assert_allclose(result, new, atol=1e-16)
            assert result.dtype == new.dtype


# ### Directly compare to numpy ###

weaks = (True, 1, 2.0, 3j)
non_weaks = (
    tnp.asarray(True),
    tnp.uint8(1),
    tnp.int8(1),
    tnp.int32(1),
    tnp.int64(1),
    tnp.float32(1),
    tnp.float64(1),
    tnp.complex64(1),
    tnp.complex128(1),
)
if HAVE_NUMPY:
    dtypes = (
        None,
        _np.bool_,
        _np.uint8,
        _np.int8,
        _np.int32,
        _np.int64,
        _np.float32,
        _np.float64,
        _np.complex64,
        _np.complex128,
    )
else:
    dtypes = (None,)


# ufunc name: [array.dtype]
corners = {
    "true_divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "divide": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "arctan2": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "copysign": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "heaviside": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "ldexp": ["bool_", "uint8", "int8", "int16", "int32", "int64"],
    "power": ["uint8"],
    "nextafter": ["float32"],
}


@skipif(not HAVE_NUMPY, reason="NumPy not found")
@instantiate_parametrized_tests
class TestCompareToNumpy(TestCase):
    @parametrize("scalar, array, dtype", itertools.product(weaks, non_weaks, dtypes))
    def test_direct_compare(self, scalar, array, dtype):
        # compare to NumPy w/ NEP 50.
        try:
            state = _np._get_promotion_state()
            _np._set_promotion_state("weak")

            if dtype is not None:
                kwargs = {"dtype": dtype}
            try:
                result_numpy = _np.add(scalar, array.tensor.numpy(), **kwargs)
            except Exception:
                return

            kwargs = {}
            if dtype is not None:
                kwargs = {"dtype": getattr(tnp, dtype.__name__)}
            result = tnp.add(scalar, array, **kwargs).tensor.numpy()
            assert result.dtype == result_numpy.dtype
            assert result == result_numpy

        finally:
            _np._set_promotion_state(state)

    @parametrize("name", tnp._ufuncs._binary)
    @parametrize("scalar, array", itertools.product(weaks, non_weaks))
    def test_compare_ufuncs(self, name, scalar, array):
        if name in corners and (
            array.dtype.name in corners[name]
            or tnp.asarray(scalar).dtype.name in corners[name]
        ):
            raise SkipTest(f"{name}(..., dtype=array.dtype)")

        try:
            state = _np._get_promotion_state()
            _np._set_promotion_state("weak")

            if name in ["matmul", "modf", "divmod", "ldexp"]:
                return
            ufunc = getattr(tnp, name)
            ufunc_numpy = getattr(_np, name)

            try:
                result = ufunc(scalar, array)
            except RuntimeError:
                # RuntimeError: "bitwise_xor_cpu" not implemented for 'ComplexDouble' etc
                result = None

            try:
                result_numpy = ufunc_numpy(scalar, array.tensor.numpy())
            except TypeError:
                # TypeError: ufunc 'hypot' not supported for the input types
                result_numpy = None

            if result is not None and result_numpy is not None:
                assert result.tensor.numpy().dtype == result_numpy.dtype

        finally:
            _np._set_promotion_state(state)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test examples for NEP 50."""import itertoolsfrom unittest import skipIf as skipif, SkipTesttry:    import numpy as _np    v = _np.__version__.split(".")    HAVE_NUMPY = int(v[0]) >= 1 and int(v[1]) >= 24except ImportError:    HAVE_NUMPY = Falseimport torch._numpy as tnpfrom torch._numpy import (  # noqa: F401    array,    bool_,    complex128,    complex64,    float32,    float64,    inf,    int16,    int32,    int64,    uint8,)from torch._numpy.testing import assert_allclosefrom torch.testing._internal.common_utils import (    instantiate_parametrized_tests,    parametrize,    run_tests,    TestCase,)uint16 = uint8  # can be anything here, see below# from numpy import array, uint8, uint16, int64, float32, float64, inf# from numpy.testing import assert_allclose# import numpy as np# np._set_promotion_state('weak')from pytest import raises as assert_raises

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNEP50Table`, `TestCompareToNumpy`

**Functions defined**: `test_nep50_exceptions`, `test_direct_compare`, `test_compare_ufuncs`

**Key imports**: itertools, skipIf as skipif, SkipTest, numpy as _np, torch._numpy as tnp, assert_allclose, array, uint8, uint16, int64, float32, float64, inf, assert_allclose, numpy as np, raises as assert_raises


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `unittest`: skipIf as skipif, SkipTest
- `numpy as _np`
- `torch._numpy as tnp`
- `torch._numpy.testing`: assert_allclose
- `numpy`: array, uint8, uint16, int64, float32, float64, inf
- `numpy.testing`: assert_allclose
- `numpy as np`
- `pytest`: raises as assert_raises


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/torch_np/test_nep50_examples.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np`):

- [`test_random.py_docs.md`](./test_random.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ufuncs_basic.py_docs.md`](./test_ufuncs_basic.py_docs.md)
- [`test_function_base.py_docs.md`](./test_function_base.py_docs.md)
- [`test_basic.py_docs.md`](./test_basic.py_docs.md)
- [`test_binary_ufuncs.py_docs.md`](./test_binary_ufuncs.py_docs.md)
- [`test_indexing.py_docs.md`](./test_indexing.py_docs.md)
- [`test_ndarray_methods.py_docs.md`](./test_ndarray_methods.py_docs.md)
- [`conftest.py_docs.md`](./conftest.py_docs.md)
- [`test_reductions.py_docs.md`](./test_reductions.py_docs.md)


## Cross-References

- **File Documentation**: `test_nep50_examples.py_docs.md`
- **Keyword Index**: `test_nep50_examples.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/torch_np`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/torch_np`, which is part of the **core PyTorch library**.



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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/torch_np/test_nep50_examples.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np`):

- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_scalars_0D_arrays.py_docs.md_docs.md`](./test_scalars_0D_arrays.py_docs.md_docs.md)
- [`test_ndarray_methods.py_docs.md_docs.md`](./test_ndarray_methods.py_docs.md_docs.md)
- [`test_scalars_0D_arrays.py_kw.md_docs.md`](./test_scalars_0D_arrays.py_kw.md_docs.md)
- [`test_function_base.py_docs.md_docs.md`](./test_function_base.py_docs.md_docs.md)
- [`test_basic.py_docs.md_docs.md`](./test_basic.py_docs.md_docs.md)
- [`test_function_base.py_kw.md_docs.md`](./test_function_base.py_kw.md_docs.md)
- [`check_tests_conform.py_kw.md_docs.md`](./check_tests_conform.py_kw.md_docs.md)
- [`test_ufuncs_basic.py_kw.md_docs.md`](./test_ufuncs_basic.py_kw.md_docs.md)
- [`test_reductions.py_docs.md_docs.md`](./test_reductions.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_nep50_examples.py_docs.md_docs.md`
- **Keyword Index**: `test_nep50_examples.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
