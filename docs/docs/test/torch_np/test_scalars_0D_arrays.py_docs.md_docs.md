# Documentation: `docs/test/torch_np/test_scalars_0D_arrays.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/test_scalars_0D_arrays.py_docs.md`
- **Size**: 6,912 bytes (6.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/test_scalars_0D_arrays.py`

## File Metadata

- **Path**: `test/torch_np/test_scalars_0D_arrays.py`
- **Size**: 3,823 bytes (3.73 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

"""
Basic tests to assert and illustrate the  behavior around the decision to use 0D
arrays in place of array scalars.

Extensive tests of this sort of functionality is in numpy_tests/core/*scalar*

Also test the isscalar function (which is deliberately a bit more lax).
"""

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


parametrize_value = parametrize(
    "value",
    [
        subtest(np.int64(42), name="int64"),
        subtest(np.array(42), name="array"),
        subtest(np.asarray(42), name="asarray"),
        subtest(np.asarray(np.int64(42)), name="asarray_int"),
    ],
)


@instantiate_parametrized_tests
class TestArrayScalars(TestCase):
    @parametrize_value
    def test_array_scalar_basic(self, value):
        assert value.ndim == 0
        assert value.shape == ()
        assert value.size == 1
        assert value.dtype == np.dtype("int64")

    @parametrize_value
    def test_conversion_to_int(self, value):
        py_scalar = int(value)
        assert py_scalar == 42
        assert isinstance(py_scalar, int)
        assert not isinstance(value, int)

    @parametrize_value
    def test_decay_to_py_scalar(self, value):
        # NumPy distinguishes array scalars and 0D arrays. For instance
        # `scalar * list` is equivalent to `int(scalar) * list`, but
        # `0D array * list` is equivalent to `0D array * np.asarray(list)`.
        # Our scalars follow 0D array behavior (because they are 0D arrays)
        lst = [1, 2, 3]

        product = value * lst
        assert isinstance(product, np.ndarray)
        assert product.shape == (3,)
        assert_equal(product, [42, 42 * 2, 42 * 3])

        # repeat with right-multiply
        product = lst * value
        assert isinstance(product, np.ndarray)
        assert product.shape == (3,)
        assert_equal(product, [42, 42 * 2, 42 * 3])

    def test_scalar_comparisons(self):
        scalar = np.int64(42)
        arr = np.array(42)

        assert arr == scalar
        assert arr >= scalar
        assert arr <= scalar

        assert scalar == 42
        assert arr == 42


# @xfailIfTorchDynamo
@instantiate_parametrized_tests
class TestIsScalar(TestCase):
    #
    # np.isscalar(...) checks that its argument is a numeric object with exactly one element.
    #
    # This differs from NumPy which also requires that shape == ().
    #
    scalars = [
        subtest(42, "literal"),
        subtest(int(42.0), "int"),
        subtest(np.float32(42), "float32"),
        subtest(np.array(42), "array_0D", decorators=[xfailIfTorchDynamo]),
        subtest([42], "list", decorators=[xfailIfTorchDynamo]),
        subtest([[42]], "list-list", decorators=[xfailIfTorchDynamo]),
        subtest(np.array([42]), "array_1D", decorators=[xfailIfTorchDynamo]),
        subtest(np.array([[42]]), "array_2D", decorators=[xfailIfTorchDynamo]),
    ]

    import math

    not_scalars = [
        int,
        np.float32,
        subtest("s", decorators=[xfailIfTorchDynamo]),
        subtest("string", decorators=[xfailIfTorchDynamo]),
        (),
        [],
        math.sin,
        np,
        np.transpose,
        [1, 2],
        np.asarray([1, 2]),
        np.float32([1, 2]),
    ]

    @parametrize("value", scalars)
    def test_is_scalar(self, value):
        assert np.isscalar(value)

    @parametrize("value", not_scalars)
    def test_is_not_scalar(self, value):
        assert not np.isscalar(value)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Basic tests to assert and illustrate the  behavior around the decision to use 0Darrays in place of array scalars.Extensive tests of this sort of functionality is in numpy_tests/core/*scalar*Also test the isscalar function (which is deliberately a bit more lax).

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestArrayScalars`, `TestIsScalar`

**Functions defined**: `test_array_scalar_basic`, `test_conversion_to_int`, `test_decay_to_py_scalar`, `test_scalar_comparisons`, `test_is_scalar`, `test_is_not_scalar`

**Key imports**: numpy as np, assert_equal, torch._numpy as np, assert_equal, math


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `numpy as np`
- `numpy.testing`: assert_equal
- `torch._numpy as np`
- `torch._numpy.testing`: assert_equal
- `math`


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

This is a test file. Run it with:

```bash
python test/torch_np/test_scalars_0D_arrays.py
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

- **File Documentation**: `test_scalars_0D_arrays.py_docs.md`
- **Keyword Index**: `test_scalars_0D_arrays.py_kw.md`
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

This is a test file. Run it with:

```bash
python docs/test/torch_np/test_scalars_0D_arrays.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np`):

- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ndarray_methods.py_docs.md_docs.md`](./test_ndarray_methods.py_docs.md_docs.md)
- [`test_scalars_0D_arrays.py_kw.md_docs.md`](./test_scalars_0D_arrays.py_kw.md_docs.md)
- [`test_function_base.py_docs.md_docs.md`](./test_function_base.py_docs.md_docs.md)
- [`test_basic.py_docs.md_docs.md`](./test_basic.py_docs.md_docs.md)
- [`test_function_base.py_kw.md_docs.md`](./test_function_base.py_kw.md_docs.md)
- [`check_tests_conform.py_kw.md_docs.md`](./check_tests_conform.py_kw.md_docs.md)
- [`test_ufuncs_basic.py_kw.md_docs.md`](./test_ufuncs_basic.py_kw.md_docs.md)
- [`test_reductions.py_docs.md_docs.md`](./test_reductions.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_scalars_0D_arrays.py_docs.md_docs.md`
- **Keyword Index**: `test_scalars_0D_arrays.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
