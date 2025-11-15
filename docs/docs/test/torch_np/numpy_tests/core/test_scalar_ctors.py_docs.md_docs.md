# Documentation: `docs/test/torch_np/numpy_tests/core/test_scalar_ctors.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/numpy_tests/core/test_scalar_ctors.py_docs.md`
- **Size**: 6,532 bytes (6.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/numpy_tests/core/test_scalar_ctors.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/core/test_scalar_ctors.py`
- **Size**: 3,328 bytes (3.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

"""
Test the scalar constructors, which also do type-coercion
"""

import functools
from unittest import skipIf as skipif

import pytest

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_almost_equal, assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_almost_equal, assert_equal


skip = functools.partial(skipif, True)


class TestFromString(TestCase):
    @xpassIfTorchDynamo_np  # (reason="XXX: floats from strings")
    def test_floating(self):
        # Ticket #640, floats from string
        fsingle = np.single("1.234")
        fdouble = np.double("1.234")
        assert_almost_equal(fsingle, 1.234)
        assert_almost_equal(fdouble, 1.234)

    @xpassIfTorchDynamo_np  # (reason="XXX: floats from strings")
    def test_floating_overflow(self):
        """Strings containing an unrepresentable float overflow"""
        fhalf = np.half("1e10000")
        assert_equal(fhalf, np.inf)
        fsingle = np.single("1e10000")
        assert_equal(fsingle, np.inf)
        fdouble = np.double("1e10000")
        assert_equal(fdouble, np.inf)

        fhalf = np.half("-1e10000")
        assert_equal(fhalf, -np.inf)
        fsingle = np.single("-1e10000")
        assert_equal(fsingle, -np.inf)
        fdouble = np.double("-1e10000")
        assert_equal(fdouble, -np.inf)

    def test_bool(self):
        with pytest.raises(TypeError):
            np.bool_(False, garbage=True)


class TestFromInt(TestCase):
    def test_intp(self):
        # Ticket #99
        assert_equal(1024, np.intp(1024))

    def test_uint64_from_negative(self):
        # NumPy test was asserting a DeprecationWarning
        assert_equal(np.uint8(-2), np.uint8(254))


int_types = [
    subtest(np.byte, name="np_byte"),
    subtest(np.short, name="np_short"),
    subtest(np.intc, name="np_intc"),
    subtest(np.int_, name="np_int_"),
    subtest(np.longlong, name="np_longlong"),
]
uint_types = [np.ubyte]
float_types = [np.half, np.single, np.double]
cfloat_types = [np.csingle, np.cdouble]


@instantiate_parametrized_tests
class TestArrayFromScalar(TestCase):
    """gh-15467"""

    def _do_test(self, t1, t2):
        x = t1(2)
        arr = np.array(x, dtype=t2)
        # type should be preserved exactly
        if t2 is None:
            assert arr.dtype.type is t1
        else:
            assert arr.dtype.type is t2

        arr1 = np.asarray(x, dtype=t2)
        if t2 is None:
            assert arr1.dtype.type is t1
        else:
            assert arr1.dtype.type is t2

    @parametrize("t1", int_types + uint_types)
    @parametrize("t2", int_types + uint_types + [None])
    def test_integers(self, t1, t2):
        return self._do_test(t1, t2)

    @parametrize("t1", float_types)
    @parametrize("t2", float_types + [None])
    def test_reals(self, t1, t2):
        return self._do_test(t1, t2)

    @parametrize("t1", cfloat_types)
    @parametrize("t2", cfloat_types + [None])
    def test_complex(self, t1, t2):
        return self._do_test(t1, t2)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test the scalar constructors, which also do type-coercion

This Python file contains 3 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFromString`, `TestFromInt`, `TestArrayFromScalar`

**Functions defined**: `test_floating`, `test_floating_overflow`, `test_bool`, `test_intp`, `test_uint64_from_negative`, `_do_test`, `test_integers`, `test_reals`, `test_complex`

**Key imports**: functools, skipIf as skipif, pytest, numpy as np, assert_almost_equal, assert_equal, torch._numpy as np, assert_almost_equal, assert_equal


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np/numpy_tests/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `unittest`: skipIf as skipif
- `pytest`
- `numpy as np`
- `numpy.testing`: assert_almost_equal, assert_equal
- `torch._numpy as np`
- `torch._numpy.testing`: assert_almost_equal, assert_equal


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
python test/torch_np/numpy_tests/core/test_scalar_ctors.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np/numpy_tests/core`):

- [`test_shape_base.py_docs.md`](./test_shape_base.py_docs.md)
- [`test_scalarinherit.py_docs.md`](./test_scalarinherit.py_docs.md)
- [`test_scalar_methods.py_docs.md`](./test_scalar_methods.py_docs.md)
- [`test_einsum.py_docs.md`](./test_einsum.py_docs.md)
- [`test_indexing.py_docs.md`](./test_indexing.py_docs.md)
- [`test_dlpack.py_docs.md`](./test_dlpack.py_docs.md)
- [`test_getlimits.py_docs.md`](./test_getlimits.py_docs.md)
- [`test_multiarray.py_docs.md`](./test_multiarray.py_docs.md)
- [`test_numerictypes.py_docs.md`](./test_numerictypes.py_docs.md)
- [`test_dtype.py_docs.md`](./test_dtype.py_docs.md)


## Cross-References

- **File Documentation**: `test_scalar_ctors.py_docs.md`
- **Keyword Index**: `test_scalar_ctors.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/torch_np/numpy_tests/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/torch_np/numpy_tests/core`, which is part of the **core PyTorch library**.



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
python docs/test/torch_np/numpy_tests/core/test_scalar_ctors.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np/numpy_tests/core`):

- [`test_scalar_methods.py_docs.md_docs.md`](./test_scalar_methods.py_docs.md_docs.md)
- [`test_einsum.py_docs.md_docs.md`](./test_einsum.py_docs.md_docs.md)
- [`test_scalarmath.py_kw.md_docs.md`](./test_scalarmath.py_kw.md_docs.md)
- [`test_scalarmath.py_docs.md_docs.md`](./test_scalarmath.py_docs.md_docs.md)
- [`test_shape_base.py_docs.md_docs.md`](./test_shape_base.py_docs.md_docs.md)
- [`test_numerictypes.py_docs.md_docs.md`](./test_numerictypes.py_docs.md_docs.md)
- [`test_scalar_methods.py_kw.md_docs.md`](./test_scalar_methods.py_kw.md_docs.md)
- [`test_indexing.py_docs.md_docs.md`](./test_indexing.py_docs.md_docs.md)
- [`test_scalar_ctors.py_kw.md_docs.md`](./test_scalar_ctors.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_scalar_ctors.py_docs.md_docs.md`
- **Keyword Index**: `test_scalar_ctors.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
