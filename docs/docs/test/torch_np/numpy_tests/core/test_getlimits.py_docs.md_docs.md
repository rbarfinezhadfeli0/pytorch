# Documentation: `docs/test/torch_np/numpy_tests/core/test_getlimits.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/numpy_tests/core/test_getlimits.py_docs.md`
- **Size**: 12,113 bytes (11.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/numpy_tests/core/test_getlimits.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/core/test_getlimits.py`
- **Size**: 7,576 bytes (7.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

"""Test functions for limits module."""

import functools
import warnings
from unittest import expectedFailure as xfail, skipIf

import numpy
from pytest import raises as assert_raises

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
    from numpy import double, finfo, half, iinfo, single
    from numpy.testing import assert_, assert_equal
else:
    import torch._numpy as np
    from torch._numpy import double, finfo, half, iinfo, single
    from torch._numpy.testing import assert_, assert_equal


skip = functools.partial(skipIf, True)

##################################################


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestPythonFloat(TestCase):
    def test_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestHalf(TestCase):
    def test_singleton(self):
        ftype = finfo(half)
        ftype2 = finfo(half)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestSingle(TestCase):
    def test_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype), id(ftype2))


@skip(reason="torch.finfo is not a singleton. Why demanding it is?")
class TestDouble(TestCase):
    def test_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype), id(ftype2))


class TestFinfo(TestCase):
    @skipIf(numpy.__version__ < "1.23", reason=".smallest_normal is new")
    def test_basic(self):
        dts = list(
            zip(
                ["f2", "f4", "f8", "c8", "c16"],
                [np.float16, np.float32, np.float64, np.complex64, np.complex128],
            )
        )
        for dt1, dt2 in dts:
            for attr in (
                "bits",
                "eps",
                "max",
                "min",
                "resolution",
                "tiny",
                "smallest_normal",
            ):
                assert_equal(getattr(finfo(dt1), attr), getattr(finfo(dt2), attr), attr)
        with assert_raises((TypeError, ValueError)):
            finfo("i4")

    @skip  # (reason="Some of these attributes are not implemented vs NP versions")
    def test_basic_missing(self):
        dt = np.float32
        for attr in [
            "epsneg",
            "iexp",
            "machep",
            "maxexp",
            "minexp",
            "negep",
            "nexp",
            "nmant",
            "precision",
            "smallest_subnormal",
        ]:
            getattr(finfo(dt), attr)


@instantiate_parametrized_tests
class TestIinfo(TestCase):
    def test_basic(self):
        dts = list(
            zip(
                ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"],
                [
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                ],
            )
        )
        for dt1, dt2 in dts:
            for attr in ("bits", "min", "max"):
                assert_equal(getattr(iinfo(dt1), attr), getattr(iinfo(dt2), attr), attr)
        with assert_raises((TypeError, ValueError)):
            iinfo("f4")

    @parametrize(
        "T",
        [
            np.uint8,
            # xfail: unsupported add (uint[16,32,64])
            subtest(np.uint16, decorators=[] if TEST_WITH_TORCHDYNAMO else [xfail]),
            subtest(np.uint32, decorators=[] if TEST_WITH_TORCHDYNAMO else [xfail]),
            subtest(np.uint64, decorators=[] if TEST_WITH_TORCHDYNAMO else [xfail]),
        ],
    )
    def test_unsigned_max(self, T):
        max_calculated = T(0) - T(1)
        assert_equal(iinfo(T).max, max_calculated)


class TestRepr(TestCase):
    def test_iinfo_repr(self):
        expected = "iinfo(min=-32768, max=32767, dtype=int16)"
        assert_equal(repr(np.iinfo(np.int16)), expected)

    @skipIf(TEST_WITH_TORCHDYNAMO, reason="repr differs")
    def test_finfo_repr(self):
        repr_f32 = repr(np.finfo(np.float32))
        assert "finfo(resolution=1e-06, min=-3.40282e+38," in repr_f32
        assert "dtype=float32" in repr_f32


def assert_ma_equal(discovered, ma_like):
    # Check MachAr-like objects same as calculated MachAr instances
    for key, value in discovered.__dict__.items():
        assert_equal(value, getattr(ma_like, key))
        if hasattr(value, "shape"):
            assert_equal(value.shape, getattr(ma_like, key).shape)
            assert_equal(value.dtype, getattr(ma_like, key).dtype)


class TestMisc(TestCase):
    @skip(reason="Instantiate {i,f}info from dtypes.")
    def test_instances(self):
        iinfo(10)
        finfo(3.0)

    @skip(reason="MachAr no implemented (does it need to)?")
    def test_known_types(self):
        # Test we are correctly compiling parameters for known types
        for ftype, ma_like in (
            (np.float16, _float_ma[16]),
            (np.float32, _float_ma[32]),
            (np.float64, _float_ma[64]),
        ):
            assert_ma_equal(_discovered_machar(ftype), ma_like)
        # Suppress warning for broken discovery of double double on PPC
        ld_ma = _discovered_machar(np.longdouble)
        bytes = np.dtype(np.longdouble).itemsize
        if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
            # 80-bit extended precision
            assert_ma_equal(ld_ma, _float_ma[80])
        elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
            # IEE 754 128-bit
            assert_ma_equal(ld_ma, _float_ma[128])

    @skip(reason="MachAr no implemented (does it need to be)?")
    def test_subnormal_warning(self):
        """Test that the subnormal is zero warning is not being raised."""
        ld_ma = _discovered_machar(np.longdouble)
        bytes = np.dtype(np.longdouble).itemsize
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if (ld_ma.it, ld_ma.maxexp) == (63, 16384) and bytes in (12, 16):
                # 80-bit extended precision
                ld_ma.smallest_subnormal
                assert len(w) == 0
            elif (ld_ma.it, ld_ma.maxexp) == (112, 16384) and bytes == 16:
                # IEE 754 128-bit
                ld_ma.smallest_subnormal
                assert len(w) == 0
            else:
                # Double double
                ld_ma.smallest_subnormal
                # This test may fail on some platforms
                assert len(w) == 0

    @xpassIfTorchDynamo_np  # (reason="None of nmant, minexp, maxexp is implemented.")
    def test_plausible_finfo(self):
        # Assert that finfo returns reasonable results for all types
        for ftype in (
            [np.float16, np.float32, np.float64, np.longdouble]
            + [
                np.complex64,
                np.complex128,
            ]
            # no complex256 in torch._numpy
            + ([np.clongdouble] if hasattr(np, "clongdouble") else [])
        ):
            info = np.finfo(ftype)
            assert_(info.nmant > 1)
            assert_(info.minexp < -1)
            assert_(info.maxexp > 1)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test functions for limits module."""import functoolsimport warningsfrom unittest import expectedFailure as xfail, skipIfimport numpyfrom pytest import raises as assert_raisesfrom torch.testing._internal.common_utils import (    instantiate_parametrized_tests,    parametrize,    run_tests,    subtest,    TEST_WITH_TORCHDYNAMO,    TestCase,    xpassIfTorchDynamo_np,)if TEST_WITH_TORCHDYNAMO:    import numpy as np    from numpy import double, finfo, half, iinfo, single    from numpy.testing import assert_, assert_equalelse:    import torch._numpy as np    from torch._numpy import double, finfo, half, iinfo, single    from torch._numpy.testing import assert_, assert_equalskip = functools.partial(skipIf, True)##################################################@skip(reason="torch.finfo is not a singleton. Why demanding it is?")class TestPythonFloat(TestCase):    def test_singleton(self):        ftype = finfo(float)        ftype2 = finfo(float)        assert_equal(id(ftype), id(ftype2))@skip(reason="torch.finfo is not a singleton. Why demanding it is?")class TestHalf(TestCase):    def test_singleton(self):        ftype = finfo(half)        ftype2 = finfo(half)

This Python file contains 8 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPythonFloat`, `TestHalf`, `TestSingle`, `TestDouble`, `TestFinfo`, `TestIinfo`, `TestRepr`, `TestMisc`

**Functions defined**: `test_singleton`, `test_singleton`, `test_singleton`, `test_singleton`, `test_basic`, `test_basic_missing`, `test_basic`, `test_unsigned_max`, `test_iinfo_repr`, `test_finfo_repr`, `assert_ma_equal`, `test_instances`, `test_known_types`, `test_subnormal_warning`, `test_plausible_finfo`

**Key imports**: functools, warnings, expectedFailure as xfail, skipIf, numpy, raises as assert_raises, numpy as np, double, finfo, half, iinfo, single, assert_, assert_equal, torch._numpy as np, double, finfo, half, iinfo, single


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np/numpy_tests/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `warnings`
- `unittest`: expectedFailure as xfail, skipIf
- `numpy`
- `pytest`: raises as assert_raises
- `numpy as np`
- `numpy.testing`: assert_, assert_equal
- `torch._numpy as np`
- `torch._numpy`: double, finfo, half, iinfo, single
- `torch._numpy.testing`: assert_, assert_equal


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
python test/torch_np/numpy_tests/core/test_getlimits.py
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
- [`test_multiarray.py_docs.md`](./test_multiarray.py_docs.md)
- [`test_numerictypes.py_docs.md`](./test_numerictypes.py_docs.md)
- [`test_dtype.py_docs.md`](./test_dtype.py_docs.md)


## Cross-References

- **File Documentation**: `test_getlimits.py_docs.md`
- **Keyword Index**: `test_getlimits.py_kw.md`
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
python docs/test/torch_np/numpy_tests/core/test_getlimits.py_docs.md
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
- [`test_scalar_ctors.py_docs.md_docs.md`](./test_scalar_ctors.py_docs.md_docs.md)
- [`test_scalar_methods.py_kw.md_docs.md`](./test_scalar_methods.py_kw.md_docs.md)
- [`test_indexing.py_docs.md_docs.md`](./test_indexing.py_docs.md_docs.md)
- [`test_scalar_ctors.py_kw.md_docs.md`](./test_scalar_ctors.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_getlimits.py_docs.md_docs.md`
- **Keyword Index**: `test_getlimits.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
