# Documentation: `test/torch_np/numpy_tests/core/test_scalar_methods.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/core/test_scalar_methods.py`
- **Size**: 8,022 bytes (7.83 KB)
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

import fractions
import functools
import types
from typing import Any
from unittest import skipIf as skipif, SkipTest

import pytest
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal


skip = functools.partial(skipif, True)


@skip(reason="XXX: scalar.as_integer_ratio not implemented")
@instantiate_parametrized_tests
class TestAsIntegerRatio(TestCase):
    # derived in part from the cpython test "test_floatasratio"

    @parametrize("ftype", [np.half, np.single, np.double])
    @parametrize(
        "f, ratio",
        [
            (0.875, (7, 8)),
            (-0.875, (-7, 8)),
            (0.0, (0, 1)),
            (11.5, (23, 2)),
        ],
    )
    def test_small(self, ftype, f, ratio):
        assert_equal(ftype(f).as_integer_ratio(), ratio)

    @parametrize("ftype", [np.half, np.single, np.double])
    def test_simple_fractions(self, ftype):
        R = fractions.Fraction
        assert_equal(R(0, 1), R(*ftype(0.0).as_integer_ratio()))
        assert_equal(R(5, 2), R(*ftype(2.5).as_integer_ratio()))
        assert_equal(R(1, 2), R(*ftype(0.5).as_integer_ratio()))
        assert_equal(R(-2100, 1), R(*ftype(-2100.0).as_integer_ratio()))

    @parametrize("ftype", [np.half, np.single, np.double])
    def test_errors(self, ftype):
        assert_raises(OverflowError, ftype("inf").as_integer_ratio)
        assert_raises(OverflowError, ftype("-inf").as_integer_ratio)
        assert_raises(ValueError, ftype("nan").as_integer_ratio)

    def test_against_known_values(self):
        R = fractions.Fraction
        assert_equal(R(1075, 512), R(*np.half(2.1).as_integer_ratio()))
        assert_equal(R(-1075, 512), R(*np.half(-2.1).as_integer_ratio()))
        assert_equal(R(4404019, 2097152), R(*np.single(2.1).as_integer_ratio()))
        assert_equal(R(-4404019, 2097152), R(*np.single(-2.1).as_integer_ratio()))
        assert_equal(
            R(4728779608739021, 2251799813685248), R(*np.double(2.1).as_integer_ratio())
        )
        assert_equal(
            R(-4728779608739021, 2251799813685248),
            R(*np.double(-2.1).as_integer_ratio()),
        )
        # longdouble is platform dependent

    @parametrize(
        "ftype, frac_vals, exp_vals",
        [
            # dtype test cases generated using hypothesis
            # first five generated cases per dtype
            (
                np.half,
                [
                    0.0,
                    0.01154830649280303,
                    0.31082276347447274,
                    0.527350517124794,
                    0.8308562335072596,
                ],
                [0, 1, 0, -8, 12],
            ),
            (
                np.single,
                [
                    0.0,
                    0.09248576989263226,
                    0.8160498218131407,
                    0.17389442853722373,
                    0.7956044195067877,
                ],
                [0, 12, 10, 17, -26],
            ),
            (
                np.double,
                [
                    0.0,
                    0.031066908499895136,
                    0.5214135908877832,
                    0.45780736035689296,
                    0.5906586745934036,
                ],
                [0, -801, 51, 194, -653],
            ),
        ],
    )
    def test_roundtrip(self, ftype, frac_vals, exp_vals):
        for frac, exp in zip(frac_vals, exp_vals):
            f = np.ldexp(ftype(frac), exp)
            assert f.dtype == ftype
            n, d = f.as_integer_ratio()

            try:
                nf = np.longdouble(n)
                df = np.longdouble(d)
            except (OverflowError, RuntimeWarning):
                # the values may not fit in any float type
                raise SkipTest("longdouble too small on this platform")  # noqa: B904

            assert_equal(nf / df, f, f"{n}/{d}")


@skip(reason="NP_VER: older numpies has problems with .is_integer")
@instantiate_parametrized_tests
class TestIsInteger(TestCase):
    @parametrize("str_value", ["inf", "nan"])
    @parametrize("code", np.typecodes["Float"])
    def test_special(self, code, str_value):
        cls = np.dtype(code).type
        value = cls(str_value)
        assert not value.is_integer()

    @parametrize(
        "code", "efd" + "Bbhil"
    )  # np.typecodes["Float"] + np.typecodes["AllInteger"])
    def test_true(self, code: str) -> None:
        float_array = np.arange(-5, 5).astype(code)
        for value in float_array:
            assert value.is_integer()

    @parametrize("code", "bhil")  # np.typecodes["Float"])
    def test_false(self, code: str) -> None:
        float_array = np.arange(-5, 5).astype(code)
        float_array *= 1.1
        for value in float_array:
            if value == 0:
                continue
            assert not value.is_integer()


@skip(reason="XXX: implementation details of the type system differ")
@instantiate_parametrized_tests
class TestClassGetItem(TestCase):
    @parametrize(
        "cls",
        [
            np.number,
            np.integer,
            np.inexact,
            np.unsignedinteger,
            np.signedinteger,
            np.floating,
        ],
    )
    def test_abc(self, cls: type[np.number]) -> None:
        alias = cls[Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    def test_abc_complexfloating(self) -> None:
        alias = np.complexfloating[Any, Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is np.complexfloating

    @parametrize("arg_len", range(4))
    def test_abc_complexfloating_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            assert np.complexfloating[arg_tup]
        else:
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            with pytest.raises(TypeError, match=match):
                np.complexfloating[arg_tup]

    @parametrize("cls", [np.generic])
    def test_abc_non_numeric(self, cls: type[np.generic]) -> None:
        with pytest.raises(TypeError):
            cls[Any]

    @parametrize("code", np.typecodes["All"])
    def test_concrete(self, code: str) -> None:
        cls = np.dtype(code).type
        with pytest.raises(TypeError):
            cls[Any]

    @parametrize("arg_len", range(4))
    def test_subscript_tuple(self, arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len == 1:
            assert np.number[arg_tup]
        else:
            with pytest.raises(TypeError):
                np.number[arg_tup]

    def test_subscript_scalar(self) -> None:
        assert np.number[Any]


@skip(reason="scalartype(...).bit_count() not implemented")
@instantiate_parametrized_tests
class TestBitCount(TestCase):
    # derived in part from the cpython test "test_bit_count"

    @parametrize(
        "itype",
        [np.int8, np.int16, np.int32, np.int64]
        + [np.uint8, np.uint16, np.uint32, np.uint64],
    )
    def test_small(self, itype):
        for a in range(max(np.iinfo(itype).min, 0), 128):
            msg = f"Smoke test for {itype}({a}).bit_count()"
            assert itype(a).bit_count() == a.bit_count(), msg

    def test_bit_count(self):
        for exp in [10, 17, 63]:
            a = 2**exp
            assert np.uint64(a).bit_count() == 1
            assert np.uint64(a - 1).bit_count() == exp
            assert np.uint64(a ^ 63).bit_count() == 7
            assert np.uint64((a - 1) ^ 510).bit_count() == exp - 8


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test the scalar constructors, which also do type-coercion

This Python file contains 4 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAsIntegerRatio`, `TestIsInteger`, `TestClassGetItem`, `TestBitCount`

**Functions defined**: `test_small`, `test_simple_fractions`, `test_errors`, `test_against_known_values`, `test_roundtrip`, `test_special`, `test_true`, `test_false`, `test_abc`, `test_abc_complexfloating`, `test_abc_complexfloating_subscript_tuple`, `test_abc_non_numeric`, `test_concrete`, `test_subscript_tuple`, `test_subscript_scalar`, `test_small`, `test_bit_count`

**Key imports**: fractions, functools, types, Any, skipIf as skipif, SkipTest, pytest, raises as assert_raises, numpy as np, assert_equal, torch._numpy as np


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np/numpy_tests/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `fractions`
- `functools`
- `types`
- `typing`: Any
- `unittest`: skipIf as skipif, SkipTest
- `pytest`
- `numpy as np`
- `numpy.testing`: assert_equal
- `torch._numpy as np`
- `torch._numpy.testing`: assert_equal


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

This is a test file. Run it with:

```bash
python test/torch_np/numpy_tests/core/test_scalar_methods.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np/numpy_tests/core`):

- [`test_shape_base.py_docs.md`](./test_shape_base.py_docs.md)
- [`test_scalarinherit.py_docs.md`](./test_scalarinherit.py_docs.md)
- [`test_einsum.py_docs.md`](./test_einsum.py_docs.md)
- [`test_indexing.py_docs.md`](./test_indexing.py_docs.md)
- [`test_dlpack.py_docs.md`](./test_dlpack.py_docs.md)
- [`test_getlimits.py_docs.md`](./test_getlimits.py_docs.md)
- [`test_multiarray.py_docs.md`](./test_multiarray.py_docs.md)
- [`test_numerictypes.py_docs.md`](./test_numerictypes.py_docs.md)
- [`test_dtype.py_docs.md`](./test_dtype.py_docs.md)


## Cross-References

- **File Documentation**: `test_scalar_methods.py_docs.md`
- **Keyword Index**: `test_scalar_methods.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
