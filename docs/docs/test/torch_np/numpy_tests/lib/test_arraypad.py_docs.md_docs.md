# Documentation: `docs/test/torch_np/numpy_tests/lib/test_arraypad.py_docs.md`

## File Metadata

- **Path**: `docs/test/torch_np/numpy_tests/lib/test_arraypad.py_docs.md`
- **Size**: 16,585 bytes (16.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/torch_np/numpy_tests/lib/test_arraypad.py`

## File Metadata

- **Path**: `test/torch_np/numpy_tests/lib/test_arraypad.py`
- **Size**: 13,648 bytes (13.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

from unittest import skipIf as skipif

from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


# If we are going to trace through these, we should use NumPy
# If testing on eager mode, we use torch._numpy
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_allclose, assert_array_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_allclose, assert_array_equal


class TestConstant(TestCase):
    @xpassIfTorchDynamo_np  # (reason="tuple values")
    def test_check_constant(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), "constant", constant_values=(10, 20))
        b = np.array(
            [
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
            ]
        )
        assert_array_equal(a, b)

    def test_check_constant_zeros(self):
        a = np.arange(100)
        a = np.pad(a, (25, 20), "constant")
        b = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        assert_array_equal(a, b)

    def test_check_constant_float(self):
        # If input array is int, but constant_values are float, the dtype of
        # the array to be padded is kept
        arr = np.arange(30).reshape(5, 6)
        test = np.pad(arr, (1, 2), mode="constant", constant_values=1.1)
        expected = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 2, 3, 4, 5, 1, 1],
                [1, 6, 7, 8, 9, 10, 11, 1, 1],
                [1, 12, 13, 14, 15, 16, 17, 1, 1],
                [1, 18, 19, 20, 21, 22, 23, 1, 1],
                [1, 24, 25, 26, 27, 28, 29, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        assert_allclose(test, expected)

    def test_check_constant_float2(self):
        # If input array is float, and constant_values are float, the dtype of
        # the array to be padded is kept - here retaining the float constants
        arr = np.arange(30).reshape(5, 6)
        arr_float = arr.astype(np.float64)
        test = np.pad(arr_float, ((1, 2), (1, 2)), mode="constant", constant_values=1.1)
        expected = np.array(
            [
                [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                [1.1, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.1, 1.1],
                [1.1, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 1.1, 1.1],
                [1.1, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 1.1, 1.1],
                [1.1, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 1.1, 1.1],
                [1.1, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 1.1, 1.1],
                [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            ]
        )
        assert_allclose(test, expected)

    @xpassIfTorchDynamo_np  # (reason="tuple values")
    def test_check_constant_float3(self):
        a = np.arange(100, dtype=float)
        a = np.pad(a, (25, 20), "constant", constant_values=(-1.1, -1.2))
        b = np.array(
            [
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                -1.1,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
                -1.2,
            ]
        )
        assert_allclose(a, b)

    def test_check_constant_odd_pad_amount(self):
        arr = np.arange(30).reshape(5, 6)
        test = np.pad(arr, ((1,), (2,)), mode="constant", constant_values=3)
        expected = np.array(
            [
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [3, 3, 0, 1, 2, 3, 4, 5, 3, 3],
                [3, 3, 6, 7, 8, 9, 10, 11, 3, 3],
                [3, 3, 12, 13, 14, 15, 16, 17, 3, 3],
                [3, 3, 18, 19, 20, 21, 22, 23, 3, 3],
                [3, 3, 24, 25, 26, 27, 28, 29, 3, 3],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            ]
        )
        assert_allclose(test, expected)

    @xpassIfTorchDynamo_np  # (reason="tuple values")
    def test_check_constant_pad_2d(self):
        arr = np.arange(4).reshape(2, 2)
        test = np.pad(
            arr, ((1, 2), (1, 3)), mode="constant", constant_values=((1, 2), (3, 4))
        )
        expected = np.array(
            [
                [3, 1, 1, 4, 4, 4],
                [3, 0, 1, 4, 4, 4],
                [3, 2, 3, 4, 4, 4],
                [3, 2, 2, 4, 4, 4],
                [3, 2, 2, 4, 4, 4],
            ]
        )
        assert_allclose(test, expected)

    @skipif(
        True, reason="passes on MacOS, fails otherwise"
    )  # (reason="int64 overflow")
    def test_check_large_integers(self):
        int64_max = 2**63 - 1
        arr = np.full(5, int64_max, dtype=np.int64)
        test = np.pad(arr, 1, mode="constant", constant_values=arr.min())
        expected = np.full(7, int64_max, dtype=np.int64)
        assert_array_equal(test, expected)

    def test_pad_empty_dimension(self):
        arr = np.zeros((3, 0, 2))
        result = np.pad(arr, [(0,), (2,), (1,)], mode="constant")
        assert result.shape == (3, 4, 4)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestConstant`

**Functions defined**: `test_check_constant`, `test_check_constant_zeros`, `test_check_constant_float`, `test_check_constant_float2`, `test_check_constant_float3`, `test_check_constant_odd_pad_amount`, `test_check_constant_pad_2d`, `test_check_large_integers`, `test_pad_empty_dimension`

**Key imports**: skipIf as skipif, numpy as np, assert_allclose, assert_array_equal, torch._numpy as np, assert_allclose, assert_array_equal


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/torch_np/numpy_tests/lib`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`: skipIf as skipif
- `numpy as np`
- `numpy.testing`: assert_allclose, assert_array_equal
- `torch._numpy as np`
- `torch._numpy.testing`: assert_allclose, assert_array_equal


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
python test/torch_np/numpy_tests/lib/test_arraypad.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/torch_np/numpy_tests/lib`):

- [`test_arraysetops.py_docs.md`](./test_arraysetops.py_docs.md)
- [`test_function_base.py_docs.md`](./test_function_base.py_docs.md)
- [`test_index_tricks.py_docs.md`](./test_index_tricks.py_docs.md)
- [`test_shape_base_.py_docs.md`](./test_shape_base_.py_docs.md)
- [`test_histograms.py_docs.md`](./test_histograms.py_docs.md)
- [`test_twodim_base.py_docs.md`](./test_twodim_base.py_docs.md)
- [`test_type_check.py_docs.md`](./test_type_check.py_docs.md)


## Cross-References

- **File Documentation**: `test_arraypad.py_docs.md`
- **Keyword Index**: `test_arraypad.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/torch_np/numpy_tests/lib`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/torch_np/numpy_tests/lib`, which is part of the **core PyTorch library**.



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
python docs/test/torch_np/numpy_tests/lib/test_arraypad.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/torch_np/numpy_tests/lib`):

- [`test_function_base.py_docs.md_docs.md`](./test_function_base.py_docs.md_docs.md)
- [`test_histograms.py_docs.md_docs.md`](./test_histograms.py_docs.md_docs.md)
- [`test_histograms.py_kw.md_docs.md`](./test_histograms.py_kw.md_docs.md)
- [`test_function_base.py_kw.md_docs.md`](./test_function_base.py_kw.md_docs.md)
- [`test_twodim_base.py_docs.md_docs.md`](./test_twodim_base.py_docs.md_docs.md)
- [`test_type_check.py_kw.md_docs.md`](./test_type_check.py_kw.md_docs.md)
- [`test_shape_base_.py_docs.md_docs.md`](./test_shape_base_.py_docs.md_docs.md)
- [`test_shape_base_.py_kw.md_docs.md`](./test_shape_base_.py_kw.md_docs.md)
- [`test_arraysetops.py_kw.md_docs.md`](./test_arraysetops.py_kw.md_docs.md)
- [`test_arraypad.py_kw.md_docs.md`](./test_arraypad.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_arraypad.py_docs.md_docs.md`
- **Keyword Index**: `test_arraypad.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
