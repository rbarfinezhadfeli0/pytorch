# Documentation: `docs/test/test_complex.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_complex.py_docs.md`
- **Size**: 17,389 bytes (16.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_complex.py`

## File Metadata

- **Path**: `test/test_complex.py`
- **Size**: 14,571 bytes (14.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: allow-untyped-defs
# Owner(s): ["module: complex"]

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCPU,
)
from torch.testing._internal.common_dtype import complex_types
from torch.testing._internal.common_utils import run_tests, set_default_dtype, TestCase


devices = (torch.device("cpu"), torch.device("cuda:0"))


class TestComplexTensor(TestCase):
    @dtypes(*complex_types())
    def test_to_list(self, device, dtype):
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(
            torch.zeros((2, 2), device=device, dtype=dtype).tolist(),
            [[0j, 0j], [0j, 0j]],
        )

    @dtypes(torch.float32, torch.float64, torch.float16)
    def test_dtype_inference(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/36834
        with set_default_dtype(dtype):
            x = torch.tensor([3.0, 3.0 + 5.0j], device=device)
        if dtype == torch.float16:
            self.assertEqual(x.dtype, torch.chalf)
        elif dtype == torch.float32:
            self.assertEqual(x.dtype, torch.cfloat)
        else:
            self.assertEqual(x.dtype, torch.cdouble)

    @dtypes(*complex_types())
    def test_conj_copy(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/106051
        x1 = torch.tensor([5 + 1j, 2 + 2j], device=device, dtype=dtype)
        xc1 = torch.conj(x1)
        x1.copy_(xc1)
        self.assertEqual(x1, torch.tensor([5 - 1j, 2 - 2j], device=device, dtype=dtype))

    @dtypes(*complex_types())
    def test_all(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/120875
        x = torch.tensor([1 + 2j, 3 - 4j, 5j, 6], device=device, dtype=dtype)

        self.assertTrue(torch.all(x))

    @dtypes(*complex_types())
    def test_any(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/120875
        x = torch.tensor(
            [0, 0j, -0 + 0j, -0 - 0j, 0 + 0j, 0 - 0j], device=device, dtype=dtype
        )

        self.assertFalse(torch.any(x))

    @onlyCPU
    @dtypes(*complex_types())
    def test_eq(self, device, dtype):
        "Test eq on complex types"
        nan = float("nan")
        # Non-vectorized operations
        for a, b in (
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-6.1278 - 8.5019j], device=device, dtype=dtype),
            ),
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-6.1278 - 2.1172j], device=device, dtype=dtype),
            ),
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-0.0610 - 8.5019j], device=device, dtype=dtype),
            ),
        ):
            actual = torch.eq(a, b)
            expected = torch.tensor([False], device=device, dtype=torch.bool)
            self.assertEqual(
                actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}"
            )

            actual = torch.eq(a, a)
            expected = torch.tensor([True], device=device, dtype=torch.bool)
            self.assertEqual(
                actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}"
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, b, out=actual)
            expected = torch.tensor([complex(0)], device=device, dtype=dtype)
            self.assertEqual(
                actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}"
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, a, out=actual)
            expected = torch.tensor([complex(1)], device=device, dtype=dtype)
            self.assertEqual(
                actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}"
            )

        # Vectorized operations
        for a, b in (
            (
                torch.tensor(
                    [
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(2.8871, nan),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(nan, -3.2650),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
                torch.tensor(
                    [
                        -6.1278 - 8.5019j,
                        0.5886 + 8.8816j,
                        complex(2.8871, nan),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                        -6.1278 - 2.1172j,
                        5.1576 + 8.8816j,
                        complex(nan, -3.2650),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
            ),
        ):
            actual = torch.eq(a, b)
            expected = torch.tensor(
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
                device=device,
                dtype=torch.bool,
            )
            self.assertEqual(
                actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}"
            )

            actual = torch.eq(a, a)
            expected = torch.tensor(
                [
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                ],
                device=device,
                dtype=torch.bool,
            )
            self.assertEqual(
                actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}"
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, b, out=actual)
            expected = torch.tensor(
                [
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}"
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, a, out=actual)
            expected = torch.tensor(
                [
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}"
            )

    @onlyCPU
    @dtypes(*complex_types())
    def test_ne(self, device, dtype):
        "Test ne on complex types"
        nan = float("nan")
        # Non-vectorized operations
        for a, b in (
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-6.1278 - 8.5019j], device=device, dtype=dtype),
            ),
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-6.1278 - 2.1172j], device=device, dtype=dtype),
            ),
            (
                torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
                torch.tensor([-0.0610 - 8.5019j], device=device, dtype=dtype),
            ),
        ):
            actual = torch.ne(a, b)
            expected = torch.tensor([True], device=device, dtype=torch.bool)
            self.assertEqual(
                actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}"
            )

            actual = torch.ne(a, a)
            expected = torch.tensor([False], device=device, dtype=torch.bool)
            self.assertEqual(
                actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}"
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, b, out=actual)
            expected = torch.tensor([complex(1)], device=device, dtype=dtype)
            self.assertEqual(
                actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}"
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, a, out=actual)
            expected = torch.tensor([complex(0)], device=device, dtype=dtype)
            self.assertEqual(
                actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}"
            )

        # Vectorized operations
        for a, b in (
            (
                torch.tensor(
                    [
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(2.8871, nan),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                        -0.0610 - 2.1172j,
                        5.1576 + 5.4775j,
                        complex(nan, -3.2650),
                        -6.6545 - 3.7655j,
                        -2.7036 - 1.4470j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
                torch.tensor(
                    [
                        -6.1278 - 8.5019j,
                        0.5886 + 8.8816j,
                        complex(2.8871, nan),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                        -6.1278 - 2.1172j,
                        5.1576 + 8.8816j,
                        complex(nan, -3.2650),
                        6.3505 + 2.2683j,
                        0.3712 + 7.9659j,
                        0.3712 + 7.989j,
                    ],
                    device=device,
                    dtype=dtype,
                ),
            ),
        ):
            actual = torch.ne(a, b)
            expected = torch.tensor(
                [
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
                device=device,
                dtype=torch.bool,
            )
            self.assertEqual(
                actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}"
            )

            actual = torch.ne(a, a)
            expected = torch.tensor(
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
                device=device,
                dtype=torch.bool,
            )
            self.assertEqual(
                actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}"
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, b, out=actual)
            expected = torch.tensor(
                [
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(1),
                    complex(0),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}"
            )

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, a, out=actual)
            expected = torch.tensor(
                [
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(0),
                    complex(1),
                    complex(0),
                    complex(0),
                    complex(0),
                ],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(
                actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}"
            )


instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestComplexTensor`

**Functions defined**: `test_to_list`, `test_dtype_inference`, `test_conj_copy`, `test_all`, `test_any`, `test_eq`, `test_ne`

**Key imports**: torch, complex_types, run_tests, set_default_dtype, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_dtype`: complex_types
- `torch.testing._internal.common_utils`: run_tests, set_default_dtype, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_complex.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_complex.py_docs.md`
- **Keyword Index**: `test_complex.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_complex.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_complex.py_docs.md_docs.md`
- **Keyword Index**: `test_complex.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
