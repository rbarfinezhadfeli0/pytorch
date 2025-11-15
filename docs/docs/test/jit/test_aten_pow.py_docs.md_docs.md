# Documentation: `docs/test/jit/test_aten_pow.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_aten_pow.py_docs.md`
- **Size**: 7,233 bytes (7.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_aten_pow.py`

## File Metadata

- **Path**: `test/jit/test_aten_pow.py`
- **Size**: 4,444 bytes (4.34 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import torch
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class TestAtenPow(TestCase):
    def test_aten_pow_zero_negative_exponent(self):
        """
        1. Testing a = int, b = int
        """

        @torch.jit.script
        def fn_int_int(a: int, b: int):
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_int_int(2, 1), 2**1)
        self.assertEqual(fn_int_int(2, 0), 2**0)
        self.assertEqual(fn_int_int(2, -2), 2 ** (-2))
        self.assertEqual(fn_int_int(-2, 2), (-2) ** 2)
        self.assertEqual(fn_int_int(-2, 0), (-2) ** 0)
        self.assertEqual(fn_int_int(-2, -2), (-2) ** (-2))
        self.assertEqual(fn_int_int(-2, -1), (-2) ** (-1))
        self.assertEqual(fn_int_int(0, 2), 0**1)
        self.assertEqual(fn_int_int(0, 0), 0**0)
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_int_int, 0, -2)

        """
        2. Testing a = int, b = float
        """

        @torch.jit.script
        def fn_int_float(a: int, b: float):
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_int_float(2, 2.5), 2**2.5)
        self.assertEqual(fn_int_float(2, -2.5), 2 ** (-2.5))
        self.assertEqual(fn_int_float(2, -0.0), 2 ** (-0.0))
        self.assertEqual(fn_int_float(2, 0.0), 2 ** (0.0))
        self.assertEqual(fn_int_float(-2, 2.0), (-2) ** 2.0)
        self.assertEqual(fn_int_float(-2, -2.0), (-2) ** (-2.0))
        self.assertEqual(fn_int_float(-2, -3.0), (-2) ** (-3.0))
        self.assertEqual(fn_int_float(-2, -0.0), (-2) ** (-0.0))
        self.assertEqual(fn_int_float(-2, 0.0), (-2) ** (0.0))
        self.assertEqual(fn_int_float(0, 2.0), 0**2.0)
        self.assertEqual(fn_int_float(0, 0.5), 0**0.5)
        self.assertEqual(fn_int_float(0, 0.0), 0**0.0)
        self.assertEqual(fn_int_float(0, -0.0), 0 ** (-0.0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_int_float, 0, -2.5)

        """
        3. Testing a = float, b = int
        """

        @torch.jit.script
        def fn_float_int(a: float, b: int):
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_float_int(2.5, 2), 2.5**2)
        self.assertEqual(fn_float_int(2.5, -2), 2.5 ** (-2))
        self.assertEqual(fn_float_int(2.5, -0), 2.5 ** (-0))
        self.assertEqual(fn_float_int(2.5, 0), 2.5**0)
        self.assertEqual(fn_float_int(-2.5, 2), 2.5**2)
        self.assertEqual(fn_float_int(-2.5, -2), (-2.5) ** (-2))
        self.assertEqual(fn_float_int(-2.5, -3), (-2.5) ** (-3))
        self.assertEqual(fn_float_int(-2.5, -0), (-2.5) ** (-0))
        self.assertEqual(fn_float_int(-2.5, 0), (-2.5) ** 0)
        self.assertEqual(fn_float_int(0.0, 2), 0**2)
        self.assertEqual(fn_float_int(0.0, 0), 0**0)
        self.assertEqual(fn_float_int(0.0, -0), 0 ** (-0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_float_int, 0.0, -2)

        """
        4. Testing a = float, b = float
        """

        @torch.jit.script
        def fn_float_float(a: float, b: float):
            return a**b

        # Existing correct behaviors of aten::pow
        self.assertEqual(fn_float_float(2.5, 2.0), 2.5**2.0)
        self.assertEqual(fn_float_float(2.5, -2.0), 2.5 ** (-2.0))
        self.assertEqual(fn_float_float(2.5, -0.0), 2.5 ** (-0.0))
        self.assertEqual(fn_float_float(2.5, 0.0), 2.5**0.0)
        self.assertEqual(fn_float_float(-2.5, 2.0), 2.5**2.0)
        self.assertEqual(fn_float_float(-2.5, -2.0), (-2.5) ** (-2.0))
        self.assertEqual(fn_float_float(-2.5, -3.0), (-2.5) ** (-3.0))
        self.assertEqual(fn_float_float(-2.5, -0.0), (-2.5) ** (-0.0))
        self.assertEqual(fn_float_float(-2.5, 0.0), (-2.5) ** 0.0)
        self.assertEqual(fn_float_float(0.0, 2.0), 0.0**2.0)
        self.assertEqual(fn_float_float(0.0, 0.0), 0.0**0.0)
        self.assertEqual(fn_float_float(0.0, -0.0), 0.0 ** (-0.0))
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_float_float, 0.0, -2.0)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

"""        1. Testing a = int, b = int

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAtenPow`

**Functions defined**: `test_aten_pow_zero_negative_exponent`, `fn_int_int`, `fn_int_float`, `fn_float_int`, `fn_float_float`

**Key imports**: torch, raise_on_run_directly, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/jit/test_aten_pow.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_dataclasses.py_docs.md`](./test_dataclasses.py_docs.md)
- [`test_recursive_script.py_docs.md`](./test_recursive_script.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_python_builtins.py_docs.md`](./test_python_builtins.py_docs.md)
- [`test_functional_blocks.py_docs.md`](./test_functional_blocks.py_docs.md)
- [`test_hooks_modules.py_docs.md`](./test_hooks_modules.py_docs.md)
- [`mydecorator.py_docs.md`](./mydecorator.py_docs.md)
- [`test_union.py_docs.md`](./test_union.py_docs.md)
- [`test_python_bindings.py_docs.md`](./test_python_bindings.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_aten_pow.py_docs.md`
- **Keyword Index**: `test_aten_pow.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/jit/test_aten_pow.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/jit`):

- [`test_attr.py_kw.md_docs.md`](./test_attr.py_kw.md_docs.md)
- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_dataclasses.py_docs.md_docs.md`](./test_dataclasses.py_docs.md_docs.md)
- [`test_aten_pow.py_kw.md_docs.md`](./test_aten_pow.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_graph_rewrite_passes.py_kw.md_docs.md`](./test_graph_rewrite_passes.py_kw.md_docs.md)
- [`test_module_containers.py_kw.md_docs.md`](./test_module_containers.py_kw.md_docs.md)
- [`test_complex.py_kw.md_docs.md`](./test_complex.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_aten_pow.py_docs.md_docs.md`
- **Keyword Index**: `test_aten_pow.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
