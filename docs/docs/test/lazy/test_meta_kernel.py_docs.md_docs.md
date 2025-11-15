# Documentation: `docs/test/lazy/test_meta_kernel.py_docs.md`

## File Metadata

- **Path**: `docs/test/lazy/test_meta_kernel.py_docs.md`
- **Size**: 4,678 bytes (4.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/lazy/test_meta_kernel.py`

## File Metadata

- **Path**: `test/lazy/test_meta_kernel.py`
- **Size**: 1,578 bytes (1.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import torch
import torch._lazy
import torch._lazy.ts_backend
from torch import float16, float32
from torch.testing._internal.common_utils import TestCase


torch._lazy.ts_backend.init()


class TestMetaKernel(TestCase):
    def test_addmm_invalid_dtype(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertTrue(input.dtype == torch.float16)

        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float32).to("lazy")

        with self.assertRaises(Exception):
            fc_nobias(input)

    def test_addmm(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertEqual(input.dtype, torch.float16)

        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float16).to("lazy")
        out_nobias = fc_nobias(input)
        self.assertEqual(out_nobias.dtype, torch.float16)

        fc_bias = torch.nn.Linear(2, 2, bias=True, dtype=float16).to("lazy")
        out_bias = fc_bias(input)
        self.assertEqual(out_bias.dtype, torch.float16)

    def test_add_invalid_device(self):
        with self.assertRaisesRegex(RuntimeError, ".*not a lazy tensor.*"):
            _ = torch.tensor([1], device="cpu") + torch.tensor([1], device="lazy")


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview

"""Tests that the addmm meta kernel returns the correct output type"""        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")        self.assertTrue(input.dtype == torch.float16)        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float32).to("lazy")        with self.assertRaises(Exception):            fc_nobias(input)    def test_addmm(self):

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestMetaKernel`

**Functions defined**: `test_addmm_invalid_dtype`, `test_addmm`, `test_add_invalid_device`

**Key imports**: torch, torch._lazy, torch._lazy.ts_backend, float16, float32, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._lazy`
- `torch._lazy.ts_backend`
- `torch.testing._internal.common_utils`: TestCase


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/lazy/test_meta_kernel.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/lazy`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ts_opinfo.py_docs.md`](./test_ts_opinfo.py_docs.md)
- [`test_functionalization.py_docs.md`](./test_functionalization.py_docs.md)
- [`test_generator.py_docs.md`](./test_generator.py_docs.md)
- [`test_bindings.py_docs.md`](./test_bindings.py_docs.md)
- [`test_extract_compiled_graph.py_docs.md`](./test_extract_compiled_graph.py_docs.md)
- [`test_reuse_ir.py_docs.md`](./test_reuse_ir.py_docs.md)
- [`test_step_closures.py_docs.md`](./test_step_closures.py_docs.md)
- [`test_debug_util.py_docs.md`](./test_debug_util.py_docs.md)


## Cross-References

- **File Documentation**: `test_meta_kernel.py_docs.md`
- **Keyword Index**: `test_meta_kernel.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/lazy`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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
python docs/test/lazy/test_meta_kernel.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/lazy`):

- [`test_reuse_ir.py_kw.md_docs.md`](./test_reuse_ir.py_kw.md_docs.md)
- [`test_extract_compiled_graph.py_kw.md_docs.md`](./test_extract_compiled_graph.py_kw.md_docs.md)
- [`test_reuse_ir.py_docs.md_docs.md`](./test_reuse_ir.py_docs.md_docs.md)
- [`test_step_closures.py_kw.md_docs.md`](./test_step_closures.py_kw.md_docs.md)
- [`test_step_closures.py_docs.md_docs.md`](./test_step_closures.py_docs.md_docs.md)
- [`test_ts_opinfo.py_kw.md_docs.md`](./test_ts_opinfo.py_kw.md_docs.md)
- [`test_meta_kernel.py_kw.md_docs.md`](./test_meta_kernel.py_kw.md_docs.md)
- [`test_bindings.py_kw.md_docs.md`](./test_bindings.py_kw.md_docs.md)
- [`test_ts_opinfo.py_docs.md_docs.md`](./test_ts_opinfo.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_meta_kernel.py_docs.md_docs.md`
- **Keyword Index**: `test_meta_kernel.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
