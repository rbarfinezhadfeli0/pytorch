# Documentation: `docs/test/jit/test_sparse.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_sparse.py_docs.md`
- **Size**: 6,914 bytes (6.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_sparse.py`

## File Metadata

- **Path**: `test/jit/test_sparse.py`
- **Size**: 3,842 bytes (3.75 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import io
import unittest

import torch
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    raise_on_run_directly,
    TEST_MKL,
)
from torch.testing._internal.jit_utils import JitTestCase


class TestSparse(JitTestCase):
    def test_freeze_sparse_coo(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()
                self.b = torch.rand(3, 4).to_sparse()

            def forward(self, x):
                return x + self.a + self.b

        x = torch.rand(3, 4).to_sparse()

        m = SparseTensorModule()
        unfrozen_result = m.forward(x)

        m.eval()
        frozen = torch.jit.freeze(torch.jit.script(m))

        frozen_result = frozen.forward(x)

        self.assertEqual(unfrozen_result, frozen_result)

        buffer = io.BytesIO()
        torch.jit.save(frozen, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(unfrozen_result, loaded_result)

    def test_serialize_sparse_coo(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()
                self.b = torch.rand(3, 4).to_sparse()

            def forward(self, x):
                return x + self.a + self.b

        x = torch.rand(3, 4).to_sparse()
        m = SparseTensorModule()
        expected_result = m.forward(x)

        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(m), buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(expected_result, loaded_result)

    @unittest.skipIf(IS_WINDOWS or not TEST_MKL, "Need MKL to run CSR matmul")
    def test_freeze_sparse_csr(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.rand(4, 4).to_sparse_csr()
                self.b = torch.rand(4, 4).to_sparse_csr()

            def forward(self, x):
                return x.matmul(self.a).matmul(self.b)

        x = torch.rand(4, 4).to_sparse_csr()

        m = SparseTensorModule()
        unfrozen_result = m.forward(x)

        m.eval()
        frozen = torch.jit.freeze(torch.jit.script(m))

        frozen_result = frozen.forward(x)

        self.assertEqual(unfrozen_result.to_dense(), frozen_result.to_dense())

        buffer = io.BytesIO()
        torch.jit.save(frozen, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(unfrozen_result.to_dense(), loaded_result.to_dense())

    @unittest.skipIf(IS_WINDOWS or not TEST_MKL, "Need MKL to run CSR matmul")
    def test_serialize_sparse_csr(self):
        class SparseTensorModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.rand(4, 4).to_sparse_csr()
                self.b = torch.rand(4, 4).to_sparse_csr()

            def forward(self, x):
                return x.matmul(self.a).matmul(self.b)

        x = torch.rand(4, 4).to_sparse_csr()
        m = SparseTensorModule()
        expected_result = m.forward(x)

        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(m), buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(expected_result.to_dense(), loaded_result.to_dense())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 5 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSparse`, `SparseTensorModule`, `SparseTensorModule`, `SparseTensorModule`, `SparseTensorModule`

**Functions defined**: `test_freeze_sparse_coo`, `__init__`, `forward`, `test_serialize_sparse_coo`, `__init__`, `forward`, `test_freeze_sparse_csr`, `__init__`, `forward`, `test_serialize_sparse_csr`, `__init__`, `forward`

**Key imports**: io, unittest, torch, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `unittest`
- `torch`
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/jit/test_sparse.py
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

- **File Documentation**: `test_sparse.py_docs.md`
- **Keyword Index**: `test_sparse.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/jit/test_sparse.py_docs.md
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

- **File Documentation**: `test_sparse.py_docs.md_docs.md`
- **Keyword Index**: `test_sparse.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
