# Documentation: `test/jit/test_tensor_creation_ops.py`

## File Metadata

- **Path**: `test/jit/test_tensor_creation_ops.py`
- **Size**: 2,930 bytes (2.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestTensorCreationOps(JitTestCase):
    """
    A suite of tests for ops that create tensors.
    """

    def test_randperm_default_dtype(self):
        def randperm(x: int):
            perm = torch.randperm(x)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert perm.dtype == torch.int64

        self.checkScript(randperm, (3,))

    def test_randperm_specifed_dtype(self):
        def randperm(x: int):
            perm = torch.randperm(x, dtype=torch.float)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert perm.dtype == torch.float

        self.checkScript(randperm, (3,))

    def test_triu_indices_default_dtype(self):
        def triu_indices(rows: int, cols: int):
            indices = torch.triu_indices(rows, cols)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int64

        self.checkScript(triu_indices, (3, 3))

    def test_triu_indices_specified_dtype(self):
        def triu_indices(rows: int, cols: int):
            indices = torch.triu_indices(rows, cols, dtype=torch.int32)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int32

        self.checkScript(triu_indices, (3, 3))

    def test_tril_indices_default_dtype(self):
        def tril_indices(rows: int, cols: int):
            indices = torch.tril_indices(rows, cols)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int64

        self.checkScript(tril_indices, (3, 3))

    def test_tril_indices_specified_dtype(self):
        def tril_indices(rows: int, cols: int):
            indices = torch.tril_indices(rows, cols, dtype=torch.int32)
            # Have to perform assertion here because TorchScript returns dtypes
            # as integers, which are not comparable against eager torch.dtype.
            assert indices.dtype == torch.int32

        self.checkScript(tril_indices, (3, 3))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

"""    A suite of tests for ops that create tensors.

This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTensorCreationOps`

**Functions defined**: `test_randperm_default_dtype`, `randperm`, `test_randperm_specifed_dtype`, `randperm`, `test_triu_indices_default_dtype`, `triu_indices`, `test_triu_indices_specified_dtype`, `triu_indices`, `test_tril_indices_default_dtype`, `tril_indices`, `test_tril_indices_specified_dtype`, `tril_indices`

**Key imports**: os, sys, torch, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


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
python test/jit/test_tensor_creation_ops.py
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

- **File Documentation**: `test_tensor_creation_ops.py_docs.md`
- **Keyword Index**: `test_tensor_creation_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
