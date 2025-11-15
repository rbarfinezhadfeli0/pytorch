# Documentation: test_sparse.py

## File Metadata
- **Path**: `test/jit/test_sparse.py`
- **Size**: 3842 bytes
- **Lines**: 128
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 5 class(es): TestSparse, SparseTensorModule, SparseTensorModule, SparseTensorModule, SparseTensorModule

### Functions
This file defines 12 function(s): test_freeze_sparse_coo, __init__, forward, test_serialize_sparse_coo, __init__, forward, test_freeze_sparse_csr, __init__, forward, test_serialize_sparse_csr, __init__, forward


## Key Components

The file contains 259 words across 128 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 3842 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
