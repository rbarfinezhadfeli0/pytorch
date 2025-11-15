# Documentation: `test/inductor/test_external_callables.py`

## File Metadata

- **Path**: `test/inductor/test_external_callables.py`
- **Size**: 3,064 bytes (2.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import TEST_CUDA


class MatMulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.eye(128, 128) * 2, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.matrix)


# torch.add performs better than torch.mm and got chosen during tuning
def matmul_cpu(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


def matmul_dup(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


def matmul_cuda(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    torch.add(a, b, out=out)


class TestInductorExternalCallable(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._saved_config = config.save_config()

    def tearDown(self):
        super().tearDown()
        config.load_config(self._saved_config)

    def test_matmul_cpu(self):
        # 2I + 2I == (2I)(2I)
        x = torch.eye(128, 128) * 2
        opt_fn = torch.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_cpu]},
        )
        opt_fn_golden = torch.compile(MatMulModule(), options={"max_autotune": True})
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_cpu}) failed",
        )

    def test_matmul_dup(self):
        # 2I + 2I == (2I)(2I)
        x = torch.eye(128, 128) * 2
        # This should only register the first external call
        opt_fn = torch.compile(
            MatMulModule(),
            options={"max_autotune": True, "external_matmul": [matmul_dup, matmul_dup]},
        )
        opt_fn_golden = torch.compile(MatMulModule(), options={"max_autotune": True})
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_dup}) failed",
        )

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @unittest.skipIf(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (7, 0),
        "Triton does not support device capability < 7.0",
    )
    def test_matmul_cuda(self):
        device = torch.device("cuda")
        x = (torch.eye(128, 128) * 2).to(device=device)
        opt_fn = torch.compile(
            MatMulModule().to(device),
            options={"max_autotune": True, "external_matmul": [matmul_cuda]},
        )
        opt_fn_golden = torch.compile(
            MatMulModule().to(device), options={"max_autotune": True}
        )
        torch.testing.assert_close(
            opt_fn(x),
            opt_fn_golden(x),
            msg=f"torch.compile(..., external_matmul = {matmul_cuda}) failed",
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MatMulModule`, `TestInductorExternalCallable`

**Functions defined**: `__init__`, `forward`, `matmul_cpu`, `matmul_dup`, `matmul_cuda`, `setUpClass`, `tearDown`, `test_matmul_cpu`, `test_matmul_dup`, `test_matmul_cuda`

**Key imports**: unittest, torch, config, run_tests, TestCase, TEST_CUDA


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch._inductor`: config
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal.common_cuda`: TEST_CUDA


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python test/inductor/test_external_callables.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_external_callables.py_docs.md`
- **Keyword Index**: `test_external_callables.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
