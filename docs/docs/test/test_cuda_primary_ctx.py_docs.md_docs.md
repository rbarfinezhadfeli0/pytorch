# Documentation: `docs/test/test_cuda_primary_ctx.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_cuda_primary_ctx.py_docs.md`
- **Size**: 7,838 bytes (7.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_cuda_primary_ctx.py`

## File Metadata

- **Path**: `test/test_cuda_primary_ctx.py`
- **Size**: 4,880 bytes (4.77 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: cuda"]

import sys
import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU
from torch.testing._internal.common_utils import NoTest, run_tests, skipIfRocm, TestCase


# NOTE: this needs to be run in a brand new process

if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCudaPrimaryCtx(TestCase):
    CTX_ALREADY_CREATED_ERR_MSG = (
        "Tests defined in test_cuda_primary_ctx.py must be run in a process "
        "where CUDA contexts are never created. Use either run_test.py or add "
        "--subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        super().setUp()
        for device in range(torch.cuda.device_count()):
            # Ensure context has not been created beforehand
            self.assertFalse(
                torch._C._cuda_hasPrimaryContext(device),
                TestCudaPrimaryCtx.CTX_ALREADY_CREATED_ERR_MSG,
            )

    @skipIfRocm(
        msg="last checked in ROCm 7, HIP runtime doesn't create context for hipSetDevice()"
    )
    def test_set_device_0(self):
        # In CUDA 12 the behavior of cudaSetDevice has changed. It eagerly creates context on target.
        # The behavior of `torch.cuda.set_device(0)` should also create context on the device 0.
        # Initially, we should not have any context on device 0.
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        torch.cuda.set_device(0)
        # Now after the device was set, the context should present in CUDA 12.
        self.assertTrue(torch._C._cuda_hasPrimaryContext(0))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_str_repr(self):
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        str(x)
        repr(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_copy(self):
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        y = torch.randn(1, device="cpu")
        y.copy_(x)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_pin_memory(self):
        x = torch.randn(1, device="cuda:1")

        # We should have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        self.assertFalse(x.is_pinned())

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.randn(3, device="cpu").pin_memory()

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        self.assertTrue(x.is_pinned())

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.randn(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.zeros(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = torch.empty(3, device="cpu", pin_memory=True)

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))

        x = x.pin_memory()

        # We should still have only created context on 'cuda:1'
        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))
        self.assertTrue(torch._C._cuda_hasPrimaryContext(1))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCudaPrimaryCtx`

**Functions defined**: `setUp`, `test_set_device_0`, `test_str_repr`, `test_copy`, `test_pin_memory`

**Key imports**: sys, unittest, torch, TEST_CUDA, TEST_MULTIGPU, NoTest, run_tests, skipIfRocm, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `unittest`
- `torch`
- `torch.testing._internal.common_cuda`: TEST_CUDA, TEST_MULTIGPU
- `torch.testing._internal.common_utils`: NoTest, run_tests, skipIfRocm, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_cuda_primary_ctx.py
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

- **File Documentation**: `test_cuda_primary_ctx.py_docs.md`
- **Keyword Index**: `test_cuda_primary_ctx.py_kw.md`
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
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_cuda_primary_ctx.py_docs.md
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

- **File Documentation**: `test_cuda_primary_ctx.py_docs.md_docs.md`
- **Keyword Index**: `test_cuda_primary_ctx.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
