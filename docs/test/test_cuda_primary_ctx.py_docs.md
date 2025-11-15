# Documentation: test_cuda_primary_ctx.py

## File Metadata
- **Path**: `test/test_cuda_primary_ctx.py`
- **Size**: 4880 bytes
- **Lines**: 129
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): TestCudaPrimaryCtx

### Functions
This file defines 5 function(s): setUp, test_set_device_0, test_str_repr, test_copy, test_pin_memory


## Key Components

The file contains 393 words across 129 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4880 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
