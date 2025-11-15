# Documentation: `docs/test/jit/test_backend_nnapi.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_backend_nnapi.py_docs.md`
- **Size**: 8,340 bytes (8.14 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_backend_nnapi.py`

## File Metadata

- **Path**: `test/jit/test_backend_nnapi.py`
- **Size**: 5,236 bytes (5.11 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys
import unittest
from pathlib import Path

import torch
import torch._C
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    raise_on_run_directly,
    skipIfTorchDynamo,
)


# hacky way to skip these tests in fbcode:
# during test execution in fbcode, test_nnapi is available during test discovery,
# but not during test execution. So we can't try-catch here, otherwise it'll think
# it sees tests but then fails when it tries to actually run them.
if not IS_FBCODE:
    from test_nnapi import TestNNAPI

    HAS_TEST_NNAPI = True
else:
    from torch.testing._internal.common_utils import TestCase as TestNNAPI

    HAS_TEST_NNAPI = False


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

"""
Unit Tests for Nnapi backend with delegate
Inherits most tests from TestNNAPI, which loads Android NNAPI models
without the delegate API.
"""
# First skip is needed for IS_WINDOWS or IS_MACOS to skip the tests.
torch_root = Path(__file__).resolve().parents[2]
lib_path = torch_root / "build" / "lib" / "libnnapi_backend.so"


@skipIfTorchDynamo("weird py38 failures")
@unittest.skipIf(
    not os.path.exists(lib_path),
    "Skipping the test as libnnapi_backend.so was not found",
)
@unittest.skipIf(IS_FBCODE, "test_nnapi.py not found")
class TestNnapiBackend(TestNNAPI):
    def setUp(self):
        super().setUp()

        # Save default dtype
        module = torch.nn.PReLU()
        self.default_dtype = module.weight.dtype
        # Change dtype to float32 (since a different unit test changed dtype to float64,
        # which is not supported by the Android NNAPI delegate)
        # Float32 should typically be the default in other files.
        torch.set_default_dtype(torch.float32)

        # Load nnapi delegate library
        torch.ops.load_library(str(lib_path))

    # Override
    def call_lowering_to_nnapi(self, traced_module, args):
        compile_spec = {"forward": {"inputs": args}}
        return torch._C._jit_to_backend("nnapi", traced_module, compile_spec)

    def test_tensor_input(self):
        # Lower a simple module
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        module = torch.nn.PReLU()
        traced = torch.jit.trace(module, args)

        # Argument input is a single Tensor
        self.call_lowering_to_nnapi(traced, args)
        # Argument input is a Tensor in a list
        self.call_lowering_to_nnapi(traced, [args])

    # Test exceptions for incorrect compile specs
    def test_compile_spec_santiy(self):
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        module = torch.nn.PReLU()
        traced = torch.jit.trace(module, args)

        errorMsgTail = r"""
method_compile_spec should contain a Tensor or Tensor List which bundles input parameters: shape, dtype, quantization, and dimorder.
For input shapes, use 0 for run/load time flexible input.
method_compile_spec must use the following format:
{"forward": {"inputs": at::Tensor}} OR {"forward": {"inputs": c10::List<at::Tensor>}}"""

        # No forward key
        compile_spec = {"backward": {"inputs": args}}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain the "forward" key.' + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # No dictionary under the forward key
        compile_spec = {"forward": 1}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain a dictionary with an "inputs" key, '
            'under it\'s "forward" key.' + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # No inputs key (in the dictionary under the forward key)
        compile_spec = {"forward": {"not inputs": args}}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain a dictionary with an "inputs" key, '
            'under it\'s "forward" key.' + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # No Tensor or TensorList under the inputs key
        compile_spec = {"forward": {"inputs": 1}}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain either a Tensor or TensorList, under it\'s "inputs" key.'
            + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)
        compile_spec = {"forward": {"inputs": [1]}}
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain either a Tensor or TensorList, under it\'s "inputs" key.'
            + errorMsgTail,
        ):
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

    def tearDown(self):
        # Change dtype back to default (Otherwise, other unit tests will complain)
        torch.set_default_dtype(self.default_dtype)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

"""Unit Tests for Nnapi backend with delegateInherits most tests from TestNNAPI, which loads Android NNAPI modelswithout the delegate API.

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNnapiBackend`

**Functions defined**: `setUp`, `call_lowering_to_nnapi`, `test_tensor_input`, `test_compile_spec_santiy`, `tearDown`

**Key imports**: os, sys, unittest, Path, torch, torch._C, TestNNAPI, TestCase as TestNNAPI


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `unittest`
- `pathlib`: Path
- `torch`
- `torch._C`
- `test_nnapi`: TestNNAPI
- `torch.testing._internal.common_utils`: TestCase as TestNNAPI


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
python test/jit/test_backend_nnapi.py
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

- **File Documentation**: `test_backend_nnapi.py_docs.md`
- **Keyword Index**: `test_backend_nnapi.py_kw.md`
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
python docs/test/jit/test_backend_nnapi.py_docs.md
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

- **File Documentation**: `test_backend_nnapi.py_docs.md_docs.md`
- **Keyword Index**: `test_backend_nnapi.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
