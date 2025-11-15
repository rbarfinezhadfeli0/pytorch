# Documentation: `docs/test/jit/test_data_parallel.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_data_parallel.py_docs.md`
- **Size**: 9,070 bytes (8.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_data_parallel.py`

## File Metadata

- **Path**: `test/jit/test_data_parallel.py`
- **Size**: 5,633 bytes (5.50 KB)
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

import torch
import torch.nn as nn
import torch.nn.parallel as dp


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA_MULTI_GPU


class TestDataParallel(JitTestCase):
    class Mpy(torch.nn.Module):
        def __init__(self) -> None:
            super(TestDataParallel.Mpy, self).__init__()
            self.m = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 2)
            )

        @torch.jit.ignore
        def forward(self, input):
            return self.m(input)

    class Mpy1(torch.nn.Module):
        def __init__(self, block):
            super(TestDataParallel.Mpy1, self).__init__()
            self.m = block

        @torch.jit.ignore
        def forward(self, input):
            return self.m.forward(input)

    class Mpy2(torch.nn.Module):
        def __init__(self, block1, block2):
            super(TestDataParallel.Mpy2, self).__init__()
            self.m1 = block1
            self.m2 = block2

        @torch.jit.ignore
        def forward(self, input):
            x = self.m1.forward(input)
            return self.m2(x)

    class Msm(torch.jit.ScriptModule):
        __constants__ = ["m"]

        def __init__(self) -> None:
            super(TestDataParallel.Msm, self).__init__()
            self.m = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 2)
            )

        @torch.jit.script_method
        def forward(self, input):
            return self.m(input)

    class Msm1(torch.jit.ScriptModule):
        def __init__(self, block):
            super(TestDataParallel.Msm1, self).__init__()
            self.block = block

        @torch.jit.script_method
        def forward(self, input):
            x = self.block(input)
            return x

    def check_replicas(self, module, replicas, input_shape=(2, 2)):
        input = torch.randn(input_shape).cuda()
        expected_output = module(input).data
        for i, replica in enumerate(replicas):
            for p in replica.parameters():
                self.assertEqual(p.get_device(), i)
            for b in replica.buffers():
                self.assertEqual(b.get_device(), i)
            replica_input = input.cuda(i)
            self.assertEqual(replica(replica_input).data, expected_output)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_python_submodule_script(self):
        module = self.Mpy1(self.Msm()).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_shared_module(self):
        s = self.Msm()
        p1 = self.Mpy1(s)
        module = self.Mpy2(p1, s).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_traced_module(self):
        module = torch.jit.trace(self.Mpy1(self.Mpy()), torch.ones(2, 2)).cuda()
        replicas = dp.replicate(module, {0, 1})
        self.check_replicas(module, replicas)

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_tensor_sharing(self):
        module = self.Msm1(self.Msm()).cuda()
        replica = dp.replicate(module, {0, 1})

        def assert_share_data(t1, t2):
            # Only checks that they point to the same memory on the same device.
            return (
                t1.device == t2.device
                and t1.storage().data_ptr() == t2.storage().data_ptr()
            )

        for p1, p2 in zip(module.parameters(), replica[0].parameters()):
            self.assertTrue(assert_share_data(p1, p2))

        for p1, p2 in zip(module.buffers(), replica[0].buffers()):
            self.assertTrue(assert_share_data(p1, p2))

        for p1, p2 in zip(module.parameters(), replica[1].parameters()):
            self.assertFalse(assert_share_data(p1, p2))

        for p1, p2 in zip(module.buffers(), replica[1].buffers()):
            self.assertFalse(assert_share_data(p1, p2))

    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_tensor_sharing_with_forward(self):
        module = self.Msm1(self.Msm()).cuda()
        replica = dp.replicate(module, {0, 1})
        x = torch.ones(2, 2, requires_grad=True).cuda()
        first_forward = module(x)
        first_forward.sum().backward()
        with torch.no_grad():
            for p in module.parameters():
                # Use .data here to avoid version counter bump.
                # The graph created by the following forward will be wrong but
                # we never backward through them so it's fine
                p.data -= 1.0 * p.grad
        second_forward = module(x)

        # replica which is on the same GPU has a shallow copy of the original
        # params and buffers
        r0_forward = replica[0](x)
        self.assertEqual(second_forward, r0_forward)

        # replica which is on a different GPU has a deep copy of the original
        # params and buffers
        x1 = torch.ones(2, 2, requires_grad=True).cuda(device=1)
        r1_forward = replica[1](x1)
        self.assertEqual(first_forward, r1_forward)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 6 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDataParallel`, `Mpy`, `Mpy1`, `Mpy2`, `Msm`, `Msm1`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `check_replicas`, `test_python_submodule_script`, `test_shared_module`, `test_traced_module`, `test_tensor_sharing`, `assert_share_data`, `test_tensor_sharing_with_forward`

**Key imports**: os, sys, unittest, torch, torch.nn as nn, torch.nn.parallel as dp, raise_on_run_directly, JitTestCase, RUN_CUDA_MULTI_GPU


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
- `torch`
- `torch.nn as nn`
- `torch.nn.parallel as dp`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase, RUN_CUDA_MULTI_GPU


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
python test/jit/test_data_parallel.py
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

- **File Documentation**: `test_data_parallel.py_docs.md`
- **Keyword Index**: `test_data_parallel.py_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/jit/test_data_parallel.py_docs.md
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

- **File Documentation**: `test_data_parallel.py_docs.md_docs.md`
- **Keyword Index**: `test_data_parallel.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
