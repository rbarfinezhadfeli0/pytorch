# Documentation: `docs/test/jit/test_convert_activation.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_convert_activation.py_docs.md`
- **Size**: 9,964 bytes (9.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_convert_activation.py`

## File Metadata

- **Path**: `test/jit/test_convert_activation.py`
- **Size**: 6,437 bytes (6.29 KB)
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
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import FileCheck


try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


activations = [
    F.celu,
    F.elu,
    F.hardsigmoid,
    F.hardswish,
    F.hardtanh,
    F.leaky_relu,
    F.relu,
    F.relu6,
    F.rrelu,
    F.selu,
    F.silu,
]


class TestFunctionalToInplaceActivation(JitTestCase):
    def test_check_no_type_promotion(self):
        dtypes = [
            torch.bool,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
        ]
        # restore_mutation.h contains a mapping from activation operators
        # to whether they allow type conversion. Use this checking to
        # guard the mapping, and if any later change breaks the assumption
        # we need to update the mapping correspondingly.
        for activation, dtype in product(activations, dtypes):
            inp = torch.normal(0, 5, size=(4, 4)).to(dtype)
            try:
                out = activation(inp)
                self.assertEqual(dtype, out.dtype)
            except RuntimeError:
                # Skip the not implemented error
                pass

    def test_functional_to_inplace_activation(self):
        for activation in activations:

            def test_basic(x):
                y = x + 1
                z = activation(y)
                return z

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)
            self.run_pass("functional_to_inplace_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}(").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            inp = torch.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

    def test_no_functional_to_inplace(self):
        # inplace conversion should not happen because sigmoid may
        # perform type conversion
        def test1():
            y = torch.ones([2, 2])
            z = torch.sigmoid(y)
            return z

        fn = torch.jit.script(test1)
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::sigmoid_").run(fn.graph)

        # inplace conversion should not happen because y is alias
        # the input x
        def test2(x):
            y = x[0]
            z = torch.relu(y)
            return z

        fn = torch.jit.script(test2)
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::relu_").run(fn.graph)

        # inplace conversion should not happen because self.x is
        # at the global scope
        class Test3(nn.Module):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def forward(self):
                y = torch.relu(self.x)
                return y

        fn = torch.jit.script(Test3(torch.rand([2, 2])).eval())
        self.run_pass("functional_to_inplace_activation", fn.graph)
        FileCheck().check_not("aten::relu_").run(fn.graph)

    @skipIfNoTorchVision
    def test_resnet18_correctness(self):
        model = torchvision.models.resnet18()
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        inp = torch.randn(N, C, H, W)
        self.run_pass("functional_to_inplace_activation", frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))


class TestInplaceToFunctionalActivation(JitTestCase):
    def test_inplace_to_functional_activation(self):
        for activation in activations:

            def test_basic(x):
                y = x + 1
                activation(y, inplace=True)
                return y

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}_").run(fn.graph)
            self.run_pass("inplace_to_functional_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}_").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__}(").run(fn.graph)

        for activation in [
            torch.relu_,
            torch.sigmoid_,
            torch.tanh_,
        ]:

            def test_basic(x):
                y = x + 1
                activation(y)
                return y

            fn = torch.jit.script(test_basic)
            self.run_pass("inline", fn.graph)
            self.run_pass("constant_propagation", fn.graph)
            FileCheck().check(f"aten::{activation.__name__}").run(fn.graph)
            self.run_pass("inplace_to_functional_activation", fn.graph)
            FileCheck().check_not(f"aten::{activation.__name__}").run(fn.graph)
            FileCheck().check(f"aten::{activation.__name__[:-1]}(").run(fn.graph)

            inp = torch.rand([2, 2])
            self.assertEqual(fn(inp), test_basic(inp))

    @skipIfNoTorchVision
    def test_resnet18_correctness(self):
        model = torchvision.models.resnet18()
        frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
        (
            N,
            C,
            H,
            W,
        ) = (
            10,
            3,
            224,
            224,
        )
        inp = torch.randn(N, C, H, W)
        self.run_pass("inplace_to_functional_activation", frozen_model.graph)
        self.assertEqual(model(inp), frozen_model(inp))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 3 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFunctionalToInplaceActivation`, `Test3`, `TestInplaceToFunctionalActivation`

**Functions defined**: `test_check_no_type_promotion`, `test_functional_to_inplace_activation`, `test_basic`, `test_no_functional_to_inplace`, `test1`, `test2`, `__init__`, `forward`, `test_resnet18_correctness`, `test_inplace_to_functional_activation`, `test_basic`, `test_basic`, `test_resnet18_correctness`

**Key imports**: os, sys, unittest, product, torch, torch.nn as nn, torch.nn.functional as F, FileCheck, torchvision, raise_on_run_directly


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
- `itertools`: product
- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.testing`: FileCheck
- `torchvision`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python test/jit/test_convert_activation.py
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

- **File Documentation**: `test_convert_activation.py_docs.md`
- **Keyword Index**: `test_convert_activation.py_kw.md`
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
- **Error Handling**: Includes exception handling
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
python docs/test/jit/test_convert_activation.py_docs.md
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

- **File Documentation**: `test_convert_activation.py_docs.md_docs.md`
- **Keyword Index**: `test_convert_activation.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
