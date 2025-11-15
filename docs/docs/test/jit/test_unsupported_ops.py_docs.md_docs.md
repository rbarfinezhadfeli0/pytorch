# Documentation: `docs/test/jit/test_unsupported_ops.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_unsupported_ops.py_docs.md`
- **Size**: 5,918 bytes (5.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_unsupported_ops.py`

## File Metadata

- **Path**: `test/jit/test_unsupported_ops.py`
- **Size**: 2,893 bytes (2.83 KB)
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


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# NOTE: FIXING FAILING TESTS
# If you are seeing a test failure from this file, congrats, you improved
# parity between JIT and Python API. Before you fix the test, you must also update
# the corresponding section in documentation that states the unsupported behavior.
# see: `jit_unsupported.rst`


class TestUnsupportedOps(JitTestCase):
    def test_factory_ops_requires_grad_fail(self):
        # Keyword argument {name} unknown is a JIT-only error message,
        # so these functions are succeeding in eager and failing in JIT

        # Complete issue and set of ops is https://github.com/pytorch/pytorch/issues/30761
        # only testing some because they should be fixed all at once
        def ones():
            return torch.ones([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.ones"
        ):
            torch.jit.script(ones)

        def randn():
            return torch.randn([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.randn"
        ):
            torch.jit.script(randn)

        def zeros():
            return torch.zeros([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.zeros"
        ):
            torch.jit.script(zeros)

    @unittest.skipIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")
    def test_init_ops(self):
        def calculate_gain():
            return torch.nn.init.calculate_gain("leaky_relu", 0.2)

        def eye_():
            return torch.nn.init.eye_(torch.zeros([2, 2]))

        def dirac_():
            return torch.nn.init.dirac_(torch.empty(3, 16, 5, 5))

        def kaiming_uniform_():
            return torch.nn.init.kaiming_normal_(torch.empty(3, 5))

        def orthogonal_():
            return torch.nn.init.orthogonal_(torch.empty(3, 5))

        def sparse():
            return torch.nn.init.sparse_(torch.empty(3, 5), sparsity=0.1)

        for func in [
            calculate_gain,
            eye_,
            dirac_,
            kaiming_uniform_,
            orthogonal_,
            sparse,
        ]:
            # doesn't error in eager
            func()
            with self.assertRaisesRegex(Exception, ""):
                torch.jit.script(func)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestUnsupportedOps`

**Functions defined**: `test_factory_ops_requires_grad_fail`, `ones`, `randn`, `zeros`, `test_init_ops`, `calculate_gain`, `eye_`, `dirac_`, `kaiming_uniform_`, `orthogonal_`, `sparse`

**Key imports**: os, sys, unittest, torch, raise_on_run_directly, JitTestCase


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
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


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
python test/jit/test_unsupported_ops.py
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

- **File Documentation**: `test_unsupported_ops.py_docs.md`
- **Keyword Index**: `test_unsupported_ops.py_kw.md`
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
python docs/test/jit/test_unsupported_ops.py_docs.md
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

- **File Documentation**: `test_unsupported_ops.py_docs.md_docs.md`
- **Keyword Index**: `test_unsupported_ops.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
