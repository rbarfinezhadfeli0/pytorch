# Documentation: `docs/test/test_jit_disabled.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_jit_disabled.py_docs.md`
- **Size**: 5,613 bytes (5.48 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_jit_disabled.py`

## File Metadata

- **Path**: `test/test_jit_disabled.py`
- **Size**: 2,396 bytes (2.34 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import sys
import os
import contextlib
import subprocess
from torch.testing._internal.common_utils import TestCase, run_tests, TemporaryFileName


@contextlib.contextmanager
def _jit_disabled():
    cur_env = os.environ.get("PYTORCH_JIT", "1")
    os.environ["PYTORCH_JIT"] = "0"
    try:
        yield
    finally:
        os.environ["PYTORCH_JIT"] = cur_env


class TestJitDisabled(TestCase):
    """
    These tests are separate from the rest of the JIT tests because we need
    run a new subprocess and `import torch` with the correct environment
    variables set.
    """

    def compare_enabled_disabled(self, src):
        """
        Runs the script in `src` with PYTORCH_JIT enabled and disabled and
        compares their stdout for equality.
        """
        # Write `src` out to a temporary so our source inspection logic works
        # correctly.
        with TemporaryFileName() as fname:
            with open(fname, 'w') as f:
                f.write(src)
                with _jit_disabled():
                    out_disabled = subprocess.check_output([
                        sys.executable,
                        fname])
                out_enabled = subprocess.check_output([
                    sys.executable,
                    fname])
                self.assertEqual(out_disabled, out_enabled)

    def test_attribute(self):
        _program_string = """
import torch

class Foo(torch.jit.ScriptModule):
    def __init__(self, x):
        super().__init__()
        self.x = torch.jit.Attribute(x, torch.Tensor)

    def forward(self, input):
        return input

s = Foo(torch.ones(2, 3))
print(s.x)
"""
        self.compare_enabled_disabled(_program_string)

    def test_script_module_construction(self):
        _program_string = """
import torch

class AModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, input):
        pass

AModule()
print("Didn't throw exception")
"""
        self.compare_enabled_disabled(_program_string)

    def test_recursive_script(self):
        _program_string = """
import torch

class AModule(torch.nn.Module):
    def forward(self, input):
        pass

sm = torch.jit.script(AModule())
print("Didn't throw exception")
"""
        self.compare_enabled_disabled(_program_string)

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()

```



## High-Level Overview

"""    These tests are separate from the rest of the JIT tests because we need    run a new subprocess and `import torch` with the correct environment    variables set.

This Python file contains 4 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestJitDisabled`, `Foo`, `AModule`, `AModule`

**Functions defined**: `_jit_disabled`, `compare_enabled_disabled`, `test_attribute`, `__init__`, `forward`, `test_script_module_construction`, `forward`, `test_recursive_script`, `forward`

**Key imports**: sys, os, contextlib, subprocess, TestCase, run_tests, TemporaryFileName, torch, torch, torch, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `os`
- `contextlib`
- `subprocess`
- `torch.testing._internal.common_utils`: TestCase, run_tests, TemporaryFileName
- `torch`


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

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_jit_disabled.py
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

- **File Documentation**: `test_jit_disabled.py_docs.md`
- **Keyword Index**: `test_jit_disabled.py_kw.md`
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

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_jit_disabled.py_docs.md
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

- **File Documentation**: `test_jit_disabled.py_docs.md_docs.md`
- **Keyword Index**: `test_jit_disabled.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
