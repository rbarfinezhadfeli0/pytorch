# Documentation: `test/jit/test_ignore_context_manager.py`

## File Metadata

- **Path**: `test/jit/test_ignore_context_manager.py`
- **Size**: 3,031 bytes (2.96 KB)
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


class TestIgnoreContextManager(JitTestCase):
    def test_with_ignore_context_manager_with_inp_out(self):
        class A(torch.nn.Module):
            def forward(self):
                a: int = 4
                b: int = 5
                c: int = 0
                d: int = 6
                with torch.jit._IgnoreContextManager(
                    a="inp:int", b="inp:int", c="out:int", d="out:int"
                ):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                    d = 9
                return c + d

        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), model())
        self.assertEqual(s(), 20)

        class B(torch.nn.Module):
            def forward(self):
                a: int = 4
                b: int = 5
                c: int = 0
                with torch.jit._IgnoreContextManager(
                    a="inp:int", b="inp:int", c="out:int"
                ):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                return c

        model = B()
        s = torch.jit.script(model)
        self.assertEqual(s(), 11)
        self.assertEqual(s(), model())

        class C(torch.nn.Module):
            def forward(self):
                a: int = 4
                b: int = 5
                with torch.jit._IgnoreContextManager(a="inp:int", b="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    b = l[0] + a
                return b

        model = C()
        s = torch.jit.script(model)
        self.assertEqual(s(), 6)
        self.assertEqual(s(), model())

    def test_with_ignore_context_manager_with_just_inp(self):
        class A(torch.nn.Module):
            def forward(self):
                a: int = 4
                b: int = 5
                with torch.jit._IgnoreContextManager(a="inp:int", b="inp:int"):
                    l = [2 + b for i in range(a) if i > 2]  # noqa: F841
                return a

        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), 4)
        self.assertEqual(s(), model())

    def test_with_ignore_context_manager_with_just_out(self):
        class A(torch.nn.Module):
            def forward(self):
                with torch.jit._IgnoreContextManager(c="out:List[int]"):
                    c = [2 for i in range(7) if i > 2]
                c[0] = 3
                return c[0] + c[1]

        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), 5)
        self.assertEqual(s(), model())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 6 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestIgnoreContextManager`, `A`, `B`, `C`, `A`, `A`

**Functions defined**: `test_with_ignore_context_manager_with_inp_out`, `forward`, `forward`, `forward`, `test_with_ignore_context_manager_with_just_inp`, `forward`, `test_with_ignore_context_manager_with_just_out`, `forward`

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
python test/jit/test_ignore_context_manager.py
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

- **File Documentation**: `test_ignore_context_manager.py_docs.md`
- **Keyword Index**: `test_ignore_context_manager.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
