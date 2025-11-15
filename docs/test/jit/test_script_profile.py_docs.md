# Documentation: `test/jit/test_script_profile.py`

## File Metadata

- **Path**: `test/jit/test_script_profile.py`
- **Size**: 2,892 bytes (2.82 KB)
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
from torch import nn


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class Sequence(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input):
        outputs = []
        h_t = torch.zeros(input.size(0), 51)
        c_t = torch.zeros(input.size(0), 51)
        h_t2 = torch.zeros(input.size(0), 51)
        c_t2 = torch.zeros(input.size(0), 51)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class TestScriptProfile(JitTestCase):
    def test_basic(self):
        seq = torch.jit.script(Sequence())
        p = torch.jit._ScriptProfile()
        p.enable()
        seq(torch.rand((10, 100)))
        p.disable()
        self.assertNotEqual(p.dump_string(), "")

    def test_script(self):
        seq = Sequence()

        p = torch.jit._ScriptProfile()
        p.enable()

        @torch.jit.script
        def fn():
            _ = seq(torch.rand((10, 100)))

        fn()
        p.disable()

        self.assertNotEqual(p.dump_string(), "")

    def test_multi(self):
        seq = torch.jit.script(Sequence())
        profiles = [torch.jit._ScriptProfile() for _ in range(5)]
        for p in profiles:
            p.enable()

        last = None
        while len(profiles) > 0:
            seq(torch.rand((10, 10)))
            p = profiles.pop()
            p.disable()
            stats = p.dump_string()
            self.assertNotEqual(stats, "")
            if last:
                self.assertNotEqual(stats, last)
            last = stats

    def test_section(self):
        seq = Sequence()

        @torch.jit.script
        def fn(max: int):
            _ = seq(torch.rand((10, max)))

        p = torch.jit._ScriptProfile()
        p.enable()
        fn(100)
        p.disable()
        s0 = p.dump_string()

        fn(10)
        p.disable()
        s1 = p.dump_string()

        p.enable()
        fn(10)
        p.disable()
        s2 = p.dump_string()

        self.assertEqual(s0, s1)
        self.assertNotEqual(s1, s2)

    def test_empty(self):
        p = torch.jit._ScriptProfile()
        p.enable()
        p.disable()
        self.assertEqual(p.dump_string(), "")


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 2 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Sequence`, `TestScriptProfile`

**Functions defined**: `__init__`, `forward`, `test_basic`, `test_script`, `fn`, `test_multi`, `test_section`, `fn`, `test_empty`

**Key imports**: os, sys, torch, nn, raise_on_run_directly, JitTestCase


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

- **Object-Oriented Design**: Uses classes and constructors


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
python test/jit/test_script_profile.py
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

- **File Documentation**: `test_script_profile.py_docs.md`
- **Keyword Index**: `test_script_profile.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
