# Documentation: `test/jit/test_module_apis.py`

## File Metadata

- **Path**: `test/jit/test_module_apis.py`
- **Size**: 5,068 bytes (4.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys
from typing import Any, Dict, List

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestModuleAPIs(JitTestCase):
    def test_default_state_dict_methods(self):
        """Tests that default state dict methods are automatically available"""

        class DefaultStateDictModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)

            def forward(self, x):
                x = self.conv(x)
                x = self.fc(x)
                return x

        m1 = torch.jit.script(DefaultStateDictModule())
        m2 = torch.jit.script(DefaultStateDictModule())
        state_dict = m1.state_dict()
        m2.load_state_dict(state_dict)

    def test_customized_state_dict_methods(self):
        """Tests that customized state dict methods are in effect"""

        class CustomStateDictModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)
                self.customized_save_state_dict_called: bool = False
                self.customized_load_state_dict_called: bool = False

            def forward(self, x):
                x = self.conv(x)
                x = self.fc(x)
                return x

            @torch.jit.export
            def _save_to_state_dict(
                self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool
            ):
                self.customized_save_state_dict_called = True
                return {"dummy": torch.ones(1)}

            @torch.jit.export
            def _load_from_state_dict(
                self,
                state_dict: Dict[str, torch.Tensor],
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: List[str],
                unexpected_keys: List[str],
                error_msgs: List[str],
            ):
                self.customized_load_state_dict_called = True
                return

        m1 = torch.jit.script(CustomStateDictModule())
        self.assertFalse(m1.customized_save_state_dict_called)
        state_dict = m1.state_dict()
        self.assertTrue(m1.customized_save_state_dict_called)

        m2 = torch.jit.script(CustomStateDictModule())
        self.assertFalse(m2.customized_load_state_dict_called)
        m2.load_state_dict(state_dict)
        self.assertTrue(m2.customized_load_state_dict_called)

    def test_submodule_customized_state_dict_methods(self):
        """Tests that customized state dict methods on submodules are in effect"""

        class CustomStateDictModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)
                self.customized_save_state_dict_called: bool = False
                self.customized_load_state_dict_called: bool = False

            def forward(self, x):
                x = self.conv(x)
                x = self.fc(x)
                return x

            @torch.jit.export
            def _save_to_state_dict(
                self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool
            ):
                self.customized_save_state_dict_called = True
                return {"dummy": torch.ones(1)}

            @torch.jit.export
            def _load_from_state_dict(
                self,
                state_dict: Dict[str, torch.Tensor],
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: List[str],
                unexpected_keys: List[str],
                error_msgs: List[str],
            ):
                self.customized_load_state_dict_called = True
                return

        class ParentModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = CustomStateDictModule()

            def forward(self, x):
                return self.sub(x)

        m1 = torch.jit.script(ParentModule())
        self.assertFalse(m1.sub.customized_save_state_dict_called)
        state_dict = m1.state_dict()
        self.assertTrue(m1.sub.customized_save_state_dict_called)

        m2 = torch.jit.script(ParentModule())
        self.assertFalse(m2.sub.customized_load_state_dict_called)
        m2.load_state_dict(state_dict)
        self.assertTrue(m2.sub.customized_load_state_dict_called)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

"""Tests that default state dict methods are automatically available"""        class DefaultStateDictModule(torch.nn.Module):            def __init__(self) -> None:                super().__init__()                self.conv = torch.nn.Conv2d(6, 16, 5)                self.fc = torch.nn.Linear(16 * 5 * 5, 120)            def forward(self, x):                x = self.conv(x)                x = self.fc(x)                return x        m1 = torch.jit.script(DefaultStateDictModule())        m2 = torch.jit.script(DefaultStateDictModule())        state_dict = m1.state_dict()        m2.load_state_dict(state_dict)    def test_customized_state_dict_methods(self):

This Python file contains 5 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestModuleAPIs`, `DefaultStateDictModule`, `CustomStateDictModule`, `CustomStateDictModule`, `ParentModule`

**Functions defined**: `test_default_state_dict_methods`, `__init__`, `forward`, `test_customized_state_dict_methods`, `__init__`, `forward`, `_save_to_state_dict`, `_load_from_state_dict`, `test_submodule_customized_state_dict_methods`, `__init__`, `forward`, `_save_to_state_dict`, `_load_from_state_dict`, `__init__`, `forward`

**Key imports**: os, sys, Any, Dict, List, torch, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `typing`: Any, Dict, List
- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/jit/test_module_apis.py
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

- **File Documentation**: `test_module_apis.py_docs.md`
- **Keyword Index**: `test_module_apis.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
