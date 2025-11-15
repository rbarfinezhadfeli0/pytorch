# Documentation: test_module_apis.py

## File Metadata
- **Path**: `test/jit/test_module_apis.py`
- **Size**: 5068 bytes
- **Lines**: 141
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 5 class(es): TestModuleAPIs, DefaultStateDictModule, CustomStateDictModule, CustomStateDictModule, ParentModule

### Functions
This file defines 15 function(s): test_default_state_dict_methods, __init__, forward, test_customized_state_dict_methods, __init__, forward, _save_to_state_dict, _load_from_state_dict, test_submodule_customized_state_dict_methods, __init__, forward, _save_to_state_dict, _load_from_state_dict, __init__, forward


## Key Components

The file contains 324 words across 141 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5068 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
