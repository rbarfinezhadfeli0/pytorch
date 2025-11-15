# Documentation: `test/jit/test_scriptmod_ann.py`

## File Metadata

- **Path**: `test/jit/test_scriptmod_ann.py`
- **Size**: 11,257 bytes (10.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys
import warnings
from typing import Dict, List, Optional

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestScriptModuleInstanceAttributeTypeAnnotation(JitTestCase):
    # NB: There are no tests for `Tuple` or `NamedTuple` here. In fact,
    # reassigning a non-empty Tuple to an attribute previously typed
    # as containing an empty Tuple SHOULD fail. See note in `_check.py`

    def test_annotated_falsy_base_type(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: int = 0

            def forward(self, x: int):
                self.x = x
                return 1

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), (1,))
        assert len(w) == 0

    def test_annotated_nonempty_container(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: List[int] = [1, 2, 3]

            def forward(self, x: List[int]):
                self.x = x
                return 1

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_empty_tensor(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: torch.Tensor = torch.empty(0)

            def forward(self, x: torch.Tensor):
                self.x = x
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), (torch.rand(2, 3),))
        assert len(w) == 0

    def test_annotated_with_jit_attribute(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.Attribute([], List[int])

            def forward(self, x: List[int]):
                self.x = x
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_annotation_only(self):
        class M(torch.nn.Module):
            x: List[int]

            def __init__(self) -> None:
                super().__init__()
                self.x = []

            def forward(self, y: List[int]):
                self.x = y
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_annotation_and_init_annotation(self):
        class M(torch.nn.Module):
            x: List[int]

            def __init__(self) -> None:
                super().__init__()
                self.x: List[int] = []

            def forward(self, y: List[int]):
                self.x = y
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_class_level_jit_annotation(self):
        class M(torch.nn.Module):
            x: List[int]

            def __init__(self) -> None:
                super().__init__()
                self.x: List[int] = torch.jit.annotate(List[int], [])

            def forward(self, y: List[int]):
                self.x = y
                return self.x

        with warnings.catch_warnings(record=True) as w:
            self.checkModule(M(), ([1, 2, 3],))
        assert len(w) == 0

    def test_annotated_empty_list(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: List[int] = []

            def forward(self, x: List[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_list_lowercase(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: list[int] = []

            def forward(self, x: list[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_dict(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: Dict[str, int] = {}

            def forward(self, x: Dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_dict_lowercase(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: dict[str, int] = {}

            def forward(self, x: dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_empty_optional(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x: Optional[str] = None

            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_list(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(List[int], [])

            def forward(self, x: List[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_list_lowercase(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(list[int], [])

            def forward(self, x: list[int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_dict(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(Dict[str, int], {})

            def forward(self, x: Dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_dict_lowercase(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(dict[str, int], {})

            def forward(self, x: dict[str, int]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Tried to set nonexistent attribute", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_jit_empty_optional(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.jit.annotate(Optional[str], None)

            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())

    def test_annotated_with_torch_jit_import(self):
        from torch import jit

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = jit.annotate(Optional[str], None)

            def forward(self, x: Optional[str]):
                self.x = x
                return 1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Wrong type for attribute assignment", "self.x = x"
        ):
            with self.assertWarnsRegex(
                UserWarning,
                "doesn't support instance-level annotations on empty non-base types",
            ):
                torch.jit.script(M())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 19 class(es) and 54 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestScriptModuleInstanceAttributeTypeAnnotation`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`, `M`

**Functions defined**: `test_annotated_falsy_base_type`, `__init__`, `forward`, `test_annotated_nonempty_container`, `__init__`, `forward`, `test_annotated_empty_tensor`, `__init__`, `forward`, `test_annotated_with_jit_attribute`, `__init__`, `forward`, `test_annotated_class_level_annotation_only`, `__init__`, `forward`, `test_annotated_class_level_annotation_and_init_annotation`, `__init__`, `forward`, `test_annotated_class_level_jit_annotation`, `__init__`

**Key imports**: os, sys, warnings, Dict, List, Optional, torch, raise_on_run_directly, JitTestCase, jit


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `warnings`
- `typing`: Dict, List, Optional
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
python test/jit/test_scriptmod_ann.py
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

- **File Documentation**: `test_scriptmod_ann.py_docs.md`
- **Keyword Index**: `test_scriptmod_ann.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
