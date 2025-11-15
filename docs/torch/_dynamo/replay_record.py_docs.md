# Documentation: `torch/_dynamo/replay_record.py`

## File Metadata

- **Path**: `torch/_dynamo/replay_record.py`
- **Size**: 4,389 bytes (4.29 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Python execution state recording and replay functionality.

This module provides mechanisms for capturing and replaying Python execution state:

- ModuleRecord: Tracks module access patterns and attribute usage
- DummyModule: Lightweight module substitute for replay
- ExecutionRecord: Manages execution context including globals, locals and builtins
- ExecutionRecorder: Records variable states and module access during execution

The module enables serialization and reproduction of Python execution environments,
particularly useful for debugging and testing frameworks that need to capture
and recreate specific program states.
"""

import dataclasses
from dataclasses import field
from io import BufferedReader, BufferedWriter
from types import CellType, CodeType, ModuleType
from typing import Any, IO, Union
from typing_extensions import Self

from torch.utils._import_utils import import_dill


dill = import_dill()


@dataclasses.dataclass
class ModuleRecord:
    module: ModuleType
    accessed_attrs: dict[str, Any] = field(default_factory=dict)


@dataclasses.dataclass
class DummyModule:
    name: str
    is_torch: bool = False
    value: object = None

    @property
    def __name__(self) -> str:
        return self.name


@dataclasses.dataclass
class ExecutionRecord:
    code: CodeType
    closure: tuple[CellType]
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)

    def dump(self, f: Union[IO[str], BufferedWriter]) -> None:
        assert dill is not None, "replay_record requires `pip install dill`"
        dill.dump(self, f)

    @classmethod
    def load(cls, f: Union[IO[bytes], BufferedReader]) -> Self:
        assert dill is not None, "replay_record requires `pip install dill`"
        return dill.load(f)


@dataclasses.dataclass
class ExecutionRecorder:
    LOCAL_MOD_PREFIX = "___local_mod_"

    code: CodeType
    closure: tuple[CellType]
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)
    name_to_modrec: dict[str, ModuleRecord] = field(default_factory=dict)

    def add_local_var(self, name: str, var: Any) -> None:
        if isinstance(var, ModuleType):
            self.locals[name] = self._add_mod(var)
        else:
            self.locals[name] = var

    def add_global_var(self, name: str, var: Any) -> None:
        if isinstance(var, ModuleType):
            self.globals[name] = self._add_mod(var)
        else:
            self.globals[name] = var

    def add_local_mod(self, name: str, mod: ModuleType) -> None:
        assert isinstance(mod, ModuleType)
        self.add_global_var(name, mod)

    def record_module_access(self, mod: ModuleType, name: str, val: Any) -> None:
        if isinstance(val, ModuleType):
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = self._add_mod(val)
            return

        if mod.__name__ in self.name_to_modrec:
            self.name_to_modrec[mod.__name__].accessed_attrs[name] = val

    def get_record(self) -> ExecutionRecord:
        return ExecutionRecord(
            self.code,
            self.closure,
            ExecutionRecorder._resolve_modules(self.globals),
            ExecutionRecorder._resolve_modules(self.locals),
            self.builtins.copy(),
            self.code_options.copy(),
        )

    def _add_mod(self, mod: ModuleType) -> ModuleRecord:
        if mod.__name__ not in self.name_to_modrec:
            self.name_to_modrec[mod.__name__] = ModuleRecord(mod)

        return self.name_to_modrec[mod.__name__]

    @classmethod
    def _resolve_modules(cls, vars: dict[str, Any]) -> dict[str, Any]:
        def resolve_module(var: Any) -> Any:
            if not isinstance(var, ModuleRecord):
                return var

            dummy_mod = DummyModule(var.module.__name__)
            for attr_name, attr_value in var.accessed_attrs.items():
                attr_value = resolve_module(attr_value)
                dummy_mod.__setattr__(attr_name, attr_value)

            return dummy_mod

        return {k: resolve_module(v) for k, v in vars.items()}

```



## High-Level Overview

"""Python execution state recording and replay functionality.This module provides mechanisms for capturing and replaying Python execution state:- ModuleRecord: Tracks module access patterns and attribute usage- DummyModule: Lightweight module substitute for replay- ExecutionRecord: Manages execution context including globals, locals and builtins- ExecutionRecorder: Records variable states and module access during executionThe module enables serialization and reproduction of Python execution environments,particularly useful for debugging and testing frameworks that need to captureand recreate specific program states.

This Python file contains 4 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ModuleRecord`, `DummyModule`, `ExecutionRecord`, `ExecutionRecorder`

**Functions defined**: `__name__`, `dump`, `load`, `add_local_var`, `add_global_var`, `add_local_mod`, `record_module_access`, `get_record`, `_add_mod`, `_resolve_modules`, `resolve_module`

**Key imports**: dataclasses, field, BufferedReader, BufferedWriter, CellType, CodeType, ModuleType, Any, IO, Union, Self, import_dill


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `io`: BufferedReader, BufferedWriter
- `types`: CellType, CodeType, ModuleType
- `typing`: Any, IO, Union
- `typing_extensions`: Self
- `torch.utils._import_utils`: import_dill


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_dynamo`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`side_effects.py_docs.md`](./side_effects.py_docs.md)
- [`package.py_docs.md`](./package.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`hooks.py_docs.md`](./hooks.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`graph_break_hints.py_docs.md`](./graph_break_hints.py_docs.md)
- [`device_interface.py_docs.md`](./device_interface.py_docs.md)
- [`graph_break_registry.json_docs.md`](./graph_break_registry.json_docs.md)
- [`current_scope_id.py_docs.md`](./current_scope_id.py_docs.md)


## Cross-References

- **File Documentation**: `replay_record.py_docs.md`
- **Keyword Index**: `replay_record.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
