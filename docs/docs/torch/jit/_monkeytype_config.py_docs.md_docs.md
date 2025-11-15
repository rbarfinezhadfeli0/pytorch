# Documentation: `docs/torch/jit/_monkeytype_config.py_docs.md`

## File Metadata

- **Path**: `docs/torch/jit/_monkeytype_config.py_docs.md`
- **Size**: 10,897 bytes (10.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file handles **configuration or setup**.

## Original Source

```markdown
# Documentation: `torch/jit/_monkeytype_config.py`

## File Metadata

- **Path**: `torch/jit/_monkeytype_config.py`
- **Size**: 7,451 bytes (7.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file handles **configuration or setup**.

## Original Source

```python
# mypy: allow-untyped-defs
import inspect
import sys
import typing
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from types import CodeType
from typing import Optional

import torch


_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype  # type: ignore[import]

    # pyrefly: ignore [import-error]
    from monkeytype import trace as monkeytype_trace
    from monkeytype.config import _startswith, LIB_PATHS  # type: ignore[import]
    from monkeytype.db.base import (  # type: ignore[import]
        CallTraceStore,
        CallTraceStoreLogger,
        CallTraceThunk,
    )
    from monkeytype.tracing import CallTrace, CodeFilter  # type: ignore[import]
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False


# Checks whether a class is defined in `torch.*` modules
def is_torch_native_class(cls):
    if not hasattr(cls, "__module__"):
        return False

    parent_modules = cls.__module__.split(".")
    if not parent_modules:
        return False

    root_module = sys.modules.get(parent_modules[0])
    return root_module is torch


def get_type(type):
    """Convert the given type to a torchScript acceptable format."""
    if isinstance(type, str):
        return type
    elif inspect.getmodule(type) == typing:
        # If the type is a type imported from typing
        # like Tuple, List, Dict then replace `typing.`
        # with a null string. This needs to be done since
        # typing.List is not accepted by TorchScript.
        type_to_string = str(type)
        return type_to_string.replace(type.__module__ + ".", "")
    elif is_torch_native_class(type):
        # If the type is a subtype of torch module, then TorchScript expects a fully qualified name
        # for the type which is obtained by combining the module name and type name.
        return type.__module__ + "." + type.__name__
    else:
        # For all other types use the name for the type.
        return type.__name__


def get_optional_of_element_type(types):
    """Extract element type, return as `Optional[element type]` from consolidated types.

    Helper function to extracts the type of the element to be annotated to Optional
    from the list of consolidated types and returns `Optional[element type]`.
    TODO: To remove this check once Union support lands.
    """
    elem_type = types[1] if type(None) is types[0] else types[0]
    elem_type = get_type(elem_type)

    # Optional type is internally converted to Union[type, NoneType], which
    # is not supported yet in TorchScript. Hence, representing the optional type as string.
    return "Optional[" + elem_type + "]"


def get_qualified_name(func):
    return func.__qualname__


if _IS_MONKEYTYPE_INSTALLED:

    class JitTypeTraceStoreLogger(CallTraceStoreLogger):
        """A JitTypeCallTraceLogger that stores logged traces in a CallTraceStore."""

        def __init__(self, store: CallTraceStore) -> None:
            super().__init__(store)

        def log(self, trace: CallTrace) -> None:
            # pyrefly: ignore [missing-attribute]
            self.traces.append(trace)

    class JitTypeTraceStore(CallTraceStore):
        def __init__(self) -> None:
            super().__init__()
            # A dictionary keeping all collected CallTrace
            # key is fully qualified name of called function
            # value is list of all CallTrace
            self.trace_records: dict[str, list] = defaultdict(list)

        def add(self, traces: Iterable[CallTrace]) -> None:
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(
            self,
            qualified_name: str,
            qualname_prefix: Optional[str] = None,
            limit: int = 2000,
        ) -> list[CallTraceThunk]:
            return self.trace_records[qualified_name]

        def analyze(self, qualified_name: str) -> dict:
            # Analyze the types for the given module
            # and create a dictionary of all the types
            # for arguments.
            records = self.trace_records[qualified_name]
            all_args = defaultdict(set)
            for record in records:
                for arg, arg_type in record.arg_types.items():
                    all_args[arg].add(arg_type)
            return all_args

        def consolidate_types(self, qualified_name: str) -> dict:
            all_args = self.analyze(qualified_name)
            # If there are more types for an argument,
            # then consolidate the type to `Any` and replace the entry
            # by type `Any`.
            for arg, types in all_args.items():
                types = list(types)
                type_length = len(types)
                if type_length == 2 and type(None) in types:
                    # TODO: To remove this check once Union support in TorchScript lands.
                    all_args[arg] = get_optional_of_element_type(types)
                elif type_length > 1:
                    all_args[arg] = "Any"
                elif type_length == 1:
                    all_args[arg] = get_type(types[0])
            return all_args

        def get_args_types(self, qualified_name: str) -> dict:
            return self.consolidate_types(qualified_name)

    class JitTypeTraceConfig(monkeytype.config.Config):
        def __init__(self, s: JitTypeTraceStore) -> None:
            super().__init__()
            self.s = s

        def trace_logger(self) -> JitTypeTraceStoreLogger:
            """Return a JitCallTraceStoreLogger that logs to the configured trace store."""
            # pyrefly: ignore [bad-argument-count]
            return JitTypeTraceStoreLogger(self.trace_store())

        def trace_store(self) -> CallTraceStore:
            return self.s

        def code_filter(self) -> Optional[CodeFilter]:
            return jit_code_filter

else:
    # When MonkeyType is not installed, we provide dummy class definitions
    # for the below classes.
    class JitTypeTraceStoreLogger:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            pass

    class JitTypeTraceStore:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            self.trace_records = None

    class JitTypeTraceConfig:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            pass

    monkeytype_trace = None  # type: ignore[assignment]  # noqa: F811


def jit_code_filter(code: CodeType) -> bool:
    """Codefilter for Torchscript to trace forward calls.

    The custom CodeFilter is required while scripting a FX Traced forward calls.
    FX Traced forward calls have `code.co_filename` start with '<' which is used
    to exclude tracing of stdlib and site-packages in the default code filter.
    Since we need all forward calls to be traced, this custom code filter
    checks for code.co_name to be 'forward' and enables tracing for all such calls.
    The code filter is similar to default code filter for monkeytype and
    excludes tracing of stdlib and site-packages.
    """
    # Filter code without a source file and exclude this check for 'forward' calls.
    if code.co_name != "forward" and (
        not code.co_filename or code.co_filename[0] == "<"
    ):
        return False

    filename = Path(code.co_filename).resolve()
    return not any(_startswith(filename, lib_path) for lib_path in LIB_PATHS)

```



## High-Level Overview

"""Convert the given type to a torchScript acceptable format."""    if isinstance(type, str):        return type    elif inspect.getmodule(type) == typing:        # If the type is a type imported from typing        # like Tuple, List, Dict then replace `typing.`

This Python file contains 8 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `JitTypeTraceStoreLogger`, `JitTypeTraceStore`, `JitTypeTraceConfig`, `JitTypeTraceStoreLogger`, `JitTypeTraceStore`, `JitTypeTraceConfig`

**Functions defined**: `is_torch_native_class`, `get_type`, `get_optional_of_element_type`, `get_qualified_name`, `__init__`, `log`, `__init__`, `add`, `filter`, `analyze`, `consolidate_types`, `get_args_types`, `__init__`, `trace_logger`, `trace_store`, `code_filter`, `__init__`, `__init__`, `__init__`, `jit_code_filter`

**Key imports**: inspect, sys, typing, defaultdict, Iterable, Path, CodeType, Optional, torch, monkeytype  


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `inspect`
- `sys`
- `typing`
- `collections`: defaultdict
- `collections.abc`: Iterable
- `pathlib`: Path
- `types`: CodeType
- `torch`
- `monkeytype  `
- `monkeytype`: trace as monkeytype_trace
- `monkeytype.config`: _startswith, LIB_PATHS  
- `monkeytype.tracing`: CallTrace, CodeFilter  


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/jit`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_decompositions.py_docs.md`](./_decompositions.py_docs.md)
- [`_dataclass_impls.py_docs.md`](./_dataclass_impls.py_docs.md)
- [`quantized.py_docs.md`](./quantized.py_docs.md)
- [`frontend.py_docs.md`](./frontend.py_docs.md)
- [`_builtins.py_docs.md`](./_builtins.py_docs.md)
- [`_trace.py_docs.md`](./_trace.py_docs.md)
- [`_serialization.py_docs.md`](./_serialization.py_docs.md)
- [`_state.py_docs.md`](./_state.py_docs.md)
- [`_await.py_docs.md`](./_await.py_docs.md)


## Cross-References

- **File Documentation**: `_monkeytype_config.py_docs.md`
- **Keyword Index**: `_monkeytype_config.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/jit`):

- [`_check.py_kw.md_docs.md`](./_check.py_kw.md_docs.md)
- [`_shape_functions.py_docs.md_docs.md`](./_shape_functions.py_docs.md_docs.md)
- [`_trace.py_kw.md_docs.md`](./_trace.py_kw.md_docs.md)
- [`_logging.py_docs.md_docs.md`](./_logging.py_docs.md_docs.md)
- [`_async.py_kw.md_docs.md`](./_async.py_kw.md_docs.md)
- [`_state.py_docs.md_docs.md`](./_state.py_docs.md_docs.md)
- [`_decomposition_utils.py_kw.md_docs.md`](./_decomposition_utils.py_kw.md_docs.md)
- [`frontend.py_docs.md_docs.md`](./frontend.py_docs.md_docs.md)
- [`_check.py_docs.md_docs.md`](./_check.py_docs.md_docs.md)
- [`_script.pyi_docs.md_docs.md`](./_script.pyi_docs.md_docs.md)


## Cross-References

- **File Documentation**: `_monkeytype_config.py_docs.md_docs.md`
- **Keyword Index**: `_monkeytype_config.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
