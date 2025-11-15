# Documentation: `torch/_dynamo/graph_bytecode_inputs.py`

## File Metadata

- **Path**: `torch/_dynamo/graph_bytecode_inputs.py`
- **Size**: 3,034 bytes (2.96 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import weakref
from collections.abc import Callable
from typing import Any

from torch._dynamo.source import Source


PyCodegen = Any

# This file is to handle types that we don't want to support
# as explicit FX graph inputs. This uses a sidetable which
# we populate in bytecode and is loaded during graph execution

# We use a dynamo-generated index as a level of indirection
# this allows us to register objects externally in pre-graph bytecode that we want
# to pass to the graph, but not support their types as graph inputs
index_to_bytecode_constructor: dict[int, Callable[[PyCodegen], None]] = {}

index_to_external_object_weakref: dict[int, weakref.ReferenceType[Any]] = {}

keep_alive: list[Any] = []


def has_user_objects() -> bool:
    return bool(index_to_bytecode_constructor)


def get_external_object_by_index(index: int) -> Any:
    assert index in index_to_external_object_weakref, (
        "Index not registered in index_to_user_object_weakref"
    )
    obj = index_to_external_object_weakref[index]()
    assert obj is not None, "User object is no longer alive"
    return index_to_external_object_weakref[index]()


def store_user_object_weakrefs(*args: Any) -> None:
    global index_to_external_object_weakref
    index_to_external_object_weakref.clear()
    index_to_external_object_weakref.update(
        {i: weakref.ref(arg) for i, arg in enumerate(args)}
    )


def reset_user_object_tracking() -> None:
    index_to_bytecode_constructor.clear()
    index_to_external_object_weakref.clear()
    keep_alive.clear()


def register_graph_created_object(
    example_value: Any, construct_fn: Callable[[int, PyCodegen], None]
) -> int:
    global index_to_bytecode_constructor
    global keep_alive
    keep_alive.append(example_value)
    index = len(index_to_bytecode_constructor)
    index_to_bytecode_constructor[index] = lambda cg: construct_fn(index, cg)
    try:
        index_to_external_object_weakref[index] = weakref.ref(example_value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to graph-created external object",
            context=f"user_object: {example_value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index


# Register a user object to be used in the graph
def register_user_object(value: Any, source: Source) -> int:
    global index_to_bytecode_constructor
    index = len(index_to_bytecode_constructor)
    index_to_bytecode_constructor[index] = lambda cg: cg(source)
    try:
        index_to_external_object_weakref[index] = weakref.ref(value)
    except TypeError as e:
        from .exc import unimplemented

        unimplemented(
            gb_type="Failed to make weakref to User Object",
            context=f"user_object: {value}",
            explanation="Object does not allow us to make a weakref to it",
            hints=[],
            from_exc=e,
        )
    return index

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `has_user_objects`, `get_external_object_by_index`, `store_user_object_weakrefs`, `reset_user_object_tracking`, `register_graph_created_object`, `register_user_object`

**Key imports**: weakref, Callable, Any, Source, unimplemented, unimplemented


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_dynamo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `weakref`
- `collections.abc`: Callable
- `typing`: Any
- `torch._dynamo.source`: Source
- `.exc`: unimplemented


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

- **File Documentation**: `graph_bytecode_inputs.py_docs.md`
- **Keyword Index**: `graph_bytecode_inputs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
