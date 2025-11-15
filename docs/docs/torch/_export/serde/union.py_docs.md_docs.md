# Documentation: `docs/torch/_export/serde/union.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_export/serde/union.py_docs.md`
- **Size**: 5,485 bytes (5.36 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_export/serde/union.py`

## File Metadata

- **Path**: `torch/_export/serde/union.py`
- **Size**: 2,941 bytes (2.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import functools
from collections.abc import Hashable
from dataclasses import dataclass, fields
from typing import TypeVar
from typing_extensions import dataclass_transform


T = TypeVar("T", bound="_Union")


class _UnionTag(str):
    __slots__ = ("_cls",)
    _cls: Hashable

    @staticmethod
    def create(t, cls):
        tag = _UnionTag(t)
        assert not hasattr(tag, "_cls")
        tag._cls = cls
        return tag

    def __eq__(self, cmp) -> bool:
        assert isinstance(cmp, str)
        other = str(cmp)
        assert other in _get_field_names(self._cls), (
            f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
        )
        return str(self) == other

    def __hash__(self):
        return hash(str(self))


@functools.cache
def _get_field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)}


# If you turn a schema class that inherits from union into a dataclass, please use
# this decorator to configure it. It's safe, faster and allows code sharing.
#
# For example, _union_dataclass customizes the __eq__ method to only check the type
# and value property instead of default implementation of dataclass which goes
# through every field in the dataclass.
@dataclass_transform(eq_default=False)
def _union_dataclass(cls: type[T]) -> type[T]:
    assert issubclass(cls, _Union), f"{cls} must inheirt from {_Union}."
    return dataclass(repr=False, eq=False)(cls)


class _Union:
    _type: _UnionTag

    @classmethod
    def create(cls, **kwargs):
        assert len(kwargs) == 1
        obj = cls(**{**{f.name: None for f in fields(cls)}, **kwargs})  # type: ignore[arg-type]
        obj._type = _UnionTag.create(next(iter(kwargs.keys())), cls)
        return obj

    def __post_init__(self):
        assert not any(
            f.name in ("type", "_type", "create", "value")
            for f in fields(self)  # type: ignore[arg-type, misc]
        )

    @property
    def type(self) -> str:
        try:
            return self._type
        except AttributeError as e:
            raise RuntimeError(
                f"Please use {type(self).__name__}.create to instantiate the union type."
            ) from e

    @property
    def value(self):
        return getattr(self, self.type)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if attr is None and name in _get_field_names(type(self)) and name != self.type:  # type: ignore[arg-type]
            raise AttributeError(f"Field {name} is not set.")
        return attr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Union):
            return False
        return self.type == other.type and self.value == other.value

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{type(self).__name__}({self.type}={getattr(self, self.type)})"

```



## High-Level Overview


This Python file contains 5 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `_UnionTag`, `_Union`

**Functions defined**: `create`, `__eq__`, `__hash__`, `_get_field_names`, `_union_dataclass`, `create`, `__post_init__`, `type`, `value`, `__getattribute__`, `__eq__`, `__str__`, `__repr__`

**Key imports**: functools, Hashable, dataclass, fields, TypeVar, dataclass_transform


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_export/serde`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `collections.abc`: Hashable
- `dataclasses`: dataclass, fields
- `typing`: TypeVar
- `typing_extensions`: dataclass_transform


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/_export/serde`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`schema.yaml_docs.md`](./schema.yaml_docs.md)
- [`export_schema.thrift_docs.md`](./export_schema.thrift_docs.md)
- [`schema.py_docs.md`](./schema.py_docs.md)
- [`dynamic_shapes.py_docs.md`](./dynamic_shapes.py_docs.md)
- [`schema_check.py_docs.md`](./schema_check.py_docs.md)
- [`serialize.py_docs.md`](./serialize.py_docs.md)


## Cross-References

- **File Documentation**: `union.py_docs.md`
- **Keyword Index**: `union.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_export/serde`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export/serde`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_export/serde`):

- [`schema_check.py_kw.md_docs.md`](./schema_check.py_kw.md_docs.md)
- [`schema.py_docs.md_docs.md`](./schema.py_docs.md_docs.md)
- [`serialize.py_kw.md_docs.md`](./serialize.py_kw.md_docs.md)
- [`serialize.py_docs.md_docs.md`](./serialize.py_docs.md_docs.md)
- [`schema.yaml_kw.md_docs.md`](./schema.yaml_kw.md_docs.md)
- [`schema.yaml_docs.md_docs.md`](./schema.yaml_docs.md_docs.md)
- [`schema.py_kw.md_docs.md`](./schema.py_kw.md_docs.md)
- [`export_schema.thrift_kw.md_docs.md`](./export_schema.thrift_kw.md_docs.md)
- [`schema_check.py_docs.md_docs.md`](./schema_check.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `union.py_docs.md_docs.md`
- **Keyword Index**: `union.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
