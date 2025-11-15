# Documentation: `docs/torchgen/api/types/types_base.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/api/types/types_base.py_docs.md`
- **Size**: 10,620 bytes (10.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/api/types/types_base.py`

## File Metadata

- **Path**: `torchgen/api/types/types_base.py`
- **Size**: 7,184 bytes (7.02 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
"""
Where should I add a new type? `types_base.py` vs `types.py`

This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.

`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.

The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn't
contain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reused
if we want to generate code for another C++ library.

Add new types to `types.py` if these types are ATen/c10 related.
Add new types to `types_base.py` if they are basic and not attached to ATen/c10.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto, Enum
from typing import TYPE_CHECKING, Union


if TYPE_CHECKING:
    from torchgen.model import Argument, SelfArgument, TensorOptionsArguments


# An ArgName is just the str name of the argument in schema;
# but in some special circumstances, we may add a little extra
# context.  The Enum SpecialArgName covers all of these cases;
# grep for their construction sites to see when they can occur.


class SpecialArgName(Enum):
    possibly_redundant_memory_format = auto()


ArgName = Union[str, SpecialArgName]


# This class shouldn't be created directly; instead, use/create one of the singletons below.
@dataclass(frozen=True)
class BaseCppType:
    ns: str | None
    name: str

    def __str__(self) -> str:
        if self.ns is None or self.ns == "":
            return self.name
        return f"{self.ns}::{self.name}"


# The set of all non-templated, valid, fully-qualified names of C++ types that are used in the codegen.
# Templated types get their own dataclass, mainly to make namespace parsing easier.
byteT = BaseCppType("", "uint8_t")
charT = BaseCppType("", "int8_t")
shortT = BaseCppType("", "int16_t")
# It would be more symmetric for this to be called intT, but it easy to mix
# this up with JIT int (which is int64_t in C++), so we intentionally don't
# define intT to make it obvious when you've stuffed it up
int32T = BaseCppType("", "int32_t")
longT = BaseCppType("", "int64_t")
doubleT = BaseCppType("", "double")
floatT = BaseCppType("", "float")
boolT = BaseCppType("", "bool")
voidT = BaseCppType("", "void")


class CType(ABC):
    @abstractmethod
    def cpp_type(self, *, strip_ref: bool = False) -> str:
        raise NotImplementedError

    @abstractmethod
    def remove_const_ref(self) -> CType:
        return self


@dataclass(frozen=True)
class BaseCType(CType):
    type: BaseCppType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return str(self.type)

    def remove_const_ref(self) -> CType:
        return self


@dataclass(frozen=True)
class ConstRefCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f"const {self.elem.cpp_type()} &"

    def remove_const_ref(self) -> CType:
        return self.elem.remove_const_ref()


@dataclass(frozen=True)
class VectorCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::vector<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return VectorCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayCType(CType):
    elem: CType
    size: int

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::array<{self.elem.cpp_type()},{self.size}>"

    def remove_const_ref(self) -> CType:
        return ArrayCType(self.elem.remove_const_ref(), self.size)


@dataclass(frozen=True)
class TupleCType(CType):
    elems: list[CType]

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::tuple<{','.join([e.cpp_type() for e in self.elems])}>"

    def remove_const_ref(self) -> CType:
        return TupleCType([e.remove_const_ref() for e in self.elems])


@dataclass(frozen=True)
class MutRefCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        if strip_ref:
            return self.elem.cpp_type(strip_ref=strip_ref)
        return f"{self.elem.cpp_type()} &"

    def remove_const_ref(self) -> CType:
        return self.elem.remove_const_ref()


# A NamedCType is short for Named C++ semantic type.  A NamedCType represents a C++ type, plus
# semantic information about what it represents.  For example, consider the
# argument "bool pin_memory"; its normal C++ type is "bool", but its C++
# semantic type also keeps track that this represents a "pin_memory"; you can't
# just use a random other boolean in a context where you need a "pin_memory"!
#


@dataclass(frozen=True)
class NamedCType:
    name: ArgName
    type: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return self.type.cpp_type(strip_ref=strip_ref)

    def remove_const_ref(self) -> NamedCType:
        return NamedCType(self.name, self.type.remove_const_ref())

    def with_name(self, name: str) -> NamedCType:
        return NamedCType(name, self.type)


# A binding represents any C++ binding site for a formal parameter.
# We don't distinguish between binding sites for different APIs;
# instead, all of the important distinctions are encoded in CType,
# which you can use to figure out if a given Binding is appropriate
# for use in another context.  (See torchgen.api.translate)


@dataclass(frozen=True)
class Binding:
    name: str
    nctype: NamedCType
    argument: Argument | TensorOptionsArguments | SelfArgument
    # TODO: maybe don't represent default here
    default: str | None = None

    def rename(self, name: str) -> Binding:
        return Binding(
            name=name,
            nctype=self.nctype,
            argument=self.argument,
            default=self.default,
        )

    @property
    def type(self) -> str:
        return self.nctype.cpp_type()

    def no_default(self) -> Binding:
        return Binding(
            name=self.name,
            nctype=self.nctype,
            default=None,
            argument=self.argument,
        )

    def decl(self, *, func_ptr_cast: bool = False) -> str:
        mb_default = ""
        if self.default is not None:
            mb_default = f"={self.default}"

        # casting only needs to know the type
        if func_ptr_cast:
            return f"{self.type}"
        else:
            return f"{self.type} {self.name}{mb_default}"

    def defn(self) -> str:
        return f"{self.type} {self.name}"

    def with_name(self, name: str) -> Binding:
        return Binding(
            name=name, nctype=self.nctype, argument=self.argument, default=self.default
        )


# An Expr is a C++ expression.  It has a C++ string representing its syntax,
# as well as a CType saying what it provides.


@dataclass(frozen=True)
class Expr:
    expr: str
    type: NamedCType

```



## High-Level Overview

"""Where should I add a new type? `types_base.py` vs `types.py`This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn'tcontain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reusedif we want to generate code for another C++ library.Add new types to `types.py` if these types are ATen/c10 related.Add new types to `types_base.py` if they are basic and not attached to ATen/c10.

This Python file contains 14 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SpecialArgName`, `BaseCppType`, `CType`, `BaseCType`, `ConstRefCType`, `VectorCType`, `ArrayCType`, `TupleCType`, `MutRefCType`, `NamedCType`, `Binding`, `Expr`

**Functions defined**: `__str__`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `with_name`, `rename`, `type`

**Key imports**: annotations, ABC, abstractmethod, dataclass, auto, Enum, TYPE_CHECKING, Union, Argument, SelfArgument, TensorOptionsArguments


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/api/types`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `abc`: ABC, abstractmethod
- `dataclasses`: dataclass
- `enum`: auto, Enum
- `typing`: TYPE_CHECKING, Union
- `torchgen.model`: Argument, SelfArgument, TensorOptionsArguments


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`torchgen/api/types`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`types.py_docs.md`](./types.py_docs.md)
- [`signatures.py_docs.md`](./signatures.py_docs.md)


## Cross-References

- **File Documentation**: `types_base.py_docs.md`
- **Keyword Index**: `types_base.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torchgen/api/types`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torchgen/api/types`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces


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

Files in the same folder (`docs/torchgen/api/types`):

- [`types.py_kw.md_docs.md`](./types.py_kw.md_docs.md)
- [`types_base.py_kw.md_docs.md`](./types_base.py_kw.md_docs.md)
- [`signatures.py_docs.md_docs.md`](./signatures.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`signatures.py_kw.md_docs.md`](./signatures.py_kw.md_docs.md)
- [`types.py_docs.md_docs.md`](./types.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `types_base.py_docs.md_docs.md`
- **Keyword Index**: `types_base.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
