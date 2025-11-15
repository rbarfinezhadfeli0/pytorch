# Documentation: `docs/torchgen/api/types/types.py_docs.md`

## File Metadata

- **Path**: `docs/torchgen/api/types/types.py_docs.md`
- **Size**: 9,099 bytes (8.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torchgen/api/types/types.py`

## File Metadata

- **Path**: `torchgen/api/types/types.py`
- **Size**: 6,210 bytes (6.06 KB)
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

from dataclasses import dataclass

from torchgen.api.types.types_base import (
    BaseCppType,
    BaseCType,
    boolT,
    byteT,
    charT,
    CType,
    doubleT,
    floatT,
    int32T,
    longT,
    shortT,
)
from torchgen.model import BaseTy, ScalarType


TENSOR_LIST_LIKE_CTYPES = [
    "at::TensorList",
    "const c10::List<::std::optional<at::Tensor>> &",
    "const at::ITensorListRef &",
]


halfT = BaseCppType("at", "Half")
complexHalfT = BaseCppType(
    "c10", "complex<c10::Half>"
)  # stuffing template param here is an abuse
complexFloatT = BaseCppType("c10", "complex<float>")
complexDoubleT = BaseCppType("c10", "complex<double>")
bfloat16T = BaseCppType("at", "BFloat16")
float8_e5m2T = BaseCppType("at", "Float8_e5m2")
float8_e5m2fnuzT = BaseCppType("at", "Float8_e5m2fnuz")
float8_e4m3fnT = BaseCppType("at", "Float8_e4m3fn")
float8_e4m3fnuzT = BaseCppType("at", "Float8_e4m3fnuz")
float8_e8m0fnuT = BaseCppType("at", "Float8_e8m0fnu")
stringT = BaseCppType("c10", "string_view")
generatorT = BaseCppType("at", "Generator")
scalarTypeT = BaseCppType("at", "ScalarType")
tensorT = BaseCppType("at", "Tensor")
optionalTensorRefT = BaseCppType("at", "OptionalTensorRef")
tensorListT = BaseCppType("at", "TensorList")
iTensorListRefT = BaseCppType("at", "ITensorListRef")
iOptTensorListRefT = BaseCppType("at", "IOptTensorListRef")
dimnameT = BaseCppType("at", "Dimname")
dimnameListT = BaseCppType("at", "DimnameList")
dimVectorT = BaseCppType("at", "DimVector")
layoutT = BaseCppType("at", "Layout")
deviceT = BaseCppType("at", "Device")
deviceIndexT = BaseCppType("at", "DeviceIndex")
scalarT = BaseCppType("at", "Scalar")
optionalScalarRefT = BaseCppType("at", "OptionalScalarRef")
memoryFormatT = BaseCppType("at", "MemoryFormat")
qschemeT = BaseCppType("at", "QScheme")
storageT = BaseCppType("at", "Storage")
streamT = BaseCppType("at", "Stream")
intArrayRefT = BaseCppType("at", "IntArrayRef")
optionalIntArrayRefT = BaseCppType("at", "OptionalIntArrayRef")
optionalSymIntArrayRefT = BaseCppType("at", "OptionalSymIntArrayRef")
tensorOptionsT = BaseCppType("at", "TensorOptions")
typeAndSizeT = BaseCppType("torch::autograd::generated", "TypeAndSize")
tensorGeometryT = BaseCppType("at", "TensorGeometry")
SymIntT = BaseCppType("c10", "SymInt")
SymBoolT = BaseCppType("c10", "SymBool")
symIntArrayRefT = BaseCppType("c10", "SymIntArrayRef")

# Types representing template parameters.  Technically, we probably shouldn't
# represent them this way in codegen, but it was pretty convenient.
scalar_t = BaseCppType("", "scalar_t")
opmath_t = BaseCppType("", "opmath_t")

ScalarTypeToCppMapping: dict[ScalarType, BaseCppType] = {
    ScalarType.Byte: byteT,
    ScalarType.Char: charT,
    ScalarType.Short: shortT,
    ScalarType.Int: int32T,
    ScalarType.Long: longT,
    ScalarType.Half: halfT,
    ScalarType.Float: floatT,
    ScalarType.Double: doubleT,
    ScalarType.ComplexHalf: complexHalfT,
    ScalarType.ComplexFloat: complexFloatT,
    ScalarType.ComplexDouble: complexDoubleT,
    ScalarType.Bool: boolT,
    ScalarType.Float8_e5m2: float8_e5m2T,
    ScalarType.Float8_e5m2fnuz: float8_e5m2fnuzT,
    ScalarType.Float8_e4m3fn: float8_e4m3fnT,
    ScalarType.Float8_e4m3fnuz: float8_e4m3fnuzT,
    ScalarType.Float8_e8m0fnu: float8_e8m0fnuT,
}

BaseTypeToCppMapping: dict[BaseTy, BaseCppType] = {
    BaseTy.int: longT,
    BaseTy.float: doubleT,
    BaseTy.bool: boolT,
    BaseTy.str: stringT,
    BaseTy.Generator: generatorT,
    BaseTy.ScalarType: scalarTypeT,
    BaseTy.Tensor: tensorT,
    BaseTy.Dimname: dimnameT,
    BaseTy.DimVector: dimVectorT,
    BaseTy.Layout: layoutT,
    BaseTy.Device: deviceT,
    BaseTy.DeviceIndex: deviceIndexT,
    BaseTy.Scalar: scalarT,
    BaseTy.MemoryFormat: memoryFormatT,
    BaseTy.QScheme: qschemeT,
    BaseTy.Storage: storageT,
    BaseTy.Stream: streamT,
    BaseTy.SymInt: SymIntT,
    BaseTy.SymBool: SymBoolT,
}

# CTypes encode C++ type structure as needed for translation.


@dataclass(frozen=True)
class OptionalCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"::std::optional<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return OptionalCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ListCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"c10::List<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return ListCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class ArrayRefCType(CType):
    elem: CType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        # Do not pass `strip_ref` recursively.
        return f"at::ArrayRef<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return ArrayRefCType(self.elem.remove_const_ref())


@dataclass(frozen=True)
class VectorizedCType(CType):
    # This template is explicitly specialized, so the only valid
    # elems are those we have specializations for (e.g., float, double, ...)
    # scalar_t is also a common argument here (when we are codegen in
    # a templated context)
    elem: BaseCType

    def cpp_type(self, *, strip_ref: bool = False) -> str:
        return f"at::vec::Vectorized<{self.elem.cpp_type()}>"

    def remove_const_ref(self) -> CType:
        return self

```



## High-Level Overview

"""Where should I add a new type? `types_base.py` vs `types.py`This file defines data model classes for torchgen typing system, as well as some base types such as int32_t.`types.py` defines ATen Tensor type and some c10 types, along with signatures that use these types.The difference between these two files, is `types_base.py` should be implementation-agnostic, meaning it shouldn'tcontain any type definition that is tight to a specific C++ library (e.g., ATen), so that it can be easily reusedif we want to generate code for another C++ library.Add new types to `types.py` if these types are ATen/c10 related.Add new types to `types_base.py` if they are basic and not attached to ATen/c10.

This Python file contains 5 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OptionalCType`, `ListCType`, `ArrayRefCType`, `VectorizedCType`

**Functions defined**: `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`, `cpp_type`, `remove_const_ref`

**Key imports**: annotations, dataclass, BaseTy, ScalarType


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torchgen/api/types`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `dataclasses`: dataclass
- `torchgen.model`: BaseTy, ScalarType


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

Files in the same folder (`torchgen/api/types`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`signatures.py_docs.md`](./signatures.py_docs.md)
- [`types_base.py_docs.md`](./types_base.py_docs.md)


## Cross-References

- **File Documentation**: `types.py_docs.md`
- **Keyword Index**: `types.py_kw.md`
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

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
- [`types_base.py_docs.md_docs.md`](./types_base.py_docs.md_docs.md)
- [`signatures.py_kw.md_docs.md`](./signatures.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `types.py_docs.md_docs.md`
- **Keyword Index**: `types.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
