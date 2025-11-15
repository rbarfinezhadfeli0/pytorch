# Documentation: `docs/torch/csrc/jit/tensorexpr/types.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/types.cpp_docs.md`
- **Size**: 5,534 bytes (5.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/types.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/types.cpp`
- **Size**: 3,072 bytes (3.00 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/tensorexpr/types.h>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/exceptions.h>

#include <c10/util/Logging.h>

namespace torch::jit::tensorexpr {

Dtype Dtype::scalar_dtype() const {
  return ToDtype(scalar_type_);
}

#define DTYPE_DEFINE(_1, n) TORCH_API Dtype k##n(ScalarType::n, 1);

AT_FORALL_SCALAR_TYPES_AND7(
    Bool,
    Half,
    BFloat16,
    Float8_e5m2,
    Float8_e5m2fnuz,
    Float8_e4m3fn,
    Float8_e4m3fnuz,
    DTYPE_DEFINE)
DTYPE_DEFINE(c10::quint8, QUInt8)
DTYPE_DEFINE(c10::qint8, QInt8)

#undef DTYPE_DEFINE

TORCH_API Dtype kHandle(ScalarType::Undefined, 1);

Dtype ToDtype(ScalarType type) {
  switch (type) {
#define TYPE_CASE(_1, n) \
  case ScalarType::n:    \
    return k##n;
    AT_FORALL_SCALAR_TYPES_AND7(
        Bool,
        Half,
        BFloat16,
        Float8_e5m2,
        Float8_e5m2fnuz,
        Float8_e4m3fn,
        Float8_e4m3fnuz,
        TYPE_CASE)
    TYPE_CASE(c10::quint8, QUInt8);
    TYPE_CASE(c10::qint8, QInt8);
#undef TYPE_CASE

    case ScalarType::Undefined:
      return kHandle;
    default:
      throw unsupported_dtype();
  }
}

TORCH_API std::ostream& operator<<(std::ostream& stream, const Dtype& dtype) {
  stream << dtype.scalar_type_;
  if (dtype.lanes() > 1) {
    stream << "x" << dtype.lanes();
    ;
  }
  return stream;
}

int Dtype::byte_size() const {
  int scalar_size = -1;
  switch (scalar_type_) {
#define TYPE_CASE(Type, Name)   \
  case ScalarType::Name:        \
    scalar_size = sizeof(Type); \
    break;

    AT_FORALL_SCALAR_TYPES_AND7(
        Bool,
        Half,
        BFloat16,
        Float8_e5m2,
        Float8_e4m3fn,
        Float8_e5m2fnuz,
        Float8_e4m3fnuz,
        TYPE_CASE);
    TYPE_CASE(c10::quint8, QUInt8);
    TYPE_CASE(c10::qint8, QInt8);
#undef TYPE_CASE
    default:
      throw std::runtime_error(
          "invalid scalar type; " + std::to_string(scalar_type_));
  }
  return static_cast<int>(scalar_size * lanes());
}

std::string Dtype::ToCppString() const {
  switch (scalar_type_) {
#define TYPE_CASE(t, n) \
  case ScalarType::n:   \
    return #t;
    AT_FORALL_SCALAR_TYPES(TYPE_CASE);
#undef TYPE_CASE
    case ScalarType::Bool:
      return "bool";
    case ScalarType::Half:
      return "half";
    case ScalarType::BFloat16:
      return "bfloat16";
    case ScalarType::Float8_e5m2:
      return "float8_e5m2";
    case ScalarType::Float8_e4m3fn:
      return "float8_e4m3fn";
    case ScalarType::Float8_e5m2fnuz:
      return "float8_e5m2fnuz";
    case ScalarType::Float8_e4m3fnuz:
      return "float8_e4m3fnuz";
    case ScalarType::QInt8:
      return "qint8";
    case ScalarType::QUInt8:
      return "quint8";
    default:
      throw unsupported_dtype();
  }
  return "invalid";
}

} // namespace torch::jit::tensorexpr

namespace std {

std::string to_string(const Dtype& dtype) {
  std::ostringstream oss;
  oss << dtype;
  return oss.str();
}

std::string to_string(const ScalarType& type) {
  std::ostringstream oss;
  oss << type;
  return oss.str();
}

} // namespace std

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `std`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/tensorexpr/types.h`
- `torch/csrc/Export.h`
- `torch/csrc/jit/tensorexpr/exceptions.h`
- `c10/util/Logging.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `types.cpp_docs.md`
- **Keyword Index**: `types.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/tensorexpr`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/jit/tensorexpr`):

- [`loopnest.h_kw.md_docs.md`](./loopnest.h_kw.md_docs.md)
- [`expr.h_docs.md_docs.md`](./expr.h_docs.md_docs.md)
- [`block_codegen.h_kw.md_docs.md`](./block_codegen.h_kw.md_docs.md)
- [`ir_cloner.cpp_kw.md_docs.md`](./ir_cloner.cpp_kw.md_docs.md)
- [`tensorexpr_init.h_docs.md_docs.md`](./tensorexpr_init.h_docs.md_docs.md)
- [`lowerings.cpp_kw.md_docs.md`](./lowerings.cpp_kw.md_docs.md)
- [`graph_opt.h_kw.md_docs.md`](./graph_opt.h_kw.md_docs.md)
- [`eval.h_kw.md_docs.md`](./eval.h_kw.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `types.cpp_docs.md_docs.md`
- **Keyword Index**: `types.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
