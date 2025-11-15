# Documentation: `aten/src/ATen/core/enum_type.h`

## File Metadata

- **Path**: `aten/src/ATen/core/enum_type.h`
- **Size**: 2,929 bytes (2.86 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/ivalue.h>

#include <utility>

namespace c10 {

struct EnumType;
using EnumTypePtr = std::shared_ptr<EnumType>;
using EnumNameValue = std::pair<std::string, IValue>;
struct TORCH_API EnumType : public NamedType {
  friend struct Type;
  static const TypeKind Kind = TypeKind::EnumType;

  static EnumTypePtr create(
      const c10::QualifiedName& qualified_class_name,
      TypePtr value,
      std::vector<EnumNameValue> enum_names_values,
      std::weak_ptr<::torch::jit::CompilationUnit> cu) {
    C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")
    switch (value->kind()) {
      case TypeKind::IntType:
      case TypeKind::FloatType:
      case TypeKind::StringType:
        return EnumTypePtr(new EnumType(
            qualified_class_name,
            std::move(value),
            std::move(enum_names_values),
            std::move(cu)));
      default:
        TORCH_CHECK(
            false,
            "Cannot create Enum with value type '",
            value->str(),
            "', only int, float and string are supported");
    }
    C10_DIAGNOSTIC_POP()
  }

  std::string str() const override {
    return "Enum<" + annotation_str() + ">";
  }

  std::string repr_str() const override {
    return str();
  }

  const TypePtr& getValueType() const {
    return value_type_;
  }

  bool equals(const Type& rhs) const override {
    if (auto* enum_rhs = rhs.castRaw<EnumType>()) {
      return name().has_value() && name() == enum_rhs->name() &&
          *getValueType() == *(enum_rhs->getValueType()) &&
          this->compilation_unit() == enum_rhs->compilation_unit();
    }
    return false;
  }

  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

  std::shared_ptr<const ::torch::jit::CompilationUnit> compilation_unit()
      const {
    auto cu = cu_.lock();
    return cu;
  }

  const QualifiedName& qualifiedClassName() const {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return name().value();
  }

  at::ArrayRef<TypePtr> containedTypes() const override {
    return value_type_;
  }

  const at::ArrayRef<EnumNameValue> enumNamesValues() const {
    return enum_names_values_;
  }

 private:
  EnumType(
      c10::QualifiedName qualified_class_name,
      TypePtr value_type,
      std::vector<EnumNameValue> enum_names_values,
      std::weak_ptr<torch::jit::CompilationUnit> cu)
      : NamedType(TypeKind::EnumType, std::move(qualified_class_name)),
        value_type_(std::move(value_type)),
        enum_names_values_(std::move(enum_names_values)),
        cu_(std::move(cu)) {}

  std::string annotation_str_impl(
      [[maybe_unused]] const TypePrinter& printer = nullptr) const override {
    return qualifiedClassName().qualifiedName();
  }

  TypePtr value_type_;
  std::vector<EnumNameValue> enum_names_values_;
  std::weak_ptr<::torch::jit::CompilationUnit> cu_;
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `EnumType`, `TORCH_API`, `Type`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `utility`


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

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `enum_type.h_docs.md`
- **Keyword Index**: `enum_type.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
