# Documentation: `docs/aten/src/ATen/core/dynamic_type.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/dynamic_type.h_docs.md`
- **Size**: 13,507 bytes (13.19 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/dynamic_type.h`

## File Metadata

- **Path**: `aten/src/ATen/core/dynamic_type.h`
- **Size**: 10,901 bytes (10.65 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>

#include <ATen/core/jit_type_base.h>
#include <optional>

namespace c10 {

using DynamicTypeBits = std::uint32_t;
#define DYNAMIC_TYPE_BIT(x) (1u << x)

constexpr DynamicTypeBits kDynamicCovariantTypeBit = DYNAMIC_TYPE_BIT(31);
constexpr DynamicTypeBits kDynamicAnyTypeBit = DYNAMIC_TYPE_BIT(30);

constexpr DynamicTypeBits kDynamicNoneTypeBit = DYNAMIC_TYPE_BIT(1);
constexpr DynamicTypeBits kDynamicIntTypeBit = DYNAMIC_TYPE_BIT(3);
constexpr DynamicTypeBits kDynamicFloatTypeBit = DYNAMIC_TYPE_BIT(4);
constexpr DynamicTypeBits kDynamicComplexTypeBit = DYNAMIC_TYPE_BIT(5);
constexpr DynamicTypeBits kDynamicListTypeBit = DYNAMIC_TYPE_BIT(7);
constexpr DynamicTypeBits kDynamicTupleTypeBit = DYNAMIC_TYPE_BIT(8);
constexpr DynamicTypeBits kDynamicClassTypeBit = DYNAMIC_TYPE_BIT(10);

#define FORALL_DYNAMIC_TYPES(_)                                              \
  _(Tensor, DYNAMIC_TYPE_BIT(0), 1)                                          \
  _(None, kDynamicNoneTypeBit, 1)                                            \
  _(Bool, DYNAMIC_TYPE_BIT(2), 1)                                            \
  _(Int, kDynamicIntTypeBit, 1)                                              \
  _(Float, kDynamicFloatTypeBit, 1)                                          \
  _(Complex, kDynamicComplexTypeBit, 1)                                      \
  _(Number,                                                                  \
    (kDynamicIntTypeBit | kDynamicFloatTypeBit | kDynamicComplexTypeBit),    \
    1)                                                                       \
  _(String, DYNAMIC_TYPE_BIT(6), 1)                                          \
  _(List, kDynamicListTypeBit, 0)                                            \
  _(Tuple, (kDynamicTupleTypeBit | kDynamicCovariantTypeBit), 0)             \
  _(Dict, DYNAMIC_TYPE_BIT(9), 0)                                            \
  _(Class, kDynamicClassTypeBit, 0)                                          \
  _(Optional,                                                                \
    (DYNAMIC_TYPE_BIT(11) | kDynamicNoneTypeBit | kDynamicCovariantTypeBit), \
    0)                                                                       \
  _(AnyList, (kDynamicListTypeBit | kDynamicAnyTypeBit), 1)                  \
  _(AnyTuple,                                                                \
    (kDynamicTupleTypeBit | kDynamicCovariantTypeBit | kDynamicAnyTypeBit),  \
    1)                                                                       \
  _(DeviceObj, DYNAMIC_TYPE_BIT(12), 1)                                      \
  _(StreamObj, DYNAMIC_TYPE_BIT(13), 1)                                      \
  _(Capsule, DYNAMIC_TYPE_BIT(14), 1)                                        \
  _(Generator, DYNAMIC_TYPE_BIT(15), 1)                                      \
  _(Storage, DYNAMIC_TYPE_BIT(16), 1)                                        \
  _(Var, DYNAMIC_TYPE_BIT(17), 0)                                            \
  _(AnyClass, (kDynamicClassTypeBit | kDynamicAnyTypeBit), 1)                \
  _(QScheme, DYNAMIC_TYPE_BIT(18), 1)                                        \
  _(Quantizer, DYNAMIC_TYPE_BIT(19), 1)                                      \
  _(AnyEnum, DYNAMIC_TYPE_BIT(20), 1)                                        \
  _(RRef, DYNAMIC_TYPE_BIT(21), 0)                                           \
  _(Future, DYNAMIC_TYPE_BIT(22), 0)                                         \
  _(Await, DYNAMIC_TYPE_BIT(23), 0)                                          \
  _(Any, 0xffffffff, 1)

#define FORALL_DYNAMIC_TYPES_FAKE(_) \
  _(ScalarType, kDynamicIntTypeBit, 1)                                \
  _(Layout, kDynamicIntTypeBit, 1)                                        \
  _(SymInt, kDynamicIntTypeBit, 1)                                        \
  _(SymBool, kDynamicIntTypeBit, 1)                                        \
  _(MemoryFormat, kDynamicIntTypeBit, 1)

#define FORWARD_DECL_TYPE(NAME, _, __) struct NAME ## Type;
  FORALL_DYNAMIC_TYPES(FORWARD_DECL_TYPE)
  FORALL_DYNAMIC_TYPES_FAKE(FORWARD_DECL_TYPE)
#undef FORWARD_DECL_TYPE

class DynamicType;
using DynamicTypePtr = std::shared_ptr<DynamicType>;

/**
 * DynamicType is designed as a low dependency type system for TorchScript. The
 * existing JIT types are used for both compilation and runtime, which makes
 * sense for server contexts because we often compile and run the model in
 * the same process, however this doesn't hold for mobile devices where we
 * always compiles a model ahead of time, therefore there will be dependencies
 * which are not needed, but built with mobile runtime causing binary size
 * bloat, by design. Every basic type like Int, Bool or String will bring their
 * vtable, typeinfo, constructor, destructor and even more data from their
 * specializations for STL types to the binary causing a long tail bloat.
 *
 * The core problem is about the complexity to implement and maintain a single
 * type system for both analysis and execution purposes. Although they should
 * have the exactly same semantics, in practice implement a unified abstraction
 * adds conceptual and representational overhead for both sides of the world.
 *
 * To address the issues, DynamicType implements a minimal subset of JIT types
 * and uses a generic algorithm to test all subtyping relations. To achieve
 * this, we assign each dynamic type a single integer tag to represent its
 * semantics. More specifically, a dynamic type is defined as a set of "control
 * bits" and "data bits", where control bits describe the special behavior when
 * testing a type and data bits map to identity of each nominal type. We use bit
 * operations to perform all the tests.
 *
 * For example, a "covariant bit" is a control bit used to describe if a type
 * is covariant, right now the most used one is tuple type, and in addition to
 * the control bit, tuple type's data bit is the 8th bit from the LSB. Control
 * bits start from MSB and data bits start from LSB.
 *
 * If two types are equal, then they are subtype of each other, also if the bits
 * from one type tag is subset of the other tag, it automatically becomes a
 * subtype of the other. This simplifies the subtyping logic a lot, and over the
 * long term it is possible to adopt this scheme on the server side as well.
 * Special cases can be added but they generally should not take too much code
 * size.
 *
 * DynamicType may or may not inherit from c10::Type because it's not the core
 * requirement of DynamicType to interface with existing JIT types, but we might
 * want to inherit from c10::Type to reduce the migration cost.
 */
class DynamicType : public SharedType {
  using ClassTypePtr = std::shared_ptr<const c10::ClassType>;

  /**
   * A implementation detail to support NamedTuple.
   */
  struct LabeledDynamicType {
    std::optional<std::string> label;
    DynamicTypePtr ty;
    explicit LabeledDynamicType(DynamicTypePtr t) : ty(std::move(t)) {}

    bool equals(const LabeledDynamicType& other) const;
    bool isSubtypeOf(const LabeledDynamicType& other) const;
  };

 public:
  // TODO Change Ptr to DynamicTypePtr when all migrations are done.
  using Ptr = TypePtr;
  using ElementType = DynamicType;
  ~DynamicType() override;

  struct Arguments {
    Arguments() = default;
    Arguments(c10::ArrayRef<TypePtr> /*args*/);
    Arguments(const std::vector<std::string_view>& /*names*/, c10::ArrayRef<TypePtr> /*args*/);
    std::vector<LabeledDynamicType> elems;
  };

  enum class Tag : DynamicTypeBits {
#define DYNAMIC_TYPE_ITEM(NAME, VAL, _) NAME = VAL,
    FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_ITEM)
    FORALL_DYNAMIC_TYPES_FAKE(DYNAMIC_TYPE_ITEM)
#undef DYNAMIC_TYPE_ITEM
  };

  bool equals(const Type& rhs) const override;
  bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;
  std::string str() const override;
  static const TypeKind Kind = TypeKind::DynamicType;
  static TORCH_API DynamicTypePtr create(Type& ty);

  explicit DynamicType(Tag /*tag*/, Arguments /*arguments*/);
  explicit DynamicType(Tag /*tag*/, std::string_view /*name*/, Arguments /*arguments*/);

  DynamicType(DynamicType&& other) = delete;
  DynamicType(const DynamicType&) = delete;
  DynamicType& operator=(const DynamicType&) = delete;
  DynamicType& operator=(DynamicType&&) = delete;

  TypePtr containedType(size_t /*i*/) const override;
  size_t containedTypeSize() const override;
  Tag tag() const {
    return tag_;
  }
  const std::optional<std::string>& name() const {
    return name_;
  }
  const Arguments& arguments() const {
    return arguments_;
  }
  TORCH_API TypeKind dynamicKind() const;

  // Should be used only on the server side to restore static type information.
#ifndef C10_MOBILE
  TORCH_API
#endif
  TypePtr fallback() const;

 private:
  bool symmetric() const override {
    return false;
  }
  friend struct Type;
  // NOTE: Here we are using SingletonOrSharedTypePtr to mean
  // "original-type-because-it-was-actually-a-DynamicType or shared".
  static SingletonOrSharedTypePtr<const DynamicType> create(const Type& ty);
  DynamicType(const Type& other);
  bool equals(const DynamicType& other) const;

  template <typename F>
  bool compareArguments(const DynamicType& other, const F& f) const {
    if (arguments_.elems.size() != other.arguments_.elems.size()) {
      return false;
    }
    for (size_t i = 0; i < arguments_.elems.size(); i++) {
      if (!f(arguments_.elems[i], other.arguments_.elems[i])) {
        return false;
      }
    }
    return true;
  }

  Tag tag_;
  std::optional<std::string> name_;
  union {
    Arguments arguments_;
    ClassTypePtr class_;
  };
};

template <typename T>
struct DynamicTypeTrait {
  C10_NOINLINE static auto tagValue() {
    TORCH_CHECK(false);
    return DynamicType::Tag::Any;
  }
};

namespace detail {
C10_NOINLINE DynamicTypePtr makeBaseType(DynamicType::Tag tag);
}

#define DYNAMIC_TYPE_TAG_VALUE(NAME, _, IS_BASE_TYPE)      \
  template <>                                              \
  struct TORCH_API DynamicTypeTrait<NAME##Type> {          \
    C10_ERASE static auto tagValue() {                     \
      return DynamicType::Tag::NAME;                       \
    }                                                      \
    static constexpr bool isBaseType = IS_BASE_TYPE;       \
    template <typename T = const DynamicTypePtr&>          \
    static std::enable_if_t<isBaseType, T> getBaseType() { \
      static auto type = detail::makeBaseType(tagValue()); \
      return type;                                         \
    }                                                      \
  }; // namespace c10
FORALL_DYNAMIC_TYPES(DYNAMIC_TYPE_TAG_VALUE)
FORALL_DYNAMIC_TYPES_FAKE(DYNAMIC_TYPE_TAG_VALUE)
#undef DYNAMIC_TYPE_TAG_VALUE

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 27 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `NAME`, `DynamicType`, `DynamicType`, `LabeledDynamicType`, `Arguments`, `Tag`, `Type`, `DynamicTypeTrait`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `memory`
- `type_traits`
- `ATen/core/jit_type_base.h`
- `optional`


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
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `dynamic_type.h_docs.md`
- **Keyword Index**: `dynamic_type.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`docs/aten/src/ATen/core`):

- [`operator_name.cpp_docs.md_docs.md`](./operator_name.cpp_docs.md_docs.md)
- [`builtin_function.h_kw.md_docs.md`](./builtin_function.h_kw.md_docs.md)
- [`QuantizerBase.h_docs.md_docs.md`](./QuantizerBase.h_docs.md_docs.md)
- [`MT19937RNGEngine.h_docs.md_docs.md`](./MT19937RNGEngine.h_docs.md_docs.md)
- [`UndefinedTensorImpl.h_docs.md_docs.md`](./UndefinedTensorImpl.h_docs.md_docs.md)
- [`IListRef_test.cpp_docs.md_docs.md`](./IListRef_test.cpp_docs.md_docs.md)
- [`CheckMemoryFormat.h_docs.md_docs.md`](./CheckMemoryFormat.h_docs.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`PythonFallbackKernel.cpp_docs.md_docs.md`](./PythonFallbackKernel.cpp_docs.md_docs.md)
- [`Dict.h_kw.md_docs.md`](./Dict.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `dynamic_type.h_docs.md_docs.md`
- **Keyword Index**: `dynamic_type.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
