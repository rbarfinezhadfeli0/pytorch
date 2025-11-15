# Documentation: `docs/aten/src/ATen/core/NestedIntSymNodeImpl.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/NestedIntSymNodeImpl.h_docs.md`
- **Size**: 8,580 bytes (8.38 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/NestedIntSymNodeImpl.h`

## File Metadata

- **Path**: `aten/src/ATen/core/NestedIntSymNodeImpl.h`
- **Size**: 5,960 bytes (5.82 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <cstdint>
#include <optional>
#include <string>

namespace c10 {

// The motivating usecase for this is to represent the ragged size structure
// of a jagged tensor [B, [s_0, s_1, s_2], D] as a single integer j0. This
// allows us to simply return [B, j0, D] if someone queries for the size of our
// tensor.
//
// Morally we define comparison between two nested ints to return true if
// that comparison holds for all corresponding elements of the arrays they
// represent. Comparison between a nested int and a plain int is defined
// similarly.
//
// To simulate this desired behavior but also avoid the O(N) cost of checking,
// we associate each raggedness pattern with an integer "id" that can be used as
// a proxy to evaluate equality. We also constrain the range of values for this
// as to enable inequality checks.
//
// We also support a positive integer scalar "coeff" that is used for computing
// strides. For example given, a [B, j0, D] tensor, it can be strided in two
// different ways: [D * j0, D, 1] and [j0, 1, sum(j0)]. The coeff is used to
// differentiate the two cases.
//
// During tracing the strides of the outputs need to be a function of the size
// and strides of the inputs so it is important that NestedIntSymNode itself is
// able to express this.
class TORCH_API NestedIntSymNodeImpl : public SymNodeImpl {
 public:
  // CAUTION: you should probably not be constructing these directly; please
  // the higher-level API in python instead (TODO: actually introduce that).
  explicit NestedIntSymNodeImpl(int64_t val, int64_t coeff)
      : val_(val), coeff_(coeff) {}

  bool bool_() override {
    return false;
  }

  bool is_int() override {
    return true;
  }

  bool is_float() override {
    return false;
  }

  bool is_bool() override {
    return false;
  }

  bool is_nested_int() const override {
    return true;
  }

  bool has_hint() override {
    return true;
  }

  c10::SymNode wrap_int(int64_t num) override {
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<int64_t>>(num));
  }

  int64_t guard_int(const char* file, int64_t line) override {
    TORCH_CHECK(false);
  }

  double guard_float(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a float");
  }

  bool guard_bool(const char* file, int64_t line) override {
    TORCH_CHECK(false, "not a bool");
  }

  int64_t int_() override {
    TORCH_CHECK(false);
  }

  std::string str() override {
    if (coeff_ == 1) {
      return "j" + std::to_string(val_);
    }
    return std::to_string(coeff_) + "*j" + std::to_string(val_);
  }

  // NOTE [ Inequalities with nested int ]
  //
  // The semantics of nested int when it comes to relations is that it is
  // treated as integer known to be within a certain range,
  //
  //     j0 \in [2, int64_t::max]
  //
  // allowing us to answer queries like j0 >= 1 (True), and j0 == 0 (False).
  // This is a useful default range for the raggedness pattern of a jagged
  // tensor (1) since sizes are non-negative, and (2) we need to get past 0/1
  // specialization checks.
  //
  // [ Indeterminate inequalities error out ]
  //
  // Given the semantic defined above, certain relations like j0 < 3 are thus
  // indeterminable. In our impl today, evaluating such relations error
  //
  // It may seem convenient to just define indeterminate relations to return
  // False, but the implementation we maintain in parallel using sympy does not
  // allow this.
  //
  // Sympy only allows overriding of Ge. The other relations (Lt, Gt, Le) are,
  // by consequence, all derived from Ge e.g., Lt(a, b) := !Ge(a, b). This
  // would mean that means that if we define the indeterminate j0 >= 3 to be
  // False, the also indeterminate j0 < 3 will be evaluated to be True!
  //
  // [ Coefficient are assumed positive ]
  //
  // For the purpose of computing inequalities, we consider the coefficient of
  // the nested int to be a positive integer.
  //
  // Thus, no modifications are needed to the logic since
  // j0 >= k implies coeff * j0 >= k
  //
  c10::SymNode eq(const c10::SymNode& other) override;
  c10::SymNode ne(const c10::SymNode& other) override;
  c10::SymNode ge(const c10::SymNode& other) override;
  c10::SymNode gt(const c10::SymNode& other) override;
  c10::SymNode lt(const c10::SymNode& other) override;
  c10::SymNode le(const c10::SymNode& other) override;
  c10::SymNode mul(const c10::SymNode& other) override;

  std::optional<int64_t> nested_int() override {
    return val_;
  }

  std::optional<int64_t> nested_int_coeff() override {
    return coeff_;
  }

  bool is_symbolic() override {
    return false;
  }

  c10::SymNode clone() override;

#define DEFINE_BINARY_NOT_SUPPORTED(name)                           \
  c10::SymNode name(const c10::SymNode& other) override {           \
    TORCH_CHECK(false, #name " not supported by NestedIntSymNode"); \
  }

  DEFINE_BINARY_NOT_SUPPORTED(add)
  DEFINE_BINARY_NOT_SUPPORTED(sub)
  DEFINE_BINARY_NOT_SUPPORTED(truediv)
  DEFINE_BINARY_NOT_SUPPORTED(pow)
  DEFINE_BINARY_NOT_SUPPORTED(floordiv)
  DEFINE_BINARY_NOT_SUPPORTED(mod)
  DEFINE_BINARY_NOT_SUPPORTED(sym_min)
  DEFINE_BINARY_NOT_SUPPORTED(sym_max)
  DEFINE_BINARY_NOT_SUPPORTED(sym_and)
  DEFINE_BINARY_NOT_SUPPORTED(sym_or)

#undef DEFINE_BINARY_NOT_SUPPORTED

#define DEFINE_NOT_SUPPORTED(name)                                     \
  c10::SymNode name() override {                                       \
    TORCH_CHECK(false, #name " is not supported by NestedIntSymNode"); \
  }

  DEFINE_NOT_SUPPORTED(sym_not)
  DEFINE_NOT_SUPPORTED(ceil)
  DEFINE_NOT_SUPPORTED(floor)
  DEFINE_NOT_SUPPORTED(neg)
  DEFINE_NOT_SUPPORTED(sym_float)

#undef DEFINE_NOT_SUPPORTED

 private:
  int64_t val_;
  int64_t coeff_;
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 30 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/ConstantSymNodeImpl.h`
- `c10/core/SymNodeImpl.h`
- `c10/macros/Export.h`
- `c10/util/Exception.h`
- `c10/util/intrusive_ptr.h`
- `cstdint`
- `optional`
- `string`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

- **File Documentation**: `NestedIntSymNodeImpl.h_docs.md`
- **Keyword Index**: `NestedIntSymNodeImpl.h_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `NestedIntSymNodeImpl.h_docs.md_docs.md`
- **Keyword Index**: `NestedIntSymNodeImpl.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
