# Documentation: `docs/aten/src/ATen/core/NamedTensor.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/NamedTensor.h_docs.md`
- **Size**: 7,788 bytes (7.61 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/NamedTensor.h`

## File Metadata

- **Path**: `aten/src/ATen/core/NamedTensor.h`
- **Size**: 5,276 bytes (5.15 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Dimname.h>
#include <c10/core/TensorImpl.h>

namespace at {

class TensorBase;

// XXX: This file exists because TensorImpl is in c10, but Dimname is in ATen.
// Due to the c10/ATen library split, TensorImpl cannot depend on Dimname,
// so we have a couple of workarounds.
//
// In the long term, we'll move Dimname to c10 and everything in this file
// can be refactored out. The main blocker for that is that "c10::Symbol"
// actually exists outside of c10 and needs to be moved in.

// TensorImpl has a unique_ptr<NamedTensorMetaInterface> field.
// XXX: Ideally we would just put std::optional<vector<Dimname>> into TensorImpl.
//
// This class has an important invariant: there must be at least ONE
// non-wildcard
struct TORCH_API NamedTensorMeta final : public c10::NamedTensorMetaInterface {
  // This enum is to remind people that the invariant on constructors is that
  // the list of dimnames must have at least one non-wildcard
  enum HAS_NON_WILDCARD {
    HasNonWildcard
  };

  explicit NamedTensorMeta(HAS_NON_WILDCARD /*unused*/, DimnameList names)
    : names_(names.vec()) {
    check_invariants();
  }
  explicit NamedTensorMeta(HAS_NON_WILDCARD /*unused*/, std::vector<Dimname>&& names)
    : names_(std::move(names)) {
    check_invariants();
  }

  std::unique_ptr<c10::NamedTensorMetaInterface> clone() const override {
    return std::make_unique<NamedTensorMeta>(HasNonWildcard, names_);
  }

  DimnameList names() const { return names_; }

  // Used for an assertion in TensorImpl.h
  int64_t slow_dim() const override {
    return static_cast<int64_t>(names_.size());
  }

  void check_invariants() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      std::any_of(names_.begin(), names_.end(), [](const Dimname& n) { return !n.isWildcard(); }));
  }

  void set_names(HAS_NON_WILDCARD /*unused*/, DimnameList new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    std::copy(new_names.begin(), new_names.end(), names_.begin());
    check_invariants();
  }

  void set_names(HAS_NON_WILDCARD /*unused*/, std::vector<Dimname>&& new_names) {
    TORCH_INTERNAL_ASSERT(new_names.size() == names_.size());
    names_ = std::move(new_names);
    check_invariants();
  }

  // INVARIANT: at least one Dimname is non-WILDCARD
  std::vector<Dimname> names_;
};

// When NamesMode is disabled, then all operations ignore tensors' names fields.
// Concretely speaking, all tensors are treated as having nullopt names.
struct TORCH_API NamesMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};


// A RAII, thread local (!) guard that enables or disables names upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API NoNamesGuard {
  NoNamesGuard() : prev_mode(NamesMode::is_enabled()) {
    NamesMode::set_enabled(false);
  }
  NoNamesGuard(const NoNamesGuard&) = delete;
  NoNamesGuard(NoNamesGuard&&) = delete;
  NoNamesGuard& operator=(const NoNamesGuard&) = delete;
  NoNamesGuard& operator=(NoNamesGuard&&) = delete;
  ~NoNamesGuard() {
    if (initialized) {
      reset();
    }
  }
  void reset() {
    TORCH_INTERNAL_ASSERT(initialized);
    NamesMode::set_enabled(prev_mode);
  }
 private:
  bool prev_mode;
  bool initialized{true};
};

void check_names_valid_for(const TensorBase& tensor, DimnameList names);
void check_names_valid_for(size_t tensor_dim, DimnameList names);

// Sets the names of `tensor` to be `names`.
TORCH_API const TensorBase& internal_set_names_inplace(const TensorBase& tensor, std::optional<DimnameList> names);
TORCH_API const TensorBase& internal_set_names_inplace(const TensorBase& tensor, std::vector<Dimname>&& names, bool validate_names);

constexpr size_t kMaxNamedTensorDim = 64;

DimnameList default_names(size_t len);

namespace impl {

// Some helper functions on TensorImpl. Useful for working with names in TH.
// XXX: Ideally these would exist as methods on TensorImpl
TORCH_API void internal_set_names_inplace(TensorImpl* impl, std::optional<DimnameList> names, bool validate_names);
TORCH_API void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names);

void check_names_valid_for(TensorImpl* impl, DimnameList names);

// Returns true if the tensor's names exist and are not all 'None'.
// Returns false if the tensor's names don't exist (were not allocated),
// or if all names are 'None'.
// We treat not-allocated-names the same as allocated names that are all 'None'.
TORCH_API bool has_names(const TensorImpl* impl);

// Returns the names of the tensor's dimensions.
// Unnamed tensors are treated as having 'None' in all dimension; this method
// would return a DimnameList of all 'None's for an unnamed tensor.
TORCH_API DimnameList get_names(const TensorImpl* impl);

// This is more of an implementation detail; one should use impl::get_names /
// Tensor::names() whenever possible because it provides a cleaner API.
// Returns the names of the tensor if they have been allocated; returns nullopt
// instead if the haven't been. The names of a tensor are not allocated if a
// tensor is constructed with names=None.
TORCH_API std::optional<DimnameList> get_opt_names(const TensorImpl* impl);

} // namespace impl

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `impl`, `at`

**Classes/Structs**: `TensorBase`, `has`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Dimname.h`
- `c10/core/TensorImpl.h`


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

- **File Documentation**: `NamedTensor.h_docs.md`
- **Keyword Index**: `NamedTensor.h_kw.md`
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

- **File Documentation**: `NamedTensor.h_docs.md_docs.md`
- **Keyword Index**: `NamedTensor.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
