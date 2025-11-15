# Documentation: `docs/aten/src/ATen/core/NamedTensor.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/NamedTensor.cpp_docs.md`
- **Size**: 7,613 bytes (7.43 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/NamedTensor.cpp`

## File Metadata

- **Path**: `aten/src/ATen/core/NamedTensor.cpp`
- **Size**: 5,143 bytes (5.02 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/NamedTensor.h>

#include <ATen/core/TensorBase.h>

namespace at {

thread_local static bool NamesMode_enabled = true;

bool NamesMode::is_enabled() {
  return NamesMode_enabled;
}

void NamesMode::set_enabled(bool enabled) {
  NamesMode_enabled = enabled;
  c10::impl::tls_set_dispatch_key_excluded(DispatchKey::Named, !enabled);
}

const TensorBase& internal_set_names_inplace(const TensorBase& tensor, std::optional<DimnameList> names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), names, /*validate_names=*/true);
  return tensor;
}

const TensorBase& internal_set_names_inplace(const TensorBase& tensor, std::vector<Dimname>&& names, bool validate_names) {
  impl::internal_set_names_inplace(tensor.unsafeGetTensorImpl(), std::move(names), validate_names);
  return tensor;
}

DimnameList default_names(size_t len) {
  static std::vector<Dimname> all_unnamed(kMaxNamedTensorDim, Dimname::wildcard());
    TORCH_INTERNAL_ASSERT(
        len <= kMaxNamedTensorDim,
        "Only tensors with up to ", kMaxNamedTensorDim, " are supported.");
  return DimnameList(&all_unnamed.front(), len);
}

static void check_unique_names(DimnameList names) {
  // Strategy: Compare each element with the ones that come after it.
  // Although this is O(N^2), in practice N is small (no more than 25).
  for (auto it = names.begin(); it != names.end(); ++it) {
    if (it->isWildcard()) continue;
    auto dup = std::find(it + 1, names.end(), *it);
    while (dup != names.end()) {
      TORCH_CHECK(false,
          "Cannot construct a tensor with duplicate names. Got names: ",
          names, ".");
    }
  }
}

void check_names_valid_for(const TensorBase& tensor, DimnameList names) {
  impl::check_names_valid_for(tensor.unsafeGetTensorImpl(), names);
}

void check_names_valid_for(size_t tensor_dim, DimnameList names) {
  TORCH_CHECK(
      tensor_dim <= kMaxNamedTensorDim,
      "Named tensors only support up to ", kMaxNamedTensorDim, " dims: "
      "Attempted to create a tensor with dim ", tensor_dim, " with names ", names);
  TORCH_CHECK(tensor_dim == names.size(),
      "Number of names (", names.size(), ") and "
      "number of dimensions in tensor (", tensor_dim, ") ",
      "do not match. Attempted to create a tensor with names ", names);
  check_unique_names(names);
}

namespace impl {

static NamedTensorMeta* get_named_tensor_meta(TensorImpl* impl) {
  if (!NamesMode::is_enabled()) {
    return nullptr;
  }
  return static_cast<NamedTensorMeta*>(impl->named_tensor_meta());
}

static const NamedTensorMeta* get_named_tensor_meta(const TensorImpl* impl) {
  if (!NamesMode::is_enabled()) {
    return nullptr;
  }
  return static_cast<const NamedTensorMeta*>(impl->named_tensor_meta());
}

void check_names_valid_for(TensorImpl* impl, DimnameList names) {
  check_names_valid_for(impl->dim(), names);
}

void internal_set_names_inplace(TensorImpl* impl, std::optional<DimnameList> names, bool validate_names) {
  TORCH_CHECK(impl->layout() == Layout::Strided,
      "NYI: named tensors only support strided layout");
  TORCH_CHECK(impl->device().is_cpu() || impl->device().is_cuda() || impl->device().is_xpu() || impl->device().is_privateuseone(),
      "NYI: named tensors only support CPU, CUDA, XPU or ", c10::get_privateuse1_backend(), " tensors.");
  if (!names) {
    impl->set_named_tensor_meta(nullptr);
    return;
  }
  if (validate_names) {
    check_names_valid_for(impl, *names);
  }
  // Do this after validation!
  if (std::all_of(names->begin(), names->end(), [](const Dimname& n) { return n.isWildcard(); })) {
    impl->set_named_tensor_meta(nullptr);
    return;
  }
  auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    // Constructor is private
    impl->set_named_tensor_meta(std::make_unique<NamedTensorMeta>(NamedTensorMeta::HasNonWildcard, *names));
  } else {
    meta->set_names(NamedTensorMeta::HasNonWildcard, *names);
  }
}

void internal_set_names_inplace(TensorImpl* impl, std::vector<Dimname>&& names, bool validate_names) {
  if (validate_names) {
    check_names_valid_for(impl, names);
  }
  // Do this after validation!
  if (std::all_of(names.begin(), names.end(), [](const Dimname& n) { return n.isWildcard(); })) {
    impl->set_named_tensor_meta(nullptr);
    return;
  }
  auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    impl->set_named_tensor_meta(std::make_unique<NamedTensorMeta>(NamedTensorMeta::HasNonWildcard, std::move(names)));
  } else {
    meta->set_names(NamedTensorMeta::HasNonWildcard, std::move(names));
  }
}

std::optional<DimnameList> get_opt_names(const TensorImpl* impl) {
  const auto* meta = get_named_tensor_meta(impl);
  if (meta == nullptr) {
    return std::nullopt;
  } else {
    return meta->names();
  }
}

DimnameList get_names(const TensorImpl* impl) {
  auto maybe_names = get_opt_names(impl);
  if (maybe_names) {
    return *maybe_names;
  }
  return default_names(impl->dim());
}

bool has_names(const TensorImpl* impl) {
  return impl->has_named_tensor_meta() && NamesMode::is_enabled();
}

} // namespace impl

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `impl`, `at`

**Classes/Structs**: `a`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/NamedTensor.h`
- `ATen/core/TensorBase.h`


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

- **File Documentation**: `NamedTensor.cpp_docs.md`
- **Keyword Index**: `NamedTensor.cpp_kw.md`
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

- **File Documentation**: `NamedTensor.cpp_docs.md_docs.md`
- **Keyword Index**: `NamedTensor.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
