# Documentation: `docs/torch/csrc/lazy/core/tensor_util.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/core/tensor_util.h_docs.md`
- **Size**: 5,073 bytes (4.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/core/tensor_util.h`

## File Metadata

- **Path**: `torch/csrc/lazy/core/tensor_util.h`
- **Size**: 2,536 bytes (2.48 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/shape.h>

#include <ATen/FunctionalTensorWrapper.h>

#include <string>
#include <vector>

namespace torch::lazy {

TORCH_API std::vector<int64_t> ComputeArrayStrides(
    c10::ArrayRef<int64_t> sizes);

TORCH_API std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<BackendDataPtr> data_handles,
    at::ScalarType dest_element_type);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
TORCH_API BackendDataPtr
TensorToDataHandle(const at::Tensor& tensor, const BackendDevice& device);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
TORCH_API std::vector<BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<BackendDevice>& devices);

// Makes a deep copy of an ATEN tensor.
inline at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

// Same as above, with an additional cast.
inline at::Tensor CopyTensor(
    const at::Tensor& ref,
    at::ScalarType dest_type,
    bool copy = true) {
  return ref.to(ref.options().dtype(dest_type), /*non_blocking=*/false, copy);
}

template <typename T, typename S>
T OptionalOr(const std::optional<S>& value, T defval) {
  return value ? static_cast<T>(*value) : defval;
}

// Unwraps tensor to target dtype if it's a wrapped number.
inline at::Tensor UnwrapNumber(const at::Tensor& tensor, at::ScalarType dtype) {
  return tensor.unsafeGetTensorImpl()->is_wrapped_number() ? tensor.to(dtype)
                                                           : tensor;
}

template <typename T>
at::Scalar MakeIntScalar(T value) {
  return at::Scalar(static_cast<int64_t>(value));
}

// Routing values to device data maximizes the changes for compilation cache
// hits, but it can prevent the compiler to perform optimizations. So tensor
// values which are within a given set, are routed to constant scalars if this
// API returns true.
TORCH_API bool IsSpecialScalar(const at::Scalar& value);

// Note: returns a reference instead of a fresh tensor to avoid refcount bumps.
inline const at::Tensor& maybe_unwrap_functional(const at::Tensor& tensor) {
  if (at::functionalization::impl::isFunctionalTensor(tensor)) {
    return at::functionalization::impl::unsafeGetFunctionalWrapper(tensor)
        ->value();
  } else {
    return tensor;
  }
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/backend/backend_interface.h`
- `torch/csrc/lazy/core/shape.h`
- `ATen/FunctionalTensorWrapper.h`
- `string`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/csrc/lazy/core`):

- [`hash.cpp_docs.md`](./hash.cpp_docs.md)
- [`shape_inference.cpp_docs.md`](./shape_inference.cpp_docs.md)
- [`tensor_impl.h_docs.md`](./tensor_impl.h_docs.md)
- [`helpers.h_docs.md`](./helpers.h_docs.md)
- [`tensor_impl.cpp_docs.md`](./tensor_impl.cpp_docs.md)
- [`ir_metadata.cpp_docs.md`](./ir_metadata.cpp_docs.md)
- [`ir_metadata.h_docs.md`](./ir_metadata.h_docs.md)
- [`trie.cpp_docs.md`](./trie.cpp_docs.md)
- [`cache.h_docs.md`](./cache.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)


## Cross-References

- **File Documentation**: `tensor_util.h_docs.md`
- **Keyword Index**: `tensor_util.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/csrc/lazy/core`):

- [`helpers.cpp_docs.md_docs.md`](./helpers.cpp_docs.md_docs.md)
- [`tensor_util.h_kw.md_docs.md`](./tensor_util.h_kw.md_docs.md)
- [`permutation_util.h_kw.md_docs.md`](./permutation_util.h_kw.md_docs.md)
- [`ir_util.cpp_kw.md_docs.md`](./ir_util.cpp_kw.md_docs.md)
- [`shape_inference.h_kw.md_docs.md`](./shape_inference.h_kw.md_docs.md)
- [`ir_builder.h_docs.md_docs.md`](./ir_builder.h_docs.md_docs.md)
- [`shape_inference.cpp_kw.md_docs.md`](./shape_inference.cpp_kw.md_docs.md)
- [`hash.h_kw.md_docs.md`](./hash.h_kw.md_docs.md)
- [`multi_wait.cpp_kw.md_docs.md`](./multi_wait.cpp_kw.md_docs.md)
- [`lazy_graph_executor.cpp_docs.md_docs.md`](./lazy_graph_executor.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `tensor_util.h_docs.md_docs.md`
- **Keyword Index**: `tensor_util.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
