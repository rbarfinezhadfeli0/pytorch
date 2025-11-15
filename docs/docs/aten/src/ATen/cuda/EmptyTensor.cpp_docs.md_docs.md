# Documentation: `docs/aten/src/ATen/cuda/EmptyTensor.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cuda/EmptyTensor.cpp_docs.md`
- **Size**: 5,736 bytes (5.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cuda/EmptyTensor.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cuda/EmptyTensor.cpp`
- **Size**: 3,260 bytes (3.18 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/EmptyTensor.h>

namespace at::detail {

TensorBase empty_cuda(
    IntArrayRef size,
    ScalarType dtype,
    std::optional<Device> device_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_cuda());
  const DeviceGuard device_guard(device);
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_generic(
      size, allocator, cuda_dks, dtype, memory_format_opt);
}

TensorBase empty_cuda(
    IntArrayRef size,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_cuda(size, dtype, device_opt, memory_format_opt);
}

TensorBase empty_cuda(
    IntArrayRef size, const TensorOptions &options) {
  return at::detail::empty_cuda(
      size,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt(),
      options.memory_format_opt());
}

TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    ScalarType dtype,
    std::optional<Device> device_opt) {
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  const auto device = device_or_default(device_opt);
  TORCH_INTERNAL_ASSERT(device.is_cuda());
  const DeviceGuard device_guard(device);
  auto* allocator = at::cuda::getCUDADeviceAllocator();
  constexpr c10::DispatchKeySet cuda_dks(c10::DispatchKey::CUDA);
  return at::detail::empty_strided_generic(
      size, stride, allocator, cuda_dks, dtype);
}

TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    std::optional<ScalarType> dtype_opt,
    std::optional<Layout> layout_opt,
    std::optional<Device> device_opt,
    std::optional<bool> pin_memory_opt) {
  TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
#ifndef NDEBUG
  // TODO: remove check for jagged, see https://github.com/pytorch/pytorch/issues/130073
  const auto layout = layout_or_default(layout_opt);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout == Layout::Strided || layout == Layout::Jagged);
#endif

  const auto dtype = dtype_or_default(dtype_opt);
  return at::detail::empty_strided_cuda(size, stride, dtype, device_opt);
}

TensorBase empty_strided_cuda(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions &options) {
  return at::detail::empty_strided_cuda(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}

}  // namespace at::detail

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cuda/EmptyTensor.h`
- `ATen/cuda/CUDAContext.h`
- `ATen/EmptyTensor.h`


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

Files in the same folder (`aten/src/ATen/cuda`):

- [`CublasHandlePool.cpp_docs.md`](./CublasHandlePool.cpp_docs.md)
- [`llvm_basic.cpp_docs.md`](./llvm_basic.cpp_docs.md)
- [`CUDABlas.h_docs.md`](./CUDABlas.h_docs.md)
- [`jiterator.cu_docs.md`](./jiterator.cu_docs.md)
- [`CUDAGraph.h_docs.md`](./CUDAGraph.h_docs.md)
- [`llvm_jit_strings.h_docs.md`](./llvm_jit_strings.h_docs.md)
- [`llvm_complex.cpp_docs.md`](./llvm_complex.cpp_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md`](./CUDAGeneratorImpl.cpp_docs.md)
- [`cub_definitions.cuh_docs.md`](./cub_definitions.cuh_docs.md)
- [`jiterator_impl.h_docs.md`](./jiterator_impl.h_docs.md)


## Cross-References

- **File Documentation**: `EmptyTensor.cpp_docs.md`
- **Keyword Index**: `EmptyTensor.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/cuda`):

- [`PhiloxCudaState.h_docs.md_docs.md`](./PhiloxCudaState.h_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_docs.md_docs.md`](./CUDAGeneratorImpl.cpp_docs.md_docs.md)
- [`Exceptions.cpp_docs.md_docs.md`](./Exceptions.cpp_docs.md_docs.md)
- [`CUDAGeneratorImpl.cpp_kw.md_docs.md`](./CUDAGeneratorImpl.cpp_kw.md_docs.md)
- [`Sleep.h_docs.md_docs.md`](./Sleep.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-2.cu_kw.md_docs.md`](./cub-RadixSortPairs-int64-2.cu_kw.md_docs.md)
- [`CUDASparseDescriptors.h_kw.md_docs.md`](./CUDASparseDescriptors.h_kw.md_docs.md)
- [`jiterator_impl.h_docs.md_docs.md`](./jiterator_impl.h_docs.md_docs.md)
- [`CUDAContext.h_docs.md_docs.md`](./CUDAContext.h_docs.md_docs.md)
- [`cub-RadixSortPairs-int64-4.cu_docs.md_docs.md`](./cub-RadixSortPairs-int64-4.cu_docs.md_docs.md)


## Cross-References

- **File Documentation**: `EmptyTensor.cpp_docs.md_docs.md`
- **Keyword Index**: `EmptyTensor.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
