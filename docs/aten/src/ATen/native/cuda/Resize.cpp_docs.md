# Documentation: `aten/src/ATen/native/cuda/Resize.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Resize.cpp`
- **Size**: 2,450 bytes (2.39 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Resize.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PeerToPeerAccess.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/resize_native.h>
#endif

namespace at::native {

void resize_bytes_cuda(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();

  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  c10::cuda::CUDAGuard guard(device.index());
  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr()) {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);

    C10_CUDA_CHECK(
        cudaMemcpyAsync(
            data.get(),
            storage->data(),
            std::min(storage->nbytes(), size_bytes),
            cudaMemcpyDeviceToDevice,
            c10::cuda::getCurrentCUDAStream()));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

const Tensor& resize_cuda_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  auto old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  resize_impl_cuda_(self_, size, /*stride=*/std::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_resize_deterministic_(self, static_cast<int64_t>(old_storage_nbytes));
  }
  return self;
}
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/cuda/Resize.h`
- `ATen/core/Tensor.h`
- `ATen/cuda/CUDAContext.h`
- `ATen/cuda/PeerToPeerAccess.h`
- `ATen/native/ResizeCommon.h`
- `c10/cuda/CUDAGuard.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/resize_native.h`


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

Files in the same folder (`aten/src/ATen/native/cuda`):

- [`LogcumsumexpKernel.cu_docs.md`](./LogcumsumexpKernel.cu_docs.md)
- [`WeightNorm.cu_docs.md`](./WeightNorm.cu_docs.md)
- [`SparseBinaryOpIntersectionKernel.cu_docs.md`](./SparseBinaryOpIntersectionKernel.cu_docs.md)
- [`jit_utils.cpp_docs.md`](./jit_utils.cpp_docs.md)
- [`ReduceNormKernel.cu_docs.md`](./ReduceNormKernel.cu_docs.md)
- [`BinaryMiscOpsKernels.cu_docs.md`](./BinaryMiscOpsKernels.cu_docs.md)
- [`RowwiseScaledMM.h_docs.md`](./RowwiseScaledMM.h_docs.md)
- [`fused_adamw_amsgrad_impl.cuh_docs.md`](./fused_adamw_amsgrad_impl.cuh_docs.md)
- [`Col2Im.cu_docs.md`](./Col2Im.cu_docs.md)
- [`DistributionRandomKernel.cu_docs.md`](./DistributionRandomKernel.cu_docs.md)


## Cross-References

- **File Documentation**: `Resize.cpp_docs.md`
- **Keyword Index**: `Resize.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
