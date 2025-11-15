# Documentation: `aten/src/ATen/native/cuda/TensorTopK.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/TensorTopK.cpp`
- **Size**: 3,110 bytes (3.04 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/TensorTopK.h>

#include <ATen/core/Tensor.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/cuda/Sort.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CUDAFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/sort_cuda_dispatch.h>
#include <ATen/ops/topk_native.h>
#endif

namespace at::native {

void topk_out_with_sort(
  const Tensor& self,
  int64_t k, int64_t dim, bool largest,
  const Tensor& values,
  const Tensor& indices
) {
  auto [sorted_values, sorted_indices] = at::cuda::sort(self, /* stable= */false, dim, largest);
  values.copy_(sorted_values.narrow(dim, 0, k));
  indices.copy_(sorted_indices.narrow(dim, 0, k));
}

bool should_use_sort(const Tensor& self, int64_t dim) {
#if defined(USE_ROCM)
  if (self.dtype() == kBool) return false; // Bool sort not supported in ROCm: https://github.com/pytorch/pytorch/issues/139972
  return (self.numel() >= 10000 && self.numel() == self.size(dim)); // based on the experiments in https://github.com/pytorch/pytorch/pull/146387
#else
  return false;
#endif
}

TORCH_IMPL_FUNC(topk_out_cuda)
  (const Tensor& self,
   int64_t k, int64_t dim, bool largest, bool sorted,
   const Tensor& values,
   const Tensor& indices) {
  TensorArg topK_arg{values, "topK", 1}, indices_arg{indices, "indices", 2}, input_arg{self, "self", 3};
  checkAllSameGPU(__func__, {topK_arg, indices_arg, input_arg});

  dim = at::maybe_wrap_dim(dim, self);

  if (should_use_sort(self, dim)) {
    topk_out_with_sort(self, k, dim, largest, values, indices);
    return;
  }

  // If k is 0 the result is an empty tensor, so we don't need to launch a kernel.
  if (k == 0) {
    return;
  }

  launch_gather_topk_kernel(self, k, dim, largest, values, indices);

  // Sort the results if the user wants them sorted, since our
  // selection routine does not ensure sorting
  if (sorted && values.numel() > 1) {
    if (should_use_small_sort(values, dim)) {
      // This avoids any memory allocations and performs all sorting
      // work inplace along the slice

      sortKeyValueInplace(values, indices, dim, largest);
    } else {
      // Depend upon the backup sort that returns indices, which we
      // can use in conjunction with gather to produce the original
      // indices.
      // This is not the most efficient implementation, especially since
      // there are memory allocations performed here. If the user desires
      // greater performance, they should torch.gather() the results
      // themselves using the reported indices, providing previously
      // allocated tensors to receive the results.

      Tensor sortedIndices = at::empty_like(indices);
      Tensor sortedValues = at::empty_like(values);
      at::cuda::sort_outf(values, /* stable= */ false, dim, largest, sortedValues, sortedIndices);
      indices.copy_(indices.gather(dim, sortedIndices));
      values.copy_(sortedValues);
    }
  }
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

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

- `ATen/native/cuda/TensorTopK.h`
- `ATen/core/Tensor.h`
- `ATen/TensorMeta.h`
- `ATen/TensorUtils.h`
- `ATen/WrapDimUtils.h`
- `ATen/native/cuda/Sort.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/CUDAFunctions.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/sort_cuda_dispatch.h`
- `ATen/ops/topk_native.h`


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

- **File Documentation**: `TensorTopK.cpp_docs.md`
- **Keyword Index**: `TensorTopK.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
