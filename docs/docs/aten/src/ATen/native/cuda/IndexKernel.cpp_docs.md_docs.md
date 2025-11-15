# Documentation: `docs/aten/src/ATen/native/cuda/IndexKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/IndexKernel.cpp_docs.md`
- **Size**: 5,873 bytes (5.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/IndexKernel.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/IndexKernel.cpp`
- **Size**: 2,955 bytes (2.89 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/IndexKernel.h>
#include <ATen/native/TensorAdvancedIndexing.h>  // For at::native::index_out
#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/CUDAFunctions.h>
#else
#include <ATen/ops/index_cuda_dispatch.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/masked_scatter_native.h>
#include <ATen/ops/masked_select_native.h>
#endif


namespace at::native {

static Tensor & masked_select_out_cuda_impl(Tensor & result, const Tensor & self, const Tensor & mask) {
  NoNamesGuard guard;

  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool,
              "masked_select: expected BoolTensor for mask");
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
              "masked_select(): self and result must have the same scalar type");

  auto mask_temp = (mask.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(mask.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(mask);
  auto self_temp = (self.dim() == 0)
    ? c10::MaybeOwned<Tensor>::owned(self.unsqueeze(0))
    : c10::MaybeOwned<Tensor>::borrowed(self);

  // Cannot reassign to mask_temp and self_temp here! if they are
  // owning and expand_outplace returns a borrow, the returned borrow
  // would dangle.
  auto [mask_expanded, self_expanded] = expand_outplace(*mask_temp, *self_temp);
  at::cuda::index_out(
      result, *self_expanded,
      c10::List<std::optional<at::Tensor>>({*std::move(mask_expanded)}));

  return result;
}

Tensor masked_select_cuda(const Tensor & self, const Tensor & mask) {
  namedinference::compute_broadcast_outnames(self, mask);
  Tensor result = at::empty({0}, self.options());
  return masked_select_out_cuda_impl(result, self, mask);
}

Tensor & masked_select_out_cuda(const Tensor & self, const Tensor & mask, Tensor & result) {
  namedinference::compute_broadcast_outnames(self, mask);
  return masked_select_out_cuda_impl(result, self, mask);
}

Tensor & masked_scatter__cuda(Tensor& self, const Tensor& mask, const Tensor& source) {
  at::assert_no_internal_overlap(self);
  TORCH_CHECK(
      self.scalar_type() == source.scalar_type(),
      "masked_scatter_: expected self and source to have same dtypes but got ",
      self.scalar_type(),
      " and ",
      source.scalar_type());
  TORCH_CHECK(mask.dtype() == ScalarType::Bool, "masked_scatter_ only supports boolean masks, "
     "but got mask with dtype ", mask.dtype());

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_scatter_");

  if (self.numel() == 0) {
    return self;
  }

  auto maskPrefixSum = at::empty(self.sizes(), mask.options().dtype(kLong));
  launch_masked_scatter_kernel(self, *b_mask, maskPrefixSum, source);

  return self;
}

}  // namespace at::native

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

- `ATen/native/cuda/IndexKernel.h`
- `ATen/native/TensorAdvancedIndexing.h`
- `ATen/core/Tensor.h`
- `ATen/core/List.h`
- `ATen/ExpandUtils.h`
- `ATen/MemoryOverlap.h`
- `ATen/NamedTensorUtils.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/CUDAFunctions.h`
- `ATen/ops/index_cuda_dispatch.h`
- `ATen/ops/empty.h`
- `ATen/ops/masked_scatter_native.h`
- `ATen/ops/masked_select_native.h`


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

- **File Documentation**: `IndexKernel.cpp_docs.md`
- **Keyword Index**: `IndexKernel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/cuda`):

- [`DeviceSqrt.cuh_kw.md_docs.md`](./DeviceSqrt.cuh_kw.md_docs.md)
- [`UnaryGeometricAsinKernel.cu_kw.md_docs.md`](./UnaryGeometricAsinKernel.cu_kw.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`fused_adamw_impl.cu_docs.md_docs.md`](./fused_adamw_impl.cu_docs.md_docs.md)
- [`TensorTopK.h_kw.md_docs.md`](./TensorTopK.h_kw.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`FusedSgdKernel.cu_docs.md_docs.md`](./FusedSgdKernel.cu_docs.md_docs.md)
- [`Distributions.cu_kw.md_docs.md`](./Distributions.cu_kw.md_docs.md)
- [`block_reduce.cuh_docs.md_docs.md`](./block_reduce.cuh_docs.md_docs.md)
- [`fused_adagrad_impl.cuh_kw.md_docs.md`](./fused_adagrad_impl.cuh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `IndexKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `IndexKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
