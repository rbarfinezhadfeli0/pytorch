# Documentation: `docs/aten/src/ATen/native/cuda/CrossKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/CrossKernel.cu_docs.md`
- **Size**: 5,965 bytes (5.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/CrossKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/CrossKernel.cu`
- **Size**: 3,310 bytes (3.23 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Cross.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

namespace at::native {

template <typename T, typename OffsetCalc, typename StrideType>
__global__ void cross_kernel(
    int numel, T* out, const T* x1, const T* x2, OffsetCalc offset_calculator,
    StrideType ostride, StrideType x1stride, StrideType x2stride) {
  CUDA_KERNEL_LOOP(i, numel) {
    const auto offsets = offset_calculator.get(i);
    auto* out_row = out + offsets[0];
    const auto* x1_row = x1 + offsets[1];
    const auto* x2_row = x2 + offsets[2];

    const T val0 = (x1_row[1 * x1stride] * x2_row[2 * x2stride] -
                    x1_row[2 * x1stride] * x2_row[1 * x2stride]);

    const T val1 = (x1_row[2 * x1stride] * x2_row[0 * x2stride] -
                    x1_row[0 * x1stride] * x2_row[2 * x2stride]);

    const T val2 = (x1_row[0 * x1stride] * x2_row[1 * x2stride] -
                    x1_row[1 * x1stride] * x2_row[0 * x2stride]);


    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

void launch_cross_kernel(const TensorIteratorBase& iter, int64_t ostride,
                         int64_t x1stride, int64_t x2stride) {
  const auto N = iter.numel();
  auto offset_calculator = make_element_offset_calculator<3>(iter);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(N > 0 && N <= std::numeric_limits<int32_t>::max());
  int64_t grid = (N + num_threads() - 1) / num_threads();
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kHalf, kBFloat16, iter.common_dtype(), "cross_cuda", [&] {
    auto out = static_cast<scalar_t*>(iter.data_ptr(0));
    auto x1 = static_cast<const scalar_t*>(iter.data_ptr(1));
    auto x2 = static_cast<const scalar_t*>(iter.data_ptr(2));
    constexpr int64_t int_max = std::numeric_limits<int>::max();
    if (ostride * 2 > int_max || x1stride * 2 > int_max || x2stride * 2 > int_max) {
      cross_kernel<<<grid, num_threads(), 0, stream>>>(
          N, out, x1, x2, offset_calculator, ostride, x1stride, x2stride);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      cross_kernel<<<grid, num_threads(), 0, stream>>>(
          N, out, x1, x2, offset_calculator,
          static_cast<int>(ostride),
          static_cast<int>(x1stride),
          static_cast<int>(x2stride));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  });
}

void cross_impl(const Tensor& result, const Tensor& x1, const Tensor& x2, int64_t dim) {
  const int64_t ostride = result.stride(dim);
  const int64_t x1stride = x1.stride(dim);
  const int64_t x2stride = x2.stride(dim);

  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_const_input(x1)
      .add_const_input(x2)
      .resize_outputs(false)
      .declare_static_shape(result.sizes(), /*squash_dims=*/dim)
      .build();

  if (iter.numel() == 0) {
    return;
  }

  if (iter.can_use_32bit_indexing()) {
    launch_cross_kernel(iter, ostride, x1stride, x2stride);
  } else {
    for (auto&& sub_iter: iter.with_32bit_indexing()) {
      launch_cross_kernel(sub_iter, ostride, x1stride, x2stride);
    }
  }
}

REGISTER_DISPATCH(cross_stub, &cross_impl)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

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

- `ATen/native/Cross.h`
- `ATen/cuda/detail/KernelUtils.h`
- `ATen/native/cuda/Loops.cuh`
- `ATen/Dispatch.h`
- `ATen/core/Tensor.h`


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

- **File Documentation**: `CrossKernel.cu_docs.md`
- **Keyword Index**: `CrossKernel.cu_kw.md`
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

- **File Documentation**: `CrossKernel.cu_docs.md_docs.md`
- **Keyword Index**: `CrossKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
