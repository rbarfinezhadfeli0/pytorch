# Documentation: `docs/aten/src/ATen/native/cuda/ActivationGluKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ActivationGluKernel.cu_docs.md`
- **Size**: 7,445 bytes (7.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ActivationGluKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ActivationGluKernel.cu`
- **Size**: 4,602 bytes (4.49 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/Loops.cuh>

namespace at::native {

// -----------------------------------
// glu forward
// -----------------------------------
void glu_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "glu_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a_, scalar_t b_) -> scalar_t {
          const opmath_t a = a_;
          const opmath_t b = b_;
          const opmath_t one = opmath_t(1);
          const opmath_t sigmoid = one / (one + std::exp(-b));
          return a * sigmoid;
        });
      });
}

// -----------------------------------
// glu forward ad
// -----------------------------------
void glu_jvp_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.dtype(), "glu_cuda", [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        gpu_kernel(
            iter,
            [] GPU_LAMBDA(
                scalar_t res_, scalar_t b_, scalar_t da_, scalar_t db_)
                -> scalar_t {
              const opmath_t res = res_;
              const opmath_t b = b_;
              const opmath_t da = da_;
              const opmath_t db = db_;
              const opmath_t one = opmath_t(1);

              const opmath_t sig_b = one / (one + std::exp(-b));
              return (da * sig_b + res * (db - sig_b * db));
            });
      });
}

// -----------------------------------
// glu backward
// -----------------------------------

// Byte offsets don't require multiplication by sizeof(T), so are slightly
// cheaper. For fixed offsets, this removes all penalty from 64-bit indexing.
template <typename T>
__device__ T* byte_offset(T* ptr, int64_t offset) {
  using byte_ptr_t = typename std::
      conditional_t<std::is_const_v<T>, const char*, char*>;
  return reinterpret_cast<T*>(reinterpret_cast<byte_ptr_t>(ptr) + offset);
}

template <typename scalar_t, typename OffsetCalc>
__global__ void glu_backward_kernel(
    int numel,
    scalar_t* gI,
    const scalar_t* I,
    const scalar_t* gO,
    OffsetCalc offset_calculator,
    int64_t gI_byte_offset,
    int64_t I_byte_offset) {
  using opmath_t = at::opmath_type<scalar_t>;

  const uint32_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear_index >= numel) {
    return;
  }
  const auto offsets = offset_calculator.get(linear_index);

  // We explicitly iterate over the first half of the input tensor, and
  // gI_byte_offset and I_byte_offset are the offsets to access the
  // corresponding index in the second half of the tensor.
  const opmath_t a = I[offsets[1]];
  const opmath_t b = *byte_offset(I + offsets[1], I_byte_offset);
  const opmath_t gO_val = gO[offsets[2]];

  const auto one = opmath_t(1);
  const opmath_t sigmoid = one / (one + std::exp(-b));

  auto* gA = gI + offsets[0];
  *gA = sigmoid * gO_val;

  auto* gB = byte_offset(gA, gI_byte_offset);
  *gB = (one - sigmoid) * sigmoid * gO_val * a;
}

void launch_glu_backward_kernel(
    const TensorIteratorBase& iter,
    int64_t gI_stride,
    int64_t I_stride) {
  const auto N = iter.numel();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      N > 0 && N <= std::numeric_limits<int32_t>::max());
  const auto offset_calculator = make_element_offset_calculator<3>(iter);
  constexpr int64_t block_size = 256;
  const int64_t grid = (N + block_size - 1) / block_size;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, iter.common_dtype(), "glu_backward_cuda", [&] {
        auto gI = static_cast<scalar_t*>(iter.data_ptr(0));
        auto I = static_cast<const scalar_t*>(iter.data_ptr(1));
        auto gO = static_cast<const scalar_t*>(iter.data_ptr(2));
        glu_backward_kernel<<<grid, block_size, 0, stream>>>(
            N,
            gI,
            I,
            gO,
            offset_calculator,
            gI_stride * sizeof(scalar_t),
            I_stride * sizeof(scalar_t));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

REGISTER_DISPATCH(glu_stub, &glu_kernel)
REGISTER_DISPATCH(glu_jvp_stub, &glu_jvp_kernel)

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

- `ATen/native/Activation.h`
- `cmath`
- `thrust/tuple.h`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/core/TensorBase.h`
- `c10/core/Scalar.h`
- `c10/cuda/CUDAMathCompat.h`
- `ATen/cuda/ApplyGridUtils.cuh`
- `ATen/cuda/detail/OffsetCalculator.cuh`
- `ATen/native/cuda/Loops.cuh`


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

- **File Documentation**: `ActivationGluKernel.cu_docs.md`
- **Keyword Index**: `ActivationGluKernel.cu_kw.md`
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

- **File Documentation**: `ActivationGluKernel.cu_docs.md_docs.md`
- **Keyword Index**: `ActivationGluKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
