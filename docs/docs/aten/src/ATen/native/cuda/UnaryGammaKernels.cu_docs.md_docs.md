# Documentation: `docs/aten/src/ATen/native/cuda/UnaryGammaKernels.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/UnaryGammaKernels.cu_docs.md`
- **Size**: 7,188 bytes (7.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/UnaryGammaKernels.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/UnaryGammaKernels.cu`
- **Size**: 4,312 bytes (4.21 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/Math.h>

namespace at::native {

#if AT_USE_JITERATOR()
constexpr char digamma_name[] = "digamma";
#endif // AT_USE_JITERATOR()
// See note [Jiterator]
void digamma_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "digamma_cuda", [&]() {
        jitted_gpu_kernel</*name=*/digamma_name,
                          /*return_dtype=*/ scalar_t,
                          /*common_dtype=*/ scalar_t,
                          /*arity=*/ 1>(iter, digamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "digamma_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return calc_digamma(a);
        });
    });
  #endif // AT_USE_JITERATOR()
}

// See note [Jiterator]
constexpr char trigamma_name[] = "trigamma";
void trigamma_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "trigamma_cuda", [&]() {
        jitted_gpu_kernel</*name=*/trigamma_name,
                          /*return_dtype=*/ scalar_t,
                          /*common_dtype=*/ scalar_t,
                          /*arity=*/ 1>(iter, trigamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "trigamma_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return calc_trigamma(a);
        });
    });
  #endif // AT_USE_JITERATOR()
}

constexpr char polygamma_name[] = "polygamma";
void polygamma_kernel_cuda(TensorIteratorBase& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel_cuda(iter);
  } else if (n == 1) {
    trigamma_kernel_cuda(iter);
  } else {
#if AT_USE_JITERATOR()
    // TODO : `unary_jitted_gpu_kernel` for cleaner UX.
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
        iter.common_dtype(), "polygamma_cuda", [&]() {
          jitted_gpu_kernel<
              /*name=*/polygamma_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(
              iter,
              polygamma_string,
              /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
              /*scalar_val=*/0,
              /*extra_args=*/std::make_tuple(n));
        });
#else
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
        iter.common_dtype(), "polygamma_cuda", [&]() {
          gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return calc_polygamma<scalar_t, /*is_cuda=*/true>(a, static_cast<int>(n));
          });
        });
#endif // AT_USE_JITERATOR()
  }
}

constexpr char lgamma_name[] = "lgamma_kernel";
void lgamma_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "lgamma_cuda", [&]() {
        jitted_gpu_kernel</*name=*/lgamma_name,
                          /*return_dtype=*/ scalar_t,
                          /*common_dtype=*/ scalar_t,
                          /*arity=*/ 1>(iter, lgamma_string);
    });
  #else
    AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(), "lgamma_cuda", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ::lgamma(a);
        });
    });
  #endif
}

REGISTER_DISPATCH(digamma_stub, &digamma_kernel_cuda)
REGISTER_DISPATCH(polygamma_stub, &polygamma_kernel_cuda)
REGISTER_DISPATCH(lgamma_stub, &lgamma_kernel_cuda)

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

- `limits`
- `ATen/native/UnaryOps.h`
- `ATen/native/cuda/JitLoops.cuh`
- `ATen/native/cuda/Loops.cuh`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/cuda/Math.cuh`
- `ATen/native/Math.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `UnaryGammaKernels.cu_docs.md`
- **Keyword Index**: `UnaryGammaKernels.cu_kw.md`
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

- **File Documentation**: `UnaryGammaKernels.cu_docs.md_docs.md`
- **Keyword Index**: `UnaryGammaKernels.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
