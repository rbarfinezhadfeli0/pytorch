# Documentation: `docs/aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu_docs.md`
- **Size**: 4,840 bytes (4.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu`
- **Size**: 1,893 bytes (1.85 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>

#include <type_traits>

namespace at::native {
namespace binary_internal {

void div_trunc_kernel_cuda(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_cuda", [&]() {
      gpu_kernel_with_scalars(
          iter,
          [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t { return a / b; });
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_cuda", [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto inv_b = accscalar_t(1.0) / iter.scalar_value<accscalar_t>(2);
          iter.remove_operand(2);
          gpu_kernel(iter, [inv_b] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return std::trunc(a * inv_b);
          });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_cuda", [&]() {
          gpu_kernel_with_scalars(
              iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
                return std::trunc(a / b);
              });
        });
  }
}
} // namespace binary_internal

REGISTER_DISPATCH(div_trunc_stub, &binary_internal::div_trunc_kernel_cuda)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `binary_internal`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/native/BinaryOps.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/TensorIterator.h`
- `c10/cuda/CUDAGuard.h`
- `c10/cuda/CUDAMathCompat.h`
- `c10/util/TypeSafeSignMath.h`
- `ATen/native/cuda/JitLoops.cuh`
- `ATen/native/cuda/Loops.cuh`
- `type_traits`


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

- **File Documentation**: `BinaryDivTruncKernel.cu_docs.md`
- **Keyword Index**: `BinaryDivTruncKernel.cu_kw.md`
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

- **File Documentation**: `BinaryDivTruncKernel.cu_docs.md_docs.md`
- **Keyword Index**: `BinaryDivTruncKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
