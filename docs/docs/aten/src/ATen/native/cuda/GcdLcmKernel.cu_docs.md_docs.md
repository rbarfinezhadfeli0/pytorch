# Documentation: `docs/aten/src/ATen/native/cuda/GcdLcmKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/GcdLcmKernel.cu_docs.md`
- **Size**: 4,786 bytes (4.67 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/GcdLcmKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/GcdLcmKernel.cu`
- **Size**: 1,956 bytes (1.91 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/cuda/jit_utils.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

// See note [Jiterator]
constexpr char gcd_name[] = "gcd";
void gcd_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_cuda", [&]() {
      jitted_gpu_kernel</*name=*/gcd_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 2>(iter, gcd_string);
    });
  #else
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "gcd_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        return calc_gcd(a, b);
      });
    });
  #endif // AT_USE_JITERATOR()
}

// See note [Jiterator]
constexpr char lcm_name[] = "lcm";
void lcm_kernel_cuda(TensorIteratorBase& iter) {
  #if AT_USE_JITERATOR()
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "lcm_cuda", [&]() {
      jitted_gpu_kernel</*name=*/lcm_name,
                        /*return_dtype=*/ scalar_t,
                        /*common_dtype=*/ scalar_t,
                        /*arity=*/ 2>(iter, lcm_string);
    });
  #else
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "lcm_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA (scalar_t a, scalar_t b) -> scalar_t {
        scalar_t g = calc_gcd(a, b);
        return (g == 0) ? 0 : ::abs(a / g * b);
      });
    });
  #endif // AT_USE_JITERATOR()
}

REGISTER_DISPATCH(gcd_stub, &gcd_kernel_cuda)
REGISTER_DISPATCH(lcm_stub, &lcm_kernel_cuda)

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

- `ATen/Dispatch.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/cuda/JitLoops.cuh`
- `ATen/native/cuda/Loops.cuh`
- `ATen/native/cuda/Math.cuh`
- `ATen/native/TensorIterator.h`
- `ATen/native/BinaryOps.h`
- `ATen/native/cuda/jit_utils.h`


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

- **File Documentation**: `GcdLcmKernel.cu_docs.md`
- **Keyword Index**: `GcdLcmKernel.cu_kw.md`
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

- **File Documentation**: `GcdLcmKernel.cu_docs.md_docs.md`
- **Keyword Index**: `GcdLcmKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
