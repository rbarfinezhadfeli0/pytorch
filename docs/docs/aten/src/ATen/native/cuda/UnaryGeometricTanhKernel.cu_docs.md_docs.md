# Documentation: `docs/aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu_docs.md`
- **Size**: 4,678 bytes (4.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/UnaryGeometricTanhKernel.cu`
- **Size**: 1,775 bytes (1.73 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/cuda/JitLoops.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Math.cuh>
#include <limits>

namespace at::native {

#if 0 && AT_USE_JITERATOR()
constexpr char tanh_name[] = "tanh_impl";
#endif

void tanh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    // Disabled due to accuracy issues
#if 0 && AT_USE_JITERATOR()
    static const auto tanh_string = jiterator_stringify(
        template <typename T> T tanh_impl(T a) { return std::tanh(a); });
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "tanh_name", [&]() {
          jitted_gpu_kernel<
              /*name=*/tanh_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, tanh_string);
        });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "tanh_name", [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return ::tanh(static_cast<opmath_t>(a));
          });
        });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "tanh_cuda",
        [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return ::tanh(a);
          });
        });
  }
}

REGISTER_DISPATCH(tanh_stub, &tanh_kernel_cuda)

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

- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/OpMathType.h`
- `ATen/native/DispatchStub.h`
- `ATen/native/TensorIterator.h`
- `ATen/native/UnaryOps.h`
- `ATen/native/cuda/JitLoops.cuh`
- `ATen/native/cuda/Loops.cuh`
- `ATen/native/cuda/Math.cuh`
- `limits`


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

- **File Documentation**: `UnaryGeometricTanhKernel.cu_docs.md`
- **Keyword Index**: `UnaryGeometricTanhKernel.cu_kw.md`
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

- **File Documentation**: `UnaryGeometricTanhKernel.cu_docs.md_docs.md`
- **Keyword Index**: `UnaryGeometricTanhKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
