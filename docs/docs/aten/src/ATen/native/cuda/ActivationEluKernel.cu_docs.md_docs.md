# Documentation: `docs/aten/src/ATen/native/cuda/ActivationEluKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ActivationEluKernel.cu_docs.md`
- **Size**: 5,454 bytes (5.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ActivationEluKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ActivationEluKernel.cu`
- **Size**: 2,590 bytes (2.53 KB)
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
namespace {

void elu_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "elu_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negcoef = alpha.to<opmath_t>() * scale.to<opmath_t>();
        auto poscoef = scale.to<opmath_t>();
        auto negiptcoef = input_scale.to<opmath_t>();
        gpu_kernel(
            iter,
            [negcoef, poscoef, negiptcoef] GPU_LAMBDA(scalar_t a) -> scalar_t {
              opmath_t aop = static_cast<opmath_t>(a);
              return aop > 0 ? aop * poscoef
                             : std::expm1(aop * negiptcoef) * negcoef;
            });
      });
}

void elu_backward_kernel(
    TensorIteratorBase& iter,
    const Scalar& alpha,
    const Scalar& scale,
    const Scalar& input_scale,
    bool is_result) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "elu_backward_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        auto negcoef = alpha.to<opmath_t>() * scale.to<opmath_t>();
        auto poscoef = scale.to<opmath_t>();
        auto negiptcoef = input_scale.to<opmath_t>();
        gpu_kernel(
            iter,
            [negcoef, poscoef, negiptcoef, is_result] GPU_LAMBDA(
                scalar_t a, scalar_t b) -> scalar_t {
              opmath_t aop = static_cast<opmath_t>(a);
              opmath_t bop = static_cast<opmath_t>(b);

              if (is_result) {
                return bop <= 0 ? aop * negiptcoef * (bop + negcoef)
                                : aop * poscoef;
              } else {
                return bop <= 0
                    ? aop * negiptcoef * negcoef * std::exp(bop * negiptcoef)
                    : aop * poscoef;
              }
            });
      });
}
} // namespace

REGISTER_DISPATCH(elu_stub, &elu_kernel)
REGISTER_DISPATCH(elu_backward_stub, &elu_backward_kernel)

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `REGISTER_DISPATCH`, `at`


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

- **File Documentation**: `ActivationEluKernel.cu_docs.md`
- **Keyword Index**: `ActivationEluKernel.cu_kw.md`
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

- **File Documentation**: `ActivationEluKernel.cu_docs.md_docs.md`
- **Keyword Index**: `ActivationEluKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
