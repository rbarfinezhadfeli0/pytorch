# Documentation: `docs/aten/src/ATen/native/cuda/ActivationHardsigmoidKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ActivationHardsigmoidKernel.cu_docs.md`
- **Size**: 5,174 bytes (5.05 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ActivationHardsigmoidKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ActivationHardsigmoidKernel.cu`
- **Size**: 2,278 bytes (2.22 KB)
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

void hardsigmoid_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t zero(0.0f);
        const opmath_t one_sixth(1.0f / 6.0f);
        const opmath_t three(3.0f);
        const opmath_t six(6.0f);
        gpu_kernel(
            iter,
            [zero, one_sixth, three, six] GPU_LAMBDA(
                scalar_t self_val) -> scalar_t {
              opmath_t x = static_cast<opmath_t>(self_val);
              return std::min<opmath_t>(std::max<opmath_t>(x + three, zero), six) * one_sixth;
            });
      });
}

void hardsigmoid_backward_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "hardsigmoid_backward_cuda",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        const opmath_t zero(0.0f);
        const opmath_t three(3.0f);
        const opmath_t neg_three(-3.0f);
        const opmath_t one_sixth(1.0f / 6.0f);
        gpu_kernel(
            iter,
            [zero, three, neg_three, one_sixth] GPU_LAMBDA(
                scalar_t grad_val_, scalar_t self_val_) -> scalar_t {
              opmath_t grad_val = static_cast<opmath_t>(grad_val_);
              opmath_t self_val = static_cast<opmath_t>(self_val_);
              return (self_val > neg_three && self_val < three)
                  ? grad_val * one_sixth
                  : zero;
            });
      });
}

} // namespace

REGISTER_DISPATCH(hardsigmoid_stub, &hardsigmoid_kernel)
REGISTER_DISPATCH(hardsigmoid_backward_stub, &hardsigmoid_backward_kernel)

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

- **File Documentation**: `ActivationHardsigmoidKernel.cu_docs.md`
- **Keyword Index**: `ActivationHardsigmoidKernel.cu_kw.md`
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

- **File Documentation**: `ActivationHardsigmoidKernel.cu_docs.md_docs.md`
- **Keyword Index**: `ActivationHardsigmoidKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
