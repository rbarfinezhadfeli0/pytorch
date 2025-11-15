# Documentation: `docs/aten/src/ATen/native/cuda/ReduceNormKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ReduceNormKernel.cu_docs.md`
- **Size**: 5,120 bytes (5.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ReduceNormKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ReduceNormKernel.cu`
- **Size**: 2,418 bytes (2.36 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/LinearAlgebra.h>
#include <c10/core/Scalar.h>

namespace at::native {

// This reduction accumulates results as the type `acc_t`. By default, when
// `scalar_t` is complex, `acc_t` is the downgraded real number type.
// Otherwise, `acc_t` and `scalar_t` are the same type.
template <typename scalar_t, typename acc_t=typename scalar_value_type<scalar_t>::type, typename out_t=typename scalar_value_type<scalar_t>::type>
void norm_kernel_cuda_impl(TensorIterator& iter, double p) {
  if (p == static_cast<double>(0)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormZeroOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(1)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOneOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(2)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormTwoOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMaxOps<scalar_t, acc_t, out_t>(), 0);
  } else if (p == static_cast<double>(-INFINITY)) {
    gpu_reduce_kernel<scalar_t, out_t>(iter, AbsMinOps<scalar_t, acc_t, out_t>(), std::numeric_limits<acc_t>::infinity());
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, NormOps<scalar_t, acc_t, out_t>{acc_t(p)}, 0);
  }
}

void norm_launch_kernel(TensorIterator& iter, double ord) {
  if (iter.dtype(0) == kHalf) {
    return norm_kernel_cuda_impl<at::Half, float>(iter, ord);
  } else if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::Half, float, float>(iter, ord);
  }
  else if(iter.dtype(0) == kBFloat16) {
    return norm_kernel_cuda_impl<at::BFloat16, float>(iter, ord);
  } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    return norm_kernel_cuda_impl<at::BFloat16, float, float>(iter, ord);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.input_dtype(), "norm_cuda", [&] {
    norm_kernel_cuda_impl<scalar_t>(iter, ord);
  });
}

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
- `ATen/TensorIterator.h`
- `ATen/native/cuda/Reduce.cuh`
- `ATen/native/DispatchStub.h`
- `ATen/native/SharedReduceOps.h`
- `ATen/native/ReduceOps.h`
- `ATen/native/LinearAlgebra.h`
- `c10/core/Scalar.h`


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
- [`BinaryMiscOpsKernels.cu_docs.md`](./BinaryMiscOpsKernels.cu_docs.md)
- [`RowwiseScaledMM.h_docs.md`](./RowwiseScaledMM.h_docs.md)
- [`fused_adamw_amsgrad_impl.cuh_docs.md`](./fused_adamw_amsgrad_impl.cuh_docs.md)
- [`Col2Im.cu_docs.md`](./Col2Im.cu_docs.md)
- [`DistributionRandomKernel.cu_docs.md`](./DistributionRandomKernel.cu_docs.md)


## Cross-References

- **File Documentation**: `ReduceNormKernel.cu_docs.md`
- **Keyword Index**: `ReduceNormKernel.cu_kw.md`
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

- **File Documentation**: `ReduceNormKernel.cu_docs.md_docs.md`
- **Keyword Index**: `ReduceNormKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
