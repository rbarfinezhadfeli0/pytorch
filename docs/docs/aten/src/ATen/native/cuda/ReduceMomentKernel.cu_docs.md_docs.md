# Documentation: `docs/aten/src/ATen/native/cuda/ReduceMomentKernel.cu_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ReduceMomentKernel.cu_docs.md`
- **Size**: 6,027 bytes (5.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ReduceMomentKernel.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ReduceMomentKernel.cu`
- **Size**: 3,247 bytes (3.17 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/ReduceOps.h>

namespace at::native {

template <typename scalar_t, typename out_t=scalar_t>
void std_var_kernel_impl(TensorIterator& iter, double correction, bool take_sqrt) {
  // reducing unrolling factor to 2 for welford kernel
  // This is necessary to lower register usage that leads to register spills.
  using accscalar_t = at::acc_type<scalar_t, true>;
  using ops_t = WelfordOps<scalar_t, accscalar_t, int32_t, thrust::pair<out_t, out_t>>;
  ops_t ops(static_cast<accscalar_t>(correction), take_sqrt);
  gpu_reduce_kernel<scalar_t, out_t, 2>(iter, ops, typename ops_t::acc_t{});
}

static void std_var_kernel_cuda(TensorIterator& iter, double correction, bool take_sqrt) {
  const auto input_dtype = iter.input_dtype();
  if (input_dtype == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::Half, float>(iter, correction, take_sqrt);
  } else if (input_dtype == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    std_var_kernel_impl<at::BFloat16, float>(iter, correction, take_sqrt);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                    iter.dtype(), "std_cuda", [&]() {
      std_var_kernel_impl<scalar_t>(iter, correction, take_sqrt);
    });
  }
}

template <typename scalar_t, typename acc_t=scalar_t, typename out_t=scalar_t>
void mean_kernel_impl(TensorIterator& iter) {
  //  returns acc_t for all non-complex dtypes and returns T for c10::complex<T>
  constexpr bool is_16_bits = sizeof(scalar_t) == 2;
  using factor_t = typename c10::scalar_value_type<acc_t>::type;
  factor_t factor = static_cast<factor_t>(iter.num_output_elements()) / iter.numel();
  if constexpr (is_16_bits) {
    gpu_reduce_kernel<scalar_t, out_t, /*vt0=*/4, /*input_vec_size=*/8>(iter, MeanOps<scalar_t, acc_t, factor_t, out_t> {factor});
  } else {
    gpu_reduce_kernel<scalar_t, out_t>(iter, MeanOps<scalar_t, acc_t, factor_t, out_t> {factor});
  }
}

static void mean_kernel_cuda(TensorIterator& iter) {
  if (iter.dtype() == kHalf) {
    mean_kernel_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kHalf && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::Half, float, float>(iter);
  } else if(iter.dtype() == kBFloat16) {
    mean_kernel_impl<at::BFloat16, float>(iter);
  } else if (iter.dtype(1) == kBFloat16 && iter.dtype() == kFloat) {
    // type promotion that does cast and reduction in a single kernel
    mean_kernel_impl<at::BFloat16, float, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cuda", [&]() {
      mean_kernel_impl<scalar_t>(iter);
    });
  }
}

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_cuda)
REGISTER_DISPATCH(mean_stub, &mean_kernel_cuda)

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
- `ATen/native/TensorIterator.h`
- `ATen/native/cuda/Reduce.cuh`
- `ATen/native/DispatchStub.h`
- `ATen/native/SharedReduceOps.h`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/native/ReduceOps.h`


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

- **File Documentation**: `ReduceMomentKernel.cu_docs.md`
- **Keyword Index**: `ReduceMomentKernel.cu_kw.md`
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

- **File Documentation**: `ReduceMomentKernel.cu_docs.md_docs.md`
- **Keyword Index**: `ReduceMomentKernel.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
