# Documentation: `docs/aten/src/ATen/native/cuda/ReduceOps.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/ReduceOps.cpp_docs.md`
- **Size**: 6,415 bytes (6.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/ReduceOps.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/ReduceOps.cpp`
- **Size**: 3,425 bytes (3.34 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/ReduceOps.h>

#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>

#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/TensorIterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/full.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/where.h>
#endif

namespace at::native {
namespace {

void norm_kernel_cuda(TensorIterator& iter, const Scalar& val) {
  double p = 0;
  if (val.isIntegral(false)) {
    p = static_cast<double>(val.to<int64_t>());
  } else if (val.isFloatingPoint()) {
    p = val.to<double>();
  } else {
    TORCH_CHECK(false, "norm_kernel_cuda_impl expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((p < 0) ? INFINITY : 0);
    return;
  }

  norm_launch_kernel(iter, p);

  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }

}

void min_kernel_impl(const Tensor& result, const Tensor& indice, const Tensor& self, int64_t dim, bool keepdim) {
  auto iter = meta::make_reduction(self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  min_launch_kernel(iter);
}

void max_kernel_impl(const Tensor& result, const Tensor& indice, const Tensor& self, int64_t dim, bool keepdim) {
  auto iter = meta::make_reduction(self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  max_launch_kernel(iter);
}

void aminmax_kernel_impl(
    const Tensor& self, int64_t dim, bool keepdim, Tensor& min_result, Tensor& max_result) {
  at::TensorIterator iter = make_reduction("aminmax_cuda", min_result,
                                           max_result, self, dim, keepdim, self.scalar_type());
  if (iter.numel() != 0) {
    aminmax_launch_kernel(iter);
  }
}

void min_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("min_all", result, input, IntArrayRef{}, false, dtype);
  min_all_launch_kernel(iter);
}

void max_all_kernel_impl(Tensor& result, const Tensor& input) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("max_all", result, input, IntArrayRef{}, false, dtype);
  max_all_launch_kernel(iter);
}

void aminmax_allreduce_kernel_impl(const Tensor& input, Tensor& min_result, Tensor& max_result) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction("aminmax_cuda", min_result, max_result, input,
                             IntArrayRef{}, false, dtype);
  TORCH_CHECK(iter.numel() > 0, "min_max on a tensor with no elements is not defined.");
  aminmax_allreduce_launch_kernel(iter);
}

}  // namespace (anonymous)

REGISTER_CUDA_DISPATCH(min_stub, &min_kernel_impl)
REGISTER_CUDA_DISPATCH(max_stub, &max_kernel_impl)
REGISTER_CUDA_DISPATCH(min_all_stub, &min_all_kernel_impl)
REGISTER_CUDA_DISPATCH(max_all_stub, &max_all_kernel_impl)
REGISTER_CUDA_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_kernel_impl)
REGISTER_CUDA_DISPATCH(aminmax_stub, &aminmax_kernel_impl)

REGISTER_CUDA_DISPATCH(norm_stub, &norm_kernel_cuda)

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

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

- `ATen/native/cuda/ReduceOps.h`
- `ATen/native/ReduceOps.h`
- `ATen/native/ReduceAllOps.h`
- `ATen/native/ReduceOpsUtils.h`
- `ATen/native/TensorCompare.h`
- `ATen/Context.h`
- `ATen/TensorUtils.h`
- `ATen/WrapDimUtils.h`
- `ATen/core/NamedTensor.h`
- `ATen/TensorIterator.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/full.h`
- `ATen/ops/imag.h`
- `ATen/ops/kthvalue_native.h`
- `ATen/ops/median_native.h`
- `ATen/ops/nanmedian_native.h`
- `ATen/ops/where.h`


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

- **File Documentation**: `ReduceOps.cpp_docs.md`
- **Keyword Index**: `ReduceOps.cpp_kw.md`
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

- **File Documentation**: `ReduceOps.cpp_docs.md_docs.md`
- **Keyword Index**: `ReduceOps.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
