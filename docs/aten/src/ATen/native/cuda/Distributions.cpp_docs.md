# Documentation: `aten/src/ATen/native/cuda/Distributions.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Distributions.cpp`
- **Size**: 3,111 bytes (3.04 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Distributions.h>
#include <ATen/TensorIterator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_dirichlet_grad_native.h>
#include <ATen/ops/_sample_dirichlet_native.h>
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/binomial_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/poisson_native.h>
#endif

namespace at::native {

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor _s_poisson_cuda(const Tensor& lambda, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  launch_poisson_cuda_kernel(ret, lambda, gen);
  return ret;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor _s_binomial_cuda(const Tensor& count, const Tensor& prob, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty(count.sizes(), count.options());
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(count)
      .add_input(prob)
      .build();
  launch_binomial_cuda_kernel(iter, gen);
  return ret;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor _s_gamma_cuda(const Tensor& alpha, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  launch_gamma_kernel(ret, alpha, gen);
  return ret;
}

// NOLINTNEXTLINE(performance-unnecessary-value-param)
Tensor _s_dirichlet_cuda(const Tensor& alpha, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, cuda::detail::getDefaultCUDAGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  launch_gamma_kernel(ret, alpha, gen);
  auto gamma_sum = ret.sum(/*dim=*/-1, /*keepdim=*/true);
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(ret)
      .add_input(gamma_sum)
      .build();
  launch_dirichlet_kernel(iter);
  return ret;
}

Tensor _standard_gamma_grad_cuda(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(self)
      .add_input(output)
      .build();
  launch_standard_gamma_grad_kernel(iter);
  return ret;
}

Tensor _dirichlet_grad_cuda(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  Tensor ret = at::empty(x.sizes(), x.options());
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(x)
      .add_input(alpha)
      .add_input(total)
      .build();
  launch_dirichlet_grad_kernel(iter);
  return ret;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

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

- `ATen/native/cuda/Distributions.h`
- `ATen/TensorIterator.h`
- `ATen/cuda/CUDAGeneratorImpl.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_dirichlet_grad_native.h`
- `ATen/ops/_sample_dirichlet_native.h`
- `ATen/ops/_standard_gamma_grad_native.h`
- `ATen/ops/_standard_gamma_native.h`
- `ATen/ops/binomial_native.h`
- `ATen/ops/empty.h`
- `ATen/ops/poisson_native.h`


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

- **File Documentation**: `Distributions.cpp_docs.md`
- **Keyword Index**: `Distributions.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
