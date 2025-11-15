# Documentation: `docs/aten/src/ATen/native/cuda/GridSampler.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/GridSampler.cpp_docs.md`
- **Size**: 6,028 bytes (5.89 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/GridSampler.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/GridSampler.cpp`
- **Size**: 3,182 bytes (3.11 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/GridSampler.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/grid_sampler_2d_backward_native.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d_backward_native.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

Tensor grid_sampler_2d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
  launch_grid_sampler_2d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

Tensor grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode,
                            bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
      input.options());
  launch_grid_sampler_3d_forward_kernel(
      output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

std::tuple<Tensor, Tensor>
grid_sampler_2d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool, 2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_2d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
                              const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
                              bool align_corners, std::array<bool,2> output_mask) {
  auto input_requires_grad = output_mask[0];
  Tensor grad_input = ([&]() {
    if (input_requires_grad) {
      return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
    } else {
      return Tensor();
    }
  })();
  auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_3d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, interpolation_mode, padding_mode, align_corners, output_mask);
  return std::make_tuple(grad_input, grad_grid);
}

}  // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

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

- `ATen/native/cuda/GridSampler.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/grid_sampler_2d_backward_native.h`
- `ATen/ops/grid_sampler_2d_native.h`
- `ATen/ops/grid_sampler_3d_backward_native.h`
- `ATen/ops/grid_sampler_3d_native.h`
- `ATen/ops/zeros_like.h`


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

- **File Documentation**: `GridSampler.cpp_docs.md`
- **Keyword Index**: `GridSampler.cpp_kw.md`
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

- **File Documentation**: `GridSampler.cpp_docs.md_docs.md`
- **Keyword Index**: `GridSampler.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
