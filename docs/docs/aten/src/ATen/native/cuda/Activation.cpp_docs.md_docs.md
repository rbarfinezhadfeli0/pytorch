# Documentation: `docs/aten/src/ATen/native/cuda/Activation.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/cuda/Activation.cpp_docs.md`
- **Size**: 6,422 bytes (6.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/cuda/Activation.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Activation.cpp`
- **Size**: 3,491 bytes (3.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/Activation.h>

#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/glu_backward_native.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#endif

namespace at::native {

// -----------------------------------
// glu backward
// -----------------------------------

Tensor& glu_backward_cuda_out(const Tensor& grad_output, const Tensor& input,
                              int64_t dim, Tensor& grad_input) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  auto input_sizes = input.sizes();
  const int64_t nIn = input_sizes[wrap_dim];
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  resize_output(grad_input, input_sizes);

  DimVector iter_shape(input_sizes);
  const auto dim_size = nIn / 2;
  iter_shape[wrap_dim] = dim_size;
  TORCH_CHECK(grad_output.sizes() == IntArrayRef{iter_shape});

  const auto iter = at::TensorIteratorConfig()
    .add_output(grad_input)
    .add_const_input(input)
    .add_const_input(grad_output)
    .resize_outputs(false)
    .declare_static_shape(iter_shape)
    .build();

  if (iter.numel() == 0) {
    return grad_input;
  }

  const auto I_stride = input.strides()[wrap_dim] * dim_size;
  const auto gI_stride = grad_input.strides()[wrap_dim] * dim_size;

  if (iter.can_use_32bit_indexing()) {
    launch_glu_backward_kernel(iter, gI_stride, I_stride);
  } else {
    for (const auto& sub_iter: iter.with_32bit_indexing()) {
      launch_glu_backward_kernel(sub_iter, gI_stride, I_stride);
    }
  }
  return grad_input;
}

Tensor glu_backward_cuda(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return glu_backward_cuda_out(grad_output, input, dim, grad_input);
}

// -----------------------------------
// log_sigmoid forward
// -----------------------------------

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_cuda(const Tensor& input, Tensor& result, Tensor& buffer) {
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  auto iter = TensorIteratorConfig()
    .add_output(result)
    .add_const_input(input)
    .build();
  launch_log_sigmoid_forward_kernel(iter);
  return std::forward_as_tuple(result, buffer);
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_cuda(const Tensor& input) {
  auto result = at::empty_like(input);
  auto buffer = at::empty({0}, input.options());
  log_sigmoid_forward_out_cuda(input, result, buffer);
  return std::forward_as_tuple(result, buffer);
}

TORCH_IMPL_FUNC(gelu_out_cuda) (
  const Tensor& /*self*/, std::string_view approximate, const Tensor& /*result*/
) {
  GeluCUDAKernelImpl(*this, get_gelutype_enum(approximate));
}

TORCH_IMPL_FUNC(gelu_backward_out_cuda) (
  const Tensor& /*grad*/, const Tensor& /*self*/, std::string_view approximate, const Tensor& /*grad_input*/
) {
  GeluBackwardCUDAKernelImpl(*this, get_gelutype_enum(approximate));
}

}  // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `ATen/native/cuda/Activation.h`
- `ATen/core/DimVector.h`
- `ATen/core/Tensor.h`
- `ATen/TensorIterator.h`
- `ATen/WrapDimUtils.h`
- `ATen/native/Resize.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/gelu_backward_native.h`
- `ATen/ops/gelu_native.h`
- `ATen/ops/glu_backward_native.h`
- `ATen/ops/log_sigmoid_forward_native.h`


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

- **File Documentation**: `Activation.cpp_docs.md`
- **Keyword Index**: `Activation.cpp_kw.md`
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

- **File Documentation**: `Activation.cpp_docs.md_docs.md`
- **Keyword Index**: `Activation.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
