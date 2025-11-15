# Documentation: `aten/src/ATen/native/cuda/Im2Col.cu`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/Im2Col.cu`
- **Size**: 4,321 bytes (4.22 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>

#include <ATen/native/cuda/im2col.cuh>
#include <ATen/native/im2col_shape_check.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/im2col_native.h>
#endif

namespace at::native {
namespace {

static void im2col_out_cuda_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TORCH_CHECK(
      kernel_size.size() == 2,
      "It is expected kernel_size equals to 2, but got size ",
      kernel_size.size());

  TORCH_CHECK(
      dilation.size() == 2,
      "It is expected dilation equals to 2, but got size ",
      dilation.size());

  TORCH_CHECK(
      padding.size() == 2,
      "It is expected padding equals to 2, but got size ",
      padding.size());

  TORCH_CHECK(
      stride.size() == 2,
      "It is expected stride equals to 2, but got size ",
      stride.size());

  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  TensorArg input_arg{input_, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU(__func__, {input_arg, output_arg});

  im2col_shape_check(
      input_,
      Tensor(),
      kernel_height,
      kernel_width,
      dilation_height,
      dilation_width,
      pad_height,
      pad_width,
      stride_height,
      stride_width);

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input = input.unsqueeze(0);
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});

  // Launch kernel
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kBool,
      input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    for (int64_t elt = 0; elt < batch_size; elt++) {
      input_n = input.select(0, elt);
      output_n = output.select(0, elt);

      im2col<scalar_t>(
          at::cuda::getCurrentCUDAStream(),
          input_n.const_data_ptr<scalar_t>(),
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width,
          kernel_height,
          kernel_width,
          pad_height,
          pad_width,
          stride_height,
          stride_width,
          dilation_height,
          dilation_width,
          output_n.mutable_data_ptr<scalar_t>());
    }

  });
  if (!batched_input) {
    output = output.squeeze(0);
  }
}

} // namespace

Tensor& im2col_out_cuda(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& output) {
  im2col_out_cuda_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

Tensor im2col_cuda(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  im2col_out_cuda_template(
      output, input, kernel_size, dilation, padding, stride);
  return output;
}

} // namespace at::native

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native/cuda`.

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/TensorUtils.h`
- `ATen/Utils.h`
- `ATen/div_rtn.h`
- `ATen/cuda/CUDAContext.h`
- `ATen/native/cuda/im2col.cuh`
- `ATen/native/im2col_shape_check.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/col2im_native.h`
- `ATen/ops/im2col_native.h`


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

- **File Documentation**: `Im2Col.cu_docs.md`
- **Keyword Index**: `Im2Col.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
