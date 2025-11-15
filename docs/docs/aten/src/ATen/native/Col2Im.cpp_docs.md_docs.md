# Documentation: `docs/aten/src/ATen/native/Col2Im.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Col2Im.cpp_docs.md`
- **Size**: 9,391 bytes (9.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Col2Im.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Col2Im.cpp`
- **Size**: 6,771 bytes (6.61 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

#include <ATen/native/im2col.h>
#include <ATen/native/im2col_shape_check.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/col2im_native.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/im2col_native.h>
#endif

// Note [im2col/col2im output padding]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Our implementations of im2col and col2im take both the input height/width as
// well as a seemingly redundant output height/width.  In principle, you could
// compute the output height/width by using the convolution shape formulas.  So,
// what's up with that?
//
// The trouble arises when one runs the backward of a transposed convolution
// with output_padding >= stride.  (BTW, output_padding is known as adj inside
// THNN.) Let's consider a simple case where we have kernel=2, dilation=2,
// stride=1, output_padding=1 for a 4x4 input:
//
// Input:  X
//
// Output: X.X.
//         ....
//         X.X.
//         ....
//
// If we compute backwards of output with a standard convolution on the output
// with the same parameters, we would end up with a 2x2 grad_input (because you
// can slide the stencil over to the right once and down once).  But that is all
// out-of-bounds if you're computing backwards for a 1x1 input.
//
// "Now Edward," you might say, "the real problem is that you set output_padding
// >= stride, surely an error should have been raised in this case."  To
// understand why it is useful to handle this case, we have to understand how we
// compute the weight gradient of a convolution.  Suppose we have a convolution
// with kernel=2, stride=2 on a 5x5 input.  Let us see all the contributions of
// weight[0][0] (which we have labeled w) in the output:
//
// Input:  a.b..  Weight: w.
//         .....          ..
//         c.d..
//         .....
//         .....
//
// Output: [ aw+...  bw+... ]
//         [ cw+...  dw+... ]
//
// From this diagram, it easy to see that we can compute the weight gradient
// by performing a *dilated* convolution between the input and the
// output gradients with kernel=2, dilation=2, stride=1.  But there's a rub: if
// we do a dilated convolution directly, we'll end up with a 3x3 weight
// gradient, when we clearly wanted a 2x2.  So how do we avoid going out
// of bounds?  We could add a notion of 'output_padding' for non-transposed
// convolution, but another simple and effective fix is to just accept
// the desired output size directly, and compute only within those bounds.
//
//
// ALSO do vol2col

namespace at::native {
namespace {

void col2im_out_cpu_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

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

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  col2im_shape_check(
      input_,
      Tensor(),
      output_height,
      output_width,
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
  if (input.dim() == 2) {
    // Force batch
    batched_input = false;
    input = input.view({1, input.size(0), input.size(1)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);

  output.resize_({batch_size, n_output_plane, output_height, output_width});

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kBFloat16, kHalf, kBool,
      input.scalar_type(), "col2im_out_cpu", [&] {
        Tensor input_n = Tensor();
        Tensor output_n = Tensor();

        int64_t height_col = (output_height + 2 * pad_height -
                              (dilation_height * (kernel_height - 1) + 1)) /
                stride_height +
            1;
        int64_t width_col = (output_width + 2 * pad_width -
                             (dilation_width * (kernel_width - 1) + 1)) /
                stride_width +
            1;

        for (const auto elt : c10::irange(batch_size)) {
          input_n = input.select(0, elt);
          output_n = output.select(0, elt);

          col2im<scalar_t>(
              input_n.const_data_ptr<scalar_t>(),
              n_output_plane,
              output_height,
              output_width,
              height_col,
              width_col,
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

        if (!batched_input) {
          output.resize_({n_output_plane, output_height, output_width});
        }
      });
}

} // namespace

Tensor& col2im_out_cpu(const Tensor& input,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& output) {
  col2im_out_cpu_template(
      output, input, output_size, kernel_size, dilation, padding, stride);
  return output;
}

Tensor col2im_cpu(
    const Tensor& input,
    IntArrayRef output_size,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
  Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  col2im_out_cpu_template(
      output, input, output_size, kernel_size, dilation, padding, stride);
  return output;
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `Tensor`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/TensorUtils.h`
- `ATen/native/im2col.h`
- `ATen/native/im2col_shape_check.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/col2im_native.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/im2col_native.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `Col2Im.cpp_docs.md`
- **Keyword Index**: `Col2Im.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Col2Im.cpp_docs.md_docs.md`
- **Keyword Index**: `Col2Im.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
