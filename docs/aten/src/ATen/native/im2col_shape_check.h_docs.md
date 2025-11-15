# Documentation: `aten/src/ATen/native/im2col_shape_check.h`

## File Metadata

- **Path**: `aten/src/ATen/native/im2col_shape_check.h`
- **Size**: 7,336 bytes (7.16 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/div_rtn.h>
#include <c10/util/safe_numerics.h>

namespace at::native {

inline void col2im_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int64_t output_height,
    int64_t output_width,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t stride_height,
    int64_t stride_width) {
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);
  TORCH_CHECK(
      stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);
  TORCH_CHECK(
      dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);
  TORCH_CHECK(
      pad_width >= 0 && pad_height >= 0,
      "padding should be non-negative, but got pad_height: ",
      pad_height,
      " pad_width: ",
      pad_width);


  int64_t ndim = input.ndimension();
  // allow dim=0 only the batch dimension.
  TORCH_CHECK(
      (ndim == 2 && input.size(0) != 0 && input.size(1) != 0) ||
      (ndim == 3 && input.size(1) != 0 && input.size(2) != 0),
      "Expected 2D or 3D (batch mode) tensor for input with possibly 0 batch size and non-zero dimensions for input, but got: ",
      input.sizes());

  int64_t batch_dim = (ndim == 3) ? 0 : -1;
  int64_t n_input_plane = input.size(batch_dim + 1);
  uint64_t prod_kernel_size = 1;

  TORCH_CHECK(!c10::mul_overflows(static_cast<uint64_t>(kernel_width), static_cast<uint64_t>(kernel_height), &prod_kernel_size),
            "Given kernel_width = ",
            kernel_width,
            " and kernel_height = ",
            kernel_height,
            " the product of kernel_width and kernel_height overflowed.");

  if (n_input_plane % (kernel_width * kernel_height) != 0) {
    TORCH_CHECK(false,
        "Expected size of input's dimension 1 to be divisible by the "
        "product of kernel_size, but got input.size(1)=",
        n_input_plane,
        " and kernel_size=(",
        kernel_height,
        ", ",
        kernel_width,
        ").");
  }

  int64_t input_length = input.size(batch_dim + 2);
  int64_t n_blocks_height =
      div_rtn<int64_t>(
          output_height + 2 * pad_height -
              dilation_height * (kernel_height - 1) - 1,
          stride_height) +
      1;
  int64_t n_blocks_width = div_rtn<int64_t>(
                                   output_width + 2 * pad_width -
                                       dilation_width * (kernel_width - 1) - 1,
                                   stride_width) +
      1;

  if (input_length != (n_blocks_height * n_blocks_width)) {
    TORCH_CHECK(false,
        "Given output_size=(",
        output_height,
        ", ",
        output_width,
        "), kernel_size=(",
        kernel_height,
        ", ",
        kernel_width,
        "), dilation=(",
        dilation_height,
        ", ",
        dilation_width,
        "), padding=(",
        pad_height,
        ", ",
        pad_width,
        "), stride=(",
        stride_height,
        ", ",
        stride_width,
        "), expected size of input's dimension 2 to match the calculated number of ",
        "sliding blocks ",
        n_blocks_height,
        " * ",
        n_blocks_width,
        " = ",
        (n_blocks_height * n_blocks_width),
        ", but got input.size(2)=",
        input_length,
        ".");
  }

  TORCH_CHECK(
    n_blocks_height >= 1 && n_blocks_width >= 1,
    "Given output_size=(", output_height, ", ", output_width, "), ",
    "kernel_size=(", kernel_height, ", ", kernel_width, "), ",
    "dilation=(", dilation_height, ", ", dilation_width, "), ",
    "padding=(", pad_height, ", ", pad_width, "), ",
    "stride=(", stride_height, ", ", stride_width, "), ",
    "calculated shape of the array of sliding blocks as ",
    "(", n_blocks_height, ", ", n_blocks_width, "), ",
    "which is too small (non-positive)");

  if (output_width < 1 || output_height < 1) {
    TORCH_CHECK(false,
        "Expected output spatial size to be positive, but got: output_size=(",
        output_height,
        ", ",
        output_width,
        ").");
  }
}

inline void im2col_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t stride_height,
    int64_t stride_width) {
  TORCH_CHECK(
      kernel_width > 0 && kernel_height > 0,
      "kernel size should be greater than zero, but got kernel_height: ",
      kernel_height,
      " kernel_width: ",
      kernel_width);

  TORCH_CHECK(
      dilation_width > 0 && dilation_height > 0,
      "dilation should be greater than zero, but got dilation_height: ",
      dilation_height,
      " dilation_width: ",
      dilation_width);

  TORCH_CHECK(
      pad_width >= 0 && pad_height >= 0,
      "padding should be non-negative, but got pad_height: ",
      pad_height,
      " pad_width: ",
      pad_width);

  TORCH_CHECK(
      stride_width > 0 && stride_height > 0,
      "stride should be greater than zero, but got stride_height: ",
      stride_height,
      " stride_width: ",
      stride_width);

  int64_t ndim = input.ndimension();

  // allow dim=0 only the batch dimension.
  bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
  TORCH_CHECK(
      (ndim == 3 && input.size(0) && valid_dims) ||
      (ndim == 4 && valid_dims && input.size(3) != 0),
      "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
      input.sizes());

  int64_t dim_batch = 0;

  if (ndim == 3) {
    dim_batch = -1;
  }

  int64_t input_height = input.size(dim_batch + 2);
  int64_t input_width = input.size(dim_batch + 3);
  int64_t output_height = div_rtn<int64_t>(
                              input_height + 2 * pad_height -
                                  (dilation_height * (kernel_height - 1) + 1),
                              stride_height) +
      1;
  int64_t output_width = div_rtn<int64_t>(
                             input_width + 2 * pad_width -
                                 (dilation_width * (kernel_width - 1) + 1),
                             stride_width) +
      1;

  if (output_height < 1 || output_width < 1) {
    TORCH_CHECK(false,
        "Given input with spatial size (",
        input_height,
        ", ",
        input_height,
        "), kernel_size=(",
        kernel_height,
        ", ",
        kernel_width,
        "), dilation=(",
        dilation_height,
        ", ",
        dilation_width,
        "), padding=(",
        pad_height,
        ", ",
        pad_width,
        "), calculated shape of the array of sliding blocks as (",
        output_height,
        ", ",
        output_width,
        "), but its components must be at least one.");
  }
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/TensorUtils.h`
- `ATen/div_rtn.h`
- `c10/util/safe_numerics.h`


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

- **File Documentation**: `im2col_shape_check.h_docs.md`
- **Keyword Index**: `im2col_shape_check.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
