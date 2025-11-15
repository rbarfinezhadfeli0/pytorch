# Documentation: `aten/src/ATen/native/ReplicationPadding.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/ReplicationPadding.cpp`
- **Size**: 10,461 bytes (10.22 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/Padding.h>
#include <c10/util/irange.h>
#include <algorithm>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/replication_pad1d_backward_native.h>
#include <ATen/ops/replication_pad1d_native.h>
#include <ATen/ops/replication_pad2d_backward_native.h>
#include <ATen/ops/replication_pad2d_native.h>
#include <ATen/ops/replication_pad3d_backward_native.h>
#include <ATen/ops/replication_pad3d_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::meta {

TORCH_META_FUNC(replication_pad1d) (
  const Tensor& input, IntArrayRef paddingSize  // no out argument!
) {
  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");

  int64_t dimw = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  at::native::padding::check_valid_input<1>(input, paddingSize);

  if (input.ndimension() == 3) {
    nbatch = input.size(0);
    dimw++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iwidth = input.size(dimw);
  int64_t owidth = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth >= 1,
      "input (W: ", iwidth, ") is too small."
      " Calculated output W: ", owidth);

  if (input.ndimension() == 2) {
    set_output_raw_strided(0, {nslices, owidth}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nslices, owidth}, {}, input.options());
  }
}

TORCH_META_FUNC(replication_pad1d_backward) (
  const Tensor& gradOutput,
  const Tensor& input,
  IntArrayRef paddingSize
) {
  int64_t dimw = 1;
  TORCH_CHECK(paddingSize.size() == 2, "padding size is expected to be 2");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];

  if (input.ndimension() == 3) {
    dimw++;
  }

  /* sizes */
  int64_t iwidth = input.size(dimw);
  int64_t owidth  = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth,
      " Got: ", gradOutput.size(dimw));

  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

TORCH_META_FUNC(replication_pad2d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
  int64_t pad_l = paddingSize[0];
  int64_t pad_r = paddingSize[1];
  int64_t pad_t = paddingSize[2];
  int64_t pad_b = paddingSize[3];
  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  at::native::padding::check_valid_input<2>(input, paddingSize);

  if (input.dim() == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth  = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth >= 1 || oheight >= 1,
      "input (H: ", iheight, ", W: ", iwidth, " ) is too small."
      " Calculated output H: ", oheight, " W: ", owidth);

  if (input.dim() == 3) {
    set_output_raw_strided(0, {nslices, oheight, owidth}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nslices, oheight, owidth}, {}, input.options());
  }
}

TORCH_META_FUNC(replication_pad3d) (
  const Tensor& input, IntArrayRef paddingSize
) {
  TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
  int64_t pleft = paddingSize[0];
  int64_t pright = paddingSize[1];
  int64_t ptop = paddingSize[2];
  int64_t pbottom = paddingSize[3];
  int64_t pfront = paddingSize[4];
  int64_t pback = paddingSize[5];
  int64_t dimw = 3;
  int64_t dimh = 2;
  int64_t dimd = 1;
  int64_t dimslices = 0;
  int64_t nbatch = 1;

  at::native::padding::check_valid_input<3>(input, paddingSize);

  if (input.dim() == 5) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimd++;
    dimslices++;
  }

  /* sizes */
  int64_t nslices = input.size(dimslices);
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t odepth = idepth + pfront + pback;
  int64_t oheight = iheight + ptop + pbottom;
  int64_t owidth  = iwidth + pleft + pright;

  TORCH_CHECK(owidth >= 1 || oheight >= 1 || odepth >= 1,
      "input (D: ", idepth, " H: ", iheight, ", W: ", iwidth,
      ") is too small."
      " Calculated output D: ", odepth, " H: ", oheight, " W: ", owidth);

  /* resize output */
  if (input.dim() == 4) {
    set_output_raw_strided(0, {nslices, odepth, oheight, owidth}, {}, input.options());
  } else {
    set_output_raw_strided(0, {nbatch, nslices, odepth, oheight, owidth}, {}, input.options());
  }
}

} // namespace at::meta

namespace at::native {

namespace {

void replication_pad2d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(paddingSize.size() == 4, "padding size is expected to be 4");
  int pad_l = paddingSize[0];
  int pad_r = paddingSize[1];
  int pad_t = paddingSize[2];
  int pad_b = paddingSize[3];
  int dimw = 2;
  int dimh = 1;

  if (input.dim() == 4) {
    dimw++;
    dimh++;
  }

  /* sizes */
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t oheight = iheight + pad_t + pad_b;
  int64_t owidth  = iwidth + pad_l + pad_r;

  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));
  TORCH_CHECK(oheight == gradOutput.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput.size(dimh));

  if (gradInput.numel() == 0) {
    return;
  }

  replication_pad2d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

void replication_pad3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  TORCH_CHECK(paddingSize.size() == 6, "padding size is expected to be 6");
  int pleft = paddingSize[0];
  int pright = paddingSize[1];
  int ptop = paddingSize[2];
  int pbottom = paddingSize[3];
  int pfront = paddingSize[4];
  int pback = paddingSize[5];
  int dimc = 0;
  int dimw = 3;
  int dimh = 2;
  int dimd = 1;

  if (input.dim() == 5) {
    dimc++;
    dimw++;
    dimh++;
    dimd++;
  }

  /* sizes */
  int64_t ichannel = input.size(dimc);
  int64_t idepth = input.size(dimd);
  int64_t iheight = input.size(dimh);
  int64_t iwidth = input.size(dimw);
  int64_t odepth = idepth + pfront + pback;
  int64_t oheight = iheight + ptop + pbottom;
  int64_t owidth  = iwidth + pleft + pright;

  at::native::padding::check_valid_input<3>(input, paddingSize);

  TORCH_CHECK(ichannel == gradOutput.size(dimc),
      "gradOutput width unexpected. Expected: ", ichannel, ", Got: ",
      gradOutput.size(dimc));
  TORCH_CHECK(owidth == gradOutput.size(dimw),
      "gradOutput width unexpected. Expected: ", owidth, ", Got: ",
      gradOutput.size(dimw));
  TORCH_CHECK(oheight == gradOutput.size(dimh),
      "gradOutput height unexpected. Expected: ", oheight, ", Got: ",
      gradOutput.size(dimh));
  TORCH_CHECK(odepth == gradOutput.size(dimd),
      "gradOutput depth unexpected. Expected: ", odepth, ", Got: ",
      gradOutput.size(dimd));

  if (gradInput.numel() == 0) {
    return;
  }

  replication_pad3d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

} // anonymous namespace

TORCH_IMPL_FUNC(replication_pad1d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  replication_pad1d_kernel(kCPU, output, input, paddingSize);
}

TORCH_IMPL_FUNC(replication_pad1d_backward_out_cpu) (
  const Tensor& gradOutput, const Tensor& input, IntArrayRef paddingSize, const Tensor& gradInput
) {
  if (gradInput.numel() == 0) {
    return;
  }
  gradInput.zero_();

  replication_pad1d_backward_kernel(kCPU, gradInput, gradOutput, paddingSize);
}

TORCH_IMPL_FUNC(replication_pad2d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  // TODO: move this to TORCH_META_FUNC when CUDA has channels last support
  output.resize_(output.sizes(), input.suggest_memory_format());

  replication_pad2d_kernel(kCPU, output, input, paddingSize);
}

Tensor& replication_pad2d_backward_out_cpu(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  gradInput.resize_as_(input, input.suggest_memory_format());
  gradInput.zero_();
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad2d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, input.suggest_memory_format());
  replication_pad2d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

TORCH_IMPL_FUNC(replication_pad3d_out_cpu) (
  const Tensor& input, IntArrayRef paddingSize, const Tensor& output
) {
  // TODO: move this to TORCH_META_FUNC when CUDA has channels last support
  output.resize_(output.sizes(), input.suggest_memory_format());

  replication_pad3d_kernel(kCPU, output, input, paddingSize);
}

Tensor& replication_pad3d_backward_out_cpu(const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize,
    Tensor& gradInput)
{
  gradInput.resize_as_(input, input.suggest_memory_format());
  gradInput.zero_();
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

Tensor replication_pad3d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef paddingSize)
{
  auto gradInput = at::zeros_like(input, input.suggest_memory_format());
  replication_pad3d_backward_out_cpu_template(
      gradInput, gradOutput, input, paddingSize);
  return gradInput;
}

DEFINE_DISPATCH(replication_pad1d_kernel);
DEFINE_DISPATCH(replication_pad1d_backward_kernel);
DEFINE_DISPATCH(replication_pad2d_kernel);
DEFINE_DISPATCH(replication_pad2d_backward_kernel);
DEFINE_DISPATCH(replication_pad3d_kernel);
DEFINE_DISPATCH(replication_pad3d_backward_kernel);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `TORCH_IMPL_FUNC`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/Parallel.h`
- `ATen/TensorMeta.h`
- `ATen/native/Padding.h`
- `c10/util/irange.h`
- `algorithm`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/replication_pad1d_backward_native.h`
- `ATen/ops/replication_pad1d_native.h`
- `ATen/ops/replication_pad2d_backward_native.h`
- `ATen/ops/replication_pad2d_native.h`
- `ATen/ops/replication_pad3d_backward_native.h`
- `ATen/ops/replication_pad3d_native.h`
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

- **File Documentation**: `ReplicationPadding.cpp_docs.md`
- **Keyword Index**: `ReplicationPadding.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
