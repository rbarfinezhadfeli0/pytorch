# Documentation: `docs/aten/src/ATen/native/ChanelShuffle.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/ChanelShuffle.cpp_docs.md`
- **Size**: 6,603 bytes (6.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/ChanelShuffle.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/ChanelShuffle.cpp`
- **Size**: 3,928 bytes (3.84 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/NamedTensorUtils.h>
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
#include <ATen/native/xnnpack/Engine.h>
#endif
#include <c10/util/Exception.h>

#include <ATen/native/cpu/ChannelShuffleKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/channel_shuffle_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/native_channel_shuffle.h>
#include <ATen/ops/native_channel_shuffle_native.h>
#endif

namespace at::native {

Tensor channel_shuffle_cpu(const Tensor& self, int64_t groups) {
  TORCH_CHECK(self.dim() > 2,
              "channel_shuffle expects input with > 2 dims, but got input with sizes ",
              self.sizes());
  int64_t c = self.size(1);
  TORCH_CHECK(groups > 0,
              "Number of groups to divide channels in must be positive.",
              " Value of groups:", groups);
  TORCH_CHECK((c % groups) == 0,
              "Number of channels must be divisible by groups. Got ",
              c, " channels and ", groups, " groups.");

  Tensor output;
  if (self.numel() == 0) {
    output = self.alias();
  } else {
    auto memory_format = self.suggest_memory_format();
    output = at::empty({0}, self.options());
    output.resize_(self.sizes(), memory_format);
    auto input = self.contiguous(memory_format);
    channel_shuffle_kernel(kCPU, output, input, groups);
  }
  return namedinference::propagate_names_if_nonempty(
      output,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

Tensor channel_shuffle(const Tensor& self, int64_t groups) {
  TORCH_CHECK(self.dim() > 2,
              "channel_shuffle expects input with > 2 dims, but got input with sizes ",
              self.sizes());
  int64_t c = self.size(1);
  TORCH_CHECK(groups > 0,
              "Number of groups to divide channels in must be positive.",
              " Value of groups:", groups);
  TORCH_CHECK((c % groups) == 0,
              "Number of channels must be divisible by groups. Got ",
              c, " channels and ", groups, " groups.");

#if defined(C10_MOBILE) && defined(USE_XNNPACK)
  if (self.is_contiguous(MemoryFormat::ChannelsLast) &&
      xnnpack::use_channel_shuffle(self, groups)) {
    auto output = self.numel() == 0 ? self.alias() : xnnpack::channel_shuffle(self, groups);
    return output;
  }
#endif

  auto output = self.numel() == 0 ? self.alias() : at::native_channel_shuffle(self, groups);
  return namedinference::propagate_names_if_nonempty(
      output,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

Tensor math_channel_shuffle(const Tensor& self, int64_t groups) {
  int64_t b = self.size(0);
  int64_t c = self.size(1);
  int64_t oc = c / groups;

  auto input_reshaped = self.view({b, groups, oc, -1});
  // TODO: contiguous can be made to preserve the memory format
  // of the input. However since the above reshape clobbers h and w
  // it may not be safe to do that, since channels_last contiguous
  // may think oc and the last dim correspond to h,w?
  // It is not clear, however from initial looking around it feels that
  // this may not be correct.
  // In this case channels last will likely require custom implementation
  // if we want to preserve the memory order.
  // XNNPACK has channel shuffle op for NHWC. For mobile usecase this is good.
  // For server we will have to do a custom implementation.
  // For ChannelsFirst, a.k.a Contiguous, memory format we will also need
  // a fast custom implementation perhaps.
  Tensor output_tensor =
      input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3})
      .contiguous()
      .reshape(self.sizes());
  return namedinference::propagate_names_if_nonempty(
      output_tensor,
      self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

DEFINE_DISPATCH(channel_shuffle_kernel);

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

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

- `ATen/NamedTensorUtils.h`
- `ATen/native/xnnpack/Engine.h`
- `c10/util/Exception.h`
- `ATen/native/cpu/ChannelShuffleKernel.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/channel_shuffle_native.h`
- `ATen/ops/empty.h`
- `ATen/ops/native_channel_shuffle.h`
- `ATen/ops/native_channel_shuffle_native.h`


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

- **File Documentation**: `ChanelShuffle.cpp_docs.md`
- **Keyword Index**: `ChanelShuffle.cpp_kw.md`
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

- **File Documentation**: `ChanelShuffle.cpp_docs.md_docs.md`
- **Keyword Index**: `ChanelShuffle.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
