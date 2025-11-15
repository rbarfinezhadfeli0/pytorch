# Documentation: `docs/aten/src/ATen/native/Cross.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Cross.cpp_docs.md`
- **Size**: 5,947 bytes (5.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Cross.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Cross.cpp`
- **Size**: 3,298 bytes (3.22 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Cross.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorMeta.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/MemoryOverlap.h>


#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cross_native.h>
#include <ATen/ops/linalg_cross.h>
#include <ATen/ops/linalg_cross_native.h>
#endif

namespace at::meta {

TORCH_META_FUNC(linalg_cross)
(const Tensor & input, const Tensor & other, int64_t dim) {
  auto x_d = input.dim();
  auto y_d = other.dim();
  // This is to avoid things like
  // linalg.cross(torch.randn(2, 3), torch.randn(5, 2, 3), dim=2)
  TORCH_CHECK(x_d == y_d, "linalg.cross: inputs must have the same number of dimensions.");
  TORCH_CHECK(input.size(dim) == 3 && other.size(dim) == 3, "linalg.cross: inputs dimension ", dim, " must have length 3. Got ", input.size(dim), " and ", other.size(dim));

  // Broadcast the batch dimension of input and other.
  // Since the non-batch dimensions agree, this is the same as broadcast all the inputs
  auto out_size = infer_size(input.sizes(), other.sizes());

  set_output_raw_strided(0, out_size, {}, input.options());
}

} // namespace at::meta
namespace at::native {

DEFINE_DISPATCH(cross_stub);

static int64_t _default_cross_dim(const std::optional<int64_t> &dimension, SymIntArrayRef sizes) {
  // If dimension is not given, it defaults to the first dimension found with the size 3.
  // Note that this behaviour might be unexpected.
  // _default_cross_dim is called internally inside the cross implementation to calculate
  // the dim and finally cross delegates to the linalg_cross implementation with this dim
  if(dimension.has_value()) {
    return *dimension;
  }

  for(auto i : c10::irange(sizes.size())) {
    if(sizes[i] == 3) {
      return i;
    }
  }
  TORCH_CHECK(false, "no dimension of size 3 in input");
}

Tensor cross(const Tensor & input, const Tensor & other, const std::optional<int64_t> dimension) {
  if (!dimension) {
    TORCH_WARN_ONCE(
      "Using torch.cross without specifying the dim arg is deprecated.\n",
      "Please either pass the dim explicitly or simply use torch.linalg.cross.\n",
      "The default value of dim will change to agree with that of linalg.cross in a future release."
    );
  }
  auto dim = _default_cross_dim(dimension, input.sym_sizes());
  return at::linalg_cross(input, other, dim);
}

Tensor & cross_out(const Tensor & input, const Tensor & other, const std::optional<int64_t> dimension, Tensor & out) {
  auto dim = _default_cross_dim(dimension, input.sym_sizes());
  return at::linalg_cross_out(out, input, other, dim);
}


TORCH_IMPL_FUNC(linalg_cross_out)
(const Tensor & input, const Tensor & other, int64_t dim, const Tensor & out) {
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, input);
  at::assert_no_overlap(out, other);
  dim = maybe_wrap_dim(dim, input.dim());
  auto out_size = out.sizes();
  Tensor input_broadcasted = input.expand(out_size);
  Tensor other_broadcasted = other.expand(out_size);

  cross_stub(input.device().type(), out, input_broadcasted, other_broadcasted, dim);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `ATen/native/Cross.h`
- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/TensorMeta.h`
- `ATen/WrapDimUtils.h`
- `ATen/ExpandUtils.h`
- `ATen/native/Resize.h`
- `ATen/MemoryOverlap.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/cross_native.h`
- `ATen/ops/linalg_cross.h`
- `ATen/ops/linalg_cross_native.h`


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

- **File Documentation**: `Cross.cpp_docs.md`
- **Keyword Index**: `Cross.cpp_kw.md`
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

- **File Documentation**: `Cross.cpp_docs.md_docs.md`
- **Keyword Index**: `Cross.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
