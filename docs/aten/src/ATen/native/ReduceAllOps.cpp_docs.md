# Documentation: `aten/src/ATen/native/ReduceAllOps.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/ReduceAllOps.cpp`
- **Size**: 2,347 bytes (2.29 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/Resize.h>

#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_aminmax_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/min.h>
#include <ATen/ops/min_native.h>
#endif

namespace at::native {

DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_all_stub);

Tensor min(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0,
              "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  min_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

Tensor& min_unary_out(const Tensor &self, Tensor& out) {
  // First check if the devices match (CPU vs GPU)
  TORCH_CHECK(self.device() == out.device());

  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));

  at::native::resize_output(out, {});

  min_all_stub(self.device().type(), out, self.contiguous());
  return out;
}

Tensor max(const Tensor &self) {
  TORCH_CHECK(self.numel() > 0,
              "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
  Tensor result = at::empty({}, self.options());
  max_all_stub(self.device().type(), result, self.contiguous());
  return result;
}

Tensor& max_unary_out(const Tensor &self, Tensor& out) {
  // First check if the devices match (CPU vs GPU)
  TORCH_CHECK(self.device() == out.device());

  TORCH_CHECK(canCast(
      typeMetaToScalarType(self.dtype()),
      typeMetaToScalarType(out.dtype())));

  at::native::resize_output(out, {});

  max_all_stub(self.device().type(), out, self.contiguous());
  return out;
}

// DEPRECATED: Use at::aminmax instead
std::tuple<Tensor, Tensor> _aminmax_all(const Tensor &self) {
  TORCH_WARN_ONCE("_aminmax is deprecated as of PyTorch 1.11 and will be removed in a future release. Use aminmax instead."
                  " This warning will only appear once per process.");
  return at::aminmax(self);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

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

- `ATen/native/ReduceAllOps.h`
- `ATen/native/Resize.h`
- `ATen/core/Tensor.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_aminmax_native.h`
- `ATen/ops/aminmax.h`
- `ATen/ops/empty.h`
- `ATen/ops/max.h`
- `ATen/ops/max_native.h`
- `ATen/ops/min.h`
- `ATen/ops/min_native.h`


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

- **File Documentation**: `ReduceAllOps.cpp_docs.md`
- **Keyword Index**: `ReduceAllOps.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
