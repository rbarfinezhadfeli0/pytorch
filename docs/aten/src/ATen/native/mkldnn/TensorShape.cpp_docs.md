# Documentation: `aten/src/ATen/native/mkldnn/TensorShape.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/mkldnn/TensorShape.cpp`
- **Size**: 3,272 bytes (3.20 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/InferSize.h>
#include <ATen/core/Tensor.h>
#include <c10/core/SymIntArrayRef.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_mkldnn_reshape_native.h>
#include <ATen/ops/_mkldnn_transpose_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/view_native.h>
#endif

#if !AT_MKLDNN_ENABLED()


namespace at::native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "mkldnn_reshape: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false, "mkldnn_reshape: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_clone(const Tensor& self, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(false, "mkldnn_clone: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "mkldnn_transpose: ATen not compiled with MKLDNN support");
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "mkldnn_transpose_: ATen not compiled with MKLDNN support");
}

} // namespace at::native


#else // AT_MKLDNN_ENABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at::native {

Tensor mkldnn_view(const Tensor& self, IntArrayRef size) {
  TORCH_CHECK(false,
      "Currently Mkldnn tensor does not support view. Change to use reshape instead");
}

Tensor mkldnn_reshape(const Tensor& self, IntArrayRef size) {
  auto inferred_size = at::infer_size(size, self.numel());
  if (self.sizes() == inferred_size) {
    return self;
  }
  const ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y{x};
  y.reshape(inferred_size);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor mkldnn_clone(const Tensor& self, std::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "unsupported memory format option ",
      optional_memory_format.value());
  ideep::tensor& src = itensor_from_mkldnn(self);
  ideep::tensor dst;
  ideep::direct_copy::compute(src, dst);
  return new_with_itensor_mkldnn(std::move(dst), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor mkldnn_transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
  auto ndims = self.dim();
  dim0 = maybe_wrap_dim(dim0, ndims);
  dim1 = maybe_wrap_dim(dim1, ndims);
  const ideep::tensor& x = itensor_from_mkldnn(self);
  ideep::tensor y;
  std::vector<int> axes(x.ndims());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[dim0], axes[dim1]);
  y.transpose_from(x, axes);
  return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor& mkldnn_transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(false, "mkldnn_transpose_: in-place mkldnn operations are not supported yet");
}

} // namespace at

#endif // AT_MKLDNN_ENABLED

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mkldnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/InferSize.h`
- `ATen/core/Tensor.h`
- `c10/core/SymIntArrayRef.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_mkldnn_reshape_native.h`
- `ATen/ops/_mkldnn_transpose_native.h`
- `ATen/ops/clone_native.h`
- `ATen/ops/view_native.h`
- `ATen/native/mkldnn/MKLDNNCommon.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`aten/src/ATen/native/mkldnn`):

- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`Gelu.cpp_docs.md`](./Gelu.cpp_docs.md)
- [`Conv.h_docs.md`](./Conv.h_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`Matmul.cpp_docs.md`](./Matmul.cpp_docs.md)
- [`RNN.cpp_docs.md`](./RNN.cpp_docs.md)
- [`RegisterMkldnnOpContextClass.cpp_docs.md`](./RegisterMkldnnOpContextClass.cpp_docs.md)
- [`Copy.cpp_docs.md`](./Copy.cpp_docs.md)


## Cross-References

- **File Documentation**: `TensorShape.cpp_docs.md`
- **Keyword Index**: `TensorShape.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
