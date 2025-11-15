# Documentation: `aten/src/ATen/native/quantized/cpu/TensorOperators.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/TensorOperators.cpp`
- **Size**: 3,510 bytes (3.43 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/eq.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/le.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/ne_native.h>
#include <ATen/ops/resize_native.h>
#endif

namespace at::native {

/*
All comparator operators will be named "<aten op name>_quantized_cpu".
'_out' will be appended for the 'out' variant of the op.

TODO: This is an inefficient implementation that uses `.dequantize`.
      Need a more efficient implementation.
*/

#define DEFINE_COMPARATOR(at_op) \
Tensor& at_op##_out_quantized_cpu(const Tensor& self, \
                                const Scalar& other, Tensor& out) { \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  auto self_dq = self.dequantize(); \
  return at:: at_op##_out(out, self_dq, other); \
} \
Tensor at_op##_quantized_cpu(const Tensor& self, const Scalar& other) { \
  auto self_dq = self.dequantize(); \
  return at:: at_op(self_dq, other); \
} \
Tensor& at_op##_out_quantized_cpu(const Tensor& self, \
                                const Tensor& other, Tensor& out) { \
  /* We infer size to make sure the tensors are compatible. */\
  infer_size_dimvector(self.sizes(), other.sizes()); \
  TORCH_CHECK(out.dtype() == at::ScalarType::Bool, \
              "The 'out' tensor must have dtype 'torch.bool'"); \
  auto self_dq = self.dequantize(); \
  auto other_dq = other.dequantize(); \
  return at:: at_op##_out(out, self_dq, other_dq); \
} \
Tensor at_op##_quantized_cpu(const Tensor& self, const Tensor& other) { \
  /* We infer size to make sure the tensors are compatible. */\
  infer_size_dimvector(self.sizes(), other.sizes()); \
  auto self_dq = self.dequantize(); \
  auto other_dq = other.dequantize(); \
  return at:: at_op(self_dq, other_dq); \
}

#define AT_FORALL_OPERATORS(_) \
_(ne)                          \
_(eq)                          \
_(ge)                          \
_(le)                          \
_(gt)                          \
_(lt)                          \

AT_FORALL_OPERATORS(DEFINE_COMPARATOR)

#undef AT_FORALL_OPERATORS
#undef DEFINE_COMPARATOR

const Tensor& quantized_resize_cpu_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because if storage is resized, new elements are uninitialized
  globalContext().alertNotDeterministic("quantized_resize_cpu_");
  TORCH_CHECK(
      !optional_memory_format.has_value(),
      "Unsupported memory format for quantized tensor resize ",
      optional_memory_format.value());
  auto qscheme = self.quantizer()->qscheme();
  TORCH_CHECK(
      qscheme == QScheme::PER_TENSOR_AFFINE ||
          qscheme == QScheme::PER_TENSOR_SYMMETRIC,
      "Can only resize quantized tensors with per-tensor schemes!");
  auto* self_ = self.unsafeGetTensorImpl();
  resize_impl_cpu_(self_, size, /*stride=*/std::nullopt);
  return self;
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

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/ExpandUtils.h`
- `ATen/native/Resize.h`
- `ATen/quantized/Quantizer.h`
- `c10/core/QScheme.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/eq.h`
- `ATen/ops/eq_native.h`
- `ATen/ops/ge.h`
- `ATen/ops/ge_native.h`
- `ATen/ops/gt.h`
- `ATen/ops/gt_native.h`
- `ATen/ops/le.h`
- `ATen/ops/le_native.h`
- `ATen/ops/lt.h`
- `ATen/ops/lt_native.h`
- `ATen/ops/ne.h`
- `ATen/ops/ne_native.h`
- `ATen/ops/resize_native.h`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `TensorOperators.cpp_docs.md`
- **Keyword Index**: `TensorOperators.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
