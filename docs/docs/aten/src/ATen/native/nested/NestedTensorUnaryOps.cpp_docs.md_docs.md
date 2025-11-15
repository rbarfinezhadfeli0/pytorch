# Documentation: `docs/aten/src/ATen/native/nested/NestedTensorUnaryOps.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/nested/NestedTensorUnaryOps.cpp_docs.md`
- **Size**: 9,573 bytes (9.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/nested/NestedTensorUnaryOps.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/nested/NestedTensorUnaryOps.cpp`
- **Size**: 6,713 bytes (6.56 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/nested/NestedTensorMath.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/layer_norm.h>
#include <ATen/native/nested/NestedTensorUtils.h>

namespace at::native {

#define DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(op_name)                 \
Tensor NestedTensor_##op_name(const Tensor& self) {      \
  return map_nt(self, at::op_name);                      \
}

// Use the macro to define operations concisely
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(abs)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(sgn)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(logical_not)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(isinf)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(isposinf)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(isneginf)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(isnan)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(relu)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(silu)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(sin)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(sqrt)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(cos)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(neg)
DEFINE_TORCH_NESTED_TENSOR_UNARY_OP(tanh)

#undef DEFINE_TORCH_NESTED_TENSOR_UNARY_OP

Tensor& NestedTensor_abs_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::abs_(buffer);
  return self;
}

Tensor NestedTensor_where(const Tensor& condition, const Tensor& self, const Tensor& other) {
  TORCH_CHECK(condition.is_nested(), "condition must be nested");
  TORCH_CHECK(other.is_nested(), "other must be nested");
  TORCH_CHECK(!self.is_nested(), "self must not be nested");

  auto condition_ptr = get_nested_tensor_impl(condition);
  auto other_ptr = get_nested_tensor_impl(other);

  int64_t ntensors = condition_ptr->size(0);
  TORCH_CHECK(other_ptr->size(0) == ntensors, "condition and other must have the same number of tensors");

  // Get the buffer and sizes of the 'other' tensor to use for the output
  const Tensor& other_buffer = other_ptr->get_unsafe_storage_as_tensor();
  const Tensor& other_sizes = other_ptr->get_nested_sizes();

  // Create output buffer with the same size as other_buffer
  Tensor output_buffer = other_buffer.new_empty(other_buffer.sizes());

  // Create the output nested tensor
  Tensor output = wrap_buffer(output_buffer, other_sizes.clone());

  // Unbind condition, other, and output into lists of tensors
  std::vector<Tensor> condition_unbind = condition.unbind();
  std::vector<Tensor> other_unbind = other.unbind();
  std::vector<Tensor> output_unbind = output.unbind();

  // Apply at::where operation on each triplet of condition, self, and other tensors
  for (int64_t i = 0; i < ntensors; i++) {
    at::where_out(
      output_unbind[i],
      condition_unbind[i],
      self,  // Note: self is not nested, so we use it directly
      other_unbind[i]);
  }

  return output;
}

Tensor& NestedTensor_where_out(const Tensor& condition, const Tensor& self, const Tensor& other, at::Tensor & out) {
  TORCH_CHECK(condition.is_nested(), "condition must be nested");
  TORCH_CHECK(other.is_nested(), "other must be nested");
  TORCH_CHECK(!self.is_nested(), "self must not be nested");
  TORCH_CHECK(out.is_nested(), "out must be nested");

  auto condition_ptr = get_nested_tensor_impl(condition);
  auto other_ptr = get_nested_tensor_impl(other);
  auto out_ptr = get_nested_tensor_impl(out);

  int64_t ntensors = condition_ptr->size(0);
  TORCH_CHECK(other_ptr->size(0) == ntensors, "condition and other must have the same number of tensors");
  TORCH_CHECK(out_ptr->size(0) == ntensors, "condition and out must have the same number of tensors");

  // Unbind condition, other, and out into lists of tensors
  std::vector<Tensor> condition_unbind = condition.unbind();
  std::vector<Tensor> other_unbind = other.unbind();
  std::vector<Tensor> output_unbind = out.unbind();

  // Apply at::where operation on each triplet of condition, self, and other tensors
  for (int64_t i = 0; i < ntensors; i++) {
    at::where_out(
      output_unbind[i],
      condition_unbind[i],
      self,  // Note: self is not nested, so we use it directly
      other_unbind[i]);
  }

  return out;
}

Tensor& NestedTensor_sgn_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  buffer.sgn_();
  return self;
}

Tensor& NestedTensor_logical_not_(Tensor& self){
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  buffer.logical_not_();
  return self;
}


Tensor& NestedTensor_relu_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::relu_(buffer);
  return self;
}

Tensor& NestedTensor_gelu_(Tensor& self, std::string_view approximate) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::gelu_(buffer, approximate);
  return self;
}

Tensor NestedTensor_gelu(const Tensor& self, std::string_view approximate) {
  return map_nt(
      self,
      [approximate](const Tensor& buffer) {
        return at::gelu(buffer, approximate);
      });
}

Tensor& NestedTensor_tanh_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::tanh_(buffer);
  return self;
}

Tensor& NestedTensor_neg_(Tensor& self) {
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::neg_(buffer);
  return self;
}

Tensor& zero_nested_(Tensor& self) {
  const auto& self_buf = get_nested_tensor_impl(self)->get_buffer();
  self_buf.fill_(0);
  return self;
}

Tensor& NestedTensor_silu_(Tensor& self){
  auto self_ptr = get_nested_tensor_impl(self);
  check_numel_equals_buffer_size(self_ptr);
  auto buffer = self_ptr->get_buffer();
  at::silu_(buffer);
  return self;
}

Tensor _pin_memory_nested(const Tensor& self, std::optional<Device> device) {
  auto* nt_input = get_nested_tensor_impl(self);
  const auto& input_buffer = nt_input->get_unsafe_storage_as_tensor();
  return wrap_buffer(
      at::_pin_memory(input_buffer, device),
      nt_input->get_nested_sizes(),
      nt_input->get_nested_strides(),
      nt_input->get_storage_offsets());
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/nested`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/nested/NestedTensorMath.h`
- `ATen/AccumulateType.h`
- `ATen/Dispatch.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/NestedTensorImpl.h`
- `ATen/ScalarOps.h`
- `ATen/TensorIndexing.h`
- `ATen/TensorOperators.h`
- `ATen/TensorUtils.h`
- `ATen/core/Tensor.h`
- `ATen/native/layer_norm.h`
- `ATen/native/nested/NestedTensorUtils.h`


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

Files in the same folder (`aten/src/ATen/native/nested`):

- [`NestedTensorBinaryOps.cpp_docs.md`](./NestedTensorBinaryOps.cpp_docs.md)
- [`NestedTensorUtils.cpp_docs.md`](./NestedTensorUtils.cpp_docs.md)
- [`NestedTensorBinaryOps.h_docs.md`](./NestedTensorBinaryOps.h_docs.md)
- [`NestedTensorFactories.cpp_docs.md`](./NestedTensorFactories.cpp_docs.md)
- [`NestedTensorBackward.cpp_docs.md`](./NestedTensorBackward.cpp_docs.md)
- [`NestedTensorMatmul.cpp_docs.md`](./NestedTensorMatmul.cpp_docs.md)
- [`NestedTensorTransformerFunctions.h_docs.md`](./NestedTensorTransformerFunctions.h_docs.md)
- [`NestedTensorMath.cpp_docs.md`](./NestedTensorMath.cpp_docs.md)
- [`NestedTensorTransformerUtils.h_docs.md`](./NestedTensorTransformerUtils.h_docs.md)


## Cross-References

- **File Documentation**: `NestedTensorUnaryOps.cpp_docs.md`
- **Keyword Index**: `NestedTensorUnaryOps.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/nested`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/nested`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/nested`):

- [`NestedTensorTransformerUtils.h_kw.md_docs.md`](./NestedTensorTransformerUtils.h_kw.md_docs.md)
- [`NestedTensorTransformerFunctions.h_kw.md_docs.md`](./NestedTensorTransformerFunctions.h_kw.md_docs.md)
- [`NestedTensorTransformerFunctions.h_docs.md_docs.md`](./NestedTensorTransformerFunctions.h_docs.md_docs.md)
- [`NestedTensorTransformerFunctions.cpp_docs.md_docs.md`](./NestedTensorTransformerFunctions.cpp_docs.md_docs.md)
- [`NestedTensorAliases.cpp_docs.md_docs.md`](./NestedTensorAliases.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`NestedTensorUtils.h_docs.md_docs.md`](./NestedTensorUtils.h_docs.md_docs.md)
- [`NestedTensorBackward.cpp_kw.md_docs.md`](./NestedTensorBackward.cpp_kw.md_docs.md)
- [`NestedTensorBinaryOps.h_docs.md_docs.md`](./NestedTensorBinaryOps.h_docs.md_docs.md)
- [`NestedTensorBinaryOps.cpp_kw.md_docs.md`](./NestedTensorBinaryOps.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `NestedTensorUnaryOps.cpp_docs.md_docs.md`
- **Keyword Index**: `NestedTensorUnaryOps.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
