# Documentation: `torch/csrc/autograd/input_metadata.h`

## File Metadata

- **Path**: `torch/csrc/autograd/input_metadata.h`
- **Size**: 3,768 bytes (3.68 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/ExpandUtils.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

namespace torch::autograd {

using SymIntSmallVec = c10::SmallVector<c10::SymInt, c10::kDimVectorStaticSize>;
using MetadataShape = std::variant<SymIntSmallVec, at::Tensor>;

/**
 * Records TensorOptions, shape of the tensor, whether or not the Python
 * dispatch key is set (tensor subclass), and, where applicable, the stream the
 * corresponding operation took place on.
 *
 * If is_valid() is false, then the corresponding input is not used and may be
 * an undefined tensor.
 */
struct TORCH_API InputMetadata {
  InputMetadata() = default;
  InputMetadata(
      const at::TensorOptions& options,
      MetadataShape input_shape,
      bool is_tensor_subclass,
      bool is_nested,
      std::optional<at::ScalarType> grad_dtype);
  InputMetadata(const at::Tensor& t);

  const at::TensorOptions& options() const {
    return options_;
  }

  caffe2::TypeMeta dtype() const {
    return options_.dtype();
  }

  at::Device device() const {
    return options_.device();
  }

  at::Layout layout() const {
    return options_.layout();
  }

  c10::Stream stream() const {
    return stream_;
  }

  bool is_tensor_subclass() const {
    return is_tensor_subclass_;
  }

  at::Tensor zeros_like() const;

  bool is_same_shape(const at::Tensor& grad) const;

  bool is_expandable_to_shape(const at::Tensor& grad) const;

  at::Tensor reduce_grad(at::Tensor& grad) const;

  at::Tensor maybe_reduce(
      const size_t index,
      at::Tensor grad,
      const std::function<std::string(const std::string&)>& format_error) const;

  std::stringstream incompatible_shape_error_message(
      const size_t index,
      const at::Tensor& grad) const;

  bool was_default_constructed() const {
    return was_default_constructed_;
  }

  bool is_cpp_nested_tensor() const;

  bool is_nested_tensor() const {
    return is_nested_;
  }

  c10::SymIntArrayRef shape_as_dim_vector() const;

  // Danger: not thread safe, caller must protect with lock
  SymIntSmallVec& mutable_shape_as_dim_vector();

  std::optional<at::ScalarType> grad_dtype() const {
    TORCH_INTERNAL_ASSERT(!was_default_constructed_);
    return grad_dtype_;
  }

  void set_grad_dtype(const std::optional<at::ScalarType>& grad_dtype) {
    TORCH_INTERNAL_ASSERT(!was_default_constructed_);
    grad_dtype_ = grad_dtype;
  }

 private:
  at::Tensor shape_as_tensor() const;
  bool is_nestedness_same(const at::Tensor& grad) const;
  bool maybe_expandable_to(const at::Tensor& grad) const;

  // NB: The engine does not use the dtype from the options, but rather the
  //     grad_dtype_ field to validate grad_output dtype.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::TensorOptions options_;
  MetadataShape shape_;
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device());
  bool is_tensor_subclass_ = false;
  bool is_nested_ = false;
  bool was_default_constructed_ = true;

  // The grad_dtype_ field is the dtype that the engine expects the grad to be.
  // When nullopt, grad_dtype_ is allowed to be any dtype.
  // This field is mutated if THPVariable_set_grad_dtype is called
  // and the AccumulateGrad has already been created.
  std::optional<at::ScalarType> grad_dtype_;
};
} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ExpandUtils.h`
- `ATen/NestedTensorImpl.h`
- `ATen/core/Tensor.h`
- `c10/core/Device.h`
- `c10/core/DeviceType.h`
- `c10/core/Stream.h`
- `c10/core/SymIntArrayRef.h`
- `c10/core/TensorImpl.h`
- `c10/core/impl/DeviceGuardImplInterface.h`
- `c10/util/DimVector.h`
- `c10/util/Exception.h`
- `c10/util/SmallVector.h`
- `ATen/Functions.h`
- `ATen/ops/zeros.h`


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

Files in the same folder (`torch/csrc/autograd`):

- [`graph_task.h_docs.md`](./graph_task.h_docs.md)
- [`python_function.cpp_docs.md`](./python_function.cpp_docs.md)
- [`profiler.h_docs.md`](./profiler.h_docs.md)
- [`TraceTypeManual.cpp_docs.md`](./TraceTypeManual.cpp_docs.md)
- [`python_autograd.h_docs.md`](./python_autograd.h_docs.md)
- [`variable_info.cpp_docs.md`](./variable_info.cpp_docs.md)
- [`jit_decomp_interface.h_docs.md`](./jit_decomp_interface.h_docs.md)
- [`input_buffer.cpp_docs.md`](./input_buffer.cpp_docs.md)
- [`python_variable.h_docs.md`](./python_variable.h_docs.md)
- [`python_nn_functions.h_docs.md`](./python_nn_functions.h_docs.md)


## Cross-References

- **File Documentation**: `input_metadata.h_docs.md`
- **Keyword Index**: `input_metadata.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
