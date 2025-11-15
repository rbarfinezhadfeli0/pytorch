# Documentation: `docs/tools/autograd/templates/variable_factories.h_docs.md`

## File Metadata

- **Path**: `docs/tools/autograd/templates/variable_factories.h_docs.md`
- **Size**: 8,350 bytes (8.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file is a **utility or tool script**.

## Original Source

```markdown
# Documentation: `tools/autograd/templates/variable_factories.h`

## File Metadata

- **Path**: `tools/autograd/templates/variable_factories.h`
- **Size**: 5,637 bytes (5.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```c
#pragma once

// ${generated_comment}

#include <ATen/core/Tensor.h>
#include <ATen/TracerMode.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/MemoryFormat.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <torch/csrc/autograd/variable.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/from_blob.h>
$ops_headers
#endif

#include <functional>
#include <initializer_list>
#include <utility>

namespace torch {

/// NOTE: Currently `torch::tensor(...)` doesn't support mixed data types
/// (i.e. `torch::tensor({{bool, 2.0}})` doesn't work). We might be able to
/// support it in the future by iterating over all sub-lists to find
/// the largest data type that can represent all of the elements, or by using
/// variadic templates.
///
/// NOTE: C++ `torch::tensor` with a floating-point type or an `at::ArrayRef` / `std::vector` /
/// (nested) braced-init-list of floating-point types always produces a tensor of dtype
/// `torch::get_default_dtype()`, matching Python `torch.tensor` behavior.
///
/// NOTE: C++ `torch::tensor` with an integer type or an `at::ArrayRef` / `std::vector` /
/// (nested) braced-init-list of integer types always produces a tensor of dtype `at::kLong`
/// (aka. int64_t), matching Python `torch.tensor` behavior.
///
/// NOTE: The following dtypes are not supported by `torch::tensor` currently:
/// - `unsigned int`
/// - `unsigned long int`
/// - `unsigned long long int`
/// - `long long int`
inline at::Tensor tensor(detail::TensorDataContainer tensor_data_container, const at::TensorOptions& options = {}) {
  return autograd::make_variable(
    // note: we remove the requires_grad setting from the TensorOptions because
    // it is ignored anyways (and we actually have an assertion that it isn't set
    // which would fail otherwise). We handle requires_grad explicitly here
    // instead of passing it through to the kernel.
    tensor_data_container.convert_to_tensor(options.requires_grad(::std::nullopt)),
    options.requires_grad());
}

/// A generic deleter function.
using Deleter = std::function<void(void*)>;
using at::MemoryFormat;

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `deleter` function (a
/// `std::function<void(void*)>`) will be called on the `data` when the Tensor
/// data would normally be deallocated. The `TensorOptions` specify additional
/// configuration options for the returned tensor, such as what type to
/// interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, strides, deleter, options.requires_grad(::std::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `TensorOptions`
/// specify additional configuration options for the returned tensor, such as
/// what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, strides, options.requires_grad(::std::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The `deleter`
/// (a `std::function<void(void*)>`) function will be called on the `data` when
/// the Tensor data would normally be deallocated. The `TensorOptions` specify
/// additional configuration options for the returned tensor, such as what type
/// to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, deleter, options.requires_grad(::std::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The
/// `TensorOptions` specify additional configuration options for the returned
/// tensor, such as what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor = ([&]() {
    at::AutoDispatchBelowAutograd guard;  // TODO: remove
    at::tracer::impl::NoTracerDispatchMode tracer_guard;
    return at::from_blob(data, sizes, options.requires_grad(::std::nullopt));
  })();
  return autograd::make_variable(tensor, options.requires_grad());
}

${function_definitions}

} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/autograd/templates`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/TracerMode.h`
- `ATen/core/grad_mode.h`
- `c10/util/ArrayRef.h`
- `c10/core/MemoryFormat.h`
- `torch/csrc/api/include/torch/detail/TensorDataContainer.h`
- `torch/csrc/autograd/variable.h`
- `ATen/Functions.h`
- `ATen/ops/from_blob.h`
- `functional`
- `initializer_list`
- `utility`


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

Files in the same folder (`tools/autograd/templates`):

- [`TraceType.cpp_docs.md`](./TraceType.cpp_docs.md)
- [`python_variable_methods.cpp_docs.md`](./python_variable_methods.cpp_docs.md)
- [`python_fft_functions.cpp_docs.md`](./python_fft_functions.cpp_docs.md)
- [`Functions.cpp_docs.md`](./Functions.cpp_docs.md)
- [`python_nn_functions.cpp_docs.md`](./python_nn_functions.cpp_docs.md)
- [`python_torch_functions.cpp_docs.md`](./python_torch_functions.cpp_docs.md)
- [`Functions.h_docs.md`](./Functions.h_docs.md)
- [`ViewFuncs.h_docs.md`](./ViewFuncs.h_docs.md)
- [`python_functions.cpp_docs.md`](./python_functions.cpp_docs.md)
- [`python_linalg_functions.cpp_docs.md`](./python_linalg_functions.cpp_docs.md)


## Cross-References

- **File Documentation**: `variable_factories.h_docs.md`
- **Keyword Index**: `variable_factories.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/tools/autograd/templates`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/tools/autograd/templates`, which contains **development tools and scripts**.



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

Files in the same folder (`docs/tools/autograd/templates`):

- [`python_fft_functions.cpp_docs.md_docs.md`](./python_fft_functions.cpp_docs.md_docs.md)
- [`Functions.cpp_docs.md_docs.md`](./Functions.cpp_docs.md_docs.md)
- [`TraceType.cpp_kw.md_docs.md`](./TraceType.cpp_kw.md_docs.md)
- [`python_return_types.cpp_docs.md_docs.md`](./python_return_types.cpp_docs.md_docs.md)
- [`python_sparse_functions.cpp_docs.md_docs.md`](./python_sparse_functions.cpp_docs.md_docs.md)
- [`python_torch_functions.cpp_docs.md_docs.md`](./python_torch_functions.cpp_docs.md_docs.md)
- [`VariableType.h_kw.md_docs.md`](./VariableType.h_kw.md_docs.md)
- [`python_nn_functions.cpp_kw.md_docs.md`](./python_nn_functions.cpp_kw.md_docs.md)
- [`python_enum_tag.cpp_kw.md_docs.md`](./python_enum_tag.cpp_kw.md_docs.md)
- [`python_nested_functions.cpp_kw.md_docs.md`](./python_nested_functions.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `variable_factories.h_docs.md_docs.md`
- **Keyword Index**: `variable_factories.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
