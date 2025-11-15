# Documentation: `docs/torch/csrc/api/include/torch/nested.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nested.h_docs.md`
- **Size**: 5,049 bytes (4.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nested.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nested.h`
- **Size**: 2,773 bytes (2.71 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ATen_fwd.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <algorithm>

namespace torch::nested {

/// Nested tensor
///
/// See
/// https://pytorch.org/docs/main/nested.html#torch.nested.nested_tensor
///
/// ```
// implemented on python object to allow torch.nested.nested_tensor to be
// constructed with arbitrarily nested python objects - for now, only arbitrary
// python lists and lists of Tensors
// See torch/csrc/autograd/python_nested_functions_manual.cpp for Python
// implementation
// See here for C++ implementation
inline at::Tensor nested_tensor(
    at::TensorList nested_tensor_data,
    const at::TensorOptions& options = {}) {
  auto out = at::_nested_tensor_from_tensor_list(
      nested_tensor_data,
      c10::typeMetaToScalarType(options.dtype()),
      std::nullopt,
      options.device(),
      options.pinned_memory());
  if (options.has_requires_grad() && options.requires_grad()) {
    out.requires_grad_(true);
  }
  return out;
}

inline at::Tensor nested_tensor(
    at::ArrayRef<detail::TensorDataContainer> nested_tensor_data,
    const at::TensorOptions& options = {}) {
  for (const auto& tdc : nested_tensor_data) {
    TORCH_CHECK(
        tdc.is_init_list(),
        "nested_tensor() not implemented for these parameters");
  }
  // Construct a TensorList using nested_tensor_data
  std::vector<at::Tensor> tensor_list(nested_tensor_data.size());
  std::transform(
      nested_tensor_data.begin(),
      nested_tensor_data.end(),
      tensor_list.begin(),
      [&](const detail::TensorDataContainer& tdc) {
        return tdc.convert_to_tensor(options);
      });
  auto out = at::_nested_tensor_from_tensor_list(
      tensor_list,
      c10::typeMetaToScalarType(options.dtype()),
      std::nullopt,
      options.device(),
      options.pinned_memory());
  if (options.has_requires_grad() && options.requires_grad()) {
    out.requires_grad_(true);
  }
  return out;
}

/// As Nested Tensor
///
/// See
/// https://pytorch.org/docs/main/nested.html#torch.nested.as_nested_tensor
///
/// ```
inline at::Tensor as_nested_tensor(
    at::TensorList list,
    std::optional<at::ScalarType> dtype = std::nullopt,
    std::optional<at::Device> device = std::nullopt) {
  return at::_nested_tensor_from_tensor_list(
      list, dtype, std::nullopt, device, std::nullopt);
}

/// Nested to padded tensor
///
/// See
/// https://pytorch.org/docs/main/nested.html#torch.nested.to_padded_tensor
///
/// ```
inline at::Tensor to_padded_tensor(
    const at::Tensor& self,
    double padding,
    at::OptionalIntArrayRef output_size = std::nullopt) {
  return at::nested_to_padded_tensor(self, padding, output_size);
}

} // namespace torch::nested

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `a`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/core/ATen_fwd.h`
- `torch/csrc/api/include/torch/detail/TensorDataContainer.h`
- `algorithm`


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

Files in the same folder (`torch/csrc/api/include/torch`):

- [`ordered_dict.h_docs.md`](./ordered_dict.h_docs.md)
- [`fft.h_docs.md`](./fft.h_docs.md)
- [`serialize.h_docs.md`](./serialize.h_docs.md)
- [`nn.h_docs.md`](./nn.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`special.h_docs.md`](./special.h_docs.md)
- [`expanding_array.h_docs.md`](./expanding_array.h_docs.md)
- [`data.h_docs.md`](./data.h_docs.md)
- [`version.h_docs.md`](./version.h_docs.md)


## Cross-References

- **File Documentation**: `nested.h_docs.md`
- **Keyword Index**: `nested.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/api/include/torch`):

- [`expanding_array.h_docs.md_docs.md`](./expanding_array.h_docs.md_docs.md)
- [`nn.h_kw.md_docs.md`](./nn.h_kw.md_docs.md)
- [`serialize.h_docs.md_docs.md`](./serialize.h_docs.md_docs.md)
- [`sparse.h_kw.md_docs.md`](./sparse.h_kw.md_docs.md)
- [`types.h_docs.md_docs.md`](./types.h_docs.md_docs.md)
- [`enum.h_docs.md_docs.md`](./enum.h_docs.md_docs.md)
- [`special.h_kw.md_docs.md`](./special.h_kw.md_docs.md)
- [`nn.h_docs.md_docs.md`](./nn.h_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `nested.h_docs.md_docs.md`
- **Keyword Index**: `nested.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
