# Documentation: `docs/torch/nativert/graph/TensorMeta.h_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/graph/TensorMeta.h_docs.md`
- **Size**: 4,938 bytes (4.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/graph/TensorMeta.h`

## File Metadata

- **Path**: `torch/nativert/graph/TensorMeta.h`
- **Size**: 2,440 bytes (2.38 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Device.h>
#include <c10/util/Logging.h>

#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>

#include <torch/csrc/utils/generated_serialization_types.h>
#include <torch/nativert/executor/Placement.h>

namespace torch::nativert {

c10::ScalarType convertJsonScalarType(
    const torch::_export::ScalarType& scalarType);
c10::MemoryFormat convertJsonMemoryFormat(
    const torch::_export::MemoryFormat& memoryFormat);
c10::Layout convertJsonLayout(const torch::_export::Layout& layout);
c10::Device convertJsonDevice(const torch::_export::Device& device);

class TensorMeta {
 public:
  explicit TensorMeta(const torch::_export::TensorMeta& tensorMeta);

  c10::IntArrayRef sizes() const {
    TORCH_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
    return sizes_;
  }

  c10::IntArrayRef strides() const {
    TORCH_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
    return strides_;
  }

  c10::Layout layout() const {
    return layout_;
  }

  c10::ScalarType dtype() const {
    return dtype_;
  }

  bool requires_grad() const {
    return requiresGrad_;
  }

  int64_t storage_offset() const {
    return storage_offset_;
  }

  int64_t dim() const {
    return sizes_.size();
  }

  int64_t numel() const {
    TORCH_CHECK(!hasSymbolicShape_, "TensorMeta has symbolic shape");
    return numel_;
  }

  c10::Device device() const {
    return device_;
  }

  // override device according to placement
  void setDevice(c10::Device device) {
    device_ = device;
  }

  c10::TensorOptions asTensorOptions() const {
    return c10::TensorOptions().dtype(dtype_).layout(layout_).requires_grad(
        requiresGrad_);
  }

  // override device according to placement
  void applyDevicePlacement(const Placement& placement) {
    device_ = placement.getMappedDevice(device_);
  }

  // NYI
  // c10::SymIntArrayRef sym_sizes() const {}
  // c10::SymIntArrayRef sym_strides() const {}
  // c10::SymInt sym_storage_offset() const {}
  // c10::SymInt sym_numel() const {}

 private:
  bool hasSymbolicShape_ = false;

  std::vector<int64_t> sizes_;
  std::vector<int64_t> strides_;
  int64_t storage_offset_ = 0;
  int64_t numel_ = 1;

  c10::ScalarType dtype_;
  c10::Layout layout_;
  bool requiresGrad_;

  c10::Device device_;
};

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TensorMeta`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/graph`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Device.h`
- `c10/util/Logging.h`
- `c10/core/Layout.h`
- `c10/core/MemoryFormat.h`
- `c10/core/ScalarType.h`
- `c10/core/TensorOptions.h`
- `c10/util/ArrayRef.h`
- `torch/csrc/utils/generated_serialization_types.h`
- `torch/nativert/executor/Placement.h`


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

Files in the same folder (`torch/nativert/graph`):

- [`GraphUtils.cpp_docs.md`](./GraphUtils.cpp_docs.md)
- [`TensorMeta.cpp_docs.md`](./TensorMeta.cpp_docs.md)
- [`Serialization.cpp_docs.md`](./Serialization.cpp_docs.md)
- [`Serialization.h_docs.md`](./Serialization.h_docs.md)
- [`GraphPasses.cpp_docs.md`](./GraphPasses.cpp_docs.md)
- [`GraphSignature.cpp_docs.md`](./GraphSignature.cpp_docs.md)
- [`GraphUtils.h_docs.md`](./GraphUtils.h_docs.md)
- [`Graph.cpp_docs.md`](./Graph.cpp_docs.md)
- [`Graph.h_docs.md`](./Graph.h_docs.md)


## Cross-References

- **File Documentation**: `TensorMeta.h_docs.md`
- **Keyword Index**: `TensorMeta.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/graph`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/graph`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/nativert/graph`):

- [`Serialization.cpp_docs.md_docs.md`](./Serialization.cpp_docs.md_docs.md)
- [`GraphSignature.h_docs.md_docs.md`](./GraphSignature.h_docs.md_docs.md)
- [`GraphSignature.cpp_kw.md_docs.md`](./GraphSignature.cpp_kw.md_docs.md)
- [`GraphPasses.h_kw.md_docs.md`](./GraphPasses.h_kw.md_docs.md)
- [`GraphSignature.h_kw.md_docs.md`](./GraphSignature.h_kw.md_docs.md)
- [`Graph.h_docs.md_docs.md`](./Graph.h_docs.md_docs.md)
- [`GraphPasses.cpp_docs.md_docs.md`](./GraphPasses.cpp_docs.md_docs.md)
- [`GraphUtils.h_docs.md_docs.md`](./GraphUtils.h_docs.md_docs.md)
- [`Graph.cpp_kw.md_docs.md`](./Graph.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `TensorMeta.h_docs.md_docs.md`
- **Keyword Index**: `TensorMeta.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
