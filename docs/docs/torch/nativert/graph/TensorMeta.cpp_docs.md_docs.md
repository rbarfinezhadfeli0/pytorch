# Documentation: `docs/torch/nativert/graph/TensorMeta.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/graph/TensorMeta.cpp_docs.md`
- **Size**: 7,983 bytes (7.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/graph/TensorMeta.cpp`

## File Metadata

- **Path**: `torch/nativert/graph/TensorMeta.cpp`
- **Size**: 5,723 bytes (5.59 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nativert/graph/TensorMeta.h>

#include <c10/util/Logging.h>

namespace torch::nativert {

c10::ScalarType convertJsonScalarType(
    const torch::_export::ScalarType& scalarType) {
  switch (scalarType) {
    case torch::_export::ScalarType::UNKNOWN:
      TORCH_CHECK(false, "scalar type is not properly set");
    case torch::_export::ScalarType::BYTE:
      return c10::ScalarType::Byte;
    case torch::_export::ScalarType::CHAR:
      return c10::ScalarType::Char;
    case torch::_export::ScalarType::SHORT:
      return c10::ScalarType::Short;
    case torch::_export::ScalarType::INT:
      return c10::ScalarType::Int;
    case torch::_export::ScalarType::LONG:
      return c10::ScalarType::Long;
    case torch::_export::ScalarType::HALF:
      return c10::ScalarType::Half;
    case torch::_export::ScalarType::FLOAT:
      return c10::ScalarType::Float;
    case torch::_export::ScalarType::DOUBLE:
      return c10::ScalarType::Double;
    case torch::_export::ScalarType::COMPLEXHALF:
      return c10::ScalarType::ComplexHalf;
    case torch::_export::ScalarType::COMPLEXFLOAT:
      return c10::ScalarType::ComplexFloat;
    case torch::_export::ScalarType::COMPLEXDOUBLE:
      return c10::ScalarType::ComplexDouble;
    case torch::_export::ScalarType::BOOL:
      return c10::ScalarType::Bool;
    case torch::_export::ScalarType::BFLOAT16:
      return c10::ScalarType::BFloat16;
    case torch::_export::ScalarType::UINT16:
      return c10::ScalarType::UInt16;
    case torch::_export::ScalarType::FLOAT8E4M3FN:
      return c10::ScalarType::Float8_e4m3fn;
    case torch::_export::ScalarType::FLOAT8E5M2:
      return c10::ScalarType::Float8_e5m2;
    case torch::_export::ScalarType::FLOAT8E4M3FNUZ:
      return c10::ScalarType::Float8_e4m3fnuz;
    case torch::_export::ScalarType::FLOAT8E5M2FNUZ:
      return c10::ScalarType::Float8_e5m2fnuz;
    default:
      TORCH_CHECK(false, "unknown scalar type", static_cast<int>(scalarType));
  }
}

c10::MemoryFormat convertJsonMemoryFormat(
    const torch::_export::MemoryFormat& memoryFormat) {
  switch (memoryFormat) {
    case torch::_export::MemoryFormat::Unknown:
      TORCH_CHECK(false, "got unknown scalar type");
    case torch::_export::MemoryFormat::ContiguousFormat:
      return c10::MemoryFormat::Contiguous;
    case torch::_export::MemoryFormat::ChannelsLast:
      return c10::MemoryFormat::ChannelsLast;
    case torch::_export::MemoryFormat::ChannelsLast3d:
      return c10::MemoryFormat::ChannelsLast3d;
    case torch::_export::MemoryFormat::PreserveFormat:
      return c10::MemoryFormat::Preserve;
    default:
      TORCH_CHECK(
          false, "unknown memory format", static_cast<int>(memoryFormat));
  }
}

c10::Layout convertJsonLayout(const torch::_export::Layout& layout) {
  switch (layout) {
    case torch::_export::Layout::Unknown:
      TORCH_CHECK(false, "got unknown layout");
    case torch::_export::Layout::SparseCoo:
      // TODO is this the right translation
      return c10::Layout::Sparse;
    case torch::_export::Layout::SparseCsr:
      return c10::Layout::SparseCsr;
    case torch::_export::Layout::SparseCsc:
      return c10::Layout::SparseCsc;
    case torch::_export::Layout::SparseBsr:
      return c10::Layout::SparseBsr;
    case torch::_export::Layout::SparseBsc:
      return c10::Layout::SparseBsc;
    case torch::_export::Layout::_mkldnn:
      return c10::Layout::Mkldnn;
    case torch::_export::Layout::Strided:
      return c10::Layout::Strided;
    default:
      TORCH_CHECK(false, "unknown layout", static_cast<int>(layout));
  }
}

c10::Device convertJsonDevice(const torch::_export::Device& device) {
  c10::Device d(device.get_type());
  if (auto index = device.get_index()) {
    d.set_index(static_cast<at::DeviceIndex>(*index));
  }
  return d;
}

TensorMeta::TensorMeta(const torch::_export::TensorMeta& tensorMeta)
    : dtype_(convertJsonScalarType(tensorMeta.get_dtype())),
      layout_(convertJsonLayout(tensorMeta.get_layout())),
      requiresGrad_(tensorMeta.get_requires_grad()),
      device_(convertJsonDevice(tensorMeta.get_device())) {
  const auto& storageOffset = tensorMeta.get_storage_offset();
  if (storageOffset.tag() == torch::_export::SymInt::Tag::AS_INT) {
    storage_offset_ = tensorMeta.get_storage_offset().get_as_int();
  } else if (storageOffset.tag() == torch::_export::SymInt::Tag::AS_EXPR) {
    // TODO: it's still unclear how SymInt shape should be used in runtime
    // setting the storage offset to 0 for now
    hasSymbolicShape_ = true;
    storage_offset_ = 0;
  }

  for (const auto& size : tensorMeta.get_sizes()) {
    if (size.tag() == torch::_export::SymInt::Tag::AS_INT) {
      int64_t val = size.get_as_int();
      sizes_.emplace_back(val);
      numel_ *= val;
    } else if (size.tag() == torch::_export::SymInt::Tag::AS_EXPR) {
      // TODO: it's still unclear how SymInt shape should be used in runtime
      // One potential use cases is for verifying inputs shape matches constrain
      // This would require unpacking the serialized constrain, which is NYI
      //
      // For the time being, we just set the symbolic dim to -1
      hasSymbolicShape_ = true;
      sizes_.emplace_back(-1);
      numel_ = -1;
    }
  }

  for (const auto& stride : tensorMeta.get_strides()) {
    if (stride.tag() == torch::_export::SymInt::Tag::AS_INT) {
      strides_.emplace_back(stride.get_as_int());
    } else if (stride.tag() == torch::_export::SymInt::Tag::AS_EXPR) {
      // TODO: it's still unclear how SymInt shape should be used in runtime
      // Setting symbolic shape to -1 for now
      hasSymbolicShape_ = true;
      strides_.emplace_back(-1);
    }
  }
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/graph`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nativert/graph/TensorMeta.h`
- `c10/util/Logging.h`


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
- [`Serialization.cpp_docs.md`](./Serialization.cpp_docs.md)
- [`Serialization.h_docs.md`](./Serialization.h_docs.md)
- [`GraphPasses.cpp_docs.md`](./GraphPasses.cpp_docs.md)
- [`GraphSignature.cpp_docs.md`](./GraphSignature.cpp_docs.md)
- [`GraphUtils.h_docs.md`](./GraphUtils.h_docs.md)
- [`TensorMeta.h_docs.md`](./TensorMeta.h_docs.md)
- [`Graph.cpp_docs.md`](./Graph.cpp_docs.md)
- [`Graph.h_docs.md`](./Graph.h_docs.md)


## Cross-References

- **File Documentation**: `TensorMeta.cpp_docs.md`
- **Keyword Index**: `TensorMeta.cpp_kw.md`
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
- [`TensorMeta.h_docs.md_docs.md`](./TensorMeta.h_docs.md_docs.md)
- [`GraphSignature.h_kw.md_docs.md`](./GraphSignature.h_kw.md_docs.md)
- [`Graph.h_docs.md_docs.md`](./Graph.h_docs.md_docs.md)
- [`GraphPasses.cpp_docs.md_docs.md`](./GraphPasses.cpp_docs.md_docs.md)
- [`GraphUtils.h_docs.md_docs.md`](./GraphUtils.h_docs.md_docs.md)
- [`Graph.cpp_kw.md_docs.md`](./Graph.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `TensorMeta.cpp_docs.md_docs.md`
- **Keyword Index**: `TensorMeta.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
