# Documentation: `docs/torch/csrc/jit/serialization/onnx.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/serialization/onnx.cpp_docs.md`
- **Size**: 10,075 bytes (9.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/serialization/onnx.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/serialization/onnx.cpp`
- **Size**: 7,477 bytes (7.30 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/irange.h>
#include <torch/csrc/jit/serialization/onnx.h>
#include <torch/csrc/onnx/onnx.h>

#include <sstream>
#include <string>

namespace torch::jit {

namespace {
namespace onnx = ::ONNX_NAMESPACE;

// Pretty printing for ONNX
constexpr char indent_char = ' ';
constexpr size_t indent_multiplier = 2;

std::string idt(size_t indent) {
  return std::string(indent * indent_multiplier, indent_char);
}

std::string nlidt(size_t indent) {
  return std::string("\n") + idt(indent);
}

void dump(const onnx::TensorProto& tensor, std::ostream& stream) {
  stream << "TensorProto shape: [";
  for (const auto i : c10::irange(tensor.dims_size())) {
    stream << tensor.dims(i) << (i == tensor.dims_size() - 1 ? "" : " ");
  }
  stream << "]";
}

void dump(const onnx::TensorShapeProto& shape, std::ostream& stream) {
  for (const auto i : c10::irange(shape.dim_size())) {
    auto& dim = shape.dim(i);
    if (dim.has_dim_value()) {
      stream << dim.dim_value();
    } else {
      stream << "?";
    }
    stream << (i == shape.dim_size() - 1 ? "" : " ");
  }
}

void dump(const onnx::TypeProto_Tensor& tensor_type, std::ostream& stream) {
  stream << "Tensor dtype: ";
  if (tensor_type.has_elem_type()) {
    stream << tensor_type.elem_type();
  } else {
    stream << "None.";
  }
  stream << ", ";
  stream << "Tensor dims: ";
  if (tensor_type.has_shape()) {
    dump(tensor_type.shape(), stream);
  } else {
    stream << "None.";
  }
}

void dump(const onnx::TypeProto& type, std::ostream& stream);

void dump(const onnx::TypeProto_Optional& optional_type, std::ostream& stream) {
  stream << "Optional<";
  if (optional_type.has_elem_type()) {
    dump(optional_type.elem_type(), stream);
  } else {
    stream << "None";
  }
  stream << ">";
}

void dump(const onnx::TypeProto_Sequence& sequence_type, std::ostream& stream) {
  stream << "Sequence<";
  if (sequence_type.has_elem_type()) {
    dump(sequence_type.elem_type(), stream);
  } else {
    stream << "None";
  }
  stream << ">";
}

void dump(const onnx::TypeProto& type, std::ostream& stream) {
  if (type.has_tensor_type()) {
    dump(type.tensor_type(), stream);
  } else if (type.has_sequence_type()) {
    dump(type.sequence_type(), stream);
  } else if (type.has_optional_type()) {
    dump(type.optional_type(), stream);
  } else {
    stream << "None";
  }
}

void dump(const onnx::ValueInfoProto& value_info, std::ostream& stream) {
  stream << "{name: \"" << value_info.name() << "\", type:";
  dump(value_info.type(), stream);
  stream << "}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent);

void dump(
    const onnx::AttributeProto& attr,
    std::ostream& stream,
    size_t indent) {
  stream << "{ name: '" << attr.name() << "', type: ";
  if (attr.has_f()) {
    stream << "float, value: " << attr.f();
  } else if (attr.has_i()) {
    stream << "int, value: " << attr.i();
  } else if (attr.has_s()) {
    stream << "string, value: '" << attr.s() << "'";
  } else if (attr.has_g()) {
    stream << "graph, value:\n";
    dump(attr.g(), stream, indent + 1);
    stream << nlidt(indent);
  } else if (attr.has_t()) {
    stream << "tensor, value:";
    dump(attr.t(), stream);
  } else if (attr.floats_size()) {
    stream << "floats, values: [";
    for (const auto i : c10::irange(attr.floats_size())) {
      stream << attr.floats(i) << (i == attr.floats_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.ints_size()) {
    stream << "ints, values: [";
    for (const auto i : c10::irange(attr.ints_size())) {
      stream << attr.ints(i) << (i == attr.ints_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.strings_size()) {
    stream << "strings, values: [";
    for (const auto i : c10::irange(attr.strings_size())) {
      stream << "'" << attr.strings(i) << "'"
             << (i == attr.strings_size() - 1 ? "" : " ");
    }
    stream << "]";
  } else if (attr.tensors_size()) {
    stream << "tensors, values: [";
    for (auto& t : attr.tensors()) {
      dump(t, stream);
    }
    stream << "]";
  } else if (attr.graphs_size()) {
    stream << "graphs, values: [";
    for (auto& g : attr.graphs()) {
      dump(g, stream, indent + 1);
    }
    stream << "]";
  } else {
    stream << "UNKNOWN";
  }
  stream << "}";
}

void dump(const onnx::NodeProto& node, std::ostream& stream, size_t indent) {
  stream << "Node {type: \"" << node.op_type() << "\", inputs: [";
  for (const auto i : c10::irange(node.input_size())) {
    stream << node.input(i) << (i == node.input_size() - 1 ? "" : ",");
  }
  stream << "], outputs: [";
  for (const auto i : c10::irange(node.output_size())) {
    stream << node.output(i) << (i == node.output_size() - 1 ? "" : ",");
  }
  stream << "], attributes: [";
  for (const auto i : c10::irange(node.attribute_size())) {
    dump(node.attribute(i), stream, indent + 1);
    stream << (i == node.attribute_size() - 1 ? "" : ",");
  }
  stream << "]}";
}

void dump(const onnx::GraphProto& graph, std::ostream& stream, size_t indent) {
  stream << idt(indent) << "GraphProto {" << nlidt(indent + 1) << "name: \""
         << graph.name() << "\"" << nlidt(indent + 1) << "inputs: [";
  for (const auto i : c10::irange(graph.input_size())) {
    dump(graph.input(i), stream);
    stream << (i == graph.input_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "outputs: [";
  for (const auto i : c10::irange(graph.output_size())) {
    dump(graph.output(i), stream);
    stream << (i == graph.output_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "value_infos: [";
  for (const auto i : c10::irange(graph.value_info_size())) {
    dump(graph.value_info(i), stream);
    stream << (i == graph.value_info_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "initializers: [";
  for (const auto i : c10::irange(graph.initializer_size())) {
    dump(graph.initializer(i), stream);
    stream << (i == graph.initializer_size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent + 1) << "nodes: [" << nlidt(indent + 2);
  for (const auto i : c10::irange(graph.node_size())) {
    dump(graph.node(i), stream, indent + 2);
    if (i != graph.node_size() - 1) {
      stream << "," << nlidt(indent + 2);
    }
  }
  stream << nlidt(indent + 1) << "]\n" << idt(indent) << "}\n";
}

void dump(
    const onnx::OperatorSetIdProto& operator_set_id,
    std::ostream& stream) {
  stream << "OperatorSetIdProto { domain: " << operator_set_id.domain()
         << ", version: " << operator_set_id.version() << "}";
}

void dump(const onnx::ModelProto& model, std::ostream& stream, size_t indent) {
  stream << idt(indent) << "ModelProto {" << nlidt(indent + 1)
         << "producer_name: \"" << model.producer_name() << "\""
         << nlidt(indent + 1) << "domain: \"" << model.domain() << "\""
         << nlidt(indent + 1) << "doc_string: \"" << model.doc_string() << "\"";
  if (model.has_graph()) {
    stream << nlidt(indent + 1) << "graph:\n";
    dump(model.graph(), stream, indent + 2);
  }
  if (model.opset_import_size()) {
    stream << idt(indent + 1) << "opset_import: [";
    for (auto& opset_imp : model.opset_import()) {
      dump(opset_imp, stream);
    }
    stream << "],\n";
  }
  stream << idt(indent) << "}\n";
}

} // namespace

std::string prettyPrint(const ::ONNX_NAMESPACE::ModelProto& model) {
  std::ostringstream ss;
  dump(model, ss, 0);
  return ss.str();
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 28 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `onnx`, `torch`, `std`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `torch/csrc/jit/serialization/onnx.h`
- `torch/csrc/onnx/onnx.h`
- `sstream`
- `string`


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

Files in the same folder (`torch/csrc/jit/serialization`):

- [`import_read.h_docs.md`](./import_read.h_docs.md)
- [`unpickler.h_docs.md`](./unpickler.h_docs.md)
- [`import_export_functions.h_docs.md`](./import_export_functions.h_docs.md)
- [`import.h_docs.md`](./import.h_docs.md)
- [`pickle.cpp_docs.md`](./pickle.cpp_docs.md)
- [`source_range_serialization_impl.h_docs.md`](./source_range_serialization_impl.h_docs.md)
- [`mobile_bytecode_generated.h_docs.md`](./mobile_bytecode_generated.h_docs.md)
- [`import_export_helpers.cpp_docs.md`](./import_export_helpers.cpp_docs.md)
- [`import_export_constants.h_docs.md`](./import_export_constants.h_docs.md)
- [`source_range_serialization.h_docs.md`](./source_range_serialization.h_docs.md)


## Cross-References

- **File Documentation**: `onnx.cpp_docs.md`
- **Keyword Index**: `onnx.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/serialization`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/serialization`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/serialization`):

- [`pickler.h_docs.md_docs.md`](./pickler.h_docs.md_docs.md)
- [`onnx.h_kw.md_docs.md`](./onnx.h_kw.md_docs.md)
- [`import_export_functions.h_docs.md_docs.md`](./import_export_functions.h_docs.md_docs.md)
- [`import_export_helpers.h_docs.md_docs.md`](./import_export_helpers.h_docs.md_docs.md)
- [`flatbuffer_serializer_jit.cpp_kw.md_docs.md`](./flatbuffer_serializer_jit.cpp_kw.md_docs.md)
- [`source_range_serialization.cpp_kw.md_docs.md`](./source_range_serialization.cpp_kw.md_docs.md)
- [`export.cpp_kw.md_docs.md`](./export.cpp_kw.md_docs.md)
- [`import_read.h_kw.md_docs.md`](./import_read.h_kw.md_docs.md)
- [`pickle.cpp_kw.md_docs.md`](./pickle.cpp_kw.md_docs.md)
- [`export_bytecode.cpp_docs.md_docs.md`](./export_bytecode.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `onnx.cpp_docs.md_docs.md`
- **Keyword Index**: `onnx.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
