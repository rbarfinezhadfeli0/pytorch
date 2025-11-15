# Keyword Index: `torch/csrc/jit/serialization/export.cpp`

## File Information

- **Original File**: [torch/csrc/jit/serialization/export.cpp](../../../../../torch/csrc/jit/serialization/export.cpp)
- **Documentation**: [`export.cpp_docs.md`](./export.cpp_docs.md)
- **Folder**: `torch/csrc/jit/serialization`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`GraphEncoder`**: [export.cpp_docs.md](./export.cpp_docs.md)

### Functions

- **`ATenAttributeKindToOnnxAttributeType`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`ATenTypeToOnnxType`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`CloseFile`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`CreateExternalFile`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`GetExternalFileName`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`GetFileRootPath`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`check_onnx_proto`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`getNodeStackTraceString`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`get_little_endian_data`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`get_onnx_node_names`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`get_raw_data_export_map`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`get_symbol_dim_param_map`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`get_use_external_data_format`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`if`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`pretty_print_onnx`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`serialize_model_proto_to_string`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`validateBlock`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`validateGraph`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`writeArchiveAndTensors`**: [export.cpp_docs.md](./export.cpp_docs.md)

### Includes

- **`ATen/ATen.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`ATen/Utils.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`ATen/core/functional.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`c10/macros/Macros.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`c10/util/Exception.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`c10/util/accumulate.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`c10/util/irange.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`memory`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`onnx/checker.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`onnx/onnx_pb.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`onnx/proto_utils.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`onnx/shape_inference/implementation.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`optional`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`regex`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`set`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`sstream`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`string`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/autograd/symbolic.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/jit_log.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/passes/dead_code_elimination.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/passes/inliner.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/runtime/instruction.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/serialization/export.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/serialization/import_export_constants.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/serialization/import_export_functions.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/serialization/import_export_helpers.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/serialization/onnx.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/jit/serialization/pickler.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/onnx/back_compat.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/csrc/onnx/onnx.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch/version.h`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`utility`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`vector`**: [export.cpp_docs.md](./export.cpp_docs.md)

### Namespaces

- **`and`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`onnx`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`onnx_torch`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`std`**: [export.cpp_docs.md](./export.cpp_docs.md)
- **`torch`**: [export.cpp_docs.md](./export.cpp_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
