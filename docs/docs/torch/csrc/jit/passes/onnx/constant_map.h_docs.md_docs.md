# Documentation: `docs/torch/csrc/jit/passes/onnx/constant_map.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/onnx/constant_map.h_docs.md`
- **Size**: 7,179 bytes (7.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/onnx/constant_map.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/constant_map.h`
- **Size**: 4,422 bytes (4.32 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/macros/Macros.h>

#include <onnx/shape_inference/implementation.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/export.h>
#include <unordered_map>

namespace torch::jit {

using ShapeDataMap =
    std::unordered_map<std::string, ::ONNX_NAMESPACE::TensorShapeProto>;

class ConstantValueMap {
 public:
  static ConstantValueMap& getInstance();
  static void SetRank(const std::string& tensorName, size_t rankValue);
  static bool HasRank(const std::string& tensorName);
  static std::optional<size_t> GetRank(const std::string& tensorName);

  static void SetAllGraphInputsStatic(bool all_static);
  static std::optional<bool> GetAllGraphInputsStatic();

  static void SetAllGraphInputsReliableComputed(bool computed);
  static bool GetAllGraphInputsReliableComputed();

  static void SetShape(
      const std::string& tensorName,
      const c10::SymbolicShape& shapeValue);
  static bool HasShape(const std::string& tensorName);
  static std::optional<c10::SymbolicShape> GetShape(
      const std::string& tensorName);

  static void SetValue(const std::string& tensorName, const at::Tensor& value);
  static bool HasValue(const std::string& tensorName);
  static std::optional<at::Tensor> GetValue(const std::string& tensorName);
  static void EraseValue(const std::string& tensorName);

  static std::vector<int64_t> GetCompleteShapeInto1DInt64Vector(
      const c10::SymbolicShape& shape);
  static std::optional<std::vector<int64_t>> GetShapeInto1DInt64Vector(
      const std::string& value_name);
  static std::optional<std::vector<int64_t>>
  GetShapeInto1DInt64VectorWithOneUnknown(const std::string& value_name);
  static std::vector<int64_t> GetValueInto1DInt64Vector(
      const std::string& value_name);

  static void SetTypeReliable(const std::string& tensorName, bool reliable);
  static bool HasTypeReliable(const std::string& tensorName);
  static std::optional<bool> GetTypeReliable(const std::string& tensorName);

  static void SetUseInferredType(
      const std::string& tensorName,
      bool useInferredType);
  static bool HasUseInferredType(const std::string& tensorName);
  static std::optional<bool> GetUseInferredType(const std::string& tensorName);

  static void SetShapeValue(
      const std::string& tensorName,
      const c10::SymbolicShape& shapeValue);
  static bool HasShapeValue(const std::string& tensorName);
  static std::optional<c10::SymbolicShape> GetShapeValue(
      const std::string& tensorName);

  static ShapeDataMap& GetInferredShapeData();

  static SymbolDimMap& GetSymbolDimMap();
  static DimSymbolMap& GetDimSymbolMap();

  static void UpdateValueName(
      const std::string& old_name,
      const std::string& new_name);

  static void PrintMaps();
  static void ClearMaps();
  ~ConstantValueMap() = default;

  ConstantValueMap& operator=(const ConstantValueMap&) = delete;

 private:
  ConstantValueMap() = default;

  std::unordered_map<std::string, size_t> rankMap;
  std::unordered_map<std::string, c10::SymbolicShape> shapeMap;
  std::unordered_map<std::string, at::Tensor> tensorValueMap;
  // This map indicates whether the current type is reliably estimated or not.
  std::unordered_map<std::string, bool> typeReliableMap;
  // This map indicates whether the current type is estimated through inference
  // or tracer.
  std::unordered_map<std::string, bool> useInferredTypeMap;
  // This map indicates a tensor value which represents a shape.
  // We assume that the rank of the tensor value <= 1, and we ensure this when
  // we write the processing logic for the operators. When the rank > 1, we
  // should be able to rewrite the model so that the rank <= 1. The difference
  // between shapeMap and shapeValueMap: shapeMap stores the shape of the tensor
  // from a node. shapeValueMap stores the value of the tensor from a node when
  // this tensor represents a shape.
  std::unordered_map<std::string, c10::SymbolicShape> shapeValueMap;
  // Stores earlier data propagation results so that they are accessible
  // during future node-level shape inference.
  ShapeDataMap inferredShapeData;
  SymbolDimMap symbolDimMap;
  DimSymbolMap dimSymbolMap;
  // Stores if all graph-level inputs have static shape
  std::optional<bool> allGraphInputsStatic;
  // True if reliable has been computed for all graph inputs
  bool allGraphInputsReliableComputed{};
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ConstantValueMap`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Macros.h`
- `onnx/shape_inference/implementation.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/serialization/export.h`
- `unordered_map`


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

Files in the same folder (`torch/csrc/jit/passes/onnx`):

- [`remove_inplace_ops_for_onnx.cpp_docs.md`](./remove_inplace_ops_for_onnx.cpp_docs.md)
- [`list_model_parameters.cpp_docs.md`](./list_model_parameters.cpp_docs.md)
- [`preprocess_for_onnx.h_docs.md`](./preprocess_for_onnx.h_docs.md)
- [`remove_inplace_ops_for_onnx.h_docs.md`](./remove_inplace_ops_for_onnx.h_docs.md)
- [`constant_fold.cpp_docs.md`](./constant_fold.cpp_docs.md)
- [`eliminate_unused_items.cpp_docs.md`](./eliminate_unused_items.cpp_docs.md)
- [`cast_all_constant_to_floating.h_docs.md`](./cast_all_constant_to_floating.h_docs.md)
- [`list_model_parameters.h_docs.md`](./list_model_parameters.h_docs.md)
- [`shape_type_inference.cpp_docs.md`](./shape_type_inference.cpp_docs.md)
- [`constant_map.cpp_docs.md`](./constant_map.cpp_docs.md)


## Cross-References

- **File Documentation**: `constant_map.h_docs.md`
- **Keyword Index**: `constant_map.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes/onnx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/passes/onnx`):

- [`constant_map.cpp_kw.md_docs.md`](./constant_map.cpp_kw.md_docs.md)
- [`deduplicate_initializers.h_docs.md_docs.md`](./deduplicate_initializers.h_docs.md_docs.md)
- [`shape_type_inference.h_docs.md_docs.md`](./shape_type_inference.h_docs.md_docs.md)
- [`function_substitution.h_kw.md_docs.md`](./function_substitution.h_kw.md_docs.md)
- [`eliminate_unused_items.h_kw.md_docs.md`](./eliminate_unused_items.h_kw.md_docs.md)
- [`prepare_division_for_onnx.cpp_docs.md_docs.md`](./prepare_division_for_onnx.cpp_docs.md_docs.md)
- [`fixup_onnx_controlflow.cpp_kw.md_docs.md`](./fixup_onnx_controlflow.cpp_kw.md_docs.md)
- [`constant_fold.cpp_docs.md_docs.md`](./constant_fold.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`onnx_log.cpp_kw.md_docs.md`](./onnx_log.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `constant_map.h_docs.md_docs.md`
- **Keyword Index**: `constant_map.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
