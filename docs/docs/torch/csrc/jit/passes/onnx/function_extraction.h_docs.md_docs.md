# Documentation: `docs/torch/csrc/jit/passes/onnx/function_extraction.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/onnx/function_extraction.h_docs.md`
- **Size**: 4,991 bytes (4.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/onnx/function_extraction.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/onnx/function_extraction.h`
- **Size**: 2,251 bytes (2.20 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/ir/ir.h>

// This api will be used by serialization/export.cpp to extract function
// information. It should do conversion on graph to
//    1. Extract subgraph pattern of functions and define as local function
//    node.
//    2. Replace subgraph pattern of functions with a single node reflecting
//    that local function node type.
// Function attribute map information is also returned, as Torch IR cannot
// represent these info inside Graph object.
// export.cpp will serialize the ONNX model with function_proto with
// above information.
namespace torch::jit::onnx {

// The following return types are used to track information regarding function
// attributes, that are unable to be traced through Torch IR.
// NodeAttrNameMap tracks mapping from attribute name of IR Node inside function
// subgraph, to function attribute name. Here's an example of exporting CELU and
// LayerNorm.
//
// clang-format off
// class M(torch.nn.Module):
//     def __init__(self) -> None:
//         super().__init__()
//         self.lns = torch.nn.ModuleList([torch.nn.LayerNorm(3, eps = i) for i in range(2)])
//         self.celu1 = torch.nn.CELU(1.0)
//         self.celu2 = torch.nn.CELU(2.0)

//     def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
//         res1 = self.celu1(x)
//         res2 = self.celu2(y)
//         for ln in self.lns:
//             z = ln(z)
//         return res1 + res2 + z
// clang-format on
//
// Returning
//
// NodeAttrNameMap:
// {
//    %1 : Float(2, 3) = onnx::Celu[alpha=2.](%y) : {
//      'alpha' : 'Celu_alpha'
//    }
// }
//
// The info here helps graph._export_onnx to construct function attributes for
// onnx local FunctionProto.
using NodeAttrNameMap = std::
    unordered_map<const Node*, std::unordered_map<std::string, std::string>>;

TORCH_API NodeAttrNameMap ONNXFunctionExtraction(
    std::shared_ptr<Graph>& graph,
    const std::unordered_set<std::string>& module_names,
    const std::vector<std::string>& param_names);

TORCH_API void ONNXClearScopeRecords();

TORCH_API void ONNXTrackScopeAttributes(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& attributes);

} // namespace torch::jit::onnx

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `M`, `function`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/onnx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/ir.h`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `function_extraction.h_docs.md`
- **Keyword Index**: `function_extraction.h_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

- **File Documentation**: `function_extraction.h_docs.md_docs.md`
- **Keyword Index**: `function_extraction.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
