# Documentation: `docs/torch/csrc/lazy/ts_backend/ts_lowering_context.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/ts_backend/ts_lowering_context.cpp_docs.md`
- **Size**: 5,705 bytes (5.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/ts_backend/ts_lowering_context.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/ts_lowering_context.cpp`
- **Size**: 3,130 bytes (3.06 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <utility>

namespace torch::lazy {

TSLoweringContext::TSLoweringContext(
    const std::string& name,
    BackendDevice device)
    : torch::lazy::LoweringContext(name, std::move(device)),
      graph_(std::make_shared<torch::jit::Graph>()),
      function_(
          std::make_shared<torch::jit::GraphFunction>(name, graph_, nullptr)) {}

TSLoweringContext::TSLoweringContext(
    const std::string& name,
    BackendDevice device,
    c10::ArrayRef<const Node*> post_order,
    Util::EmissionMap emit_status)
    : torch::lazy::LoweringContext(
          name,
          std::move(device),
          post_order,
          std::move(emit_status)),
      graph_(std::make_shared<torch::jit::Graph>()),
      function_(
          std::make_shared<torch::jit::GraphFunction>(name, graph_, nullptr)) {
  for (auto node : post_order) {
    Lower(node);
  }
}

void TSLoweringContext::Lower(const Node* node) {
  if (auto* tsnode = dynamic_cast<const torch::lazy::TsNode*>(node)) {
    // First, we call the node lowering function, which exists for newly
    // codegenned or refactored nodes
    TSOpVector ops = tsnode->Lower(function_, this);
    TORCH_CHECK(!ops.empty(), "Failed to lower: ", *node);
    TORCH_CHECK_EQ(node->num_outputs(), ops.size());
    for (size_t i = 0; i < ops.size(); ++i) {
      AssignOutputOp(torch::lazy::Output(node, i), ops[i]);
    }
  } else {
    TORCH_CHECK(
        false, "Expected torch::lazy::TsNode but could not dynamic cast");
  }
}

void TSLoweringContext::AssignOutputOp(
    const Output& output,
    torch::jit::Value* op) {
  const TsNode* ts_node = static_cast<const TsNode*>(output.node);
  std::string stack_trace = ts_node->getPythonStacktrace();
  if (!stack_trace.empty()) {
    op->node()->s_(c10::Symbol::attr("source"), stack_trace);
  }
  emitted_outputs_[output] = op;
}

torch::jit::Value* TSLoweringContext::GetParameter(const BackendDataPtr& data) {
  const auto ts_data = std::static_pointer_cast<TSData>(data);
  BackendData::Handle handle = ts_data->GetHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    torch::jit::Value* param =
        graph_->addInput(c10::str("p", parameters_.size()));
    const auto& scalar = ts_data->scalar;
    if (scalar.has_value()) {
      auto scalarType = scalar.value().type();
      if (isFloatingType(scalarType)) {
        param->setType(c10::FloatType::get());
      } else if (isIntegralType(scalarType, /*includeBool=*/true)) {
        param->setType(c10::IntType::get());
      } else {
        TORCH_CHECK(
            false, "Unhandled scalar type: ", c10::toString(scalarType));
      }
    }
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(ts_data);
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/ScalarType.h`
- `c10/util/Exception.h`
- `torch/csrc/lazy/ts_backend/ts_backend_impl.h`
- `torch/csrc/lazy/ts_backend/ts_lowering_context.h`
- `torch/csrc/lazy/ts_backend/ts_node.h`
- `utility`


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

Files in the same folder (`torch/csrc/lazy/ts_backend`):

- [`ts_node.h_docs.md`](./ts_node.h_docs.md)
- [`dynamic_ir.cpp_docs.md`](./dynamic_ir.cpp_docs.md)
- [`ts_backend_impl.h_docs.md`](./ts_backend_impl.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)
- [`ts_autograd_functions.cpp_docs.md`](./ts_autograd_functions.cpp_docs.md)
- [`ts_eager_fallback.h_docs.md`](./ts_eager_fallback.h_docs.md)
- [`dynamic_ir.h_docs.md`](./dynamic_ir.h_docs.md)
- [`tensor_aten_ops.cpp_docs.md`](./tensor_aten_ops.cpp_docs.md)
- [`tensor_aten_ops.h_docs.md`](./tensor_aten_ops.h_docs.md)


## Cross-References

- **File Documentation**: `ts_lowering_context.cpp_docs.md`
- **Keyword Index**: `ts_lowering_context.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/ts_backend`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/lazy/ts_backend`):

- [`ts_native_functions.cpp_kw.md_docs.md`](./ts_native_functions.cpp_kw.md_docs.md)
- [`ts_native_functions.cpp_docs.md_docs.md`](./ts_native_functions.cpp_docs.md_docs.md)
- [`ts_autograd_functions.cpp_docs.md_docs.md`](./ts_autograd_functions.cpp_docs.md_docs.md)
- [`ir_builder.h_docs.md_docs.md`](./ir_builder.h_docs.md_docs.md)
- [`ts_node.cpp_docs.md_docs.md`](./ts_node.cpp_docs.md_docs.md)
- [`ts_lowering_context.h_kw.md_docs.md`](./ts_lowering_context.h_kw.md_docs.md)
- [`ts_lowering_context.cpp_kw.md_docs.md`](./ts_lowering_context.cpp_kw.md_docs.md)
- [`tensor_aten_ops.cpp_kw.md_docs.md`](./tensor_aten_ops.cpp_kw.md_docs.md)
- [`tensor_aten_ops.cpp_docs.md_docs.md`](./tensor_aten_ops.cpp_docs.md_docs.md)
- [`config.h_docs.md_docs.md`](./config.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ts_lowering_context.cpp_docs.md_docs.md`
- **Keyword Index**: `ts_lowering_context.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
