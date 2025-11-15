# Documentation: `docs/torch/csrc/lazy/ts_backend/ts_node_lowering.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/ts_backend/ts_node_lowering.cpp_docs.md`
- **Size**: 8,066 bytes (7.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/ts_backend/ts_node_lowering.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/ts_node_lowering.cpp`
- **Size**: 5,098 bytes (4.98 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/lazy/ts_backend/ts_node_lowering.h>

#include <ATen/Functions.h>
#include <torch/csrc/jit/frontend/sugared_value.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#include <torch/csrc/lazy/core/permutation_util.h>
#include <torch/csrc/lazy/ts_backend/ir_builder.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch::lazy {

static TSOpVector LowerBuiltin(
    const torch::lazy::Node* node,
    const std::shared_ptr<torch::jit::GraphFunction>& function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  return LowerTSBuiltin(function, node->op().op, arguments, kwarguments);
}
static TSOpVector LowerBuiltin(
    c10::Symbol sym,
    const std::shared_ptr<torch::jit::GraphFunction>& function,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {}) {
  return LowerTSBuiltin(function, sym, arguments, kwarguments);
}

TSOpVector LowerTSBuiltin(
    const std::shared_ptr<torch::jit::GraphFunction>& function,
    c10::Symbol sym,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments) {
  auto builtin =
      std::make_shared<torch::jit::BuiltinFunction>(sym, std::nullopt);
  auto magic_method = std::make_shared<torch::jit::MagicMethod>("", builtin);
  auto ret = magic_method->call({}, *function, arguments, kwarguments, 0);
  auto& sv = dynamic_cast<torch::jit::SimpleValue&>(*ret);
  if (sv.getValue()->type()->kind() == c10::TypeKind::TupleType) {
    const auto tuple_call_result = sv.asTuple({}, *function);
    TSOpVector tuple_result;
    for (const auto& tuple_component : tuple_call_result) {
      auto tuple_component_sv =
          dynamic_cast<torch::jit::SimpleValue*>(tuple_component.get());
      tuple_result.push_back(tuple_component_sv->getValue());
    }
    return tuple_result;
  }
  return {sv.getValue()};
}

static torch::jit::Value* GenerateClone(
    torch::jit::Value* val,
    const std::shared_ptr<torch::jit::GraphFunction>& function) {
  std::vector<torch::jit::NamedValue> clone_arguments;
  clone_arguments.emplace_back(val);
  TSOpVector cloned = LowerBuiltin(at::aten::clone, function, clone_arguments);
  TORCH_CHECK_EQ(cloned.size(), 1);
  return cloned.front();
}

// Node Lowerings

// Default node lowering
TSOpVector TsNode::Lower(
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  for (const torch::lazy::Output& output : operands()) {
    arguments.emplace_back(loctx->GetOutputOp(output));
  }
  return LowerBuiltin(this, function, arguments);
}

// Non-native ops
torch::lazy::TSOpVector Cast::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(dtype);
  return LowerBuiltin(at::aten::to, function, arguments);
}

torch::lazy::TSOpVector DeviceData::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  auto infoptr = data_->info();
  auto deviceDataInfoPtr =
      (torch::lazy::LazyGraphExecutor::DeviceDataInfo*)infoptr;
  if (GRAPH_DUMP_ENABLED) {
    LOG(ERROR) << "Lowering device data node, tensor id "
               << deviceDataInfoPtr->tensor_id << '\n';
  }
  return {loctx->GetParameter(data_)};
}

torch::lazy::TSOpVector Expand::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(size);
  auto expand_out = LowerBuiltin(this, function, arguments);
  if (is_scalar_expand) {
    // The aten::expand operations sets all strides to 0 when the original is
    // of rank 0. This leads to false positives when checking for internal
    // memory overlap, because at::has_internal_overlap returns
    // MemOverlap::YES when a stride is set to 0.
    TORCH_CHECK_EQ(expand_out.size(), 1);
    return {GenerateClone(expand_out.front(), function)};
  }
  return expand_out;
}

torch::lazy::TSOpVector Scalar::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    torch::lazy::TSLoweringContext* loctx) const {
  auto options =
      at::TensorOptions()
          .device(torch::lazy::getBackend()->EagerFallbackDeviceType())
          .dtype(shape().scalar_type());
  return {loctx->graph()->insertConstant(at::scalar_tensor(value, options))};
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

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

- `torch/csrc/lazy/ts_backend/ts_node_lowering.h`
- `ATen/Functions.h`
- `torch/csrc/jit/frontend/sugared_value.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/lazy/backend/backend_interface.h`
- `torch/csrc/lazy/core/helpers.h`
- `torch/csrc/lazy/core/internal_ops/ltc_ops.h`
- `torch/csrc/lazy/core/ir_builder.h`
- `torch/csrc/lazy/core/lazy_graph_executor.h`
- `torch/csrc/lazy/core/ops/utils.h`
- `torch/csrc/lazy/core/permutation_util.h`
- `torch/csrc/lazy/ts_backend/ir_builder.h`
- `torch/csrc/lazy/ts_backend/ts_lowering_context.h`


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
- [`ts_lowering_context.cpp_docs.md`](./ts_lowering_context.cpp_docs.md)


## Cross-References

- **File Documentation**: `ts_node_lowering.cpp_docs.md`
- **Keyword Index**: `ts_node_lowering.cpp_kw.md`
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

- **File Documentation**: `ts_node_lowering.cpp_docs.md_docs.md`
- **Keyword Index**: `ts_node_lowering.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
