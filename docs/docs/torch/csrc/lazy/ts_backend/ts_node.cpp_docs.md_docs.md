# Documentation: `docs/torch/csrc/lazy/ts_backend/ts_node.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/ts_backend/ts_node.cpp_docs.md`
- **Size**: 5,451 bytes (5.32 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/ts_backend/ts_node.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/ts_node.cpp`
- **Size**: 2,946 bytes (2.88 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/env.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace {
std::string GetFirstUserFrameInPythonIfEnabled() {
  static const auto LTC_ENABLE_SOURCE_INFO =
      c10::utils::has_env("LTC_ENABLE_SOURCE_INFO");
  if (LTC_ENABLE_SOURCE_INFO) {
    return {};
  }

  return torch::lazy::GetFirstUserFrameInPython();
}
} // namespace

namespace torch::lazy {

static hash_t OperandHashes(
    const OpList& operands,
    const c10::ArrayRef<Shape>& shapes,
    const hash_t& seed,
    bool bakeInSizes) {
  hash_t hash = seed;
  for (auto& operand : operands) {
    if (!operand) {
      hash = HashCombine(hash, static_cast<uint64_t>(kNullOpt));
      continue;
    }
    auto operand_hash = bakeInSizes ? operand.shapeHash() : operand.hash();
    hash = HashCombine(hash, operand_hash);
  }
  for (auto& shape : shapes) {
    hash = HashCombine(hash, shape.hash(bakeInSizes));
  }
  return hash;
}

TsNode::TsNode(
    OpKind op,
    OpList operands,
    std::vector<Shape>&& shapes,
    size_t num_outputs,
    hash_t hash_seed)
    : Node(op, operands, std::move(shapes), num_outputs) {
  hash_seed = HashCombine(op.hash(), hash_seed);
  shape_hash_ = OperandHashes(operands, this->shapes(), hash_seed, true);
  dag_hash_ =
      (enableDynamicShape()
           ? OperandHashes(operands, this->shapes(), hash_seed, false)
           : shape_hash_);
}

TsNode::TsNode(
    OpKind op,
    OpList operands,
    const std::function<Shape()>& shape_fn,
    size_t num_outputs,
    hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {
  addComputedShape(shape_fn);
}

TsNode::TsNode(OpKind op, OpList operands, size_t num_outputs, hash_t hash_seed)
    : TsNode(op, operands, std::vector<Shape>{}, num_outputs, hash_seed) {}

TsNode::TsNode(OpKind op, Shape shape, size_t num_outputs, hash_t hash_seed)
    : TsNode(op, {}, {std::move(shape)}, num_outputs, hash_seed) {}

hash_t TsNode::hash() const {
  return dag_hash_;
}

hash_t TsNode::shapeHash() const {
  return shape_hash_;
}

const std::string TsNode::getPythonStacktrace() const {
  return GetFirstUserFrameInPythonIfEnabled();
}

TensorList::TensorList(OpList values)
    : TsNode(
          /*op=*/ClassOpKind(),
          /*operands=*/values,
          /*shapes=*/std::vector<Shape>(),
          /*num_outputs=*/1,
          /*hash_seed=*/kHashSeed) {}

TSOpVector TensorList::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {
  std::vector<torch::jit::Value*> tensor_list;
  TORCH_CHECK(!operands().empty());
  for (const torch::lazy::Output& operand : operands()) {
    tensor_list.emplace_back(loctx->GetOutputOp(operand));
  }
  auto graph = function->graph();
  auto listnode =
      graph->insertNode(graph->createList(tensor_list[0]->type(), tensor_list));
  return {listnode->output()};
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/env.h`
- `torch/csrc/lazy/core/debug_util.h`
- `torch/csrc/lazy/ts_backend/ts_node.h`


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

- **File Documentation**: `ts_node.cpp_docs.md`
- **Keyword Index**: `ts_node.cpp_kw.md`
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
- [`ts_lowering_context.h_kw.md_docs.md`](./ts_lowering_context.h_kw.md_docs.md)
- [`ts_lowering_context.cpp_kw.md_docs.md`](./ts_lowering_context.cpp_kw.md_docs.md)
- [`tensor_aten_ops.cpp_kw.md_docs.md`](./tensor_aten_ops.cpp_kw.md_docs.md)
- [`tensor_aten_ops.cpp_docs.md_docs.md`](./tensor_aten_ops.cpp_docs.md_docs.md)
- [`config.h_docs.md_docs.md`](./config.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ts_node.cpp_docs.md_docs.md`
- **Keyword Index**: `ts_node.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
