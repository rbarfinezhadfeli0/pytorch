# Documentation: `docs/torch/csrc/lazy/ts_backend/dynamic_ir.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/ts_backend/dynamic_ir.cpp_docs.md`
- **Size**: 5,628 bytes (5.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/ts_backend/dynamic_ir.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/dynamic_ir.cpp`
- **Size**: 3,221 bytes (3.15 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>

#include <utility>

static const torch::lazy::DimensionNode* DimCast(torch::lazy::Output output) {
  return dynamic_cast<const torch::lazy::DimensionNode*>(output.node);
}

namespace torch::lazy {

TSOpVector SizeNode::Lower(
    std::shared_ptr<torch::jit::GraphFunction> function,
    TSLoweringContext* loctx) const {
  std::vector<torch::jit::NamedValue> arguments;
  std::vector<torch::jit::NamedValue> kwarguments;
  arguments.reserve(2);
  auto index = loctx->graph()->insertConstant(static_cast<int64_t>(this->dim_));
  arguments.emplace_back(loctx->GetOutputOp(operand(0)));
  arguments.emplace_back(index);
  torch::lazy::TSOpVector size_out =
      torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
  TORCH_CHECK_EQ(size_out.size(), 1);
  return size_out;
}

SizeNode::SizeNode(Value input, size_t dim)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::size")},
          {std::move(input)},
          std::vector<Shape>{},
          1,
          MHash(dim)),
      dim_(dim) {}

int64_t SizeNode::getStaticValue() const {
  return dynamic_cast<const TsNode*>(operand(0).node)
      ->shape(0)
      .size(static_cast<int64_t>(dim_));
}
bool SizeNode::isSymbolic() const {
  auto symbolic_vec =
      dynamic_cast<const TsNode*>(operand(0).node)->shape(0).is_symbolic();
  if (!symbolic_vec.has_value()) {
    return true;
  }
  return symbolic_vec->at(dim_);
}

std::string SizeNode::ToString() const {
  return "SizeNode";
}

SizeAdd::SizeAdd(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::add")},
          {std::move(a), std::move(b)},
          std::vector<Shape>{},
          1) {}

int64_t SizeAdd::getStaticValue() const {
  return DimCast(operand(0))->getStaticValue() +
      DimCast(operand(1))->getStaticValue();
}

bool SizeAdd::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeAdd::ToString() const {
  return "SizeAdd";
}

SizeMul::SizeMul(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::mul")},
          {std::move(a), std::move(b)},
          std::vector<Shape>{},
          1) {}

int64_t SizeMul::getStaticValue() const {
  return DimCast(operand(0))->getStaticValue() *
      DimCast(operand(1))->getStaticValue();
}

bool SizeMul::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeMul::ToString() const {
  return "SizeMul";
}

SizeDiv::SizeDiv(Value a, Value b)
    : TsNode(
          OpKind{c10::Symbol::fromQualString("aten::div")},
          {std::move(a), std::move(b)},
          std::vector<Shape>{},
          1) {}

int64_t SizeDiv::getStaticValue() const {
  TORCH_CHECK(
      DimCast(operand(1))->getStaticValue() != 0,
      "Can't divide a dimension by zero");
  return DimCast(operand(0))->getStaticValue() /
      DimCast(operand(1))->getStaticValue();
}

bool SizeDiv::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeDiv::ToString() const {
  return "SizeDiv";
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

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

- `torch/csrc/lazy/ts_backend/dynamic_ir.h`
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
- [`ts_backend_impl.h_docs.md`](./ts_backend_impl.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)
- [`ts_autograd_functions.cpp_docs.md`](./ts_autograd_functions.cpp_docs.md)
- [`ts_eager_fallback.h_docs.md`](./ts_eager_fallback.h_docs.md)
- [`dynamic_ir.h_docs.md`](./dynamic_ir.h_docs.md)
- [`tensor_aten_ops.cpp_docs.md`](./tensor_aten_ops.cpp_docs.md)
- [`tensor_aten_ops.h_docs.md`](./tensor_aten_ops.h_docs.md)
- [`ts_lowering_context.cpp_docs.md`](./ts_lowering_context.cpp_docs.md)


## Cross-References

- **File Documentation**: `dynamic_ir.cpp_docs.md`
- **Keyword Index**: `dynamic_ir.cpp_kw.md`
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

- **File Documentation**: `dynamic_ir.cpp_docs.md_docs.md`
- **Keyword Index**: `dynamic_ir.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
