# Documentation: `docs/torch/csrc/lazy/ts_backend/ir_builder.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/ts_backend/ir_builder.h_docs.md`
- **Size**: 5,157 bytes (5.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/ts_backend/ir_builder.h`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/ir_builder.h`
- **Size**: 2,391 bytes (2.33 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/lazy/core/internal_ops/ltc_ops.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_builder.h>
#include <torch/csrc/lazy/core/shape_inference.h>
#include <torch/csrc/lazy/generated/LazyNonNativeIr.h>
#include <torch/csrc/lazy/ts_backend/dynamic_ir.h>
#include <torch/csrc/lazy/ts_backend/ops/device_data.h>
#include <torch/csrc/lazy/ts_backend/ops/generic.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch::lazy {

struct TorchScriptIrBuilder : IrBuilder {
  NodePtr MakeDeviceData(
      const std::shared_ptr<BackendData>& data) const override {
    return DeviceData::Create(data);
  }
  // TODO: Scalar node is not currently used by ts_backend. Enable reusing
  // Scalar node later if needed.
  NodePtr MakeScalar(const at::Scalar& value, const at::ScalarType& type)
      const override {
    return MakeNode<Scalar>(value, type);
  }
  NodePtr MakeExpand(
      const Value& input0,
      const std::vector<int64_t>& size,
      const bool& is_scalar_expand) const override {
    return ReuseOrMakeNode<Expand>(input0, size, is_scalar_expand);
  }
  NodePtr MakeCast(
      const Value& input0,
      const at::ScalarType& dtype,
      const std::optional<at::ScalarType>& stype =
          std::nullopt) const override {
    return ReuseOrMakeNode<Cast>(input0, dtype, stype);
  }
  NodePtr MakeTensorList(const OpList& inputs) const override {
    return ReuseOrMakeNode<TensorList>(inputs);
  }
  // Generic needs cleanup
  NodePtr MakeGeneric(
      const OpKind& op,
      const OpList& operands,
      const Shape& shape,
      const size_t& num_outputs = 1,
      const hash_t& hash_seed =
          static_cast<uint32_t>(0x5a2d296e9)) const override {
    return MakeNode<Generic>(op, operands, shape, num_outputs, hash_seed);
  }

  // dynamic ir nodes
  // TODO: verify if IR node reusing works for Dynamic shape ops
  NodePtr MakeSizeNode(const Value& input, size_t dim) const override {
    return MakeNode<SizeNode>(input, dim);
  }
  NodePtr MakeSizeAdd(const Value& a, const Value& b) const override {
    return MakeNode<SizeAdd>(a, b);
  }
  NodePtr MakeSizeMul(const Value& a, const Value& b) const override {
    return MakeNode<SizeMul>(a, b);
  }
  NodePtr MakeSizeDiv(const Value& a, const Value& b) const override {
    return MakeNode<SizeDiv>(a, b);
  }
};

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TorchScriptIrBuilder`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/core/internal_ops/ltc_ops.h`
- `torch/csrc/lazy/core/ir.h`
- `torch/csrc/lazy/core/ir_builder.h`
- `torch/csrc/lazy/core/shape_inference.h`
- `torch/csrc/lazy/generated/LazyNonNativeIr.h`
- `torch/csrc/lazy/ts_backend/dynamic_ir.h`
- `torch/csrc/lazy/ts_backend/ops/device_data.h`
- `torch/csrc/lazy/ts_backend/ops/generic.h`
- `torch/csrc/lazy/ts_backend/ts_node.h`


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

- **File Documentation**: `ir_builder.h_docs.md`
- **Keyword Index**: `ir_builder.h_kw.md`
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
- [`ts_node.cpp_docs.md_docs.md`](./ts_node.cpp_docs.md_docs.md)
- [`ts_lowering_context.h_kw.md_docs.md`](./ts_lowering_context.h_kw.md_docs.md)
- [`ts_lowering_context.cpp_kw.md_docs.md`](./ts_lowering_context.cpp_kw.md_docs.md)
- [`tensor_aten_ops.cpp_kw.md_docs.md`](./tensor_aten_ops.cpp_kw.md_docs.md)
- [`tensor_aten_ops.cpp_docs.md_docs.md`](./tensor_aten_ops.cpp_docs.md_docs.md)
- [`config.h_docs.md_docs.md`](./config.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ir_builder.h_docs.md_docs.md`
- **Keyword Index**: `ir_builder.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
