# Documentation: `docs/torch/csrc/lazy/ts_backend/dynamic_ir.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/ts_backend/dynamic_ir.h_docs.md`
- **Size**: 5,061 bytes (4.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/ts_backend/dynamic_ir.h`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/dynamic_ir.h`
- **Size**: 2,356 bytes (2.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/symbol.h>

#include <memory>
#include <string>

#include <c10/core/ScalarType.h>
#include <c10/util/Flags.h>
#include <torch/csrc/lazy/core/dynamic_ir.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/ts_backend/ts_node.h>

TORCH_DECLARE_bool(ltc_enable_dynamic_shapes);

namespace torch::lazy {

/**
 * The goal of "dynamic" Nodes is to patch a hole in our tracing.
 * Previously, if a user called `sizes` on a Tensor, it would leak out
 * of our tracing system, as `sizes` returns a torch.Size or an int. To
 * prevent this from happening, we introduce DimensionNode, a new type
 * of Node that abstracts the operation of getting the dimensions of a
 * Tensor.
 *
 * Consider the following example:
 * ```
 * numel = x.shape()[0] * x.shape()[1]
 * ```
 *
 * Here, `x.shape()[i]` will be a SizeNode (subclass of DimensionNode),
 * and the multiplication of the two SizeNodes will be represented by
 * a SizeMul (also a subclass of DimensionNode). Through this, we can
 * prevent `numel` from being represented as a Python int and thus
 * burned into the Graph.
 */

// Represents the result of calling `size` on a Tensor
class TORCH_API SizeNode : public TsNode, public DimensionNode {
 public:
  SizeNode(Value input, size_t dim);
  int64_t getStaticValue() const override;
  bool isSymbolic() const override;
  std::string ToString() const override;
  size_t dim_ = 0;
  torch::lazy::TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override;
};

class TORCH_API SizeAdd : public TsNode, public DimensionNode {
 public:
  SizeAdd(Value a, Value b);
  int64_t getStaticValue() const override;
  bool isSymbolic() const override;
  std::string ToString() const override;
};

class TORCH_API SizeMul : public TsNode, public DimensionNode {
 public:
  SizeMul(Value a, Value b);
  int64_t getStaticValue() const override;
  bool isSymbolic() const override;
  std::string ToString() const override;
};

class TORCH_API SizeDiv : public TsNode, public DimensionNode {
 public:
  SizeDiv(Value a, Value b);
  int64_t getStaticValue() const override;
  bool isSymbolic() const override;
  std::string ToString() const override;
};

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 6 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `of`, `of`, `TORCH_API`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/symbol.h`
- `memory`
- `string`
- `c10/core/ScalarType.h`
- `c10/util/Flags.h`
- `torch/csrc/lazy/core/dynamic_ir.h`
- `torch/csrc/lazy/core/hash.h`
- `torch/csrc/lazy/core/ir.h`
- `torch/csrc/lazy/core/ir_metadata.h`
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
- [`tensor_aten_ops.cpp_docs.md`](./tensor_aten_ops.cpp_docs.md)
- [`tensor_aten_ops.h_docs.md`](./tensor_aten_ops.h_docs.md)
- [`ts_lowering_context.cpp_docs.md`](./ts_lowering_context.cpp_docs.md)


## Cross-References

- **File Documentation**: `dynamic_ir.h_docs.md`
- **Keyword Index**: `dynamic_ir.h_kw.md`
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

- **File Documentation**: `dynamic_ir.h_docs.md_docs.md`
- **Keyword Index**: `dynamic_ir.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
