# Documentation: `torch/csrc/lazy/ts_backend/ts_node.h`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/ts_node.h`
- **Size**: 3,351 bytes (3.27 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>

namespace torch::lazy {

using TSOpVector = std::vector<torch::jit::Value*>;

class TORCH_API TsNode : public lazy::Node {
 public:
  TsNode(
      OpKind op,
      OpList operands,
      std::vector<Shape>&& shapes,
      size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  TsNode(
      OpKind op,
      OpList operands,
      const std::function<Shape()>& shape_fn,
      size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  TsNode(
      OpKind op,
      OpList operands,
      size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  TsNode(
      OpKind op,
      Shape shape,
      size_t num_outputs,
      hash_t hash_seed = kHashSeed);

  ~TsNode() override = default;

  hash_t hash() const override;

  hash_t shapeHash() const override;

  const std::string getPythonStacktrace() const;

  // Lower is a backend-specific method since it returns a backend specific
  // type. hence, it is convenient to define it differently per-backend rather
  // than at Node API
  virtual TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const;

 private:
  // The hash of the dag WITH size info. Used for shape caching
  hash_t shape_hash_;
  // The hash of the dag used to look up the compiled graph by a hash
  // in this case, we will use the dag hash WITHOUT size info if dynamic shape
  // is enabled and use the dag hash WITH size info otherwise.
  hash_t dag_hash_;
};

// Note: this OpKind is separate from ltc_ops.h since it would be a circular
// import otherwise, I like leaving TensorList in this file, and I think most of
// ltc_ops special cases will be deleted anyway
const OpKind tensor_list_opkind = OpKind::Get("lazy_tensors::tensor_list");

// TensorList represents an at::TensorList which is a vector[Tensor] but is also
// a first-class IValue and can be fed as a single input to a TS program.  It is
// much easier to handle TensorLists in Lazy Tensor code if they are represented
// as a single Node so there can be more than one TensorList and more than one
// Tensor side-by-side as operands to an op.
//
// Note: shape is undefined for TensorList.  We assert in some places that
// #shapes matches #outputs and this stems from
//       the fact that currently all IR nodes represent tensors (there is no
//       type system for this IR).  Because of this, TensorList is a bit of a
//       hack.
//
// TODO(whc) once Shape() API is moved to Node base, also make it virtual, and
// then implement it as NotImplemented for TensorList, also fixing the assertion
// that would fail.
struct TORCH_API TensorList : public TsNode {
  static OpKind ClassOpKind() {
    return tensor_list_opkind;
  }

  TensorList() = delete;
  TensorList(OpList values);

  bool CanBeReused(OpList values) const {
    return operands() == std::vector<Output>(values.begin(), values.end());
  }

  TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override;
};

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `IValue`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/ArrayRef.h`
- `torch/csrc/jit/api/function_impl.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/lazy/backend/lowering_context.h`
- `torch/csrc/lazy/core/ir.h`
- `torch/csrc/lazy/core/shape.h`
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

- **File Documentation**: `ts_node.h_docs.md`
- **Keyword Index**: `ts_node.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
