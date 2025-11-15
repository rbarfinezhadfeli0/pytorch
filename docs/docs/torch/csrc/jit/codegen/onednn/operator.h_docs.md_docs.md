# Documentation: `docs/torch/csrc/jit/codegen/onednn/operator.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/codegen/onednn/operator.h_docs.md`
- **Size**: 6,445 bytes (6.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/codegen/onednn/operator.h`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/onednn/operator.h`
- **Size**: 3,922 bytes (3.83 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit::fuser::onednn {

class Operator {
 public:
  Operator(const Node* node, dnnl::graph::op::kind kind)
      : n(node), o(getId(node), kind, node->kind().toQualString()), k(kind) {}

  // Returns output index if the Value is a graph output.
  // Otherwise returns -1
  int32_t graphOutputIdx(Value* v) {
    int32_t i = 0;
    for (const Value* output : v->owningGraph()->outputs()) {
      if (v == output) {
        return i;
      }
      i++;
    }
    return -1;
  }

  Operator& setInputValue(Value* v) {
    if (v->mustNotBeNone()) {
      if (v->type()->kind() == c10::TensorType::Kind) {
        o.add_input(createLogicalTensor(v));
      }
    }
    return *this;
  }

  Operator& setInput(size_t offset) {
    return setInputValue(n->input(offset));
  }

  template <typename... Ts>
  Operator& setInput(size_t offset, Ts... other) {
    setInput(offset);
    return setInput(other...);
  }

  Operator& setOutputValue(Value* v) {
    if (v->mustNotBeNone()) {
      o.add_output(createLogicalTensor(v));
    }
    return *this;
  }

  // setOutputValue & setOutput require a pointer to the LLGA graph, as output
  // logical tensors that are graph outputs should be connected to an End LLGA
  // op. A value of NULL can be provided for the graph pointer in order to
  // maintain the legacy functionality of this function.
  Operator& setOutputValue(Value* v, std::unique_ptr<dnnl::graph::graph>& g) {
    if (v->mustNotBeNone()) {
      auto output_tensor = createLogicalTensor(v);
      o.add_output(output_tensor);
      if (g) {
        int32_t outputIndex = graphOutputIdx(v);
        if (outputIndex != -1) {
          dnnl::graph::op newEndNode(
              LONG_MAX - outputIndex,
              dnnl::graph::op::kind::End,
              "EndNodeForGraphOutput");
          newEndNode.add_input(output_tensor);
          g->add_op(newEndNode);
        }
      }
    }
    return *this;
  }

  Operator& setOutput(std::unique_ptr<dnnl::graph::graph>& g, size_t offset) {
    return setOutputValue(n->output(offset), g);
  }

  Operator& setOutput(size_t offset) {
    return setOutputValue(n->output(offset));
  }

  template <typename... Ts>
  Operator& setOutput(
      std::unique_ptr<dnnl::graph::graph>& g,
      size_t offset,
      Ts... other) {
    setOutput(g, offset);
    return setOutput(g, other...);
  }

  template <typename Attr>
  Operator& setAttr(dnnl::graph::op::attr name, Attr&& attr) {
    o.set_attr(name, std::forward<Attr>(attr));
    return *this;
  }

  template <typename F>
  Operator& setAttr(dnnl::graph::op::attr name, const F& fn, size_t offset) {
    return setAttr(name, fn(n, offset));
  }

  static float ScalarToFloat(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toScalar().to<float>();
  }

  static std::vector<int64_t> Ints(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toIntVector();
  }

  static int64_t Int(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toInt();
  }

  static float Float(const Node* node, size_t offset) {
    return static_cast<float>(toIValue(node->input(offset))->toDouble());
  }

  static bool Bool(const Node* node, size_t offset) {
    return toIValue(node->input(offset))->toBool();
  }

  static uint64_t getId(const Node* node) {
    return reinterpret_cast<uint64_t>(node); // cast node address as op id
  }

  dnnl::graph::op::kind kind() const {
    return k;
  }

  dnnl::graph::op llgaOp() const {
    return o;
  }

 private:
  dnnl::graph::logical_tensor createLogicalTensor(Value* value) const {
    return LlgaTensorDesc(value).logical_tensor();
  }

  const Node* n;
  dnnl::graph::op o;
  dnnl::graph::op::kind k;
};

} // namespace torch::jit::fuser::onednn

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Operator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `oneapi/dnnl/dnnl_graph.hpp`
- `torch/csrc/jit/codegen/onednn/LlgaTensorImpl.h`
- `torch/csrc/jit/ir/ir.h`


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

Files in the same folder (`torch/csrc/jit/codegen/onednn`):

- [`graph_rewriter.cpp_docs.md`](./graph_rewriter.cpp_docs.md)
- [`guard_shape.cpp_docs.md`](./guard_shape.cpp_docs.md)
- [`prepare_binary.h_docs.md`](./prepare_binary.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`graph_fuser.h_docs.md`](./graph_fuser.h_docs.md)
- [`kernel.h_docs.md`](./kernel.h_docs.md)
- [`decompose_silu.cpp_docs.md`](./decompose_silu.cpp_docs.md)
- [`prepare_binary.cpp_docs.md`](./prepare_binary.cpp_docs.md)
- [`graph_helper.cpp_docs.md`](./graph_helper.cpp_docs.md)
- [`register_interface.cpp_docs.md`](./register_interface.cpp_docs.md)


## Cross-References

- **File Documentation**: `operator.h_docs.md`
- **Keyword Index**: `operator.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/codegen/onednn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/codegen/onednn`):

- [`graph_rewriter.cpp_docs.md_docs.md`](./graph_rewriter.cpp_docs.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)
- [`decompose_silu.cpp_kw.md_docs.md`](./decompose_silu.cpp_kw.md_docs.md)
- [`defer_size_check.h_kw.md_docs.md`](./defer_size_check.h_kw.md_docs.md)
- [`graph_fuser.h_kw.md_docs.md`](./graph_fuser.h_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`graph_fuser.h_docs.md_docs.md`](./graph_fuser.h_docs.md_docs.md)
- [`interface.h_docs.md_docs.md`](./interface.h_docs.md_docs.md)
- [`layout_propagation.h_kw.md_docs.md`](./layout_propagation.h_kw.md_docs.md)
- [`graph_helper.cpp_kw.md_docs.md`](./graph_helper.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `operator.h_docs.md_docs.md`
- **Keyword Index**: `operator.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
