# Documentation: `docs/torch/csrc/jit/codegen/onednn/graph_helper.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/codegen/onednn/graph_helper.h_docs.md`
- **Size**: 5,099 bytes (4.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/codegen/onednn/graph_helper.h`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/onednn/graph_helper.h`
- **Size**: 2,464 bytes (2.41 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/codegen/onednn/operator.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit::fuser::onednn {

#define STRIDED_LAYOUT 0
#define OPAQUE_LAYOUT 1

struct OpPartitionMap {
  void add(uint64_t opId, uint64_t partitionId) {
    opmap_[opId] = partitionId;
  }
  void add(Node* n, uint64_t partitionId) {
    add(Operator::getId(n), partitionId);
  }
  bool has(uint64_t opId) {
    return opmap_.count(opId) > 0;
  }
  bool has(Node* n) {
    return has(Operator::getId(n));
  }
  uint64_t get(uint64_t opId) {
    return opmap_[opId];
  }
  uint64_t get(Node* n) {
    auto opId = Operator::getId(n);
    TORCH_CHECK(
        has(opId),
        "Node ",
        n->kind().toQualString(),
        " does not belong to any LLGA partition");
    return get(opId);
  }

 private:
  std::unordered_map<uint64_t, uint64_t> opmap_;
};

class LlgaGraphHelper {
 public:
  LlgaGraphHelper(
      const std::shared_ptr<Graph>& graph,
      dnnl::graph::partition::policy policy =
          dnnl::graph::partition::policy::fusion);

  bool shouldMerge(Node* toMerge, Node* subgraph);

  bool shouldConsiderForMerge(Node* node);

  bool checkForSingleOpPartition(Node* node);

  Node* createSingletonSubgraph(Node* n, AliasDb& db);

  void mergeNodeIntoSubgraph(Node* toMerge, Node* subgraphNode, AliasDb& db);

  void unmergeIfAnyNodeIsMissing(Node* subgraphNode);

  static bool isLlgaSubgraph(const Node* node);

  Operator makeEltwiseOp(Node* node, dnnl::graph::op::kind kind);

  Operator makeBinaryOp(Node* node, dnnl::graph::op::kind kind);

  std::vector<dnnl::graph::partition> getPartitions() const;

  std::map<size_t, Value*> getTensorIdToValue() const;

  Operator createOperator(Node* node);

 private:
  size_t countSupportedOps(const std::shared_ptr<Graph>& graph) const;
  std::unique_ptr<dnnl::graph::graph> dnnl_graph_ = nullptr;
  std::unique_ptr<torch::jit::AliasDb> aliasDb_ = nullptr;
  OpPartitionMap opToOwningPartition_;
  std::vector<dnnl::graph::partition> partitions_;
  std::map<size_t, Value*>
      tensorIdToValue_; // map from tensorId to torch::jit::Value
};

class LlgaNodeWrapper {
 public:
  LlgaNodeWrapper(const Node* node);

  void setOpaqueLayout(size_t offset);

  bool useOpaqueLayout(size_t offset) const;

  friend class LlgaGraphHelper;

 private:
  Node* n;
};

} // namespace torch::jit::fuser::onednn

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `OpPartitionMap`, `LlgaGraphHelper`, `LlgaNodeWrapper`, `LlgaGraphHelper`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `oneapi/dnnl/dnnl_graph.hpp`
- `torch/csrc/jit/codegen/onednn/operator.h`
- `torch/csrc/jit/ir/alias_analysis.h`
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

- **File Documentation**: `graph_helper.h_docs.md`
- **Keyword Index**: `graph_helper.h_kw.md`
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

- **File Documentation**: `graph_helper.h_docs.md_docs.md`
- **Keyword Index**: `graph_helper.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
