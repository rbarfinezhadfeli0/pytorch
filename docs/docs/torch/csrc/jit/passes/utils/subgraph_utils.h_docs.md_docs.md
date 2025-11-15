# Documentation: `docs/torch/csrc/jit/passes/utils/subgraph_utils.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/utils/subgraph_utils.h_docs.md`
- **Size**: 4,839 bytes (4.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/utils/subgraph_utils.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/utils/subgraph_utils.h`
- **Size**: 2,360 bytes (2.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>

// Utilities for dealing with nodes that contain subgraphs.
//
// They handle the complexity of editing inputs/outputs as you merge nodes in
// and out of subgraphs.
namespace torch::jit::SubgraphUtils {

// Create a new subgraph node that contains only `n`. The new subgraph will have
// `subgraphKind` as its type.
//
// `n` is destroyed.
//
// Returns the new subgraph node.
TORCH_API Node* createSingletonSubgraph(Node* n, Symbol subgraphKind);

// Creates a new subgraph that only contains `n`, amd updates the new outputs
// of the subgraph to have the aliasing properties of the original `n` outputs
TORCH_API Node* createSingletonSubgraphAndUpdateAliasing(
    Node* to_merge,
    Symbol subgraphKind,
    AliasDb& db);

// Merge a node into a subgraph node. If `toMerge` is also a subgraph, the
// subgraphs are merged.
// If `destroyNode` is true `toMerge` is destroyed.
// An optional argument 'vmap' could be used to retrieve value mappings.
// Values will be mapped to their new subgraph values
TORCH_API void mergeNodeIntoSubgraph(
    Node* toMerge,
    Node* subgraphNode,
    bool destroyNode = true);

// Merges a node into a subgraph node, and updates the new outputs of the
// subgraph to have the aliasing properties of the corresponding `to_merge`
// outputs
TORCH_API void mergeNodeIntoSubgraphAndUpdateAliasing(
    Node* to_merge,
    Node* subgraphNode,
    AliasDb& db);

TORCH_API std::vector<Node*> unmergeAliasedOutputs(
    Node* subgraphNode,
    AliasDb& db);

// Move nodes from a subgraph node to the outer graph.
// `subgraphNode` is destroyed.
TORCH_API void unmergeSubgraph(Node* subgraphNode);

// Move `node_to_unmerge` and its descendants after `subgraphNode`
// promotes any dependencies of `node_to_unmerge` to subgraphNode outputs
TORCH_API void unmergeNode(Node* node_to_unmerge, Node* subgraphNode);

TORCH_API bool unmergeOutputsAlisingInputs(Node* subgraphNode);

TORCH_API bool unmergeAliasedOutputs(Node* subgraphNode);

// Convenience function
std::shared_ptr<Graph> getSubgraph(Node* n);

TORCH_API std::string generateNameForGraph(
    const std::shared_ptr<Graph>& graph,
    size_t maxlen = 40,
    const std::string& prefix = "fused");

} // namespace torch::jit::SubgraphUtils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
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

Files in the same folder (`torch/csrc/jit/passes/utils`):

- [`op_registry.h_docs.md`](./op_registry.h_docs.md)
- [`optimization_utils.h_docs.md`](./optimization_utils.h_docs.md)
- [`optimization_utils.cpp_docs.md`](./optimization_utils.cpp_docs.md)
- [`subgraph_utils.cpp_docs.md`](./subgraph_utils.cpp_docs.md)
- [`check_alias_annotation.h_docs.md`](./check_alias_annotation.h_docs.md)
- [`memory_dag.h_docs.md`](./memory_dag.h_docs.md)
- [`memory_dag.cpp_docs.md`](./memory_dag.cpp_docs.md)
- [`op_registry.cpp_docs.md`](./op_registry.cpp_docs.md)
- [`check_alias_annotation.cpp_docs.md`](./check_alias_annotation.cpp_docs.md)


## Cross-References

- **File Documentation**: `subgraph_utils.h_docs.md`
- **Keyword Index**: `subgraph_utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes/utils`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/passes/utils`):

- [`op_registry.h_kw.md_docs.md`](./op_registry.h_kw.md_docs.md)
- [`optimization_utils.cpp_docs.md_docs.md`](./optimization_utils.cpp_docs.md_docs.md)
- [`subgraph_utils.cpp_docs.md_docs.md`](./subgraph_utils.cpp_docs.md_docs.md)
- [`check_alias_annotation.cpp_kw.md_docs.md`](./check_alias_annotation.cpp_kw.md_docs.md)
- [`memory_dag.cpp_docs.md_docs.md`](./memory_dag.cpp_docs.md_docs.md)
- [`optimization_utils.h_docs.md_docs.md`](./optimization_utils.h_docs.md_docs.md)
- [`op_registry.cpp_kw.md_docs.md`](./op_registry.cpp_kw.md_docs.md)
- [`op_registry.h_docs.md_docs.md`](./op_registry.h_docs.md_docs.md)
- [`check_alias_annotation.h_kw.md_docs.md`](./check_alias_annotation.h_kw.md_docs.md)
- [`memory_dag.h_docs.md_docs.md`](./memory_dag.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `subgraph_utils.h_docs.md_docs.md`
- **Keyword Index**: `subgraph_utils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
