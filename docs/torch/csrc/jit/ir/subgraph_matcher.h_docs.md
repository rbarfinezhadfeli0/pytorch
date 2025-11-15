# Documentation: `torch/csrc/jit/ir/subgraph_matcher.h`

## File Metadata

- **Path**: `torch/csrc/jit/ir/subgraph_matcher.h`
- **Size**: 3,127 bytes (3.05 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/ir/ir.h>

#include <unordered_map>
#include <vector>

namespace torch::jit {

/**
 * \brief A structure describing a match of a pattern in a graph.
 *
 * The structure contains an anchor node, from which the match was found, and
 * match-maps for nodes and values. A match-map specifies the correspondence
 * between nodes in the pattern graph (match-map keys) with nodes in the actual
 * graph (match-map values). We keep such maps for both nodes and values.
 */
struct Match {
  Node* anchor;
  std::unordered_map<const Node*, Node*> nodes_map;
  std::unordered_map<const Value*, Value*> values_map;
};

/**
 * \brief Find all matches of a \p PATTERN in a \p GRAPH.
 *
 * The function returns a vector of match-descriptors (see description of
 * `struct Match`).
 *
 * Matching rules:
 *  - Pattern graph must contain a single block.
 *  - Matched subgraphs do not span across different blocks.
 *  - No uses outside the match are allowed, except for Param and Return nodes.
 *  Basically, we're matching hammocks, not arbitrary subgraphs.
 *  - The pattern graph must return only one value (i.e. it must have a single
 *  node leading to return).
 *  - Nodes that are not used in computation of the return value in the pattern
 * graph are ignored during matching (IOW, we're essentially performing DCE on
 * the pattern).
 *  - Pattern graph nodes cannot alias. TODO: the check not implemented yet.
 *  - Aliasing nodes in the graph cannot constitute a match (i.e. through all
 * found matches, no nodes in the subgraph alias with each other). TODO: check
 * not implemented yet.
 *  - The matcher will not mutate either the pattern graph or the matched graph.
 * The matched graph is taken as non-const so that Match may contain non-const
 * pointers.  This enables clients of this API to use Match to drive mutations.
 *
 * Note [Multi-output Patterns]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Subgraph matcher provides limited support for multi-output patterns. With a
 * single output pattern, a single scan through the graph is sufficient to
 * find all the matches: given a starting node (an "anchor"), we can
 * deterministically check whether a pattern matches a subgraph corresponding to
 * this anchor node. For a general case of multi-output patterns, we would have
 * N anchors, which would result in M^N comparisons (M is the size of the
 * graph). Clearly this is computationally prohibitive.
 *
 * To overcome this, we impose some constraints on the multi-output patterns
 * that we accept. We require that checking whether the pattern matches a
 * subgraph would still be fully determined by a single node in the graph. To
 * achieve this, we designate the first output in the pattern as the "main"
 * output and assume that we can traverse up from this node to match the
 * entire pattern.
 *
 * Corrolary 1: the order of outputs in the pattern matters!
 * Corollary 2: patterns cannot contain any nodes not participating in the main
 * output computation.
 */
std::vector<Match> TORCH_API
findPatternMatches(const Graph& pattern, Graph& graph);

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Match`, `Match`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/ir`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/ir/ir.h`
- `unordered_map`
- `vector`


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

Files in the same folder (`torch/csrc/jit/ir`):

- [`node_hashing.h_docs.md`](./node_hashing.h_docs.md)
- [`constants.cpp_docs.md`](./constants.cpp_docs.md)
- [`scope.cpp_docs.md`](./scope.cpp_docs.md)
- [`graph_node_list.h_docs.md`](./graph_node_list.h_docs.md)
- [`type_hashing.cpp_docs.md`](./type_hashing.cpp_docs.md)
- [`ir.h_docs.md`](./ir.h_docs.md)
- [`ir.cpp_docs.md`](./ir.cpp_docs.md)
- [`irparser.cpp_docs.md`](./irparser.cpp_docs.md)
- [`node_hashing.cpp_docs.md`](./node_hashing.cpp_docs.md)


## Cross-References

- **File Documentation**: `subgraph_matcher.h_docs.md`
- **Keyword Index**: `subgraph_matcher.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
