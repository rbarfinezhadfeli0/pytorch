# Documentation: `torch/csrc/jit/passes/utils/memory_dag.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/utils/memory_dag.h`
- **Size**: 6,391 bytes (6.24 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/jit_type.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/sparse_bitset.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/csrc/Export.h>

// Uses a compressed index representation for faster comparisons
typedef c10::SparseBitVector<256> MemoryLocations;
namespace torch::jit {

struct Value;

using AliasTypeSet = std::vector<TypePtr>;

// `Element` represents a vertex in the points-to graph. It represents
// anything that could have an aliasing relationship--mostly IR
// `Value`s, but also wildcards or the type inside a container (e.g. `T`
// in `List[T]`)
struct Element {
  Element(const Value* value_, unsigned index_);
  // wildcard constructor
  explicit Element(unsigned index_);

  // Index into the owning DAG's bit vector that represents this element.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  unsigned index;

  // All elements that this element *may* point to. It's possible to have
  // multiple elements that you might point to due to control flow/complex ops
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  MemoryLocations pointsTo;
  // Backreference for points-to.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  MemoryLocations pointedFrom;

  // Elements can contain other elements (e.g. List[Tensor])
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  MemoryLocations containedElements;

  // The values that this element corresponds to. May be empty if this element
  // doesn't represent a first-class value.
  // This is for debug information only.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unordered_set<const Value*> values;

 private:
  // Make `from` point at `to`.
  void makePointerTo(Element* from, Element* to);

  friend class MemoryDAG;
  // We memoize the results of `getMemoryLocations` to speed up queries.
  // A nullopt means that this cache is not yet populated. Since `MemoryDAG` is
  // immutable, this cache should never need to be invalidated.
  mutable std::optional<MemoryLocations> cachedMemoryLocations_;

  mutable std::optional<MemoryLocations> cachedAllContainedMemoryLocations_;
};

// class MemoryDAG
//
// This class tracks the "A points to B" graph for all values. It is used by
// AliasDb to provide a higher-level API.
//
// We maintain a DAG where:
//   - Vertices (called "Elements") represent Values and
//     other aliasing entities (e.g. the stuff inside a list)
//   - Edges represent a "points-to" relationship.
//
// Leaves in this DAG are entities that don't point to anything, and thus
// correspond to unique "memory locations".
//
// So, by traversing the "points-to" graph to the leaves, you can determine
// which memory locations an element may point to.
class TORCH_API MemoryDAG {
 public:
  explicit MemoryDAG(std::vector<std::unique_ptr<Element>> indexToElementMap)
      : indexToElementMap_(std::move(indexToElementMap)) {}
  // explicitly delete copy constructor because otherwise windows build is
  // confused for an exported class see
  // https://stackoverflow.com/a/51033485/105137
  MemoryDAG(const MemoryDAG&) = delete;
  MemoryDAG& operator=(const MemoryDAG&) = delete;

  // Return the unique memory locations that `Element` might represent.
  const MemoryLocations& getMemoryLocations(const Element* e) const;

  // Do `a` and `b` potentially share a memory location?
  bool mayAlias(const Element* a, const Element* b) const;

  // Does `a` hold reference to any memory that is stored in `b`, or vice versa?
  bool mayContainAlias(const Element* a, const Element* b) const;

  bool mayContainAlias(const Element* a, const at::ArrayRef<Element*> b) const;

  bool mayContainAlias(
      const at::ArrayRef<Element*> a,
      const at::ArrayRef<Element*> b) const;

  // Converts from the compressed index representation
  const Element* fromIndex(unsigned x) const;
  Element* fromIndex(unsigned x);
  void collectAllContainedMemoryLocations(
      const Element* elem,
      MemoryLocations& cont) const;

  /**
   * The following methods are special cases where we need to mutate the
   * internals of MemoryDAG for efficiency reasons. Don't call them unless you
   * know what you're doing! In particular, don't add new mutating methods
   * without ensuring that you are maintaining cache consistency for memory
   * locations.
   */

  // Adding wildcards can trigger extremely expensive cache invalidations. This
  // method adds them in a more efficient cache-aware way.
  void setWildcards(
      const std::unordered_set<const Value*>& wildcards,
      const ska::flat_hash_map<const Value*, Element*>& elementMap,
      const std::function<Element*(const Value*)>& getWildcardElement);
  Element* unsafeMakeFreshValue(const Value* v);

 private:
  const MemoryLocations& getAllContainedMemoryLocations(
      const Element* elem) const;
  void collectAllContainedMemoryLocationsImpl(
      const Element* elem,
      MemoryLocations& cont) const;
  std::vector<std::unique_ptr<Element>> indexToElementMap_;
};

/**
 * Helper to build up the points-to graph.
 *
 * We separate the "building" into a different class because it allows us to
 * cache internally to MemoryDAG without worrying about how the DAG structure
 * is mutated.
 */
class TORCH_API MemoryDAGBuilder {
 public:
  MemoryDAGBuilder() = default;
  MemoryDAGBuilder(const MemoryDAGBuilder&) = delete;
  MemoryDAGBuilder& operator=(const MemoryDAGBuilder&) = delete;

  // Make `from` point at `to`.
  void makePointerTo(Element* from, Element* to);

  void addToContainedElements(Element* contained, Element* container);

  std::unique_ptr<MemoryDAG> createMemoryDAG() && {
    return std::make_unique<MemoryDAG>(std::move(indexToElementMap_));
  }

  // Make a fresh Element (i.e. an Element that doesn't point to anything) and
  // return it.
  Element* makeFreshValue(const Value* v);

  friend MemoryDAG;

 private:
  // `MemoryDAGBuilder` builds up `indexToElementMap_`, then uses
  // the map to construct the `MemoryDAG`
  std::vector<std::unique_ptr<Element>> indexToElementMap_;
};
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 8 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Value`, `Element`, `value`, `MemoryDAG`, `MemoryDAG`, `tracks`, `TORCH_API`, `see`, `because`, `TORCH_API`, `the`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/jit_type.h`
- `c10/util/ArrayRef.h`
- `c10/util/flat_hash_map.h`
- `c10/util/sparse_bitset.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/type_hashing.h`
- `memory`
- `optional`
- `unordered_map`
- `unordered_set`
- `vector`
- `torch/csrc/Export.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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
- [`memory_dag.cpp_docs.md`](./memory_dag.cpp_docs.md)
- [`subgraph_utils.h_docs.md`](./subgraph_utils.h_docs.md)
- [`op_registry.cpp_docs.md`](./op_registry.cpp_docs.md)
- [`check_alias_annotation.cpp_docs.md`](./check_alias_annotation.cpp_docs.md)


## Cross-References

- **File Documentation**: `memory_dag.h_docs.md`
- **Keyword Index**: `memory_dag.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
