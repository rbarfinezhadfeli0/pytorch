# Documentation: `docs/torch/csrc/jit/passes/utils/memory_dag.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/utils/memory_dag.cpp_docs.md`
- **Size**: 10,170 bytes (9.93 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/utils/memory_dag.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/utils/memory_dag.cpp`
- **Size**: 7,631 bytes (7.45 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/utils/memory_dag.h>

#include <c10/util/flat_hash_map.h>
#include <algorithm>
#include <queue>

namespace torch::jit {
namespace {

void makePointerToImpl(Element* from, Element* to) {
  from->pointsTo.set(to->index);
  to->pointedFrom.set(from->index);
}

Element* makeFreshValueImpl(
    const Value* v,
    std::vector<std::unique_ptr<Element>>& indexToElementMap_) {
  if (v == nullptr) {
    // Create a wildcard element, with no corresponding value
    indexToElementMap_.emplace_back(
        std::make_unique<Element>(indexToElementMap_.size()));
    return indexToElementMap_.back().get();
  }
  indexToElementMap_.emplace_back(
      std::make_unique<Element>(v, indexToElementMap_.size()));
  return indexToElementMap_.back().get();
}
} // namespace

Element::Element(const Value* value_, unsigned index_)
    : index(index_), values({value_}) {}
Element::Element(unsigned index_) : index(index_), values({}) {}

const Element* MemoryDAG::fromIndex(unsigned x) const {
  TORCH_INTERNAL_ASSERT(x < indexToElementMap_.size());
  return indexToElementMap_[x].get();
}

Element* MemoryDAG::fromIndex(unsigned x) {
  TORCH_INTERNAL_ASSERT(x < indexToElementMap_.size());
  return indexToElementMap_[x].get();
}

bool MemoryDAG::mayAlias(const Element* a, const Element* b) const {
  const auto& aMemLoc = getMemoryLocations(a);
  const auto& bMemLoc = getMemoryLocations(b);

  return aMemLoc.intersects(bMemLoc);
}

bool MemoryDAG::mayContainAlias(const Element* a, const Element* b) const {
  return getAllContainedMemoryLocations(a).intersects(
      getAllContainedMemoryLocations(b));
}

const MemoryLocations& MemoryDAG::getAllContainedMemoryLocations(
    const Element* elem) const {
  if (C10_UNLIKELY(!elem->cachedAllContainedMemoryLocations_.has_value())) {
    MemoryLocations cache;
    elem->cachedAllContainedMemoryLocations_ = MemoryLocations();
    collectAllContainedMemoryLocationsImpl(
        elem, *elem->cachedAllContainedMemoryLocations_);
  }
  return *elem->cachedAllContainedMemoryLocations_;
}

void MemoryDAG::collectAllContainedMemoryLocations(
    const Element* elem,
    MemoryLocations& cont) const {
  // we have already recursed on this element
  unsigned compIdx = elem->index;
  if (cont.test(compIdx)) {
    return;
  }

  if (C10_UNLIKELY(!elem->cachedAllContainedMemoryLocations_.has_value())) {
    MemoryLocations cache;
    collectAllContainedMemoryLocationsImpl(elem, cache);
    elem->cachedAllContainedMemoryLocations_ = std::move(cache);
  }
  cont |= *elem->cachedAllContainedMemoryLocations_;
}

void MemoryDAG::collectAllContainedMemoryLocationsImpl(
    const Element* elem,
    MemoryLocations& cont) const {
  unsigned compIdx = elem->index;
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!cont.test(compIdx));
  cont.set(compIdx);

  for (const auto& mem_loc : getMemoryLocations(elem)) {
    collectAllContainedMemoryLocations(fromIndex(mem_loc), cont);
  }

  for (const auto& contained : elem->containedElements) {
    collectAllContainedMemoryLocations(fromIndex(contained), cont);
  }
}

bool MemoryDAG::mayContainAlias(
    const Element* a,
    const at::ArrayRef<Element*> b) const {
  if (b.empty()) {
    return false;
  }

  const auto& a_contained = getAllContainedMemoryLocations(a);
  return std::any_of(b.begin(), b.end(), [this, &a_contained](Element* b_elem) {
    return a_contained.intersects(this->getAllContainedMemoryLocations(b_elem));
  });
}

bool MemoryDAG::mayContainAlias(
    const at::ArrayRef<Element*> a,
    const at::ArrayRef<Element*> b) const {
  if (a.empty() || b.empty()) {
    return false;
  }

  MemoryLocations all_a_mlocs;
  for (const auto& elem : a) {
    collectAllContainedMemoryLocations(elem, all_a_mlocs);
  }

  MemoryLocations all_b_mlocs;
  for (const auto& elem : b) {
    collectAllContainedMemoryLocations(elem, all_b_mlocs);
  }

  return all_a_mlocs.intersects(all_b_mlocs);
}

void MemoryDAGBuilder::makePointerTo(Element* from, Element* to) {
  makePointerToImpl(from, to);
}

void MemoryDAGBuilder::addToContainedElements(
    Element* elem,
    Element* container) {
  TORCH_INTERNAL_ASSERT(
      elem != container, "Elements cannot contain themselves");
  container->containedElements.set(elem->index);
}

// Give `v` a fresh alias (i.e. it does not point to any value)
Element* MemoryDAGBuilder::makeFreshValue(const Value* v) {
  return makeFreshValueImpl(v, indexToElementMap_);
}

// This function builds up a bitset representing the "alias set" for
// `e` (`MemoryLocations` is just a typedef'd c10::SparseBitVector).
const MemoryLocations& MemoryDAG::getMemoryLocations(const Element* e) const {
  // Note on cache invalidation: all mutation should occur through
  // MemoryDAGBuilder. Thus, once we consume the builder to create an
  // immutable MemoryDAG, we can cache here without worrying that we
  // might potentially get invalidated.
  if (e->cachedMemoryLocations_) {
    return *e->cachedMemoryLocations_;
  }

  MemoryLocations ret;
  if (e->pointsTo.empty()) {
    // Base case: if we don't point to anything, this element is a memory
    // location. Return itself.
    ret.set(e->index);
  } else {
    for (auto el : e->pointsTo) {
      ret |= getMemoryLocations(fromIndex(el));
    }
  }

  e->cachedMemoryLocations_ = std::move(ret);
  return *e->cachedMemoryLocations_;
}

void MemoryDAG::setWildcards(
    const std::unordered_set<const Value*>& wildcards,
    const ska::flat_hash_map<const Value*, Element*>& elementMap,
    const std::function<Element*(const Value*)>& getWildcardElement) {
  std::unordered_map<Element*, MemoryLocations> cacheUpdates;
  // If an element is set as a wildcard, that means that all its memory
  // locations must point to the wildcard element.
  for (const Value* v : wildcards) {
    auto wildcardElement = getWildcardElement(v);
    TORCH_INTERNAL_ASSERT(wildcardElement);

    const MemoryLocations& pointeeSet = getMemoryLocations(elementMap.at(v));
    for (const auto& pointee : pointeeSet) {
      auto from = this->fromIndex(pointee);
      // avoid cycles where the wildcard points to itself
      if (from != wildcardElement) {
        makePointerToImpl(from, wildcardElement);
      }
    }
    // Track which memory locations we edited with a new pointer to the wildcard
    // element.
    cacheUpdates[wildcardElement] |= pointeeSet;
  }

  // Update caches in-place.
  // We take advantage of the fact that we only edited memory locations.
  //
  // Say we added a pointer from `MemoryLocationFoo -> WildcardBar`.
  // For every element, if the cache contains `MemoryLocationFoo`, then we must
  // add `WildcardBar` to it.
  for (const std::unique_ptr<Element>& e : this->indexToElementMap_) {
    e->cachedAllContainedMemoryLocations_.reset();
    if (e->values.empty()) {
      // This element is a wildcard element, we can skip it.
      continue;
    }

    auto wildcardElement = getWildcardElement(*(e->values.begin()));
    if (!wildcardElement) {
      // This value is not a wildcard.
      continue;
    }
    auto it = cacheUpdates.find(wildcardElement);
    if (it == cacheUpdates.end()) {
      // We didn't rewrite any MemoryLocations to point to this element.
      continue;
    }
    // If this element contains an edited memory location, update the cache to
    // contain the pointed-to wildcard element as well.
    if (getMemoryLocations(e.get()).intersects(it->second)) {
      e->cachedMemoryLocations_->set(wildcardElement->index);
    }
  }
}

Element* MemoryDAG::unsafeMakeFreshValue(const Value* v) {
  return makeFreshValueImpl(v, indexToElementMap_);
}
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `Element`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/utils/memory_dag.h`
- `c10/util/flat_hash_map.h`
- `algorithm`
- `queue`


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
- [`memory_dag.h_docs.md`](./memory_dag.h_docs.md)
- [`subgraph_utils.h_docs.md`](./subgraph_utils.h_docs.md)
- [`op_registry.cpp_docs.md`](./op_registry.cpp_docs.md)
- [`check_alias_annotation.cpp_docs.md`](./check_alias_annotation.cpp_docs.md)


## Cross-References

- **File Documentation**: `memory_dag.cpp_docs.md`
- **Keyword Index**: `memory_dag.cpp_kw.md`
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

- Implements or uses **caching** mechanisms.
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
- [`optimization_utils.h_docs.md_docs.md`](./optimization_utils.h_docs.md_docs.md)
- [`op_registry.cpp_kw.md_docs.md`](./op_registry.cpp_kw.md_docs.md)
- [`op_registry.h_docs.md_docs.md`](./op_registry.h_docs.md_docs.md)
- [`check_alias_annotation.h_kw.md_docs.md`](./check_alias_annotation.h_kw.md_docs.md)
- [`memory_dag.h_docs.md_docs.md`](./memory_dag.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `memory_dag.cpp_docs.md_docs.md`
- **Keyword Index**: `memory_dag.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
