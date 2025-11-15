# Documentation: `aten/src/ATen/core/alias_info.h`

## File Metadata

- **Path**: `aten/src/ATen/core/alias_info.h`
- **Size**: 4,582 bytes (4.47 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include <ATen/core/symbol.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>

namespace c10 {
/**
 * class AliasInfo
 *
 * Data structure to hold aliasing information for an `Argument`. They can be
 * nested to represent aliasing information on contained types.
 *
 * There is a `beforeSet` which describes the aliasing information before the
 * operator executes, and an `afterSet` that describes aliasing info
 * after execution.
 */
class AliasInfo {
 public:
  AliasInfo() = default;
  AliasInfo(bool is_write, const std::set<std::string>& before_qual_strings, const std::set<std::string>& after_qual_strings) : isWrite_(is_write) {
    for (const auto& s: before_qual_strings) {
      beforeSets_.insert(Symbol::fromQualString(s));
    }
    for (const auto& s : after_qual_strings) {
      afterSets_.insert(Symbol::fromQualString(s));
    }
  }
  // Symbol for the set that can alias anything
  static Symbol wildcardSet() {
    static const Symbol wc = Symbol::fromQualString("alias::*");
    return wc;
  }

  void setIsWrite(bool isWrite) {
    isWrite_ = isWrite;
  }

  bool isWrite() const {
    return isWrite_;
  }

  void addBeforeSet(Symbol aliasSet) {
    beforeSets_.insert(aliasSet);
  }

  void addAfterSet(Symbol aliasSet) {
    afterSets_.insert(aliasSet);
  }

  const std::unordered_set<Symbol>& beforeSets() const {
    return beforeSets_;
  }

  const std::unordered_set<Symbol>& afterSets() const {
    return afterSets_;
  }

  Symbol beforeSet() const {
    AT_ASSERT(beforeSets_.size() == 1);
    return *beforeSets_.begin();
  }

  bool isWildcardBefore() const {
    return beforeSets_.count(wildcardSet()) != 0;
  }

  bool isWildcardAfter() const {
    return afterSets_.count(wildcardSet()) != 0;
  }

  // the alias info for the contained types of the type
  // e.g. if this is an annotation on List[T], `sets` refers to
  // the alias sets that the list may be in
  // while containedTypes()[0] refers to the sets that members of the list
  // may be in
  void addContainedType(AliasInfo aliasInfo) {
    containedTypes_.push_back(std::move(aliasInfo));
  }
  const std::vector<AliasInfo>& containedTypes() const {
    return containedTypes_;
  }

 private:
  std::unordered_set<Symbol> beforeSets_;
  std::unordered_set<Symbol> afterSets_;
  std::vector<AliasInfo> containedTypes_;
  bool isWrite_ = false;
};

inline bool operator==(const AliasInfo& lhs, const AliasInfo& rhs) {
  return lhs.isWrite() == rhs.isWrite()
      && lhs.beforeSets() == rhs.beforeSets()
      && lhs.afterSets() == rhs.afterSets()
      && lhs.containedTypes() == rhs.containedTypes();
}

// this does match the way things are represented in the schema
inline std::ostream& operator<<(std::ostream& out, const AliasInfo& aliasInfo) {
  out << "(";
  bool first = true;
  for (const auto& set : aliasInfo.beforeSets()) {
    if (first) {
      first = false;
    } else {
      out << "|";
    }
    out << set.toUnqualString();
  }
  if (aliasInfo.isWrite()) {
    out << "!";
  }
  if (aliasInfo.beforeSets() != aliasInfo.afterSets()) {
    out << " -> ";
    first = true;
    for (const auto& set : aliasInfo.afterSets()) {
      if (first) {
        first = false;
      } else {
        out << "|";
      }
      out << set.toUnqualString();
    }
  }
  out << ")";
  return out;
}
} // namespace c10

namespace std {
template <>
  struct hash<c10::AliasInfo> {
    size_t operator()(const c10::AliasInfo& aliasInfo) const {
      auto hash = std::hash<bool>()(aliasInfo.isWrite());

      // NOTE: for unordered_set hashes, we couldn't use hash_combine
      // because hash_combine is order dependent. Instead, we choose to
      // use XOR as the combining function as XOR is commutative.
      size_t before_set_hash_seed = 0;
      for (auto &e: aliasInfo.beforeSets()) {
        auto symbol_hash = std::hash<c10::Symbol>()(e);
        before_set_hash_seed = before_set_hash_seed ^ symbol_hash;
      }
      size_t after_set_hash_seed = 0;
      for (auto &e: aliasInfo.afterSets()) {
        auto symbol_hash = std::hash<c10::Symbol>()(e);
        after_set_hash_seed = after_set_hash_seed ^ symbol_hash;
      }

      hash = c10::hash_combine(hash, before_set_hash_seed);
      hash = c10::hash_combine(hash, after_set_hash_seed);
      for (auto &e: aliasInfo.containedTypes()) {
        auto contained_type_hash = std::hash<c10::AliasInfo>()(e);
        hash = c10::hash_combine(hash, contained_type_hash);
      }
      return hash;
    }
  };
}

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `std`, `c10`

**Classes/Structs**: `AliasInfo`, `AliasInfo`, `hash`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `set`
- `string`
- `unordered_set`
- `vector`
- `ATen/core/symbol.h`
- `c10/util/Exception.h`
- `c10/util/hash.h`


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

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `alias_info.h_docs.md`
- **Keyword Index**: `alias_info.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
