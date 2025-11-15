# Documentation: `docs/torch/nativert/detail/ITree.h_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/detail/ITree.h_docs.md`
- **Size**: 6,276 bytes (6.13 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/detail/ITree.h`

## File Metadata

- **Path**: `torch/nativert/detail/ITree.h`
- **Size**: 4,248 bytes (4.15 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*
 * A C++ extension bridge with the Python pytree
 * serialization/unserialization format for torch.export.
 */

#pragma once

#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <ATen/core/ivalue.h>
#include <nlohmann/json.hpp>
#include <torch/nativert/graph/Graph.h>

namespace torch::nativert::detail {

class ITreeSpec;

using ITreeFlattenFn =
    void (*)(const c10::IValue&, const ITreeSpec&, std::vector<c10::IValue>&);
using ITreeUnflattenFn =
    c10::IValue (*)(std::vector<c10::IValue>, const nlohmann::json&);

using ContextLoadFn = nlohmann::json (*)(std::string_view);

using ITreeMapFn = c10::function_ref<c10::IValue(const c10::IValue&)>;
using ITreeMapNoReturnFn =
    c10::function_ref<void(const c10::IValue&, const Value*)>;

using IValueApplyFn =
    void (*)(ITreeMapNoReturnFn, const c10::IValue&, const ITreeSpec&);

nlohmann::json defaultContextLoadFn(std::string_view /*context*/);

struct NodeDef {
  ITreeFlattenFn flattenFn;
  ITreeUnflattenFn unflattenFn;
  IValueApplyFn ivalueApplyFn;

  ContextLoadFn contextLoadFn = defaultContextLoadFn;
};

class ITreeSpec {
 public:
  // Leaf node.
  ITreeSpec(const Value* value = nullptr, bool isUsed = true)
      : numIValues_(1), value_(value), isUsed_(isUsed) {}

  // Non leaf node.
  ITreeSpec(
      std::string_view uniformName,
      nlohmann::json context,
      std::vector<ITreeSpec> children,
      NodeDef nodeDefCache);

  bool isIValue() const {
    return !uniformName_;
  }

  std::string_view uniformName() const {
    TORCH_CHECK(uniformName_);
    return uniformName_.value();
  }

  const nlohmann::json& context() const {
    return context_;
  }

  const std::vector<c10::IValue>& contextKeys() const {
    return contextKeys_;
  }

  const auto& children() const {
    return children_;
  }

  const ITreeSpec& children(size_t i) const {
    return children_[i];
  }

  const NodeDef& nodeDefCache() const {
    return nodeDefCache_;
  }

  size_t numIValues() const {
    return numIValues_;
  }

  bool allIValues() const {
    return allIValues_;
  }

  c10::TypePtr toAtenType() const;

  bool isUsed() const {
    return isUsed_;
  }

  const Value* value() const {
    return value_;
  }

 private:
  // Only non leaf nodes have names.
  // Examples of uniform name: "builtins.tuple", "builtins.dict".
  std::optional<std::string> uniformName_;
  nlohmann::json context_;
  std::vector<ITreeSpec> children_;

  std::vector<c10::IValue> contextKeys_;

  // Cached fields.
  NodeDef nodeDefCache_;
  size_t numIValues_;
  bool allIValues_ = true;

  const Value* value_;
  bool isUsed_;
};

void registerPytreeNode(std::string_view typeName, NodeDef nodeDef);

// Serialized json tree spec should be dumped from treespec_dumps() in
// torch.utils._pytree directly .
ITreeSpec itreeSpecLoads(
    std::string_view json,
    const std::vector<const Value*>& values);

c10::IValue itreeUnflatten(
    std::vector<c10::IValue> ivalues,
    const ITreeSpec& spec);

std::vector<c10::IValue> itreeFlatten(
    const c10::IValue& nested,
    const ITreeSpec& spec);

std::vector<c10::IValue> itreeFlattenFromArgs(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const ITreeSpec& spec);

std::vector<at::Tensor> itreeFlattenToTensorList(
    const c10::IValue& nested,
    const ITreeSpec& spec);

c10::IValue itreeMap(
    ITreeMapFn f,
    const c10::IValue& nested,
    const ITreeSpec& spec);

c10::IValue TORCH_API argsToIValue(
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs);

std::
    pair<std::vector<c10::IValue>, std::unordered_map<std::string, c10::IValue>>
    itreeMapArgs(
        ITreeMapFn f,
        const std::vector<c10::IValue>& args,
        const std::unordered_map<std::string, c10::IValue>& kwargs,
        const ITreeSpec& spec);

void ivalueApply(
    ITreeMapNoReturnFn f,
    const c10::IValue& nested,
    const ITreeSpec& spec);

void ivalueApplyFromArgs(
    ITreeMapNoReturnFn fn,
    const std::vector<c10::IValue>& args,
    const std::unordered_map<std::string, c10::IValue>& kwargs,
    const ITreeSpec& spec);

} // namespace torch::nativert::detail

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ITreeSpec`, `NodeDef`, `ITreeSpec`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/detail`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `optional`
- `string_view`
- `unordered_map`
- `vector`
- `ATen/core/ivalue.h`
- `nlohmann/json.hpp`
- `torch/nativert/graph/Graph.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`torch/nativert/detail`):

- [`ITree.cpp_docs.md`](./ITree.cpp_docs.md)
- [`MPMCQueue.h_docs.md`](./MPMCQueue.h_docs.md)


## Cross-References

- **File Documentation**: `ITree.h_docs.md`
- **Keyword Index**: `ITree.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/detail`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/detail`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/nativert/detail`):

- [`ITree.cpp_kw.md_docs.md`](./ITree.cpp_kw.md_docs.md)
- [`MPMCQueue.h_docs.md_docs.md`](./MPMCQueue.h_docs.md_docs.md)
- [`ITree.h_kw.md_docs.md`](./ITree.h_kw.md_docs.md)
- [`MPMCQueue.h_kw.md_docs.md`](./MPMCQueue.h_kw.md_docs.md)
- [`ITree.cpp_docs.md_docs.md`](./ITree.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ITree.h_docs.md_docs.md`
- **Keyword Index**: `ITree.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
