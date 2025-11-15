# Documentation: `docs/torch/csrc/jit/ir/named_value.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/ir/named_value.h_docs.md`
- **Size**: 4,858 bytes (4.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/ir/named_value.h`

## File Metadata

- **Path**: `torch/csrc/jit/ir/named_value.h`
- **Size**: 2,396 bytes (2.34 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/utils/variadic.h>

namespace torch::jit {

struct Value;

/**
 * A value with optional extra name and location information. Used during
 * schema matching to provide extra error information and resolve kwargs.
 */
struct NamedValue {
  NamedValue(const SourceRange& loc, const std::string& name, Value* value)
      : loc_(loc), name_(name), value_(value) {}
  NamedValue(const SourceRange& loc, Value* value) : loc_(loc), value_(value) {}

  /* implicit */ NamedValue(Value* value) : value_(value) {}
  NamedValue(const std::string& name, Value* value)
      : name_(name), value_(value) {}

  /* implicit */ NamedValue(IValue value) : ivalue_(std::move(value)) {}

  NamedValue(const std::string& name, IValue value)
      : name_(name), ivalue_(std::move(value)) {}

  template <
      typename T,
      typename = std::enable_if_t<
          (!std::is_same_v<std::decay_t<T>, NamedValue> &&
           !std::is_same_v<std::decay_t<T>, Value*> &&
           !std::is_same_v<std::decay_t<T>, IValue>)>>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  NamedValue(T&& t) : NamedValue(IValue(std::forward<T>(t))) {}

  template <
      typename T,
      typename = std::enable_if_t<
          (!std::is_same_v<std::decay_t<T>, Value*> &&
           !std::is_same_v<std::decay_t<T>, IValue>)>>
  NamedValue(const std::string& name, T&& t)
      : NamedValue(name, IValue(std::forward<T>(t))) {}

  SourceRange locOr(const SourceRange& backup_location) const {
    if (!loc_)
      return backup_location;
    return loc();
  }

  // note: this will insert a constant node into the graph at the current
  // insert point if this NamedValue is actually a constant
  Value* value(Graph& g) const {
    if (!value_)
      return insertConstant(
          g, ivalue_); // use insertConstant to remove need to include ir.h here
    return value_;
  }

  const std::string& name() const {
    AT_ASSERT(name_);
    return *name_;
  }

  const SourceRange& loc() const {
    AT_ASSERT(loc_);
    return *loc_;
  }

  at::TypePtr type() const;

 private:
  std::optional<SourceRange> loc_;
  std::optional<std::string> name_;
  Value* value_{nullptr};
  // only valid if value_ == nullptr;
  IValue ivalue_;
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Value`, `NamedValue`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/ir`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `torch/csrc/jit/frontend/source_range.h`
- `torch/csrc/jit/ir/constants.h`
- `torch/csrc/utils/variadic.h`


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
- [`subgraph_matcher.h_docs.md`](./subgraph_matcher.h_docs.md)
- [`scope.cpp_docs.md`](./scope.cpp_docs.md)
- [`graph_node_list.h_docs.md`](./graph_node_list.h_docs.md)
- [`type_hashing.cpp_docs.md`](./type_hashing.cpp_docs.md)
- [`ir.h_docs.md`](./ir.h_docs.md)
- [`ir.cpp_docs.md`](./ir.cpp_docs.md)
- [`irparser.cpp_docs.md`](./irparser.cpp_docs.md)
- [`node_hashing.cpp_docs.md`](./node_hashing.cpp_docs.md)


## Cross-References

- **File Documentation**: `named_value.h_docs.md`
- **Keyword Index**: `named_value.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/ir`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/ir`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/ir`):

- [`subgraph_matcher.h_docs.md_docs.md`](./subgraph_matcher.h_docs.md_docs.md)
- [`node_hashing.h_kw.md_docs.md`](./node_hashing.h_kw.md_docs.md)
- [`subgraph_matcher.h_kw.md_docs.md`](./subgraph_matcher.h_kw.md_docs.md)
- [`graph_utils.h_kw.md_docs.md`](./graph_utils.h_kw.md_docs.md)
- [`irparser.cpp_docs.md_docs.md`](./irparser.cpp_docs.md_docs.md)
- [`constants.h_docs.md_docs.md`](./constants.h_docs.md_docs.md)
- [`scope.h_kw.md_docs.md`](./scope.h_kw.md_docs.md)
- [`scope.h_docs.md_docs.md`](./scope.h_docs.md_docs.md)
- [`irparser.cpp_kw.md_docs.md`](./irparser.cpp_kw.md_docs.md)
- [`scope.cpp_docs.md_docs.md`](./scope.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `named_value.h_docs.md_docs.md`
- **Keyword Index**: `named_value.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
