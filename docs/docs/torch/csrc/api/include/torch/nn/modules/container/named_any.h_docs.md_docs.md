# Documentation: `docs/torch/csrc/api/include/torch/nn/modules/container/named_any.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/modules/container/named_any.h_docs.md`
- **Size**: 4,882 bytes (4.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/modules/container/named_any.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/container/named_any.h`
- **Size**: 2,442 bytes (2.38 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/modules/container/any.h>
#include <torch/types.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace torch::nn {

/// Stores a type erased `Module` with name.
///
/// The `NamedAnyModule` class enables the following API for constructing
/// `nn::Sequential` with named submodules:
/// \rst
/// .. code-block:: cpp
///
///   struct M : torch::nn::Module {
///     explicit M(int value_) : value(value_) {}
///     int value;
///     int forward() {
///       return value;
///     }
///   };
///
///   Sequential sequential({
///     {"m1", std::make_shared<M>(1)},  // shared pointer to `Module` is
///     supported {std::string("m2"), M(2)},  // `Module` is supported
///     {"linear1", Linear(10, 3)}  // `ModuleHolder` is supported
///   });
/// \endrst
class NamedAnyModule {
 public:
  /// Creates a `NamedAnyModule` from a (boxed) `Module`.
  template <typename ModuleType>
  NamedAnyModule(std::string name, std::shared_ptr<ModuleType> module_ptr)
      : NamedAnyModule(std::move(name), AnyModule(std::move(module_ptr))) {}

  /// Creates a `NamedAnyModule` from a `Module`, moving or copying it
  /// into a `shared_ptr` internally.
  // NOTE: We need to use `std::remove_reference_t<M>` to get rid of
  // any reference components for make_unique.
  template <typename M, typename = torch::detail::enable_if_module_t<M>>
  NamedAnyModule(std::string name, M&& module)
      : NamedAnyModule(
            std::move(name),
            std::make_shared<std::remove_reference_t<M>>(
                std::forward<M>(module))) {}

  /// Creates a `NamedAnyModule` from a `Module` that is unwrapped from
  /// a `ModuleHolder`.
  template <typename M>
  NamedAnyModule(std::string name, const ModuleHolder<M>& module_holder)
      : NamedAnyModule(std::move(name), module_holder.ptr()) {}

  /// Creates a `NamedAnyModule` from a type-erased `AnyModule`.
  NamedAnyModule(std::string name, AnyModule any_module)
      : name_(std::move(name)), module_(std::move(any_module)) {}

  /// Returns a reference to the name.
  const std::string& name() const noexcept {
    return name_;
  }

  /// Returns a reference to the module.
  AnyModule& module() noexcept {
    return module_;
  }

  /// Returns a const reference to the module.
  const AnyModule& module() const noexcept {
    return module_;
  }

 private:
  std::string name_;
  AnyModule module_;
};

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `enables`, `M`, `NamedAnyModule`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules/container`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/modules/container/any.h`
- `torch/types.h`
- `memory`
- `type_traits`
- `utility`


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

Files in the same folder (`torch/csrc/api/include/torch/nn/modules/container`):

- [`moduledict.h_docs.md`](./moduledict.h_docs.md)
- [`sequential.h_docs.md`](./sequential.h_docs.md)
- [`any_module_holder.h_docs.md`](./any_module_holder.h_docs.md)
- [`parameterlist.h_docs.md`](./parameterlist.h_docs.md)
- [`modulelist.h_docs.md`](./modulelist.h_docs.md)
- [`functional.h_docs.md`](./functional.h_docs.md)
- [`any.h_docs.md`](./any.h_docs.md)
- [`any_value.h_docs.md`](./any_value.h_docs.md)
- [`parameterdict.h_docs.md`](./parameterdict.h_docs.md)


## Cross-References

- **File Documentation**: `named_any.h_docs.md`
- **Keyword Index**: `named_any.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn/modules/container`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn/modules/container`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/api/include/torch/nn/modules/container`):

- [`sequential.h_docs.md_docs.md`](./sequential.h_docs.md_docs.md)
- [`functional.h_docs.md_docs.md`](./functional.h_docs.md_docs.md)
- [`modulelist.h_docs.md_docs.md`](./modulelist.h_docs.md_docs.md)
- [`any.h_docs.md_docs.md`](./any.h_docs.md_docs.md)
- [`moduledict.h_docs.md_docs.md`](./moduledict.h_docs.md_docs.md)
- [`modulelist.h_kw.md_docs.md`](./modulelist.h_kw.md_docs.md)
- [`parameterdict.h_docs.md_docs.md`](./parameterdict.h_docs.md_docs.md)
- [`parameterlist.h_kw.md_docs.md`](./parameterlist.h_kw.md_docs.md)
- [`named_any.h_kw.md_docs.md`](./named_any.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `named_any.h_docs.md_docs.md`
- **Keyword Index**: `named_any.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
