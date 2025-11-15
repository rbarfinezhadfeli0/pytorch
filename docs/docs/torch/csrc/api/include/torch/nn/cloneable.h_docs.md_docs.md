# Documentation: `docs/torch/csrc/api/include/torch/nn/cloneable.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/cloneable.h_docs.md`
- **Size**: 6,191 bytes (6.05 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/cloneable.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/cloneable.h`
- **Size**: 3,901 bytes (3.81 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>

#include <memory>
#include <utility>

namespace torch::nn {
/// The `clone()` method in the base `Module` class does not have knowledge of
/// the concrete runtime type of its subclasses. Therefore, `clone()` must
/// either be called from within the subclass, or from a base class that has
/// knowledge of the concrete type. `Cloneable` uses the CRTP to gain
/// knowledge of the subclass' static type and provide an implementation of the
/// `clone()` method. We do not want to use this pattern in the base class,
/// because then storing a module would always require templatizing it.
template <typename Derived>
// NOLINTNEXTLINE(bugprone-exception-escape)
class Cloneable : public Module {
 public:
  using Module::Module;

  /// `reset()` must perform initialization of all members with reference
  /// semantics, most importantly parameters, buffers and submodules.
  virtual void reset() = 0;

  /// Performs a recursive "deep copy" of the `Module`, such that all parameters
  /// and submodules in the cloned module are different from those in the
  /// original module.
  std::shared_ptr<Module> clone(
      const std::optional<Device>& device = std::nullopt) const override {
    NoGradGuard no_grad;

    const auto& self = static_cast<const Derived&>(*this);
    auto copy = std::make_shared<Derived>(self);
    copy->parameters_.clear();
    copy->buffers_.clear();
    copy->children_.clear();
    copy->reset();
    TORCH_CHECK(
        copy->parameters_.size() == parameters_.size(),
        "The cloned module does not have the same number of "
        "parameters as the original module after calling reset(). "
        "Are you sure you called register_parameter() inside reset() "
        "and not the constructor?");
    for (const auto& parameter : named_parameters(/*recurse=*/false)) {
      auto& tensor = *parameter;
      auto data = device && tensor.device() != *device ? tensor.to(*device)
                                                       : tensor.clone();
      copy->parameters_[parameter.key()].set_data(data);
    }
    TORCH_CHECK(
        copy->buffers_.size() == buffers_.size(),
        "The cloned module does not have the same number of "
        "buffers as the original module after calling reset(). "
        "Are you sure you called register_buffer() inside reset() "
        "and not the constructor?");
    for (const auto& buffer : named_buffers(/*recurse=*/false)) {
      auto& tensor = *buffer;
      auto data = device && tensor.device() != *device ? tensor.to(*device)
                                                       : tensor.clone();
      copy->buffers_[buffer.key()].set_data(data);
    }
    TORCH_CHECK(
        copy->children_.size() == children_.size(),
        "The cloned module does not have the same number of "
        "child modules as the original module after calling reset(). "
        "Are you sure you called register_module() inside reset() "
        "and not the constructor?");
    for (const auto& child : children_) {
      copy->children_[child.key()]->clone_(*child.value(), device);
    }
    return copy;
  }

 private:
  void clone_(Module& other, const std::optional<Device>& device) final {
    // Here we are *pretty* certain that `other's` type is `Derived` (because it
    // was registered under the same name as `this`), but you never know what
    // crazy things `reset()` does, so `dynamic_cast` just to be safe.
    auto clone = std::dynamic_pointer_cast<Derived>(other.clone(device));
    TORCH_CHECK(
        clone != nullptr,
        "Attempted to clone submodule, but it is of a "
        "different type than the submodule it was to be cloned into");
    static_cast<Derived&>(*this) = *clone;
  }
};

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `does`, `that`, `Cloneable`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/module.h`
- `torch/types.h`
- `torch/utils.h`
- `c10/core/TensorOptions.h`
- `c10/util/Exception.h`
- `memory`
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

Files in the same folder (`torch/csrc/api/include/torch/nn`):

- [`module.h_docs.md`](./module.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`modules.h_docs.md`](./modules.h_docs.md)
- [`init.h_docs.md`](./init.h_docs.md)
- [`options.h_docs.md`](./options.h_docs.md)
- [`functional.h_docs.md`](./functional.h_docs.md)
- [`pimpl-inl.h_docs.md`](./pimpl-inl.h_docs.md)
- [`pimpl.h_docs.md`](./pimpl.h_docs.md)


## Cross-References

- **File Documentation**: `cloneable.h_docs.md`
- **Keyword Index**: `cloneable.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/api/include/torch/nn`):

- [`functional.h_docs.md_docs.md`](./functional.h_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`cloneable.h_kw.md_docs.md`](./cloneable.h_kw.md_docs.md)
- [`options.h_docs.md_docs.md`](./options.h_docs.md_docs.md)
- [`module.h_kw.md_docs.md`](./module.h_kw.md_docs.md)
- [`module.h_docs.md_docs.md`](./module.h_docs.md_docs.md)
- [`utils.h_kw.md_docs.md`](./utils.h_kw.md_docs.md)
- [`functional.h_kw.md_docs.md`](./functional.h_kw.md_docs.md)
- [`options.h_kw.md_docs.md`](./options.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `cloneable.h_docs.md_docs.md`
- **Keyword Index**: `cloneable.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
