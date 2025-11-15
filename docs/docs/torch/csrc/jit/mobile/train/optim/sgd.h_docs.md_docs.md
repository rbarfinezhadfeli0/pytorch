# Documentation: `docs/torch/csrc/jit/mobile/train/optim/sgd.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/mobile/train/optim/sgd.h_docs.md`
- **Size**: 6,304 bytes (6.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/mobile/train/optim/sgd.h`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/train/optim/sgd.h`
- **Size**: 4,331 bytes (4.23 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/arg.h>
#include <torch/types.h>

#include <utility>
#include <vector>

namespace torch::jit::mobile {

class SGDParamState {
  TORCH_ARG(torch::Tensor, momentum_buffer);

 public:
  std::unique_ptr<SGDParamState> clone() const {
    return std::make_unique<SGDParamState>(
        static_cast<const SGDParamState&>(*this));
  }
  friend bool operator==(const SGDParamState& lhs, const SGDParamState& rhs);
};

struct TORCH_API SGDOptions {
  /* implicit */ SGDOptions(double lr);
  TORCH_ARG(double, lr);
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(double, dampening) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, nesterov) = false;

 public:
  std::unique_ptr<SGDOptions> clone() const {
    return std::make_unique<SGDOptions>(static_cast<const SGDOptions&>(*this));
  }
  TORCH_API friend bool operator==(
      const SGDOptions& lhs,
      const SGDOptions& rhs);
};

/// Stores parameters in the param_group and stores a pointer to the SGDOptions
class TORCH_API SGDParamGroup {
 public:
  // NOTE: In order to store `SGDParamGroup` in a `std::vector`, it has to be
  // copy-constructible.
  SGDParamGroup(const SGDParamGroup& param_group)
      : params_(param_group.params()),
        options_(
            param_group.has_options() ? param_group.options().clone()
                                      : nullptr) {}
  SGDParamGroup& operator=(const SGDParamGroup& param_group) {
    this->params_ = param_group.params();
    this->options_ =
        param_group.has_options() ? param_group.options().clone() : nullptr;
    return *this;
  }
  /* implicit */ SGDParamGroup(std::vector<Tensor> params)
      : params_(std::move(params)) {}
  SGDParamGroup(std::vector<Tensor> params, std::unique_ptr<SGDOptions> options)
      : params_(std::move(params)), options_(std::move(options)) {}

  bool has_options() const;
  SGDOptions& options();
  const SGDOptions& options() const;
  void set_options(std::unique_ptr<SGDOptions> options);
  std::vector<Tensor>& params();
  const std::vector<Tensor>& params() const;

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<Tensor> params_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<SGDOptions> options_;
};

class TORCH_API SGD {
 public:
  explicit SGD(
      const std::vector<torch::jit::mobile::SGDParamGroup>& param_groups,
      SGDOptions defaults)
      : defaults_(std::make_unique<SGDOptions>(defaults)) {
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    }
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        !defaults.nesterov() ||
            (defaults.momentum() > 0 && defaults.dampening() == 0),
        "Nesterov momentum requires a momentum and zero dampening");
  }

  explicit SGD(std::vector<Tensor> params, SGDOptions defaults)
      : SGD({SGDParamGroup(std::move(params))}, defaults) {}

  /// Adds the given param_group to the optimizer's param_group list.
  void add_param_group(const SGDParamGroup& param_group);

  ~SGD() = default;

  using LossClosure = std::function<Tensor()>;
  /// A loss function closure, which is expected to return the loss value.
  torch::Tensor step(const LossClosure& closure = nullptr);

  /// Zeros out the gradients of all parameters.
  void zero_grad();

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<SGDParamGroup> param_groups_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  ska::flat_hash_map<void*, std::unique_ptr<SGDParamState>> state_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<SGDOptions> defaults_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<Tensor> params_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<SGDOptions> options_;
};
} // namespace torch::jit::mobile

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `SGDParamState`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile/train/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/arg.h`
- `torch/types.h`
- `utility`
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

Files in the same folder (`torch/csrc/jit/mobile/train/optim`):

- [`sgd.cpp_docs.md`](./sgd.cpp_docs.md)


## Cross-References

- **File Documentation**: `sgd.h_docs.md`
- **Keyword Index**: `sgd.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/mobile/train/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/mobile/train/optim`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/mobile/train/optim`):

- [`sgd.h_kw.md_docs.md`](./sgd.h_kw.md_docs.md)
- [`sgd.cpp_docs.md_docs.md`](./sgd.cpp_docs.md_docs.md)
- [`sgd.cpp_kw.md_docs.md`](./sgd.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `sgd.h_docs.md_docs.md`
- **Keyword Index**: `sgd.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
