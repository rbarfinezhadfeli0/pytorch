# Documentation: `torch/csrc/api/src/optim/optimizer.cpp`

## File Metadata

- **Path**: `torch/csrc/api/src/optim/optimizer.cpp`
- **Size**: 6,489 bytes (6.34 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/optim/optimizer.h>

#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>

#include <string>
#include <utility>
#include <vector>

namespace torch::optim {

bool OptimizerParamGroup::has_options() const {
  return options_ != nullptr;
}

OptimizerOptions& OptimizerParamGroup::options() {
  TORCH_CHECK(has_options());
  return *options_;
}

const OptimizerOptions& OptimizerParamGroup::options() const {
  TORCH_CHECK(has_options());
  return *options_;
}

void OptimizerParamGroup::set_options(
    std::unique_ptr<OptimizerOptions> options) {
  options_ = std::move(options);
}

std::vector<Tensor>& OptimizerParamGroup::params() {
  return params_;
}

const std::vector<Tensor>& OptimizerParamGroup::params() const {
  return params_;
}

std::unique_ptr<OptimizerParamState> OptimizerParamState::clone() const {
  TORCH_CHECK(
      false,
      "clone() has not been implemented for torch::optim::OptimizerParamState. ",
      "Subclass torch::optim::OptimizerCloneableParamState<YourOptimizerParamState> ",
      "instead of torch::optim::OptimizerParamState to inherit the ability to clone.");
}

void OptimizerParamState::serialize(torch::serialize::InputArchive& archive) {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::InputArchive& archive) has not been implemented for torch::optim::OptimizerParamState. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
}

void OptimizerParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::OutputArchive& archive) has not been implemented for torch::optim::OptimizerParamState. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableParamState<YourOptimizerParamState>.");
}

double OptimizerOptions::get_lr() const {
  TORCH_CHECK(
      false,
      "double get_lr() has not been overridden and implemented in subclass of torch::optim::OptimizerOptions, you must override it in your subclass.");
}

void OptimizerOptions::set_lr(const double lr) {
  TORCH_CHECK(
      false,
      "double set_lr() has not been overridden and implemented in subclass of torch::optim::OptimizerOptions, you must override it in your subclass.");
}

std::unique_ptr<OptimizerOptions> OptimizerOptions::clone() const {
  TORCH_CHECK(
      false,
      "clone() has not been implemented for torch::optim::OptimizerOptions. ",
      "Subclass torch::optim::OptimizerCloneableOptions<YourOptimizerOptions> ",
      "instead of torch::optim::OptimizerOptions to inherit the ability to clone.");
}

void OptimizerOptions::serialize(torch::serialize::InputArchive& archive) {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::InputArchive& archive) has not been implemented for torch::optim::OptimizerOptions. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

void OptimizerOptions::serialize(
    torch::serialize::OutputArchive& archive) const {
  TORCH_CHECK(
      false,
      "void serialize(torch::serialize::OutputArchive& archive) has not been implemented for torch::optim::OptimizerOptions. ",
      "You must override it in your subclass of torch::optim::OptimizerCloneableOptions<YourOptimizerOptions>.");
}

void Optimizer::add_param_group(const OptimizerParamGroup& param_group) {
  for (const auto& param : param_group.params()) {
    TORCH_CHECK(param.is_leaf(), "can't optimize a non-leaf Tensor");
  }
  TORCH_INTERNAL_ASSERT(defaults_ != nullptr);
  OptimizerParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  for (const auto& p : param_group_.params()) {
    TORCH_CHECK(
        state_.count(p.unsafeGetTensorImpl()) == 0,
        "some parameters appear in more than one parameter group");
  }
  param_groups_.emplace_back(std::move(param_group_));
}

void Optimizer::add_parameters(const std::vector<Tensor>& parameters) {
  TORCH_WARN("Optimizer::add_parameters() will be removed in PyTorch 1.6");
  auto& parameters_ = param_groups_[0].params();
  parameters_.insert(parameters_.end(), parameters.begin(), parameters.end());
}

void Optimizer::zero_grad(bool set_to_none) {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (p.mutable_grad().defined()) {
        p.mutable_grad().detach_();
        if (set_to_none)
          p.mutable_grad().reset();
        else
          p.mutable_grad().zero_();
      }
    }
  }
}

const std::vector<Tensor>& Optimizer::parameters() const noexcept {
  TORCH_WARN("Optimizer::parameters() will be removed in PyTorch 1.6");
  return param_groups_.at(0).params();
}

std::vector<Tensor>& Optimizer::parameters() noexcept {
  TORCH_WARN("Optimizer::parameters() will be removed in PyTorch 1.6");
  return param_groups_.at(0).params();
}

size_t Optimizer::size() const noexcept {
  TORCH_WARN("Optimizer::size() will be removed in PyTorch 1.6");
  size_t count = 0;
  for (const auto& group : param_groups_) {
    count += group.params().size();
  }
  return count;
}

OptimizerOptions& Optimizer::defaults() noexcept {
  return *defaults_;
}

const OptimizerOptions& Optimizer::defaults() const noexcept {
  return *defaults_;
}

std::vector<OptimizerParamGroup>& Optimizer::param_groups() noexcept {
  return param_groups_;
}

const std::vector<OptimizerParamGroup>& Optimizer::param_groups()
    const noexcept {
  return param_groups_;
}

ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& Optimizer::
    state() noexcept {
  return state_;
}

const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
Optimizer::state() const noexcept {
  return state_;
}

void Optimizer::save(serialize::OutputArchive& archive) const {}
void Optimizer::load(serialize::InputArchive& archive) {}

/// Serializes an `Optimizer` into an `OutputArchive`.
serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Optimizer& optimizer) {
  optimizer.save(archive);
  return archive;
}

/// Deserializes a `Tensor` from an `InputArchive`.
serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Optimizer& optimizer) {
  optimizer.load(archive);
  return archive;
}

} // namespace torch::optim

```



## High-Level Overview


This C++ file contains approximately 8 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `torch`, `of`, `of`, `of`, `of`, `torch`, `of`, `of`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/src/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/optim/optimizer.h`
- `torch/csrc/autograd/generated/variable_factories.h`
- `torch/types.h`
- `string`
- `utility`
- `vector`


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

Files in the same folder (`torch/csrc/api/src/optim`):

- [`adam.cpp_docs.md`](./adam.cpp_docs.md)
- [`lbfgs.cpp_docs.md`](./lbfgs.cpp_docs.md)
- [`sgd.cpp_docs.md`](./sgd.cpp_docs.md)
- [`serialize.cpp_docs.md`](./serialize.cpp_docs.md)
- [`rmsprop.cpp_docs.md`](./rmsprop.cpp_docs.md)
- [`adagrad.cpp_docs.md`](./adagrad.cpp_docs.md)
- [`adamw.cpp_docs.md`](./adamw.cpp_docs.md)


## Cross-References

- **File Documentation**: `optimizer.cpp_docs.md`
- **Keyword Index**: `optimizer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
