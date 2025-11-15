# Documentation: `torch/csrc/api/include/torch/nn/modules/container/parameterlist.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/container/parameterlist.h`
- **Size**: 5,558 bytes (5.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>

#include <vector>

namespace torch::nn {
class ParameterListImpl : public Cloneable<ParameterListImpl> {
 public:
  using Iterator =
      std::vector<OrderedDict<std::string, torch::Tensor>::Item>::iterator;
  using ConstIterator = std::vector<
      OrderedDict<std::string, torch::Tensor>::Item>::const_iterator;

  ParameterListImpl() = default;

  /// Constructs the `ParameterList` from a variadic list of ParameterList.
  template <typename... Tensors>
  explicit ParameterListImpl(Tensors&&... params) {
    parameters_.reserve(sizeof...(Tensors));
    push_back_var(std::forward<Tensors>(params)...);
  }

  template <typename... Tensors>
  explicit ParameterListImpl(const Tensors&... params) {
    parameters_.reserve(sizeof...(Tensors));
    push_back_var(std::forward<Tensors>(params)...);
  }

  /// `reset()` is empty for `ParameterList`, since it does not have parameters
  /// of its own.
  void reset() override {}

  /// Pretty prints the `ParameterList` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "torch::nn::ParameterList(" << '\n';
    for (const auto& pair : parameters_) {
      stream << "(" << pair.key() << ")"
             << ": Parameter containing: [" << pair.value().scalar_type()
             << " of size " << pair.value().sizes() << "]";
      ;
      stream << '\n';
    }
    stream << ")";
  }

  /// push the a given parameter at the end of the list
  void append(torch::Tensor&& param) {
    bool requires_grad = param.requires_grad();
    register_parameter(
        std::to_string(parameters_.size()), std::move(param), requires_grad);
  }

  /// push the a given parameter at the end of the list
  void append(const torch::Tensor& param) {
    bool requires_grad = param.requires_grad();
    register_parameter(
        std::to_string(parameters_.size()), param, requires_grad);
  }

  /// push the a given parameter at the end of the list
  /// And the key of the pair will be discarded, only the value
  /// will be added into the `ParameterList`
  void append(const OrderedDict<std::string, torch::Tensor>::Item& pair) {
    register_parameter(
        std::to_string(parameters_.size()),
        pair.value(),
        pair.value().requires_grad());
  }

  /// extend parameters from a container to the end of the list
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& param : container) {
      append(param);
    }
  }

  /// Returns an iterator to the start of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string,
  /// torch::Tensor>::Item`
  Iterator begin() {
    return parameters_.begin();
  }

  /// Returns a const iterator to the start of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string,
  /// torch::Tensor>::Item`
  ConstIterator begin() const {
    return parameters_.begin();
  }

  /// Returns an iterator to the end of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string,
  /// torch::Tensor>::Item`
  Iterator end() {
    return parameters_.end();
  }

  /// Returns a const iterator to the end of the ParameterList
  /// the iterator returned will be type of `OrderedDict<std::string,
  /// torch::Tensor>::Item`
  ConstIterator end() const {
    return parameters_.end();
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterList`. Check contains(key) before
  /// for a non-throwing way of access
  at::Tensor& at(size_t idx) {
    TORCH_CHECK(idx < size(), "Index out of range");
    return parameters_[std::to_string(idx)];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterList`. Check contains(key) before
  /// for a non-throwing way of access
  const at::Tensor& at(size_t idx) const {
    TORCH_CHECK(idx < size(), "Index out of range");
    return parameters_[std::to_string(idx)];
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterList`. Check contains(key) before
  /// for a non-throwing way of access
  at::Tensor& operator[](size_t idx) {
    return at(idx);
  }

  /// Returns the value associated with the given `key`. Throws an exception if
  /// no such key is stored in the `ParameterList`. Check contains(key) before
  /// for a non-throwing way of access
  const at::Tensor& operator[](size_t idx) const {
    return at(idx);
  }

  /// Return the size of the ParameterList
  size_t size() const noexcept {
    return parameters_.size();
  }
  /// True if the ParameterList is empty
  bool is_empty() const noexcept {
    return parameters_.is_empty();
  }

  /// Overload the +=, so that two ParameterList could be incrementally added
  template <typename Container>
  Container& operator+=(const Container& other) {
    extend(other);
    return *this;
  }

 private:
  template <typename Head, typename... Tail>
  void push_back_var(Head&& head, Tail&&... tail) {
    append(std::forward<Head>(head));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `push_back()` a final time (above).
    push_back_var(std::forward<Tail>(tail)...);
  }

  /// The base case, when the list of modules is empty.
  void push_back_var() {}
};
TORCH_MODULE(ParameterList);
} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 23 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ParameterListImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules/container`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/cloneable.h`
- `torch/nn/module.h`
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

Files in the same folder (`torch/csrc/api/include/torch/nn/modules/container`):

- [`moduledict.h_docs.md`](./moduledict.h_docs.md)
- [`sequential.h_docs.md`](./sequential.h_docs.md)
- [`any_module_holder.h_docs.md`](./any_module_holder.h_docs.md)
- [`modulelist.h_docs.md`](./modulelist.h_docs.md)
- [`named_any.h_docs.md`](./named_any.h_docs.md)
- [`functional.h_docs.md`](./functional.h_docs.md)
- [`any.h_docs.md`](./any.h_docs.md)
- [`any_value.h_docs.md`](./any_value.h_docs.md)
- [`parameterdict.h_docs.md`](./parameterdict.h_docs.md)


## Cross-References

- **File Documentation**: `parameterlist.h_docs.md`
- **Keyword Index**: `parameterlist.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
