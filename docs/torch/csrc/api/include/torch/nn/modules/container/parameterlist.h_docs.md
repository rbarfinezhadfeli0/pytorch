# Documentation: parameterlist.h

## File Metadata
- **Path**: `torch/csrc/api/include/torch/nn/modules/container/parameterlist.h`
- **Size**: 5558 bytes
- **Lines**: 167
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
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

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): ParameterListImpl


## Key Components

The file contains 664 words across 167 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5558 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
