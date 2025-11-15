# Documentation: `torch/csrc/api/include/torch/nn/modules/container/functional.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/container/functional.h`
- **Size**: 3,343 bytes (3.26 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <torch/nn/cloneable.h>
#include <torch/types.h>

#include <functional>
#include <utility>

namespace torch::nn {

/// Wraps a function in a `Module`.
///
/// The `Functional` module allows wrapping an arbitrary function or function
/// object in an `nn::Module`. This is primarily handy for usage in
/// `Sequential`.
///
/// \rst
/// .. code-block:: cpp
///
///   Sequential sequential(
///     Linear(3, 4),
///     Functional(torch::relu),
///     BatchNorm1d(3),
///     Functional(torch::elu, /*alpha=*/1));
/// \endrst
///
/// While a `Functional` module only accepts a single `Tensor` as input, it is
/// possible for the wrapped function to accept further arguments. However,
/// these have to be bound *at construction time*. For example, if
/// you want to wrap `torch::leaky_relu`, which accepts a `slope` scalar as its
/// second argument, with a particular value for its `slope` in a `Functional`
/// module, you could write
///
/// \rst
/// .. code-block:: cpp
///
///   Functional(torch::leaky_relu, /*slope=*/0.5)
/// \endrst
///
/// The value of `0.5` is then stored within the `Functional` object and
/// supplied to the function call at invocation time. Note that such bound
/// values are evaluated eagerly and stored a single time. See the documentation
/// of [std::bind](https://en.cppreference.com/w/cpp/utility/functional/bind)
/// for more information on the semantics of argument binding.
///
/// \rst
/// .. attention::
///   After passing any bound arguments, the function must accept a single
///   tensor and return a single tensor.
/// \endrst
///
/// Note that `Functional` overloads the call operator (`operator()`) such that
/// you can invoke it with `my_func(...)`.
class TORCH_API FunctionalImpl : public torch::nn::Cloneable<FunctionalImpl> {
 public:
  using Function = std::function<Tensor(Tensor)>;

  /// Constructs a `Functional` from a function object.
  explicit FunctionalImpl(Function function);

  template <
      typename SomeFunction,
      typename... Args,
      typename = std::enable_if_t<(sizeof...(Args) > 0)>>
  explicit FunctionalImpl(SomeFunction original_function, Args&&... args)
      // NOLINTNEXTLINE(modernize-avoid-bind)
      : function_(std::bind(
            original_function,
            /*input=*/std::placeholders::_1,
            std::forward<Args>(args)...)) {
    // std::bind is normally evil, but (1) gcc is broken w.r.t. handling
    // parameter pack expansion in lambdas and (2) moving parameter packs into
    // a lambda only works with C++14, so std::bind is the more move-aware
    // solution here.
  }

  void reset() override;

  /// Pretty prints the `Functional` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// Forwards the `input` tensor to the underlying (bound) function object.
  Tensor forward(Tensor input);

  /// Calls forward(input).
  Tensor operator()(Tensor input);

  bool is_serializable() const override;

 private:
  Function function_;
};

/// A `ModuleHolder` subclass for `FunctionalImpl`.
/// See the documentation for `FunctionalImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
TORCH_MODULE(Functional);

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `for`, `to`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules/container`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/nn/cloneable.h`
- `torch/types.h`
- `functional`
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
- [`named_any.h_docs.md`](./named_any.h_docs.md)
- [`any.h_docs.md`](./any.h_docs.md)
- [`any_value.h_docs.md`](./any_value.h_docs.md)
- [`parameterdict.h_docs.md`](./parameterdict.h_docs.md)


## Cross-References

- **File Documentation**: `functional.h_docs.md`
- **Keyword Index**: `functional.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
