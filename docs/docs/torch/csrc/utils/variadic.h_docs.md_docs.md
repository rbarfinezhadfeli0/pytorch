# Documentation: `docs/torch/csrc/utils/variadic.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/variadic.h_docs.md`
- **Size**: 5,818 bytes (5.68 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/variadic.h`

## File Metadata

- **Path**: `torch/csrc/utils/variadic.h`
- **Size**: 3,376 bytes (3.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/Variadic.h>
#include <torch/csrc/autograd/variable.h>

#include <type_traits>
#include <utility>

namespace torch {

using at::IterArgs;

struct CountTensors : IterArgs<CountTensors> {
  size_t out = 0;
  void operator()(const at::Tensor& x) {
    out += 1;
  }
  void operator()(const std::optional<at::Tensor>& x) {
    out += x.has_value();
  }
  void operator()(at::ArrayRef<at::Tensor> xs) {
    out += xs.size();
  }
};

template <typename... Args>
size_t count_tensors(Args&&... args) {
  return CountTensors().apply(std::forward<Args>(args)...).out;
}

struct CountVariables : IterArgs<CountVariables> {
  size_t out = 0;
  void operator()(const autograd::Variable& x) {
    out += 1;
  }
  void operator()(at::ArrayRef<autograd::Variable> xs) {
    out += xs.size();
  }
};

template <typename... Args>
inline size_t count_variables(Args&&... args) {
  return CountVariables().apply(std::forward<Args>(args)...).out;
}

//===----------------------------------------------------------------------===//
//                std::index_sequence shim for C++11
//===----------------------------------------------------------------------===//

// A container of type-template parameter indices.
template <size_t... Is>
struct Indices {};

// Decrements the index N, adds N-1 to the list of indices and forwards
// whatever we already have.
template <size_t N, size_t... Is>
struct MakeIndices : MakeIndices<N - 1, N - 1, Is...> {};

// Partial specialization that forms our base case. When N is zero, we stop
// and define a typedef that will be visible to earlier classes due to
// inheritance. The typedef we define is an index list containing the numbers
// 0 through N-1.
template <size_t... Is>
struct MakeIndices<0, Is...> {
  using indices = Indices<Is...>;
};

//===----------------------------------------------------------------------===//
//                                 Utilities
//===----------------------------------------------------------------------===//

template <typename Function, typename... Ts>
void apply(Function function, Ts&&... ts) {
  // https://stackoverflow.com/questions/13978916/inserting-a-variadic-argument-list-into-a-vector
  // Creates a dummy array, so that each function call is evaluated in order.
  // `(function(), 0)` is because `function` should (!) return `void`, so
  // according to the comma operator, it is evaluated and its result (`void`)
  // is discarded. Then the zero is evaluated and used as an element in the
  // array. The first zero ensures the array is not empty.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  int _[]{0, (function(std::forward<Ts>(ts)), 0)...};
  (void)_;
}

template <
    typename ReturnType,
    typename... Ts,
    typename Function,
    typename Accessor>
ReturnType unpack(Function function, Accessor accessor) {
  return ReturnType(unpack<ReturnType, Ts...>(
      std::move(function),
      std::move(accessor),
      typename MakeIndices<sizeof...(Ts)>::indices()));
}

template <
    typename ReturnType,
    typename... Ts,
    typename Function,
    typename Accessor,
    size_t... Is>
ReturnType unpack(
    Function function,
    Accessor accessor,
    Indices<Is...> /*unused*/) {
  return ReturnType(function(accessor.template operator()<Ts>(Is)...));
}

} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `CountTensors`, `CountVariables`, `Indices`, `MakeIndices`, `MakeIndices`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/core/Variadic.h`
- `torch/csrc/autograd/variable.h`
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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`disable_torch_function.cpp_docs.md`](./disable_torch_function.cpp_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `variadic.h_docs.md`
- **Keyword Index**: `variadic.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/utils`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/utils`):

- [`python_tuples.h_kw.md_docs.md`](./python_tuples.h_kw.md_docs.md)
- [`six.h_kw.md_docs.md`](./six.h_kw.md_docs.md)
- [`tensor_types.cpp_docs.md_docs.md`](./tensor_types.cpp_docs.md_docs.md)
- [`tensor_list.h_kw.md_docs.md`](./tensor_list.h_kw.md_docs.md)
- [`verbose.h_kw.md_docs.md`](./verbose.h_kw.md_docs.md)
- [`invalid_arguments.cpp_kw.md_docs.md`](./invalid_arguments.cpp_kw.md_docs.md)
- [`tensor_apply.h_kw.md_docs.md`](./tensor_apply.h_kw.md_docs.md)
- [`cuda_enabled.h_docs.md_docs.md`](./cuda_enabled.h_docs.md_docs.md)
- [`tensor_layouts.h_docs.md_docs.md`](./tensor_layouts.h_docs.md_docs.md)
- [`variadic.h_kw.md_docs.md`](./variadic.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `variadic.h_docs.md_docs.md`
- **Keyword Index**: `variadic.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
