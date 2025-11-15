# Documentation: `docs/torch/csrc/jit/backends/backend.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/backends/backend.h_docs.md`
- **Size**: 6,377 bytes (6.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/backends/backend.h`

## File Metadata

- **Path**: `torch/csrc/jit/backends/backend.h`
- **Size**: 3,832 bytes (3.74 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/builtin_function.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/backends/backend_interface.h>
#include <torch/custom_class.h>

namespace torch::jit {
namespace {
inline c10::FunctionSchema getIsAvailableSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument available("available", c10::BoolType::get());
  c10::FunctionSchema preprocessor_schema(
      "is_available",
      /*overload_name=*/"",
      /*arguments=*/{self},
      /*returns=*/{available});
  return preprocessor_schema;
}

constexpr static auto kBackendsNamespace = "__backends__";

inline c10::FunctionSchema getCompileSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument mod("processed", c10::AnyType::get());
  auto any_dict_ty =
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get());
  c10::Argument method_compile_spec("method_compile_spec", any_dict_ty);
  c10::Argument handles("handles", any_dict_ty);

  c10::FunctionSchema compile_schema(
      "compile",
      /*overload_name=*/"",
      /*arguments=*/{self, mod, method_compile_spec},
      /*returns=*/{handles});
  return compile_schema;
}

inline c10::FunctionSchema getExecuteSchema() {
  auto any_list_ty = c10::ListType::create(c10::AnyType::get());
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument handle("handle", c10::AnyType::get());
  c10::Argument input("input", any_list_ty);
  c10::Argument output("output", any_list_ty);
  return c10::FunctionSchema(
      "execute",
      /*overload_name=*/"",
      /*arguments=*/{self, handle, input},
      /*returns=*/{output});
}

template <typename TBackendInterface>
std::function<void(Stack&)> getIsAvailableFunc() {
  return [](Stack& stack) {
    auto self = pop(stack).toCustomClass<TBackendInterface>();
    auto ret = self->is_available();
    push(stack, ret);
  };
}

template <typename TBackendInterface>
std::function<void(Stack&)> getCompileFunc() {
  return [](Stack& stack) {
    auto method_compile_spec = pop(stack).toGenericDict();
    auto processed = pop(stack);
    auto self = pop(stack).toCustomClass<TBackendInterface>();
    auto ret = self->compile(processed, method_compile_spec);
    push(stack, ret);
  };
}

template <typename TBackendInterface>
std::function<void(Stack&)> getExecuteFunc() {
  return [](Stack& stack) {
    auto args = pop(stack);
    auto handle = pop(stack);
    auto self = pop(stack);
    auto backend = self.toCustomClass<TBackendInterface>();
    auto res = backend->execute(handle, args.toList());
    push(stack, res);
  };
}
} // namespace

// Static registration API for backends.
template <class TBackendInterface>
class backend {
  static_assert(
      std::is_base_of_v<PyTorchBackendInterface, TBackendInterface>,
      "torch::jit::backend<T> requires T to inherit from PyTorchBackendInterface");
  std::string backend_name_;

 public:
  // Registers a new backend with /p name, and the given /p preprocess
  // function.
  backend(const std::string& name) : backend_name_(name) {
    static auto cls = torch::class_<TBackendInterface>(kBackendsNamespace, name)
                          .def(torch::init<>())
                          ._def_unboxed(
                              "is_available",
                              getIsAvailableFunc<TBackendInterface>(),
                              getIsAvailableSchema())
                          ._def_unboxed(
                              "compile",
                              getCompileFunc<TBackendInterface>(),
                              getCompileSchema())
                          ._def_unboxed(
                              "execute",
                              getExecuteFunc<TBackendInterface>(),
                              getExecuteSchema());
  }
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TBackendInterface`, `backend`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/backends`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/builtin_function.h`
- `ATen/core/stack.h`
- `torch/csrc/jit/backends/backend_interface.h`
- `torch/custom_class.h`


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

Files in the same folder (`torch/csrc/jit/backends`):

- [`backend_resolver.h_docs.md`](./backend_resolver.h_docs.md)
- [`backend_debug_handler.cpp_docs.md`](./backend_debug_handler.cpp_docs.md)
- [`backend_detail.h_docs.md`](./backend_detail.h_docs.md)
- [`backend_debug_info.cpp_docs.md`](./backend_debug_info.cpp_docs.md)
- [`backend_init.h_docs.md`](./backend_init.h_docs.md)
- [`backend_detail.cpp_docs.md`](./backend_detail.cpp_docs.md)
- [`backend_exception.h_docs.md`](./backend_exception.h_docs.md)
- [`backend_resolver.cpp_docs.md`](./backend_resolver.cpp_docs.md)
- [`backend_interface.cpp_docs.md`](./backend_interface.cpp_docs.md)


## Cross-References

- **File Documentation**: `backend.h_docs.md`
- **Keyword Index**: `backend.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/backends`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/backends`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/backends`):

- [`backend_interface.cpp_kw.md_docs.md`](./backend_interface.cpp_kw.md_docs.md)
- [`backend_init.h_kw.md_docs.md`](./backend_init.h_kw.md_docs.md)
- [`backend_debug_handler.cpp_kw.md_docs.md`](./backend_debug_handler.cpp_kw.md_docs.md)
- [`backend_exception.h_kw.md_docs.md`](./backend_exception.h_kw.md_docs.md)
- [`backend_detail.cpp_docs.md_docs.md`](./backend_detail.cpp_docs.md_docs.md)
- [`backend_init.h_docs.md_docs.md`](./backend_init.h_docs.md_docs.md)
- [`backend_init.cpp_kw.md_docs.md`](./backend_init.cpp_kw.md_docs.md)
- [`backend_debug_info.h_kw.md_docs.md`](./backend_debug_info.h_kw.md_docs.md)
- [`backend_resolver.cpp_docs.md_docs.md`](./backend_resolver.cpp_docs.md_docs.md)
- [`backend_debug_info.h_docs.md_docs.md`](./backend_debug_info.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `backend.h_docs.md_docs.md`
- **Keyword Index**: `backend.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
