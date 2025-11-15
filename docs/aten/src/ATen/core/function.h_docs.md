# Documentation: `aten/src/ATen/core/function.h`

## File Metadata

- **Path**: `aten/src/ATen/core/function.h`
- **Size**: 3,510 bytes (3.43 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/qualified_name.h>
#include <c10/util/Exception.h>
#include <c10/util/FunctionRef.h>

namespace c10 {
struct FunctionSchema;
}

namespace at {
TORCH_API void launch(std::function<void()> func);
}

namespace torch::jit {

struct Graph;
struct Code;

namespace mobile {
struct Code;
}

using Stack = std::vector<at::IValue>;
using Kwargs = std::unordered_map<std::string, at::IValue>;
struct RecursiveMethodCallError : public std::exception {};
using TaskLauncher = std::function<void(std::function<void()>)>;

TORCH_API void preoptimizeGraph(
    std::shared_ptr<Graph>& graph,
    bool disable_autocast = false);

// A Function is a pure Graph with no implicit `self` object bound.
// It contains schema information and the executor that manages the
// execution of the function. Method is a wrapper around an
// underlying Function that also provides a `self` object.
struct TORCH_API Function {
  Function() = default;
  Function(const Function&) = default;
  Function& operator=(const Function&) = default;
  Function(Function&&) noexcept = default;
  Function& operator=(Function&&) noexcept = default;
  virtual std::string_view doc_string() const {
    static constexpr std::string_view no_doc_string;
    return no_doc_string;
  }

  virtual bool isGraphFunction() const {
    return false;
  }

  virtual void run(Stack& stack) = 0;

  virtual c10::intrusive_ptr<c10::ivalue::Future> runAsync(
      Stack& /*stack*/,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      [[maybe_unused]] TaskLauncher taskLauncher = at::launch) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return {};
  }

  at::IValue operator()(Stack stack, const Kwargs& kwargs = Kwargs()) {
    getSchema().checkAndNormalizeInputs(stack, kwargs);
    run(stack);
    return stack.front();
  }

  virtual const c10::QualifiedName& qualname() const = 0;

  const std::string& name() const {
    return qualname().name();
  }

  // if this isn't yet defined, run its method_creator function
  virtual void ensure_defined() = 0;

  virtual const c10::FunctionSchema& getSchema() const = 0;

  virtual size_t num_inputs() const = 0;

  virtual Function& setSchema(c10::FunctionSchema schema) = 0;

  // call() defines how different interpreter implementations interacts with
  // Function objects. Basically interpreters need to provide a callback to
  // communicate to Functions what to do if provided a Code object.
  // Alternatively we could design the signature to return an optional Code
  // object, but that requires special handling the null case in interpreter
  // and the fallback behavior is not well defined by interpreter but rather
  // Function themselves, so a callback approach is more reasonable than
  // returning values.
  // If call() returns true, then callback completes successfully, otherwise
  // call() returns false.

  // Overload for server interpreter, a bailout size is needed for graph
  // executor.
  virtual bool call(
      Stack& /*unused*/,
      std::optional<size_t> /*unused*/,
      c10::function_ref<void(const Code&)> /*unused*/) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return false;
  }

  // Overload for mobile interpreter.
  virtual bool call(Stack& /*unused*/, c10::function_ref<void(const mobile::Code&)> /*unused*/) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(false);
    return false;
  }

  virtual ~Function() = default;
};
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`, `mobile`, `torch`, `c10`

**Classes/Structs**: `FunctionSchema`, `Graph`, `Code`, `Code`, `RecursiveMethodCallError`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/function_schema.h`
- `ATen/core/ivalue.h`
- `ATen/core/qualified_name.h`
- `c10/util/Exception.h`
- `c10/util/FunctionRef.h`


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

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `function.h_docs.md`
- **Keyword Index**: `function.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
