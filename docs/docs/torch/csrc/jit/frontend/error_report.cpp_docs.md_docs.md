# Documentation: `docs/torch/csrc/jit/frontend/error_report.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/frontend/error_report.cpp_docs.md`
- **Size**: 6,296 bytes (6.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/frontend/error_report.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/frontend/error_report.cpp`
- **Size**: 3,684 bytes (3.60 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/frontend/error_report.h>

#include <torch/csrc/jit/frontend/tree.h>

namespace torch::jit {

// Avoid storing objects with destructor in thread_local for mobile build.
#ifndef C10_MOBILE
// [NOTE: Thread-safe CallStack]
// `calls` maintains a stack of Python calls that resulted in the
// currently compiled TorchScript code. RAII ErrorReport::CallStack
// push and pop from the `calls` object during compilation to track
// these stacks so that they can be used to report compilation errors
//
// Q: Why can't this just be a thread_local vector<Call> (as it was previously)?
//
// A: Sometimes a CallStack RAII guard is created in Python in a given
//    thread (say, thread A). Then later, someone can call
//    sys._current_frames() from another thread (thread B), which causes
//    thread B to hold references to the CallStack guard. e.g.
//    1. CallStack RAII guard created by thread A
//    2. CallStack guard now has a reference from thread B
//    3. thread A releases guard, but thread B still holds a reference
//    4. thread B releases guard, refcount goes to 0, and we
//       call the destructor
//    under this situation, **we pop an element off the wrong `call`
//    object (from the wrong thread!)
//
//    To fix this:
//    * in CallStack, store a reference to which thread's `calls`
//      the CallStack corresponds to, so you can pop from the correct
//      `calls` object.
//    * make it a shared_ptr and add a mutex to make this thread safe
//      (since now multiple threads access a given thread_local calls object)
static thread_local std::shared_ptr<ErrorReport::Calls> calls =
    std::make_shared<ErrorReport::Calls>();
#endif // C10_MOBILE

ErrorReport::ErrorReport(const ErrorReport& e)
    : ss(e.ss.str()),
      context(e.context),
      the_message(e.the_message),
      error_stack(e.error_stack.begin(), e.error_stack.end()) {}

#ifndef C10_MOBILE
ErrorReport::ErrorReport(const SourceRange& r)
    : context(r), error_stack(calls->get_stack()) {}

void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {
  calls->update_pending_range(range);
}

ErrorReport::CallStack::CallStack(
    const std::string& name,
    const SourceRange& range) {
  source_callstack_ = calls;
  source_callstack_->push_back({name, range});
}

ErrorReport::CallStack::~CallStack() {
  if (source_callstack_) {
    source_callstack_->pop_back();
  }
}
#else // defined C10_MOBILE
ErrorReport::ErrorReport(const SourceRange& r) : context(r) {}

void ErrorReport::CallStack::update_pending_range(const SourceRange& range) {}

ErrorReport::CallStack::CallStack(
    const std::string& name,
    const SourceRange& range) {}

ErrorReport::CallStack::~CallStack() {}
#endif // C10_MOBILE

static std::string get_stacked_errors(const std::vector<Call>& error_stack) {
  std::stringstream msg;
  if (!error_stack.empty()) {
    for (auto it = error_stack.rbegin(); it != error_stack.rend() - 1; ++it) {
      auto callee = it + 1;

      msg << "'" << it->fn_name
          << "' is being compiled since it was called from '" << callee->fn_name
          << "'\n";
      callee->caller_range.highlight(msg);
    }
  }
  return msg.str();
}

std::string ErrorReport::current_call_stack() {
#ifndef C10_MOBILE
  return get_stacked_errors(calls->get_stack());
#else
  TORCH_CHECK(false, "Call stack not supported on mobile");
#endif // C10_MOBILE
}

const char* ErrorReport::what() const noexcept {
  std::stringstream msg;
  msg << "\n" << ss.str();
  msg << ":\n";
  context.highlight(msg);

  msg << get_stacked_errors(error_stack);

  the_message = msg.str();
  return the_message.c_str();
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/frontend/error_report.h`
- `torch/csrc/jit/frontend/tree.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/jit/frontend`):

- [`canonicalize_modified_loop.cpp_docs.md`](./canonicalize_modified_loop.cpp_docs.md)
- [`schema_matching.cpp_docs.md`](./schema_matching.cpp_docs.md)
- [`source_range.h_docs.md`](./source_range.h_docs.md)
- [`exit_transforms.h_docs.md`](./exit_transforms.h_docs.md)
- [`function_schema_parser.h_docs.md`](./function_schema_parser.h_docs.md)
- [`inline_loop_condition.h_docs.md`](./inline_loop_condition.h_docs.md)
- [`mini_environment.h_docs.md`](./mini_environment.h_docs.md)
- [`tree_views.cpp_docs.md`](./tree_views.cpp_docs.md)
- [`function_schema_parser.cpp_docs.md`](./function_schema_parser.cpp_docs.md)
- [`tracer.cpp_docs.md`](./tracer.cpp_docs.md)


## Cross-References

- **File Documentation**: `error_report.cpp_docs.md`
- **Keyword Index**: `error_report.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/frontend`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/frontend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/jit/frontend`):

- [`strtod.h_kw.md_docs.md`](./strtod.h_kw.md_docs.md)
- [`tree_views.cpp_docs.md_docs.md`](./tree_views.cpp_docs.md_docs.md)
- [`function_schema_parser.cpp_docs.md_docs.md`](./function_schema_parser.cpp_docs.md_docs.md)
- [`tree.h_kw.md_docs.md`](./tree.h_kw.md_docs.md)
- [`versioned_symbols.cpp_kw.md_docs.md`](./versioned_symbols.cpp_kw.md_docs.md)
- [`parser.cpp_kw.md_docs.md`](./parser.cpp_kw.md_docs.md)
- [`lexer.h_kw.md_docs.md`](./lexer.h_kw.md_docs.md)
- [`parser.cpp_docs.md_docs.md`](./parser.cpp_docs.md_docs.md)
- [`convert_to_ssa.h_docs.md_docs.md`](./convert_to_ssa.h_docs.md_docs.md)
- [`error_report.cpp_kw.md_docs.md`](./error_report.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `error_report.cpp_docs.md_docs.md`
- **Keyword Index**: `error_report.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
