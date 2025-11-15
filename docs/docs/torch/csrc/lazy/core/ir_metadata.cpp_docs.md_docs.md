# Documentation: `docs/torch/csrc/lazy/core/ir_metadata.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/core/ir_metadata.cpp_docs.md`
- **Size**: 4,849 bytes (4.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/core/ir_metadata.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/core/ir_metadata.cpp`
- **Size**: 2,416 bytes (2.36 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <functional>

namespace torch::lazy {

void EmitShortFrameInfo(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames) {
  if (!frames.empty()) {
    const SourceLocation& frame = frames.front();
    std::string::size_type pos = frame.file.find_last_of('/');
    if (pos == std::string::npos) {
      pos = 0;
    } else {
      ++pos;
    }
    stream << ", location=" << frame.function << "@" << frame.file.substr(pos)
           << ":" << frame.line;
  }
}

std::ostream& operator<<(
    std::ostream& stream,
    const std::vector<SourceLocation>& frames) {
  stream << "Frames:\n";
  for (auto& location : frames) {
    stream << "  " << location.function << " (" << location.file << ":"
           << location.line << ")\n";
  }
  return stream;
}

namespace {

struct ScopeEntry {
  std::string name;
  size_t saved_next_id = 1;
};

struct ScopeContext {
  std::vector<ScopeEntry> scopes;
  size_t next_id = 1;
};

thread_local ScopeContext g_scope_context;

std::string GetCurrentScope() {
  std::string scope;
  for (auto& scope_entry : g_scope_context.scopes) {
    if (scope.empty()) {
      scope = scope_entry.name;
    } else {
      scope += "/" + scope_entry.name;
    }
  }
  return scope;
}

void PushScope(const std::string& name) {
  size_t id = g_scope_context.next_id;
  g_scope_context.scopes.push_back(
      {c10::str(name, ".", id), g_scope_context.next_id + 1});
  g_scope_context.next_id = 1;
}

void PopScope() {
  TORCH_CHECK(!g_scope_context.scopes.empty());
  g_scope_context.next_id = g_scope_context.scopes.back().saved_next_id;
  g_scope_context.scopes.pop_back();
}

void ResetScopeContext() {
  if (!g_scope_context.scopes.empty()) {
    TORCH_CHECK(
        false, "Expecting scope to be empty but it is " + GetCurrentScope());
  }
  g_scope_context.next_id = 1;
}
} // namespace

ScopePusher::ScopePusher(const std::string& name) {
  PushScope(name);
}

ScopePusher::~ScopePusher() {
  PopScope();
}

void ScopePusher::ResetScopes() {
  ResetScopeContext();
}

MetaData GetMetaDataIfDebugging() {
  if (!FLAGS_torch_lazy_ir_debug) {
    return MetaData();
  }
  MetaData meta;
  meta.scope = GetCurrentScope();
  meta.frame_info = torch::lazy::GetPythonFramesFunction()();
  return meta;
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ScopePusher`, `torch`

**Classes/Structs**: `ScopeEntry`, `ScopeContext`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/core/config.h`
- `torch/csrc/lazy/core/debug_util.h`
- `torch/csrc/lazy/core/ir_metadata.h`
- `functional`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/csrc/lazy/core`):

- [`hash.cpp_docs.md`](./hash.cpp_docs.md)
- [`shape_inference.cpp_docs.md`](./shape_inference.cpp_docs.md)
- [`tensor_impl.h_docs.md`](./tensor_impl.h_docs.md)
- [`helpers.h_docs.md`](./helpers.h_docs.md)
- [`tensor_impl.cpp_docs.md`](./tensor_impl.cpp_docs.md)
- [`ir_metadata.h_docs.md`](./ir_metadata.h_docs.md)
- [`trie.cpp_docs.md`](./trie.cpp_docs.md)
- [`cache.h_docs.md`](./cache.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)


## Cross-References

- **File Documentation**: `ir_metadata.cpp_docs.md`
- **Keyword Index**: `ir_metadata.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/csrc/lazy/core`):

- [`helpers.cpp_docs.md_docs.md`](./helpers.cpp_docs.md_docs.md)
- [`tensor_util.h_kw.md_docs.md`](./tensor_util.h_kw.md_docs.md)
- [`permutation_util.h_kw.md_docs.md`](./permutation_util.h_kw.md_docs.md)
- [`ir_util.cpp_kw.md_docs.md`](./ir_util.cpp_kw.md_docs.md)
- [`shape_inference.h_kw.md_docs.md`](./shape_inference.h_kw.md_docs.md)
- [`ir_builder.h_docs.md_docs.md`](./ir_builder.h_docs.md_docs.md)
- [`shape_inference.cpp_kw.md_docs.md`](./shape_inference.cpp_kw.md_docs.md)
- [`hash.h_kw.md_docs.md`](./hash.h_kw.md_docs.md)
- [`multi_wait.cpp_kw.md_docs.md`](./multi_wait.cpp_kw.md_docs.md)
- [`lazy_graph_executor.cpp_docs.md_docs.md`](./lazy_graph_executor.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ir_metadata.cpp_docs.md_docs.md`
- **Keyword Index**: `ir_metadata.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
