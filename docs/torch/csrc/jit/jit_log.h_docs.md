# Documentation: `torch/csrc/jit/jit_log.h`

## File Metadata

- **Path**: `torch/csrc/jit/jit_log.h`
- **Size**: 4,806 bytes (4.69 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <c10/util/StringUtil.h>
#include <torch/csrc/Export.h>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

// `TorchScript` offers a simple logging facility that can enabled by setting an
// environment variable `PYTORCH_JIT_LOG_LEVEL`.

// Logging is enabled on a per file basis. To enable logging in
// `dead_code_elimination.cpp`, `PYTORCH_JIT_LOG_LEVEL` should be
// set to `dead_code_elimination.cpp` or, simply, to `dead_code_elimination`
// (i.e. `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination`).

// Multiple files can be logged by separating each file name with a colon `:` as
// in the following example,
// `PYTORCH_JIT_LOG_LEVEL=dead_code_elimination:guard_elimination`

// There are 3 logging levels available for your use ordered by the detail level
// from lowest to highest.

// * `GRAPH_DUMP` should be used for printing entire graphs after optimization
// passes
// * `GRAPH_UPDATE` should be used for reporting graph transformations (i.e.
// node deletion, constant folding, etc)
// * `GRAPH_DEBUG` should be used for providing information useful for debugging
//   the internals of a particular optimization pass or analysis

// The default logging level is `GRAPH_DUMP` meaning that only `GRAPH_DUMP`
// statements will be enabled when one specifies a file(s) in
// `PYTORCH_JIT_LOG_LEVEL`.

// `GRAPH_UPDATE` can be enabled by prefixing a file name with an `>` as in
// `>alias_analysis`.
// `GRAPH_DEBUG` can be enabled by prefixing a file name with an `>>` as in
// `>>alias_analysis`.
// `>>>` is also valid and **currently** is equivalent to `GRAPH_DEBUG` as there
// is no logging level that is higher than `GRAPH_DEBUG`.

namespace torch::jit {

struct Node;
struct Graph;

enum class JitLoggingLevels {
  GRAPH_DUMP = 0,
  GRAPH_UPDATE,
  GRAPH_DEBUG,
};

TORCH_API std::string get_jit_logging_levels();

TORCH_API void set_jit_logging_levels(std::string level);

TORCH_API void set_jit_logging_output_stream(std::ostream& out_stream);

TORCH_API std::ostream& get_jit_logging_output_stream();

TORCH_API std::string getHeader(const Node* node);

TORCH_API std::string log_function(const std::shared_ptr<Graph>& graph);

TORCH_API ::torch::jit::JitLoggingLevels jit_log_level();

// Prefix every line in a multiline string \p IN_STR with \p PREFIX.
TORCH_API std::string jit_log_prefix(
    const std::string& prefix,
    const std::string& in_str);

TORCH_API std::string jit_log_prefix(
    ::torch::jit::JitLoggingLevels level,
    const char* fn,
    int l,
    const std::string& in_str);

TORCH_API bool is_enabled(
    const char* cfname,
    ::torch::jit::JitLoggingLevels level);

TORCH_API std::ostream& operator<<(
    std::ostream& out,
    ::torch::jit::JitLoggingLevels level);

#define JIT_LOG(level, ...)                                         \
  if (is_enabled(__FILE__, level)) {                                \
    ::torch::jit::get_jit_logging_output_stream()                   \
        << ::torch::jit::jit_log_prefix(                            \
               level, __FILE__, __LINE__, ::c10::str(__VA_ARGS__)); \
  }

// tries to reconstruct original python source
#define SOURCE_DUMP(MSG, G)                       \
  JIT_LOG(                                        \
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP, \
      MSG,                                        \
      "\n",                                       \
      ::torch::jit::log_function(G));
// use GRAPH_DUMP for dumping graphs after optimization passes
#define GRAPH_DUMP(MSG, G) \
  JIT_LOG(                 \
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP, MSG, "\n", (G)->toString());
// use GRAPH_UPDATE for reporting graph transformations (i.e. node deletion,
// constant folding, CSE)
#define GRAPH_UPDATE(...) \
  JIT_LOG(::torch::jit::JitLoggingLevels::GRAPH_UPDATE, __VA_ARGS__);
// use GRAPH_DEBUG to provide information useful for debugging a particular opt
// pass
#define GRAPH_DEBUG(...) \
  JIT_LOG(::torch::jit::JitLoggingLevels::GRAPH_DEBUG, __VA_ARGS__);
// use GRAPH_EXPORT to export a graph so that the IR can be loaded by a script
#define GRAPH_EXPORT(MSG, G)                       \
  JIT_LOG(                                         \
      ::torch::jit::JitLoggingLevels::GRAPH_DEBUG, \
      MSG,                                         \
      "\n<GRAPH_EXPORT>\n",                        \
      (G)->toString(),                             \
      "</GRAPH_EXPORT>");

#define GRAPH_DUMP_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_DUMP))
#define GRAPH_UPDATE_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_UPDATE))
#define GRAPH_DEBUG_ENABLED \
  (is_enabled(__FILE__, ::torch::jit::JitLoggingLevels::GRAPH_DEBUG))
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Node`, `Graph`, `JitLoggingLevels`, `original`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/StringUtil.h`
- `torch/csrc/Export.h`
- `memory`
- `ostream`
- `string`
- `unordered_map`


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

Files in the same folder (`torch/csrc/jit`):

- [`jit_opt_limit.cpp_docs.md`](./jit_opt_limit.cpp_docs.md)
- [`jit_log.cpp_docs.md`](./jit_log.cpp_docs.md)
- [`resource_guard.h_docs.md`](./resource_guard.h_docs.md)
- [`JIT-AUTOCAST.md_docs.md`](./JIT-AUTOCAST.md_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`OVERVIEW.md_docs.md`](./OVERVIEW.md_docs.md)
- [`jit_opt_limit.h_docs.md`](./jit_opt_limit.h_docs.md)


## Cross-References

- **File Documentation**: `jit_log.h_docs.md`
- **Keyword Index**: `jit_log.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
