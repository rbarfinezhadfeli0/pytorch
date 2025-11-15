# Documentation: `docs/torch/csrc/jit/tensorexpr/exceptions.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/exceptions.h_docs.md`
- **Size**: 5,788 bytes (5.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/exceptions.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/exceptions.h`
- **Size**: 3,221 bytes (3.15 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/fwd_decls.h>

#include <stdexcept>

// Forward declarations of types

namespace torch::jit::tensorexpr {
class Expr;
class Stmt;
} // namespace torch::jit::tensorexpr

// Forward declarations of functions
namespace std {
TORCH_API std::string to_string(
    const torch::jit::tensorexpr::ExprPtr& /*expr*/);
TORCH_API std::string to_string(
    const torch::jit::tensorexpr::StmtPtr& /*stmt*/);
} // namespace std

namespace torch::jit::tensorexpr {

class unsupported_dtype : public std::runtime_error {
 public:
  explicit unsupported_dtype() : std::runtime_error("UNSUPPORTED DTYPE") {}
  explicit unsupported_dtype(const std::string& err)
      : std::runtime_error("UNSUPPORTED DTYPE: " + err) {}
};

class out_of_range_index : public std::runtime_error {
 public:
  explicit out_of_range_index() : std::runtime_error("OUT OF RANGE INDEX") {}
  explicit out_of_range_index(const std::string& err)
      : std::runtime_error("OUT OF RANGE INDEX: " + err) {}
};

class unimplemented_lowering : public std::runtime_error {
 public:
  explicit unimplemented_lowering()
      : std::runtime_error("UNIMPLEMENTED LOWERING") {}
  explicit unimplemented_lowering(const ExprPtr& expr)
      : std::runtime_error("UNIMPLEMENTED LOWERING: " + std::to_string(expr)) {}
  explicit unimplemented_lowering(const StmtPtr& stmt)
      : std::runtime_error("UNIMPLEMENTED LOWERING: " + std::to_string(stmt)) {}
};

class malformed_input : public std::runtime_error {
 public:
  explicit malformed_input() : std::runtime_error("MALFORMED INPUT") {}
  explicit malformed_input(const std::string& err)
      : std::runtime_error("MALFORMED INPUT: " + err) {}
  explicit malformed_input(const ExprPtr& expr)
      : std::runtime_error("MALFORMED INPUT: " + std::to_string(expr)) {}
  explicit malformed_input(const std::string& err, const ExprPtr& expr)
      : std::runtime_error(
            "MALFORMED INPUT: " + err + " - " + std::to_string(expr)) {}
  explicit malformed_input(const StmtPtr& stmt)
      : std::runtime_error("MALFORMED INPUT: " + std::to_string(stmt)) {}
  explicit malformed_input(const std::string& err, const StmtPtr& stmt)
      : std::runtime_error(
            "MALFORMED INPUT: " + err + " - " + std::to_string(stmt)) {}
};

class malformed_ir : public std::runtime_error {
 public:
  explicit malformed_ir() : std::runtime_error("MALFORMED IR") {}
  explicit malformed_ir(const std::string& err)
      : std::runtime_error("MALFORMED IR: " + err) {}
  explicit malformed_ir(const ExprPtr& expr)
      : std::runtime_error("MALFORMED IR: " + std::to_string(expr)) {}
  explicit malformed_ir(const std::string& err, const ExprPtr& expr)
      : std::runtime_error(
            "MALFORMED IR: " + err + " - " + std::to_string(expr)) {}
  explicit malformed_ir(const StmtPtr& stmt)
      : std::runtime_error("MALFORMED IR: " + std::to_string(stmt)) {}
  explicit malformed_ir(const std::string& err, const StmtPtr& stmt)
      : std::runtime_error(
            "MALFORMED IR: " + err + " - " + std::to_string(stmt)) {}
};

TORCH_API std::string buildErrorMessage(const std::string& s = "");

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 7 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `std`

**Classes/Structs**: `Expr`, `Stmt`, `unsupported_dtype`, `out_of_range_index`, `unimplemented_lowering`, `malformed_input`, `malformed_ir`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/csrc/jit/tensorexpr/fwd_decls.h`
- `stdexcept`


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

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `exceptions.h_docs.md`
- **Keyword Index**: `exceptions.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/tensorexpr`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/tensorexpr`):

- [`loopnest.h_kw.md_docs.md`](./loopnest.h_kw.md_docs.md)
- [`expr.h_docs.md_docs.md`](./expr.h_docs.md_docs.md)
- [`block_codegen.h_kw.md_docs.md`](./block_codegen.h_kw.md_docs.md)
- [`ir_cloner.cpp_kw.md_docs.md`](./ir_cloner.cpp_kw.md_docs.md)
- [`types.cpp_docs.md_docs.md`](./types.cpp_docs.md_docs.md)
- [`tensorexpr_init.h_docs.md_docs.md`](./tensorexpr_init.h_docs.md_docs.md)
- [`lowerings.cpp_kw.md_docs.md`](./lowerings.cpp_kw.md_docs.md)
- [`graph_opt.h_kw.md_docs.md`](./graph_opt.h_kw.md_docs.md)
- [`eval.h_kw.md_docs.md`](./eval.h_kw.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `exceptions.h_docs.md_docs.md`
- **Keyword Index**: `exceptions.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
