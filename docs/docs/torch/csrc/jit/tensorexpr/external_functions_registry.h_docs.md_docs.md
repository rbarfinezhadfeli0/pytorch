# Documentation: `docs/torch/csrc/jit/tensorexpr/external_functions_registry.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/external_functions_registry.h_docs.md`
- **Size**: 4,808 bytes (4.70 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/external_functions_registry.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/external_functions_registry.h`
- **Size**: 2,306 bytes (2.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/Export.h>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace torch::jit::tensorexpr {

// The external functions that could be called from NNC must have the same
// signature defined by `NNCExternalFunction`.
//
// Why this signature?
// It was picked for two reasons: 1) it should be generic enough to represent
// most of the ops we might want to call, 2) it should be possible to generate a
// code for this call in LLVM codegen.
// The first 5 parameters allow to pass any number of contiguous CPU tensors in
// case we need to run aten ops (TODO: support different devices). The first
// buffer in the array is assumed to be the output buffer. We couldn't use
// `at::Tensor` (or `c10::IValue`) type there directly as it would mean that
// we'd need to declare it in LLVM codegen in LLVM IR form, which would be very
// cumbersome and hard to maintain. Note that the dimensions of all tensors are
// concatenated into a single array buf_dims. We do not need to pass its length,
// since it can be deduced from total number of buffers and their ranks.
//
// The last 2 arguments allow to pass any non-tensor arguments encoded as an
// array of int64_t values. The way they are encoded is not specified and could
// be arbitrary - whatever the most convenient for the specific bridge function
// is.
//
// The bridge functions must not throw exceptions - properly propagating them
// from the generated code is too cumbersome, and thus all calls to functions
// that could throw must be wrapped with try-catch blocks.
using NNCExternalFunction = void (*)(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args);

// Return a global map "function-name" -> "function-pointer" for all registered
// in NNC external functions
TORCH_API std::unordered_map<std::string, NNCExternalFunction>&
getNNCFunctionRegistry();

// To register a new external function in NNC one needs to create an instance of
// this struct
struct RegisterNNCExternalFunction {
  RegisterNNCExternalFunction(const std::string& name, NNCExternalFunction fn) {
    getNNCFunctionRegistry()[name] = fn;
  }
};

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `struct`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `cstdint`
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

- **File Documentation**: `external_functions_registry.h_docs.md`
- **Keyword Index**: `external_functions_registry.h_kw.md`
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

- **File Documentation**: `external_functions_registry.h_docs.md_docs.md`
- **Keyword Index**: `external_functions_registry.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
