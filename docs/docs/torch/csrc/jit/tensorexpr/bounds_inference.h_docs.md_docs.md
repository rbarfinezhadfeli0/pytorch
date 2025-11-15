# Documentation: `docs/torch/csrc/jit/tensorexpr/bounds_inference.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/bounds_inference.h_docs.md`
- **Size**: 4,711 bytes (4.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/bounds_inference.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/bounds_inference.h`
- **Size**: 2,170 bytes (2.12 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <unordered_map>
#include <vector>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/tensorexpr/mem_dependency_checker.h>

namespace torch::jit::tensorexpr {

class Expr;
class Buf;
class Stmt;

enum C10_API_ENUM TensorAccessKind { kLoad, kStore, kMutate };

struct TORCH_API TensorAccessBoundsInfo {
  TensorAccessKind kind;
  std::vector<ExprPtr> start;
  std::vector<ExprPtr> stop;
};

using BoundsInfo =
    std::unordered_map<BufPtr, std::vector<TensorAccessBoundsInfo>>;

TORCH_API BoundsInfo
inferBounds(const StmtPtr& s, bool distinctAccessKinds = true);

// Bounds inference caching the analysis. The MemDependencyChecker must already
// have been run.
TORCH_API BoundsInfo getInferredBounds(
    analysis::MemDependencyChecker& analyzer,
    const StmtPtr& s,
    bool distinctAccessKinds = true);
TORCH_API BoundsInfo getInferredBounds(
    analysis::MemDependencyChecker& analyzer,
    const ExprPtr& e,
    bool distinctAccessKinds = true);

TORCH_API void printBoundsInfo(const BoundsInfo& v);

TORCH_API std::vector<ExprPtr> getBoundExtents(
    const std::vector<TensorAccessBoundsInfo>& infos);

// The kind of dependency found, in increasing order of exclusivity.
enum class HazardKind {
  ReadAfterWrite,
  WriteAfterRead,
  WriteAfterWrite,
  NoDependency,
};
TORCH_API HazardKind getPotentialHazards(
    analysis::MemDependencyChecker& analyzer,
    const StmtPtr& A,
    const StmtPtr& B);

// Returns true if there is a conflicting overlap between accesses in
// statements A and B. A conflicting overlap is an overlap in buffer accesses
// where at least one of the accesses is a Store.
TORCH_API bool hasConflictingOverlap(
    analysis::MemDependencyChecker& analyzer,
    const StmtPtr& A,
    const StmtPtr& B);
// Same as above, between accesses in stores S1 and S2.
TORCH_API bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    const StorePtr& S1,
    const StorePtr& S2);
// Same as above, between accesses in store S and load L.
TORCH_API bool isOverlapping(
    analysis::MemDependencyChecker& analyzer,
    const StorePtr& S,
    const LoadPtr& L);

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Expr`, `Buf`, `Stmt`, `TORCH_API`, `HazardKind`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `unordered_map`
- `vector`
- `torch/csrc/Export.h`
- `torch/csrc/jit/tensorexpr/mem_dependency_checker.h`


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

- **File Documentation**: `bounds_inference.h_docs.md`
- **Keyword Index**: `bounds_inference.h_kw.md`
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

- **File Documentation**: `bounds_inference.h_docs.md_docs.md`
- **Keyword Index**: `bounds_inference.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
