# Documentation: `docs/torch/csrc/jit/tensorexpr/ir_cloner.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/ir_cloner.h_docs.md`
- **Size**: 4,888 bytes (4.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/ir_cloner.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/ir_cloner.h`
- **Size**: 2,343 bytes (2.29 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <vector>

#include <torch/csrc/jit/tensorexpr/ir_mutator.h>

namespace torch::jit::tensorexpr {

class TORCH_API IRCloner : public IRMutator {
 public:
  ~IRCloner() override = default;
  ExprPtr mutate(const AddPtr& v) override;
  ExprPtr mutate(const SubPtr& v) override;
  ExprPtr mutate(const MulPtr& v) override;
  ExprPtr mutate(const DivPtr& v) override;
  ExprPtr mutate(const ModPtr& v) override;
  ExprPtr mutate(const MaxPtr& v) override;
  ExprPtr mutate(const MinPtr& v) override;
  ExprPtr mutate(const AndPtr& v) override;
  ExprPtr mutate(const OrPtr& v) override;
  ExprPtr mutate(const XorPtr& v) override;
  ExprPtr mutate(const LshiftPtr& v) override;
  ExprPtr mutate(const RshiftPtr& v) override;
  ExprPtr mutate(const CompareSelectPtr& v) override;
#define IMM_MUTATE_DECLARE(Type, Name) \
  ExprPtr mutate(const Name##ImmPtr& v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_MUTATE_DECLARE)
#undef IMM_MUTATE_DECLARE
  ExprPtr mutate(const CastPtr& v) override;
  ExprPtr mutate(const BitCastPtr& v) override;
  ExprPtr mutate(const VarPtr& v) override;
  ExprPtr mutate(const BufPtr& v) override;
  ExprPtr mutate(const RampPtr& v) override;
  ExprPtr mutate(const LoadPtr& v) override;
  ExprPtr mutate(const BroadcastPtr& v) override;
  ExprPtr mutate(const IfThenElsePtr& v) override;
  ExprPtr mutate(const IntrinsicsPtr& v) override;

  ExprPtr mutate(const TermPtr& v) override;
  ExprPtr mutate(const PolynomialPtr& v) override;
  ExprPtr mutate(const RoundOffPtr& v) override;
  ExprPtr mutate(const MaxTermPtr& v) override;
  ExprPtr mutate(const MinTermPtr& v) override;

  ExprPtr mutate(const ReduceOpPtr& v) override;

  StmtPtr mutate(const ForPtr& v) override;
  StmtPtr mutate(const BlockPtr& v) override;
  StmtPtr mutate(const StorePtr& v) override;
  StmtPtr mutate(const AtomicAddPtr& v) override;
  StmtPtr mutate(const SyncThreadsPtr& v) override;
  StmtPtr mutate(const ExternalCallPtr& v) override;
  StmtPtr mutate(const ExternalCallWithAllocPtr& v) override;

  StmtPtr mutate(const AllocatePtr& v) override;
  StmtPtr mutate(const FreePtr& v) override;
  StmtPtr mutate(const LetPtr& v) override;
  StmtPtr mutate(const CondPtr& v) override;
};

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 41 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/ScalarType.h`
- `torch/csrc/Export.h`
- `vector`
- `torch/csrc/jit/tensorexpr/ir_mutator.h`


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

- **File Documentation**: `ir_cloner.h_docs.md`
- **Keyword Index**: `ir_cloner.h_kw.md`
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

- **File Documentation**: `ir_cloner.h_docs.md_docs.md`
- **Keyword Index**: `ir_cloner.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
