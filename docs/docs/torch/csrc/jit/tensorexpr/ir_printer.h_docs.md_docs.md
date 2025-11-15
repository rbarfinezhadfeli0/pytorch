# Documentation: `docs/torch/csrc/jit/tensorexpr/ir_printer.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/ir_printer.h_docs.md`
- **Size**: 6,931 bytes (6.77 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/ir_printer.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/ir_printer.h`
- **Size**: 4,261 bytes (4.16 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ostream>

#include <torch/csrc/jit/tensorexpr/fwd_decls.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

namespace torch::jit::tensorexpr {

class Tensor;

class TORCH_API IRPrinter : public IRVisitor {
 public:
  explicit IRPrinter(std::ostream& os) : printer_os_(this, os) {}

  void print(ExprHandle /*expr*/);
  void print(Expr& /*expr*/);
  void print(Stmt& /*stmt*/);
  void visit(const AddPtr& v) override;
  void visit(const SubPtr& v) override;
  void visit(const MulPtr& v) override;
  void visit(const DivPtr& v) override;
  void visit(const ModPtr& v) override;
  void visit(const MaxPtr& v) override;
  void visit(const MinPtr& v) override;
  void visit(const AndPtr& v) override;
  void visit(const OrPtr& v) override;
  void visit(const XorPtr& v) override;
  void visit(const LshiftPtr& v) override;
  void visit(const RshiftPtr& v) override;
  void visit(const CompareSelectPtr& v) override;
#define IMM_PRINT_VISIT(Type, Name) void visit(const Name##ImmPtr& v) override;
  AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_PRINT_VISIT)
#undef IMM_PRINT_VISIT
  void visit(const CastPtr& v) override;
  void visit(const BitCastPtr& v) override;
  void visit(const VarPtr& v) override;
  void visit(const BufPtr& v) override;
  void visit(const RampPtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const BroadcastPtr& v) override;
  void visit(const IfThenElsePtr& v) override;
  void visit(const IntrinsicsPtr& v) override;
  void visit(const TermPtr& v) override;
  void visit(const PolynomialPtr& v) override;
  void visit(const RoundOffPtr& v) override;
  void visit(const MaxTermPtr& v) override;
  void visit(const MinTermPtr& v) override;
  void visit(const ReduceOpPtr& v) override;

  void visit(const AtomicAddPtr& v) override;
  void visit(const SyncThreadsPtr& v) override;
  void visit(const ExternalCallPtr& v) override;
  void visit(const ExternalCallWithAllocPtr& v) override;
  void visit(const StorePtr& v) override;
  void visit(const ForPtr& v) override;
  void visit(const CondPtr& v) override;
  void visit(const BlockPtr& v) override;
  void visit(const AllocatePtr& v) override;
  void visit(const FreePtr& v) override;
  void visit(const FreeExtPtr& v) override;
  void visit(const PlacementAllocatePtr& v) override;
  void visit(const LetPtr& v) override;

  // A child class may have a difference rule for generating dtype
  // string, e.g. CUDA needs int64_t to be generated as long long.
  virtual std::string dtypeToCppString(const Dtype& dtype);

  std::ostream& os() {
    return printer_os_;
  }

  class PrinterStream : public std::ostream {
   public:
    PrinterStream(IRPrinter* printer, std::ostream& os)
        : std::ostream(os.rdbuf()), printer_(printer) {
      initialize_imbue();
    }

    void initialize_imbue();

    IRPrinter* printer() {
      return printer_;
    }

   private:
    IRPrinter* printer_ = nullptr;
  };

 protected:
  std::string to_string(CompareSelectOperation op);

  UniqueNameManager* name_manager() {
    return &name_manager_;
  }
  void emitIndent();

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  int indent_ = 0;

 private:
  PrinterStream printer_os_;
  UniqueNameManager name_manager_;
};

TORCH_API std::ostream& operator<<(std::ostream& stream, const Expr& /*expr*/);
TORCH_API std::ostream& operator<<(
    std::ostream& stream,
    const ExprHandle& /*expr*/);
TORCH_API std::ostream& operator<<(std::ostream& stream, const Stmt& /*stmt*/);
TORCH_API std::ostream& operator<<(std::ostream& stream, const Tensor& /*t*/);

TORCH_API void print(const ExprPtr& expr);
TORCH_API void print(const StmtPtr& stmt);
TORCH_API void print(const Tensor& t);

} // namespace torch::jit::tensorexpr

namespace std {

using torch::jit::tensorexpr::Expr;
using torch::jit::tensorexpr::ExprPtr;
using torch::jit::tensorexpr::Stmt;
using torch::jit::tensorexpr::StmtPtr;
using torch::jit::tensorexpr::Tensor;

TORCH_API std::string to_string(const ExprPtr& expr);
TORCH_API std::string to_string(const StmtPtr& stmt);
TORCH_API std::string to_string(const Tensor& t);
} // namespace std

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 57 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `std`

**Classes/Structs**: `Tensor`, `TORCH_API`, `may`, `PrinterStream`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ostream`
- `torch/csrc/jit/tensorexpr/fwd_decls.h`
- `torch/csrc/jit/tensorexpr/ir.h`
- `torch/csrc/jit/tensorexpr/ir_visitor.h`
- `torch/csrc/jit/tensorexpr/unique_name_manager.h`


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

- **File Documentation**: `ir_printer.h_docs.md`
- **Keyword Index**: `ir_printer.h_kw.md`
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

- **File Documentation**: `ir_printer.h_docs.md_docs.md`
- **Keyword Index**: `ir_printer.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
