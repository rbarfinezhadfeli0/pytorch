# Documentation: `docs/torch/csrc/jit/tensorexpr/block_codegen.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/block_codegen.h_docs.md`
- **Size**: 7,089 bytes (6.92 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/block_codegen.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/block_codegen.h`
- **Size**: 4,290 bytes (4.19 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <ATen/ATen.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_printer.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

namespace torch::jit::tensorexpr {

// A class that analyzes the given program relevant for Block backend.
class BlockAnalysis : public IRVisitor {
 public:
  bool is_buf_store_target(const BufPtr& buf) const {
    return store_targets_.count(buf) > 0;
  }

  const std::unordered_set<BufPtr>& loads() const {
    return loads_;
  }

  const std::unordered_set<BufPtr>& stores() const {
    return store_targets_;
  }

  int64_t block_size() const {
    return block_size_;
  }

  bool areBufsInMap(const std::unordered_set<BufPtr>& bufs) const;

  BufPtr getMultiDimBuf(const BufPtr& buf) const;

  std::string getInputName(const BufPtr& buf) const;

  std::string getFlatInputName(const BufPtr& buf) const {
    return getInputName(buf) + "_flat";
  }

  std::unordered_map<std::string, BufPtr> getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  void visit(const StorePtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const ForPtr& v) override;

  std::unordered_map<std::string, BufPtr> map_input_to_tensor_bufs_;
  std::unordered_set<BufPtr> store_targets_;
  std::unordered_set<BufPtr> loads_;
  int64_t block_size_ = 32;
};

// A class that overrides the underlying IRPrinter to produce Block.
class BlockPrinter : public IRPrinter {
 public:
  BlockPrinter(std::ostream* os, BlockAnalysis* block_analysis)
      : IRPrinter(*os), block_analysis_(block_analysis) {}

  using IRPrinter::name_manager;
  using IRPrinter::visit;

 private:
  BlockAnalysis* block_analysis_;
  std::unordered_map<std::string, int> dim_values_map;
  std::vector<std::string> dim_names = {"N", "H", "W", "C"};
  std::vector<std::string> flat_dim_names = {"N", "NH", "NHW", "NHWC"};
  void PrintTensorInfo(const std::unordered_set<BufPtr>& bufs);
  void PrintArguments(const std::unordered_set<BufPtr>& bufs);
  void PrintBufferInfo(const std::unordered_set<BufPtr>& bufs);
  void PrintDistribution(const std::unordered_set<BufPtr>& bufs);
  void PrintLoop(const std::unordered_set<BufPtr>& bufs, bool block_idx = true);
  void PrintReshapeInfo(
      const std::unordered_set<BufPtr>& bufs,
      bool reverse = false);
  void PrintDMAs(const std::unordered_set<BufPtr>& bufs);
  void PrintAdjustBuffers(const std::unordered_set<BufPtr>& bufs);

  void visit(const ForPtr& v) override;
  void visit(const LoadPtr& v) override;
  void visit(const StorePtr& v) override;
  void visit(const BlockPtr& v) override;
  void visit(const AddPtr& v) override;
  void visit(const MulPtr& v) override;
};

class TORCH_API BlockCodeGen : public CodeGen {
 public:
  template <typename... Ts>
  /* implicit */
  BlockCodeGen(StmtPtr stmt, Ts... ts)
      : CodeGen(
            stmt,
            std::vector<BufferArg>({BufferArg(ts)...}),
            at::Device(at::kCPU)) {
    Initialize();
  }

  BlockCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& buffer_args,
      at::Device device = at::Device(at::kCPU),
      const std::string& kernel_func_name = "func")
      : CodeGen(std::move(stmt), buffer_args, device, kernel_func_name) {
    Initialize();
  }

  ~BlockCodeGen() override;

  void call(const std::vector<CallArg>& args) override;
  void call_raw(const std::vector<void*>& args) override;

  void Initialize();

  std::string getCodeText(const std::string& attr = "") override {
    return oss_.str();
  }

 private:
  UniqueNameManager* name_manager() {
    if (!printer_) {
      throw std::runtime_error("Null IRPrinter is not expected");
    }
    return printer_->name_manager();
  }

  std::ostream& os() {
    return printer_->os();
  }

  std::ostringstream oss_;
  std::unique_ptr<BlockPrinter> printer_;
  std::unique_ptr<BlockAnalysis> block_analysis_;

  std::string GetUniqueFuncName(const std::string& func_prefix);
};
} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 29 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `that`, `BlockAnalysis`, `that`, `BlockPrinter`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `string`
- `unordered_map`
- `unordered_set`
- `utility`
- `ATen/ATen.h`
- `torch/csrc/jit/resource_guard.h`
- `torch/csrc/jit/tensorexpr/analysis.h`
- `torch/csrc/jit/tensorexpr/codegen.h`
- `torch/csrc/jit/tensorexpr/ir.h`
- `torch/csrc/jit/tensorexpr/ir_printer.h`
- `torch/csrc/jit/tensorexpr/ir_visitor.h`
- `torch/csrc/jit/tensorexpr/unique_name_manager.h`


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

- **File Documentation**: `block_codegen.h_docs.md`
- **Keyword Index**: `block_codegen.h_kw.md`
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

- **File Documentation**: `block_codegen.h_docs.md_docs.md`
- **Keyword Index**: `block_codegen.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
