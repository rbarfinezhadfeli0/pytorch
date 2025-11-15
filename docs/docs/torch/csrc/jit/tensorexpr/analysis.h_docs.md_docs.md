# Documentation: `docs/torch/csrc/jit/tensorexpr/analysis.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/analysis.h_docs.md`
- **Size**: 11,677 bytes (11.40 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/analysis.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/analysis.h`
- **Size**: 8,982 bytes (8.77 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/stmt.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <utility>

namespace torch::jit::tensorexpr {
class HasRand : public IRVisitor {
 public:
  HasRand(StmtPtr stmt) : stmt_(std::move(stmt)) {
    stmt_->accept(this);
  }

  bool has_rand() const {
    return has_rand_;
  }

 private:
  void visit(const IntrinsicsPtr& v) override {
    if (v->op_type() == IntrinsicsOp::kRand) {
      has_rand_ = true;
    } else {
      IRVisitor::visit(v);
    }
  }
  StmtPtr stmt_;
  bool has_rand_ = false;
};

template <typename Op>
class NodeFinder : public IRVisitor {
 public:
  void visit(const NodePtr<Op>& v) override {
    nodes.push_back((NodePtr<Op>)v);
    IRVisitor::visit(v);
  }

  static std::vector<NodePtr<Op>> find(const StmtPtr& s) {
    NodeFinder<Op> nf;
    s->accept(&nf);
    return nf.nodes;
  }

  static std::vector<NodePtr<Op>> find(const ExprPtr& e) {
    NodeFinder<Op> nf;
    e->accept(&nf);
    return nf.nodes;
  }

  std::vector<NodePtr<Op>> nodes;
};

class VarFinder : public IRVisitor {
 public:
  void visit(const VarPtr& v) override {
    vars_.insert(v);
    IRVisitor::visit(v);
  }

  static std::unordered_set<VarPtr> find(const StmtPtr& s) {
    VarFinder nf;
    s->accept(&nf);
    return nf.vars();
  }

  static std::unordered_set<VarPtr> find(const ExprPtr& e) {
    VarFinder nf;
    e->accept(&nf);
    return nf.vars();
  }

  const std::unordered_set<VarPtr>& vars() {
    return vars_;
  }

 private:
  std::unordered_set<VarPtr> vars_;
};

class BufFinder : public IRVisitor {
 public:
  void visit(const BufPtr& v) override {
    bufs_.insert(v);
    IRVisitor::visit(v);
  }

  static std::unordered_set<BufPtr> find(const StmtPtr& s) {
    BufFinder nf;
    s->accept(&nf);
    return nf.bufs();
  }

  static std::unordered_set<BufPtr> find(const ExprPtr& e) {
    BufFinder nf;
    e->accept(&nf);
    return nf.bufs();
  }

  const std::unordered_set<BufPtr>& bufs() {
    return bufs_;
  }

 private:
  std::unordered_set<BufPtr> bufs_;
};

// Finds all kinds of write operations to the provided Buf.
class WritesToBuf : public IRVisitor {
 public:
  WritesToBuf(BufPtr target) : target_(std::move(target)) {}

  std::vector<StmtPtr> writes() {
    return writes_;
  }

  static std::vector<StmtPtr> find(const StmtPtr& s, BufPtr b) {
    WritesToBuf finder(std::move(b));
    s->accept(&finder);
    return finder.writes();
  }

 private:
  void visit(const StorePtr& v) override {
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  void visit(const AtomicAddPtr& v) override {
    if (v->buf() == target_) {
      writes_.push_back(v);
    }
  }

  BufPtr target_;
  std::vector<StmtPtr> writes_;
};

class StmtsReadingBuf : public IRVisitor {
 public:
  StmtsReadingBuf(BufPtr target) : target_(std::move(target)) {}

  std::vector<StmtPtr> reads() {
    return reads_;
  }

  static std::vector<StmtPtr> find(const StmtPtr& s, BufPtr b) {
    StmtsReadingBuf finder(std::move(b));
    s->accept(&finder);
    return finder.reads();
  }

 private:
  bool readsBuffer(const StmtPtr& s) {
    auto loads = NodeFinder<Load>::find(s);
    for (const auto& l : loads) {
      if (l->buf() == target_) {
        return true;
      }
    }
    return false;
  }

  void visit(const StorePtr& v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(const LetPtr& v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(const CondPtr& v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  void visit(const AtomicAddPtr& v) override {
    if (readsBuffer(v)) {
      reads_.push_back(v);
    }
  }

  BufPtr target_;
  std::vector<StmtPtr> reads_;
};

class ExternalAllocBufFinder : public IRVisitor {
 public:
  void visit(const ExternalCallWithAllocPtr& v) override {
    const auto& bufs_out = v->buf_out_args();
    bufs_.insert(bufs_out.begin(), bufs_out.end());
    IRVisitor::visit(v);
  }

  static std::unordered_set<BufPtr> find(const StmtPtr& s) {
    ExternalAllocBufFinder f;
    s->accept(&f);
    return f.bufs();
  }

  static std::unordered_set<BufPtr> find(const ExprPtr& e) {
    ExternalAllocBufFinder f;
    e->accept(&f);
    return f.bufs();
  }

  const std::unordered_set<BufPtr>& bufs() {
    return bufs_;
  }

 private:
  std::unordered_set<BufPtr> bufs_;
};

// Traverses the IR to determine if a particular Var is modified within it.
class ModifiesVarChecker : public IRVisitor {
 public:
  ModifiesVarChecker(VarPtr v) : var_(std::move(v)) {}

  static bool check(const StmtPtr& s, VarPtr v) {
    ModifiesVarChecker checker(std::move(v));
    s->accept(&checker);
    return checker.found();
  }

  bool found() {
    return found_;
  }

 private:
  void visit(const StorePtr& v) override {
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(const AtomicAddPtr& v) override {
    if (v->buf()->base_handle() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(const LetPtr& v) override {
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  void visit(const ForPtr& v) override {
    if (v->var() == var_) {
      found_ = true;
      return;
    }
    IRVisitor::visit(v);
  }

  VarPtr var_;
  bool found_{false};
};

// Traverse the Block stmt to identify the live range of the specified buf. The
// live range, indicated by a pair of integers, specifies the first and last
// stmt in block stmts that access to the buf.
class BufLiveRange : public IRVisitor {
 public:
  BufLiveRange(BufPtr b) : buf_(std::move(b)) {}

  static std::tuple<int32_t, int32_t> liveRange(const StmtPtr& s, BufPtr b) {
    BlockPtr block = to<Block>(s);
    // We Only analyze buffer live ranges for block stmts.
    if (!block) {
      return std::make_tuple(0, 0);
    }

    BufLiveRange analyzer(std::move(b));
    block->accept(&analyzer);
    return analyzer.getLiveRange();
  }

 private:
  std::tuple<int32_t, int32_t> getLiveRange() {
    return std::make_tuple(begin_, end_);
  }

  bool hasBufReads(const StmtPtr& s) {
    auto loads1 = NodeFinder<Load>::find(s);
    for (const auto& l : loads1) {
      if (l->buf() == buf_) {
        return true;
      }
    }
    auto loads2 = NodeFinder<ExternalCall>::find(s);
    for (const auto& l : loads2) {
      for (const auto& lb : l->buf_args()) {
        if (lb == buf_) {
          return true;
        }
      }
    }
    auto loads3 = NodeFinder<ExternalCallWithAlloc>::find(s);
    for (const auto& l : loads3) {
      for (const auto& lb : l->buf_args()) {
        if (lb == buf_) {
          return true;
        }
      }
    }
    return false;
  }

  bool hasBufWrites(const StmtPtr& s) {
    auto writes1 = NodeFinder<Store>::find(s);
    for (const auto& w : writes1) {
      if (w->buf() == buf_) {
        return true;
      }
    }
    auto writes2 = NodeFinder<ExternalCall>::find(s);
    for (const auto& w : writes2) {
      if (w->buf() == buf_) {
        return true;
      }
    }
    auto writes3 = NodeFinder<ExternalCallWithAlloc>::find(s);
    for (const auto& w : writes3) {
      for (const auto& wb : w->buf_out_args()) {
        if (wb == buf_) {
          return true;
        }
      }
    }
    return false;
  }

  void findAccAndUpdateLiveRange(const StmtPtr& s) {
    bool has_reads = hasBufReads(s), has_writes = hasBufWrites(s);
    if (has_reads || has_writes) {
      if (begin_ == -1) {
        begin_ = curr_index_;
      };
      end_ = curr_index_;
    }
  }

  void visit(const BlockPtr& v) override {
    for (const StmtPtr& s : *v) {
      curr_index_ += 1;
      findAccAndUpdateLiveRange(s);
    }
  }

  BufPtr buf_;
  int32_t begin_ = -1;
  int32_t end_ = -1;
  int32_t curr_index_ = -1;
};

// A class that analyzes the given program relevant for Block backend
// It creates a map of multi dim buffers and their flat versions
class CreateBufferMap : public IRVisitor {
 public:
  const std::unordered_map<std::string, BufPtr>& getBufferMap() const {
    return map_input_to_tensor_bufs_;
  }

 private:
  void visit(const StorePtr& v) override {
    auto load_node = to<Load>(v->value());
    if (load_node) {
      auto t_buf = load_node->buf();
      map_input_to_tensor_bufs_.emplace(t_buf->name_hint(), v->buf());
    } else {
      auto add_node = to<Add>(v->value());
      auto mul_node = to<Mul>(v->value());
      // This means for now, v->value() can be Add or Mul
      TORCH_INTERNAL_ASSERT(add_node || mul_node, buildErrorMessage());
      map_input_to_tensor_bufs_.emplace(v->buf()->name_hint(), v->buf());
    }
    v->value()->accept(this);
  }
  std::unordered_map<std::string, BufPtr> map_input_to_tensor_bufs_;
};

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 11 class(es)/struct(s) and 29 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `HasRand`, `NodeFinder`, `VarFinder`, `BufFinder`, `WritesToBuf`, `StmtsReadingBuf`, `ExternalAllocBufFinder`, `ModifiesVarChecker`, `BufLiveRange`, `that`, `CreateBufferMap`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/tensorexpr/ir.h`
- `torch/csrc/jit/tensorexpr/ir_visitor.h`
- `torch/csrc/jit/tensorexpr/stmt.h`
- `torch/csrc/jit/tensorexpr/tensor.h`
- `utility`


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

- **File Documentation**: `analysis.h_docs.md`
- **Keyword Index**: `analysis.h_kw.md`
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

- **File Documentation**: `analysis.h_docs.md_docs.md`
- **Keyword Index**: `analysis.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
