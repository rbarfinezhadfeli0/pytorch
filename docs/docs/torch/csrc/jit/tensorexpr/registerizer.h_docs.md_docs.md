# Documentation: `docs/torch/csrc/jit/tensorexpr/registerizer.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/registerizer.h_docs.md`
- **Size**: 15,346 bytes (14.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/registerizer.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/registerizer.h`
- **Size**: 12,519 bytes (12.23 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/irange.h>
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/tensorexpr/hash_provider.h>
#include <torch/csrc/jit/tensorexpr/ir_mutator.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <utility>
#include <vector>

namespace torch::jit::tensorexpr {
namespace registerizer {

/* The Registerizer performs scalar replacement by looking for common Stores and
Loads to a single item in a buffer and replacing them with a local temporary
scalar which is cheaper to write.

For example it can replace:

{
  A[0] = 0;
  for(const auto x : c10::irange(10)) {
    A[0] = (A[0]) + x;
  }
}

with:

{
  int A_ = 0;
  for(const auto x : c10::irange(10)) {
    A_ = x + A_;
  }
  A[0] = A_;
}

This is particularly useful on GPUs when parallelizing, since after replacing
loops with metavars we have a lot of accesses like this. */

class Scope;

/*  Holds analysis information about accesses to a specific range of a
 buffer, including the number of loads and stores and the lowest common parent
 Block.
 */
class AccessInfo {
 public:
  AccessInfo() = default;
  AccessInfo(
      SimplifierHashType h,
      BufPtr b,
      std::vector<ExprPtr> i,
      size_t accessOrder)
      : hash_(h),
        buf_(std::move(b)),
        indices_(std::move(i)),
        store_cost_(alloc<IntImm>(0)),
        load_cost_(alloc<IntImm>(0)),
        accessOrder_(accessOrder) {}

  // Adds a Store to this access, which is in the provided scope.
  void addStore(const StorePtr& store, const std::shared_ptr<Scope>& scope);

  // Adds a Load to this access, which occurs in the usage Stmt in the provided
  // scope.
  void addLoad(
      const LoadPtr& load,
      const std::shared_ptr<Scope>& scope,
      const StmtPtr& usage);

  // Merge another AccessInfo into this one.
  void merge(const std::shared_ptr<AccessInfo>& other);

  // Returns true if the other AccessInfo's bounds may overlap this one.
  bool overlaps(const std::shared_ptr<AccessInfo>& other);

  // Returns true if the indices of this access depend on the provided Var.
  bool dependsOnVar(const VarPtr& v);

  // Clone this AccessInfo, and set this as the new accesses' hiddenAccess.
  static std::shared_ptr<AccessInfo> cloneWithHiddenInfo(
      const std::shared_ptr<AccessInfo>& orig);

  // print for debugging.
  void print() const;

  SimplifierHashType hash() const {
    return hash_;
  }

  BufPtr buf() const {
    return buf_;
  }

  const std::vector<ExprPtr>& indices() const {
    return indices_;
  }

  BlockPtr block() const {
    return block_;
  }

  void setEnclosingBlock(BlockPtr b) {
    block_ = std::move(b);
  }

  StmtPtr first_usage() const {
    return first_usage_;
  }
  StmtPtr last_usage() const {
    return last_usage_;
  }

  void setUsageMarks(StmtPtr first, StmtPtr last) {
    first_usage_ = std::move(first);
    last_usage_ = std::move(last);
  }

  bool firstUsageOverlapped() const {
    return firstUsageOverlapped_;
  }

  ExprPtr store_cost() const {
    return store_cost_;
  }

  ExprPtr load_cost() const {
    return load_cost_;
  }

  const std::vector<StorePtr>& stores() const {
    return stores_;
  }

  const std::vector<LoadPtr>& loads() const {
    return loads_;
  }

  void hoistCosts(const ExprPtr& extent) {
    store_cost_ = IRSimplifier::simplify(alloc<Mul>(store_cost_, extent));
    load_cost_ = IRSimplifier::simplify(alloc<Mul>(load_cost_, extent));
  }

  size_t conditionId() const {
    return conditionId_;
  }

  void setConditionId(size_t c) {
    conditionId_ = c;
  }

  size_t accessOrder() const {
    return accessOrder_;
  }

  std::shared_ptr<AccessInfo> hiddenAccess() const {
    return hiddenAccess_;
  }

  // Holds state relating to the scalar variable we will insert to replace some
  // number of loads and stores.
  struct ScalarReplacement {
    VarPtr var{nullptr};
    BufPtr var_wrapper{nullptr};
    LetPtr initializer{nullptr};
  };

  ScalarReplacement& replacement() {
    return replacement_;
  }

 private:
  SimplifierHashType hash_;
  BufPtr buf_;
  std::vector<ExprPtr> indices_;
  BlockPtr block_{nullptr};

  StmtPtr first_usage_{nullptr};
  StmtPtr last_usage_{nullptr};

  // Whether or not this access is overlapped in the first Stmt it appears. This
  // means we cannot use it's first Store as the initializer.
  bool firstUsageOverlapped_{false};

  // The cost in real ops that this access represents, to enable
  // filtering accesses that won't save any loads or stores.
  ExprPtr store_cost_;
  ExprPtr load_cost_;

  // The actual Stores and Loads which represent this access.
  // Be careful with these, any mutator will invalidate these pointers.
  std::vector<StorePtr> stores_;
  std::vector<LoadPtr> loads_;

  // An identifier representing the conditional block, if any, this access
  // depends on.
  size_t conditionId_{0};

  // An identifier representing the order this access was first encountered, for
  // sorting returned results.
  size_t accessOrder_{0};

  // Sometimes when traversing the tree we need to record what would happen if
  // we hoisted an access, but sometimes it doesn't work out. This lets us
  // "undo" some mutation and return to the internal hidden AccessInfo.
  // It will be removed after any further additions to this AccessInfo.
  std::shared_ptr<AccessInfo> hiddenAccess_;

  ScalarReplacement replacement_;
};

using AccessHashMap =
    std::unordered_map<SimplifierHashType, std::shared_ptr<AccessInfo>>;

// Represents a scope block and holds all accesses contained within it.
class Scope {
 public:
  Scope(BlockPtr b, std::shared_ptr<Scope> parent, size_t conditionId = 0)
      : block_(std::move(b)),
        parent_(std::move(parent)),
        conditionId_(conditionId) {}

  AccessHashMap& getAccessMapByBuf(const BufPtr& b);

  std::unordered_map<BufPtr, AccessHashMap>& openAccesses() {
    return openAccesses_;
  }

  std::vector<std::shared_ptr<AccessInfo>>& closedAccesses() {
    return closedAccesses_;
  }

  BlockPtr block() const {
    return block_;
  }

  std::shared_ptr<Scope> parent() const {
    return parent_;
  }

  size_t conditionId() const {
    return conditionId_;
  }

  const std::unordered_set<VarPtr>& localVars() const {
    return localVars_;
  }
  void addLocalVar(VarPtr v) {
    localVars_.insert(std::move(v));
  }

  void closeAccess(const std::shared_ptr<AccessInfo>& info);

  void filterClosed();

 private:
  // Map of map to access, narrowing by Buf then by hash(Buf+Indices).
  // This allows us to find a candidate access easily, and also check for
  // overlap with other accesses to the same buf. Buf ->
  //    Hash ->
  //        Access
  std::unordered_map<BufPtr, AccessHashMap> openAccesses_;
  std::vector<std::shared_ptr<AccessInfo>> closedAccesses_;

  // The Block object this scope represents.
  BlockPtr block_;

  // The enclosing scope object.
  std::shared_ptr<Scope> parent_;

  // An identifier representing the condition block this scope depends on.
  size_t conditionId_;

  // A set of variables local to this scope (e.g. loop vars).
  std::unordered_set<VarPtr> localVars_;
};

/* Analyzes the graph and collects accesses to the same symbolic tensor element
 * which can be replaced by a single local scalar.
 *
 * This works by recursively walking the tree in postfix order, building sets of
 * accesses to the same symbolic element by scope and then merging lower scopes
 * into their enclosing scope.
 *
 * It is safe to move two accesses of the same Tensor element to a local scalar
 * Var if between all usages of the element there are no other Loads or Stores
 * that may refer to it. In the comments I refer to this as overlapping the
 * access, or "cutting" the existing AccessInfo. In the case where a candidate
 * for registerization is cut, it may be possible to finalize the access early
 * by writing it back to the Tensor and then create a new scalar variable after
 * the overlapping access is complete. We will attempt to do this when it saves
 * memory accesses.
 *
 * There are a few cases that make this more challenging:
 *
 *  - For: Loops change the number of real usages of a buffer by the loop
 * extent, but only if we can pull the definition and finalization of the scalar
 * variable out of the loop block.
 *
 * - Cond: Conditions complicate lifting scalars out of internal scopes.
 * Generally we cannot lift an access outside of a conditional scope unless
 * there is already a reference to that same access at the higher scope, since
 * we don't know if the condition was guarding an array access not safe at the
 * higher scope. In the comments I refer to this as the condition "hiding" the
 * access, and the outer access "unhiding" it.
 *
 * - IfThenElse: Same situation as Cond, except since IfThenElse is an Expr
 * rather than a Stmt we cannot insert the scalar definition or finalizer
 * within the conditional scope. Accesses inside an IfThenElse can be safely
 * combined with external accesses but cannot exist completely within.
 *
 * - Let: Accesses dependent on local variables via Let Stmts, or loop vars,
 * cannot be raised outside of the scope of the dependent var.
 */
class TORCH_API RegisterizerAnalysis : public IRVisitor {
 public:
  RegisterizerAnalysis()
      : currentScope_(std::make_shared<Scope>(nullptr, nullptr, 0)) {}
  ~RegisterizerAnalysis() override = default;

  void visit(const ForPtr& v) override;

  void visit(const CondPtr& v) override;

  void visit(const BlockPtr& v) override;

  void visit(const StorePtr& v) override;

  void visit(const LoadPtr& v) override;

  void visit(const IfThenElsePtr& v) override;

  void visit(const LetPtr& v) override;

#define STMT_ON_STACK(Op)                 \
  void visit(const Op##Ptr& v) override { \
    stmtStack_.push_front(v);             \
    IRVisitor::visit(v);                  \
    stmtStack_.pop_front();               \
  }

  STMT_ON_STACK(AtomicAdd)
  STMT_ON_STACK(Allocate)
  STMT_ON_STACK(Free)

#undef STMT_ON_STACK

  std::vector<std::shared_ptr<AccessInfo>> getCandidates();

 private:
  void mergeCurrentScopeIntoParent();
  void mergeHiddenScope(bool allowClosed);
  void closeAccessIntoScope(
      const std::shared_ptr<AccessInfo>& info,
      const std::shared_ptr<Scope>& scope);

  std::unordered_set<size_t> exprConditionals_;

  // A stack of enclosing Stmts for tracking the usage Stmt of Loads.
  std::deque<StmtPtr> stmtStack_;

  // The current scope being analyzed.
  std::shared_ptr<Scope> currentScope_;

  HashProvider hasher_;

  size_t conditionId_{0};
  size_t accessOrder_{0};
};

/* Replaces each registerizable access with a Scalar variable, including
 * definition, initializer and finalizer.
 */
class TORCH_API RegisterizerReplacer : public IRMutator {
 public:
  RegisterizerReplacer(std::vector<std::shared_ptr<AccessInfo>>& vec)
      : infoSet_(vec) {
    buildReplacements();
  }

  ExprPtr mutate(const LoadPtr& v) override;

  StmtPtr mutate(const StorePtr& v) override;

  StmtPtr mutate(const BlockPtr& v) override;

 private:
  struct ReplacerScope {
    std::unordered_map<StmtPtr, std::deque<std::shared_ptr<AccessInfo>>>
        initializerPoints_;
    std::unordered_map<StmtPtr, std::deque<std::shared_ptr<AccessInfo>>>
        finalizePoints_;
  };

  // Creates the various ReplacerScope objects and builds internal maps.
  void buildReplacements();

  // State relating to the accesses yet to be replaced.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  std::vector<std::shared_ptr<AccessInfo>>& infoSet_;
  std::unordered_map<StorePtr, std::shared_ptr<AccessInfo>> storeToAccess_;
  std::unordered_map<LoadPtr, std::shared_ptr<AccessInfo>> loadToAccess_;
  std::unordered_map<BlockPtr, ReplacerScope> parentToAccesses_;

  // Holds the set of Stores that should be pulled into an initializer, so they
  // can be eliminated.
  std::set<StorePtr> eliminatedIntializers_;

  // Tracks the number of times we've seen each buffer, so we can name the
  // scalar Vars appropriately.
  std::unordered_map<BufPtr, unsigned int> bufferAccessCounts_;
  unsigned int getBufferAccessCount(const BufPtr& b) {
    return ++bufferAccessCounts_[b];
  }
};
} // namespace registerizer

// Apply scalar replacement to all accesses in s.
// To produce safe code, this must occur after handling parallelized axes and
// atomics.
TORCH_API StmtPtr registerize(StmtPtr s);

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 45 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `registerizer`

**Classes/Structs**: `Scope`, `AccessInfo`, `ScalarReplacement`, `Scope`, `TORCH_API`, `TORCH_API`, `ReplacerScope`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/ScalarType.h`
- `c10/util/irange.h`
- `torch/csrc/Export.h`
- `torch/csrc/jit/tensorexpr/hash_provider.h`
- `torch/csrc/jit/tensorexpr/ir_mutator.h`
- `torch/csrc/jit/tensorexpr/ir_simplifier.h`
- `torch/csrc/jit/tensorexpr/ir_visitor.h`
- `utility`
- `vector`


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

- **File Documentation**: `registerizer.h_docs.md`
- **Keyword Index**: `registerizer.h_kw.md`
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

- **File Documentation**: `registerizer.h_docs.md_docs.md`
- **Keyword Index**: `registerizer.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
