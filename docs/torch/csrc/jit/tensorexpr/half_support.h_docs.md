# Documentation: `torch/csrc/jit/tensorexpr/half_support.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/half_support.h`
- **Size**: 5,889 bytes (5.75 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch::jit::tensorexpr {

// Walk the Statement looking for Half size loads/stores.
class HalfChecker : public IRVisitor {
 public:
  HalfChecker(const std::vector<CodeGen::BufferArg>& args) {
    for (const auto& BA : args) {
      hasHalf_ |= BA.dtype().scalar_type() == ScalarType::Half;
    }
  }

  bool hasHalf() const {
    return hasHalf_;
  }

  bool hasBFloat16() const {
    return hasBFloat16_;
  }

  void visit(const LoadPtr& v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

  void visit(const StorePtr& v) override {
    hasHalf_ |= v->buf()->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->buf()->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

  void visit(const HalfImmPtr& v) override {
    hasHalf_ = true;
  }

  void visit(const BFloat16ImmPtr& v) override {
    hasBFloat16_ = true;
  }

  void visit(const CastPtr& v) override {
    hasHalf_ |= v->dtype().scalar_type() == ScalarType::Half;
    hasBFloat16_ |= v->dtype().scalar_type() == ScalarType::BFloat16;
    IRVisitor::visit(v);
  }

 private:
  bool hasHalf_{false};
  bool hasBFloat16_{false};
};

class HalfRewriter : public IRMutator {
  ExprPtr mutate(const LoadPtr& v) override {
    ExprPtr child = IRMutator::mutate(v);
    if (!isHalf(child)) {
      return child;
    }

    ExprPtr ret = alloc<Cast>(
        child->dtype().cloneWithScalarType(ScalarType::Float), child);

    inserted_half_casts_.insert(ret);
    return ret;
  }

  StmtPtr mutate(const StorePtr& v) override {
    // Since mutation changes the `value()` expression in-place, we need to
    // get the dtype of the `value()` before that is mutated.
    auto newType = v->value()->dtype();
    ExprPtr new_val = v->value()->accept_mutator(this);
    auto bufType = v->buf()->dtype();

    if (isHalf(newType.scalar_type())) {
      new_val = alloc<Cast>(newType, new_val);
      inserted_half_casts_.insert(new_val);
    }

    // The scalar_type of value is not Half while the buf is Half
    if (!isHalf(newType.scalar_type()) && isHalf(bufType.scalar_type())) {
      new_val = alloc<Cast>(
          newType.cloneWithScalarType(bufType.scalar_type()), new_val);
      inserted_half_casts_.insert(new_val);
    }

    v->set_value(new_val);
    return v;
  }

  ExprPtr mutate(const HalfImmPtr& v) override {
    return alloc<Cast>(kFloat, v);
  }

  ExprPtr mutate(const BFloat16ImmPtr& v) override {
    return alloc<Cast>(kFloat, v);
  }

  ExprPtr mutate(const CastPtr& v) override {
    ExprPtr child = v->src_value()->accept_mutator(this);

    // just don't allow half casts we didn't insert.
    if (isHalf(v)) {
      if (inserted_half_casts_.count(v) < 1) {
        v->set_src_value(child);
        v->set_dtype(v->dtype().cloneWithScalarType(c10::kFloat));
        return v;
      }
    }

    // Remove Half(Float()) and friends.
    CastPtr cast_child = to<Cast>(child);
    if (cast_child) {
      auto cast_to_double = v->dtype().scalar_type() == ScalarType::Double;
      auto from_half = isHalf(cast_child->src_value());
      // Cannot simplify the double(float(half)) to double(half) as NNC does
      // not support cast BF16 to double directly.
      auto not_cast_half_to_doulbe = !(cast_to_double && from_half);
      if (v->dtype().is_floating_point() &&
          cast_child->dtype().is_floating_point() && not_cast_half_to_doulbe) {
        return alloc<Cast>(v->dtype(), cast_child->src_value());
      }
    }

    if (child == v->src_value()) {
      return v;
    }

    return alloc<Cast>(v->dtype(), child);
  }

  StmtPtr mutate(const LetPtr& v) override {
    if (isHalf(v->var()->dtype().scalar_type())) {
      VarPtr load_new_var = alloc<Var>(v->var()->name_hint(), kFloat);
      ExprPtr new_value = alloc<Cast>(
          v->var()->dtype().cloneWithScalarType(ScalarType::Float),
          v->value()->accept_mutator(this));
      var_map[v->var()] = load_new_var;

      return alloc<Let>(load_new_var, new_value);
    }

    return IRMutator::mutate(v);
  }

  ExprPtr mutate(const VarPtr& v) override {
    auto it = var_map.find(v);
    if (it != var_map.end()) {
      return it->second;
    }

    return v;
  }

  template <typename T>
  ExprPtr mutateArithmetic(T v) {
    IRMutator::mutate(v);
    if (isHalf(v)) {
      v->set_dtype(v->dtype().cloneWithScalarType(c10::kFloat));
    }
    return v;
  }

  ExprPtr mutate(const AddPtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const SubPtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const MulPtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const DivPtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const MaxPtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const MinPtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const CompareSelectPtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const BroadcastPtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const IfThenElsePtr& v) override {
    return mutateArithmetic(v);
  }
  ExprPtr mutate(const IntrinsicsPtr& v) override {
    return mutateArithmetic(v);
  }

 private:
  static bool isHalf(ScalarType st) {
    return st == ScalarType::Half || st == ScalarType::BFloat16;
  }

  static bool isHalf(const ExprPtr& v) {
    return isHalf(v->dtype().scalar_type());
  }

  std::unordered_set<ExprPtr> inserted_half_casts_;
  std::unordered_map<VarPtr, VarPtr> var_map;
};

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 42 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `HalfChecker`, `HalfRewriter`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/tensorexpr/codegen.h`
- `torch/csrc/jit/tensorexpr/ir.h`
- `torch/csrc/jit/tensorexpr/ir_visitor.h`
- `torch/csrc/jit/tensorexpr/tensor.h`


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

- **File Documentation**: `half_support.h_docs.md`
- **Keyword Index**: `half_support.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
