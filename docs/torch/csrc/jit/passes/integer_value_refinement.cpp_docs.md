# Documentation: `torch/csrc/jit/passes/integer_value_refinement.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/integer_value_refinement.cpp`
- **Size**: 8,311 bytes (8.12 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/integer_value_refinement.h>
#include <torch/csrc/jit/passes/value_refinement_utils.h>

#include <utility>

namespace torch::jit {

using IntegerRefinement = std::unordered_map<Value*, int64_t>;

// see [value refinement algorithm] for full explanation.
// When a comparison like `cond = x == 4` or `cond = x != 4` is made,
// `cond` value carries information (refinements) about the value of `x`.
// in an example like:
// if x == 1:
//    ...
// we can substitute all uses of x dominated by the true block
// with 1.

struct IntegerValueRefiner {
  IntegerValueRefiner(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run() {
    if (!blockHasIntComparisons(graph_->block())) {
      return false;
    }
    IntegerRefinement refinements;
    RefineIntegerValues(graph_->block(), std::move(refinements));
    return changed_;
  }

  bool blockHasIntComparisons(Block* b) {
    for (Node* n : b->nodes()) {
      if (n->matches("aten::eq(int a, int b) -> bool") ||
          n->matches("aten::ne(int a, int b) -> bool")) {
        for (size_t const_index : {0, 1}) {
          auto non_const_index = 1 - const_index;
          if (n->inputs().at(const_index)->node()->kind() == prim::Constant &&
              n->inputs().at(non_const_index)->uses().size() > 1) {
            return true;
          }
        }
      }
      for (Block* block : n->blocks()) {
        if (blockHasIntComparisons(block)) {
          return true;
        }
      }
    }
    return false;
  }

  void removeIfNodeOutputsWithRefinements(
      Node* if_node,
      IntegerRefinement& true_block_refinements,
      IntegerRefinement& false_block_refinements) {
    // we are looking for cases where we can replace both block outputs with the
    // same value, which opens up further optimization opportunities. The pass
    // will already handle if both outputs are refined to the same constant.
    // Here, we look for cases where one block output has been refined in the
    // other block to be equal to the same constant value as the other other
    // block output:
    //  graph(%y.1 : int):
    //   %one_constant : int = prim::Constant[value=1]()
    //   %3 : bool = aten::eq(%y.1, %one_constant)
    //   %15 : int = prim::If(%3)
    //     block0():
    //       -> (%one_constant)
    //     block1():
    //       -> (%y.1)
    //   return (%15)
    // %15 can always be safely replaced with %y.1
    // this is an important case for symbolic shape analysis
    for (size_t block_index : {0, 1}) {
      Block* if_block = if_node->blocks().at(block_index);
      Block* other_if_block = if_node->blocks().at(1 - block_index);
      for (size_t i = 0; i < if_node->outputs().size(); ++i) {
        Value* block_output = if_block->outputs().at(i);
        if (!block_output->type()->cast<IntType>()) {
          continue;
        }
        // Value must be in scope for both blocks
        // in example above, %y.1 cannot be defined in block1
        if (!if_node->isDominatedBy(block_output->node())) {
          continue;
        }
        // one constant value one not - we are looking for the pattern
        // where y.1 is refined to the existing block output %one_constant
        auto other_output = other_if_block->outputs().at(i);
        auto other_const_value = other_output->type()->cast<IntType>()
            ? constant_as<int64_t>(other_output)
            : std::nullopt;
        if (!other_const_value ||
            block_output->node()->kind() == prim::Constant) {
          continue;
        }
        // here, we are looking in refinements in the other block of our
        // current output. in the example, we are looking for refinements of
        // %y.1 in `block0`, and we are checking that %y.1 is refined
        // to the constant value of %one_constant
        const auto& other_block_refinements =
            block_index == 0 ? false_block_refinements : true_block_refinements;
        if (!other_block_refinements.count(block_output)) {
          continue;
        }
        if (other_block_refinements.at(block_output) == *other_const_value) {
          if_node->outputs().at(i)->replaceAllUsesWith(block_output);
          changed_ = true;
        }
      }
    }
  }

  // iteratively look through the block `b` for refinements or Value uses that
  // can be refined, `block_refinements` are the refinements present starting at
  // this block (and for all blocks dominated by this block).
  IntegerRefinement RefineIntegerValues(
      Block* b,
      IntegerRefinement block_refinements) {
    active_refinements_.push_back(&block_refinements);
    for (Node* n : b->nodes()) {
      if (n->matches("aten::eq(int a, int b) -> bool") ||
          n->matches("aten::ne(int a, int b) -> bool")) {
        for (size_t const_index : {0, 1}) {
          if (auto ival = constant_as<int64_t>(n->inputs().at(const_index))) {
            IntegerRefinement refine;
            refine[n->inputs().at(1 - const_index)] = *ival;
            info_[n->output()] = n->kind() == aten::eq
                ? BooleanRefinementMapping::TrueRefinements(std::move(refine))
                : BooleanRefinementMapping::FalseRefinements(std::move(refine));
          }
        }
      }
      for (size_t input = 0; input < n->inputs().size(); ++input) {
        Value* input_v = n->inputs().at(input);
        if (!input_v->type()->cast<IntType>()) {
          continue;
        }

        if (auto refine = tryFindRefinement(input_v)) {
          WithInsertPoint guard(n);
          auto refine_constant =
              graph_->insertConstant(static_cast<int64_t>(*refine));
          n->replaceInputWith(input_v, refine_constant);
          changed_ = true;
        }
      }

      if (n->kind() == prim::If) {
        IfView if_n(n);
        bool has_cond_ref = info_.count(if_n.cond()) != 0;
        IntegerRefinement empty;
        auto true_block_refinements = RefineIntegerValues(
            if_n.thenBlock(),
            has_cond_ref ? info_[if_n.cond()].true_refine() : empty);
        auto false_block_refinements = RefineIntegerValues(
            if_n.elseBlock(),
            has_cond_ref ? info_[if_n.cond()].false_refine() : empty);

        removeIfNodeOutputsWithRefinements(
            n, true_block_refinements, false_block_refinements);

        joinIfRefinements(
            n,
            throwing_blocks_,
            block_refinements,
            true_block_refinements,
            false_block_refinements,
            info_);
      } else {
        handleCommonRefinentOperators(n, throwing_blocks_, info_);
      }
    }

    // iterating over all nodes in the block will not iterate over
    // block outputs, so we need to add handling of them.
    // %3 : int = prim::Constant[value=3]()
    // %4 : bool = aten::eq(%y.1, %3)
    // %a : int = prim::If(%4)
    //   block0():
    //     -> (%y.1)
    // Here, we can replace y.1 with 3

    for (size_t i = 0; i < b->outputs().size(); ++i) {
      Value* output_v = b->outputs().at(i);
      if (!output_v->type()->cast<IntType>()) {
        continue;
      }

      if (auto refine = tryFindRefinement(output_v)) {
        WithInsertPoint guard(b);
        auto refine_constant =
            graph_->insertConstant(static_cast<int64_t>(*refine));
        b->replaceOutput(i, refine_constant);
        changed_ = true;
      }
    }

    active_refinements_.pop_back();
    return block_refinements;
  }

  std::optional<int64_t> tryFindRefinement(Value* v) {
    for (const auto& ref : active_refinements_) {
      auto maybe_refinement = ref->find(v);
      if (maybe_refinement != ref->end()) {
        return maybe_refinement->second;
      }
    }
    return std::nullopt;
  }

  std::shared_ptr<Graph> graph_;
  // A stack of active refinements, one for each block
  std::vector<IntegerRefinement*> active_refinements_;
  // A map from Boolean Value * -> associated refinements
  std::unordered_map<Value*, BooleanRefinementMapping> info_;
  std::unordered_set<Block*> throwing_blocks_;
  bool changed_ = false;
};

bool RefineIntegerValues(const std::shared_ptr<Graph>& graph) {
  return IntegerValueRefiner(graph).run();
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `IntegerValueRefiner`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/jit_type.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/integer_value_refinement.h`
- `torch/csrc/jit/passes/value_refinement_utils.h`
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

Files in the same folder (`torch/csrc/jit/passes`):

- [`inline_fork_wait.h_docs.md`](./inline_fork_wait.h_docs.md)
- [`subgraph_rewrite.cpp_docs.md`](./subgraph_rewrite.cpp_docs.md)
- [`value_refinement_utils.cpp_docs.md`](./value_refinement_utils.cpp_docs.md)
- [`create_autodiff_subgraphs.cpp_docs.md`](./create_autodiff_subgraphs.cpp_docs.md)
- [`update_differentiable_graph_requires_grad.h_docs.md`](./update_differentiable_graph_requires_grad.h_docs.md)
- [`inplace_check.h_docs.md`](./inplace_check.h_docs.md)
- [`common_subexpression_elimination.h_docs.md`](./common_subexpression_elimination.h_docs.md)
- [`dtype_analysis.cpp_docs.md`](./dtype_analysis.cpp_docs.md)
- [`canonicalize.h_docs.md`](./canonicalize.h_docs.md)
- [`add_if_then_else.h_docs.md`](./add_if_then_else.h_docs.md)


## Cross-References

- **File Documentation**: `integer_value_refinement.cpp_docs.md`
- **Keyword Index**: `integer_value_refinement.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
