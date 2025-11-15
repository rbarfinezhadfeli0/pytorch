# Documentation: `docs/torch/csrc/jit/passes/normalize_ops.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/passes/normalize_ops.cpp_docs.md`
- **Size**: 8,046 bytes (7.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/passes/normalize_ops.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/passes/normalize_ops.cpp`
- **Size**: 5,432 bytes (5.30 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/passes/normalize_ops.h>

#include <c10/util/Exception.h>

namespace torch::jit {

namespace {

// having multiple ops in our IR that do the same thing makes the IR more
// difficult to consumer for downstream user of the IR, such as our own
// optimization passes here, we convert op aliases into a standard form
bool normalizeOpAliases(graph_node_list_iterator& iter) {
  auto alias = getOperatorAliasMap().find(iter->kind());
  if (alias != getOperatorAliasMap().end()) {
    iter->replaceWithNewSymbol(alias->second);
    iter.destroyCurrent();
    return true;
  }
  return false;
}

// Normalize rsub such that `rsub(x,y) = sub(x,y)`
bool normalizeRSub(graph_node_list_iterator& iter) {
  if (iter->matches(
          "aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor")) {
    ArrayRef<Value*> args = iter->inputs();
    Node* newSub = iter->replaceWithNewSymbol(aten::sub);
    newSub->replaceInput(0, args[1]);
    newSub->replaceInput(1, args[0]);
    iter.destroyCurrent();
    return true;
  }
  return false;
}

// Normalizes a `__is__` comparison with a bool to `eq` (and same with
// `__isnot__`)
bool normalizeIsBool(graph_node_list_iterator& iter) {
  ArrayRef<Value*> args = iter->inputs();
  if (args.size() == 2 && args[0]->type() == BoolType::get() &&
      args[1]->type() == BoolType::get()) {
    if (iter->kind() == aten::__is__) {
      iter->replaceWithNewSymbol(aten::eq);
      iter.destroyCurrent();
      return true;
    }
    if (iter->kind() == aten::__isnot__) {
      iter->replaceWithNewSymbol(aten::ne);
      iter.destroyCurrent();
      return true;
    }
  }
  return false;
}

void NormalizeOps(Block* block) {
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    for (auto sub : it->blocks()) {
      NormalizeOps(sub);
    }

    if (normalizeRSub(it)) {
      continue;
    }

    if (normalizeOpAliases(it)) {
      continue;
    }

    if (normalizeIsBool(it)) {
      continue;
    }

    it++;
  }
}

} // namespace

const std::unordered_map<Symbol, Symbol>& getOperatorAliasMap() {
  // map from op alias -> normalized op
  static const std::unordered_map<Symbol, Symbol> alias_map = {
      {aten::absolute, aten::abs},
      {aten::absolute_, aten::abs_},
      {aten::clip, aten::clamp},
      {aten::clip_, aten::clamp_},
      {aten::det, aten::linalg_det},
      {aten::matrix_power, aten::linalg_matrix_power},
      {aten::matrix_exp, aten::linalg_matrix_exp},
      {aten::ger, aten::outer},
      {aten::arccos, aten::acos},
      {aten::arccos_, aten::acos_},
      {aten::arcsin, aten::asin},
      {aten::arcsin_, aten::asin_},
      {aten::arctan, aten::atan},
      {aten::arctan_, aten::atan_},
      {aten::arctan2, aten::atan2},
      {aten::arctan2_, aten::atan2_},
      {aten::arccosh, aten::acosh},
      {aten::arccosh_, aten::acosh_},
      {aten::arcsinh, aten::asinh},
      {aten::arcsinh_, aten::asinh_},
      {aten::arctanh, aten::atanh},
      {aten::arctanh_, aten::atanh_},
      {aten::fix, aten::trunc},
      {aten::fix_, aten::trunc_},
      {aten::negative, aten::neg},
      {aten::negative_, aten::neg_},
      {aten::subtract, aten::sub},
      {aten::subtract_, aten::sub_},
      {aten::greater_equal, aten::ge},
      {aten::greater_equal_, aten::ge_},
      {aten::greater, aten::gt},
      {aten::greater_, aten::gt_},
      {aten::less_equal, aten::le},
      {aten::less_equal_, aten::le_},
      {aten::less, aten::lt},
      {aten::less_, aten::lt_},
      {aten::not_equal, aten::ne},
      {aten::not_equal_, aten::ne_},
      {aten::divide, aten::div},
      {aten::divide_, aten::div_},
      {aten::multiply, aten::mul},
      {aten::multiply_, aten::mul_},
      {aten::linalg_matmul, aten::matmul},
      {aten::inverse, aten::linalg_inv},
      {aten::true_divide, aten::div},
      {aten::true_divide_, aten::div_},
      {aten::concat, aten::cat},
      {aten::concatenate, aten::cat},
      {aten::row_stack, aten::vstack},
      {aten::swapdims, aten::transpose},
      {aten::swapdims_, aten::transpose_},
      {aten::swapaxes, aten::transpose},
      {aten::swapaxes_, aten::transpose_},
      {aten::moveaxis, aten::movedim},
      {aten::special_erf, aten::erf},
      {aten::special_erfc, aten::erfc},
      {aten::special_erfinv, aten::erfinv},
      {aten::special_expit, aten::sigmoid},
      {aten::special_exp2, aten::exp2},
      {aten::special_expm1, aten::expm1},
      {aten::special_logit, aten::logit},
      {aten::special_logsumexp, aten::logsumexp},
      {aten::special_round, aten::round},
      {aten::special_log1p, aten::log1p},
      {aten::special_sinc, aten::sinc},
      {aten::special_digamma, aten::digamma},
      {aten::special_psi, aten::digamma},
      {aten::special_i0, aten::i0},
      {aten::special_xlogy, aten::xlogy},
      {aten::special_log_softmax, aten::log_softmax},
      {aten::orgqr, aten::linalg_householder_product},
      {aten::adjoint, aten::mH},
      {aten::special_multigammaln, aten::mvlgamma},
      {aten::special_polygamma, aten::polygamma},
      {aten::special_softmax, aten::softmax},
      {aten::special_gammainc, aten::igamma},
      {aten::special_gammaincc, aten::igammac},
      {aten::special_gammaln, aten::lgamma}};
  return alias_map;
}

void NormalizeOps(const std::shared_ptr<Graph>& graph) {
  NormalizeOps(graph->block());
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `const`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/passes/normalize_ops.h`
- `c10/util/Exception.h`


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

- **File Documentation**: `normalize_ops.cpp_docs.md`
- **Keyword Index**: `normalize_ops.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/passes`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/jit/passes`):

- [`peephole_dict_idioms.h_docs.md_docs.md`](./peephole_dict_idioms.h_docs.md_docs.md)
- [`remove_redundant_profiles.h_kw.md_docs.md`](./remove_redundant_profiles.h_kw.md_docs.md)
- [`loop_unrolling.cpp_kw.md_docs.md`](./loop_unrolling.cpp_kw.md_docs.md)
- [`onnx.h_kw.md_docs.md`](./onnx.h_kw.md_docs.md)
- [`guard_elimination.h_docs.md_docs.md`](./guard_elimination.h_docs.md_docs.md)
- [`frozen_conv_add_relu_fusion.cpp_docs.md_docs.md`](./frozen_conv_add_relu_fusion.cpp_docs.md_docs.md)
- [`hoist_conv_packed_params.h_kw.md_docs.md`](./hoist_conv_packed_params.h_kw.md_docs.md)
- [`lift_closures.h_kw.md_docs.md`](./lift_closures.h_kw.md_docs.md)
- [`frozen_conv_folding.h_kw.md_docs.md`](./frozen_conv_folding.h_kw.md_docs.md)
- [`frozen_graph_optimizations.h_docs.md_docs.md`](./frozen_graph_optimizations.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `normalize_ops.cpp_docs.md_docs.md`
- **Keyword Index**: `normalize_ops.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
