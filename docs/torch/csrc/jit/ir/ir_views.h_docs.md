# Documentation: `torch/csrc/jit/ir/ir_views.h`

## File Metadata

- **Path**: `torch/csrc/jit/ir/ir_views.h`
- **Size**: 4,624 bytes (4.52 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

struct IfView {
  explicit IfView(Node* node) : node_(node) {
    AT_ASSERT(node->kind() == ::c10::prim::If);
  }
  Value* cond() const {
    return node_->input(0);
  }
  Block* thenBlock() const {
    return node_->blocks().at(0);
  }
  Block* elseBlock() const {
    return node_->blocks().at(1);
  }
  ArrayRef<Value*> thenOutputs() const {
    return thenBlock()->outputs();
  }
  ArrayRef<Value*> elseOutputs() const {
    return elseBlock()->outputs();
  }
  ArrayRef<Value*> outputs() const {
    return node_->outputs();
  }
  Node* node() const {
    return node_;
  }
  operator Node*() const {
    return node_;
  }

  void permuteOutputs(const std::vector<size_t>& new_output_order) {
    node_->permuteOutputs(new_output_order);
    thenBlock()->permuteOutputs(new_output_order);
    elseBlock()->permuteOutputs(new_output_order);
  }

 private:
  Node* node_;
};

struct LoopView {
  explicit LoopView(Node* node) : node_(node) {
    AT_ASSERT(
        node->kind() == ::c10::prim::Loop || node->kind() == ::c10::onnx::Loop);
  }
  Block* bodyBlock() const {
    return node_->blocks().at(0);
  }
  Value* cond() const {
    return node_->input(0);
  }
  Value* maxTripCount() const {
    return node_->input(0);
  }
  Value* inputCond() const {
    return node_->input(1);
  }
  Value* nextCond() const {
    return bodyBlock()->outputs().at(0);
  }
  Value* currentTripCount() const {
    return bodyBlock()->inputs().at(0);
  }
  ArrayRef<Value*> carriedInputs() const {
    // skip trip count and cond
    return node_->inputs().slice(2);
  }
  ArrayRef<Value*> carriedInputsWithCond() const {
    // skip trip count and cond
    return node_->inputs().slice(1);
  }
  ArrayRef<Value*> carriedOutputs() const {
    return node_->outputs();
  }
  ArrayRef<Value*> bodyCarriedInputs() const {
    // skip trip count and cond
    return bodyBlock()->inputs().slice(1);
  }
  ArrayRef<Value*> bodyCarriedOutputs() const {
    return bodyBlock()->outputs().slice(1);
  }
  Node* node() const {
    return node_;
  }
  operator Node*() const {
    return node_;
  }

  void permuteLoopCarried(const std::vector<size_t>& new_output_order) {
    node_->permuteOutputs(new_output_order);
    // skip trip count and cond
    node_->permuteInputs(adjustIndices(2, new_output_order));
    auto adjusted_block_order = adjustIndices(1, new_output_order);
    bodyBlock()->permuteOutputs(adjusted_block_order);
    bodyBlock()->permuteInputs(adjusted_block_order);
  }

  void replaceMaxTripCount(Value* new_max_trip_count) {
    node_->replaceInput(0, new_max_trip_count);
  }
  void replaceInputCondition(Value* new_input_condition) {
    node_->replaceInput(1, new_input_condition);
  }

  // our way of encoding loops makes them difficult to turn back into python
  // syntax. we have to check properties of the condition and trip count inputs
  // to figure out which one it initially was. ModifiedLoops are not directly
  // mappable to either For or While
  enum LoopType { While, For, ModifiedLoop };

  LoopType loopType() {
    auto trip_count = toIValue(maxTripCount());
    auto cond_input = toIValue(inputCond());
    auto cond_next = toIValue(nextCond());

    bool condition_is_always_true =
        cond_input && cond_input->toBool() && cond_next && cond_next->toBool();
    bool trip_count_is_specified = !trip_count || // trip is not a constant
        trip_count->toInt() !=
            std::numeric_limits<int64_t>::max() || // it is a constant but not
                                                   // the default one
        !currentTripCount()
             ->uses()
             .empty(); // it is actually being used in the body.

    if (condition_is_always_true) {
      // if the trip count was not specified this was a user-written while True:
      return trip_count_is_specified ? For : While;
    } else {
      if (trip_count_is_specified) {
        return ModifiedLoop;
      }
      return While;
    }
  }

 private:
  Node* node_;

  // adjust index_ordering by adding indices 0 - thorough adjust, and
  // incrementing all existing inputs by adjust
  static std::vector<size_t> adjustIndices(
      size_t adjust,
      const std::vector<size_t>& index_ordering) {
    std::vector<size_t> adjusted;
    adjusted.reserve(adjust + index_ordering.size());
    for (const auto i : c10::irange(adjust)) {
      adjusted.push_back(i);
    }
    for (auto index : index_ordering) {
      adjusted.push_back(index + adjust);
    }
    return adjusted;
  }
};
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `IfView`, `LoopView`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/ir`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/irange.h`
- `torch/csrc/jit/ir/ir.h`


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

Files in the same folder (`torch/csrc/jit/ir`):

- [`node_hashing.h_docs.md`](./node_hashing.h_docs.md)
- [`constants.cpp_docs.md`](./constants.cpp_docs.md)
- [`subgraph_matcher.h_docs.md`](./subgraph_matcher.h_docs.md)
- [`scope.cpp_docs.md`](./scope.cpp_docs.md)
- [`graph_node_list.h_docs.md`](./graph_node_list.h_docs.md)
- [`type_hashing.cpp_docs.md`](./type_hashing.cpp_docs.md)
- [`ir.h_docs.md`](./ir.h_docs.md)
- [`ir.cpp_docs.md`](./ir.cpp_docs.md)
- [`irparser.cpp_docs.md`](./irparser.cpp_docs.md)
- [`node_hashing.cpp_docs.md`](./node_hashing.cpp_docs.md)


## Cross-References

- **File Documentation**: `ir_views.h_docs.md`
- **Keyword Index**: `ir_views.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
