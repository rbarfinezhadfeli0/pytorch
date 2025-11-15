# Documentation: `docs/torch/csrc/jit/runtime/profiling_record.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/profiling_record.cpp_docs.md`
- **Size**: 12,644 bytes (12.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/profiling_record.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/profiling_record.cpp`
- **Size**: 9,669 bytes (9.44 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <ATen/core/symbol.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

namespace torch::jit {

namespace {

class ProfileRegistry {
 public:
  static ProfileRegistry* getRegistry() {
    static ProfileRegistry profile_registry_;
    return &profile_registry_;
  }

  void registerProfileNode(const std::function<bool(const Node*)>& func) {
    std::lock_guard<std::mutex> guard(mutex_);
    registry_funcs_.push_back(func);
  }

  bool shouldProfileNode(const Node* node) {
    std::lock_guard<std::mutex> guard(mutex_);
    // to guard differentiable graphs, we want profiling information
    // (in particular requires_grad) for nodes handled by autodiff
    if (isDifferentiable(node)) {
      return true;
    }
    for (const auto& func : registry_funcs_) {
      if (func(node)) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<std::function<bool(const Node*)>> registry_funcs_;
  std::mutex mutex_;
};

} // namespace

void RegisterProfilingNode(const std::function<bool(const Node*)>& func) {
  ProfileRegistry::getRegistry()->registerProfileNode(func);
}

bool ShapeSymbolTable::bindSymbolicShapes(
    at::IntArrayRef new_sizes,
    const c10::SymbolicShape& sym_shapes) {
  if (!sym_shapes.rank().has_value()) {
    return true;
  }
  if (*sym_shapes.rank() != new_sizes.size()) {
    return false;
  }
  for (const auto i : c10::irange(new_sizes.size())) {
    auto symbol = (*sym_shapes.sizes())[i];
    if (!symbol.is_static()) {
      continue;
    }

    if (!isBound(symbol)) {
      assign(symbol, new_sizes[i]);
      continue;
    }

    if (getValue(symbol) != new_sizes[i]) {
      return false;
    }
  }
  return true;
}

ProfilingRecord::ProfilingRecord(std::shared_ptr<Graph> g)
    : profiled_graph_(std::move(g)), profiling_count_(getNumProfiledRuns()) {}

ProfileOp* ProfilingRecord::createProfileNode(
    const std::function<void(Stack&)>& fp,
    at::ArrayRef<Value*> inputs) {
  auto pn = new ProfileOp(profiled_graph_.get(), fp);

  for (auto in : inputs) {
    pn->addInput(in);
  }
  return pn;
}

ProfileIValueOp* ProfilingRecord::createProfileIValueNode(Value* in_val) {
  auto pn = new ProfileIValueOp(this->profiled_graph_.get(), nullptr);
  pn->addInput(in_val);
  auto pno = pn->addOutput();
  pno->setType(in_val->type());
  return pn;
}

ProfileIValueOp* ProfilingRecord::createProfileIValueNode(
    ArrayRef<Value*> inputs) {
  auto pn = new ProfileIValueOp(this->profiled_graph_.get(), nullptr);
  for (auto inp : inputs) {
    pn->addInput(inp);
    auto pno = pn->addOutput();
    pno->setType(inp->type());
  }
  return pn;
}

namespace {
bool isOptionalTensorType(const TypePtr& type) {
  if (type->kind() != c10::TypeKind::OptionalType) {
    return false;
  }
  const auto& kind = type->expectRef<OptionalType>().getElementType()->kind();
  return kind == c10::TypeKind::TensorType;
}
} // namespace

// Inserts profiling nodes.
//
// The prim::profile node profiles Tensor and Optional[Tensor].
//
// It stores two fields:
// 1. attr::seen_none, an integer, which is initially 0 and is set to 1 if the
// profiled value is ever `None`
// 2. attr::profiled_type, which is the most specific Tensor type that matches
// all the non-null inputs observed during profiling.
void ProfilingRecord::insertShapeProfile(
    Node* n,
    size_t offset,
    const TypePtr& input_type) {
  Value* i = n->input(offset);
  auto pn = createProfileNode(nullptr, {i});
  auto pno = pn->addOutput();
  pn->ty_(attr::profiled_type, TensorType::get());
  pn->i_(attr::seen_none, 0);
  if (isOptionalTensorType(input_type)) {
    pno->setType(OptionalType::create(TensorType::get()));
  } else if (input_type->kind() == c10::TypeKind::TensorType) {
    pno->setType(TensorType::get());
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "Trying to profile an unsupported type (neither Tensor or Optional[Tensor]): ",
        input_type->str());
  }
  std::function<void(Stack&)> shape_profiler = [this, pn, pno](Stack& stack) {
    int64_t frame_id = 0;
    pop(stack, frame_id);
    IValue v;
    pop(stack, v);

    TensorTypePtr new_tensor_type = nullptr;
    if (v.isTensor()) {
      auto& t = v.toTensor();
      new_tensor_type = tensorTypeInCurrentExecutionContext(t);
    }

    if (v.isTensor() || v.isNone()) {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (profiling_count_ > 0) {
        GRAPH_DEBUG(
            "In run ",
            frame_id,
            " annotating %",
            pno->debugName(),
            " with ",
            *new_tensor_type);

        if (new_tensor_type != nullptr) {
          if (pn->hasSeenTensor()) {
            const auto& existing_tensor_type =
                pn->ty(attr::profiled_type)->expectRef<TensorType>();
            GRAPH_DEBUG(
                "Existing type for %",
                pno->debugName(),
                ": ",
                existing_tensor_type);
            auto merged_type = new_tensor_type->merge(existing_tensor_type);
            GRAPH_DEBUG(
                "Merged type for %", pno->debugName(), ": ", *merged_type);
            pn->ty_(attr::profiled_type, std::move(merged_type));
          } else {
            pn->setHasSeenTensor(true);
            pn->ty_(attr::profiled_type, std::move(new_tensor_type));
          }
        }
        if (v.isNone()) {
          pn->i_(attr::seen_none, 1);
        }
      }
    }
    // passing t through
    push(stack, v);
  };

  pn->setCallback(shape_profiler);
  pn->insertBefore(n);
  n->replaceInput(offset, pn->output());
}

static bool needsProfiledInputs(Node* n) {
  if (tensorexpr::isSupported(n)) {
    return true;
  }

  switch (n->kind()) {
    // specialize_autogradzero
    case prim::AutogradAdd:
    case prim::AutogradAnyNonZero:
    case prim::AutogradAllNonZero:
    case prim::AutogradAllZero:
    case prim::AutogradZero:
    // peephole
    case aten::dim:
    case aten::size:
    case aten::expand:
    case prim::dtype:
    case prim::device:
    case prim::is_cuda:
    case aten::is_floating_point:
    case aten::type_as:
    // TODO: hack to make `test_lstm_gates_permutations_cuda`
    // pass.
    case aten::t:
    case aten::mm:
      return true;
    default:
      return ProfileRegistry::getRegistry()->shouldProfileNode(n);
  }
}

static bool needsProfiledOutput(Node* n) {
  if (tensorexpr::isSupported(n)) {
    return true;
  }

  switch (n->kind()) {
    case prim::AutogradAdd:
    case prim::AutogradZero:
      return true;
    default:
      return ProfileRegistry::getRegistry()->shouldProfileNode(n);
  }
}

void ProfilingRecord::removeProfileCounter(Block* b) {
  for (auto it = b->nodes().rbegin(); it != b->nodes().rend();) {
    auto n = *it;
    if (n->kind() == prim::profile && n->inputs().empty()) {
      it.destroyCurrent();
      // there is only one counter node
      return;
    } else {
      it++;
    }
  }
}

void ProfilingRecord::instrumentBlock(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    for (const auto offset : c10::irange(n->inputs().size())) {
      auto i = n->input(offset);
      if ((needsProfiledInputs(n) || needsProfiledOutput(i->node()))) {
        if (i->type()->kind() == c10::TypeKind::TensorType ||
            isOptionalTensorType(i->type())) {
          insertShapeProfile(n, offset, i->type());
        }
      }
    }

    for (auto b : n->blocks()) {
      instrumentBlock(b);
    }
  }

  // inserting profile nodes on block outputs
  // allows us to eliminate more guards as
  // the use of a guard is now in the same
  // block as opposed to being separated from
  // the definition by block boundaries
  for (size_t offset = 0; offset < block->return_node()->inputs().size();
       offset++) {
    auto i = block->return_node()->input(offset);
    if (i->type()->isSubtypeOf(*TensorType::get()) ||
        isOptionalTensorType(i->type())) {
      insertShapeProfile(block->return_node(), offset, i->type());
    }
  }
}

void ProfilingRecord::removeProfilingNodes(Block* b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::profile || it->kind() == prim::profile_ivalue) {
      it->output()->replaceAllUsesWith(it->input());
      it.destroyCurrent();
    } else {
      for (Block* ib : it->blocks()) {
        removeProfilingNodes(ib);
      }
    }
  }
}

bool ProfilingRecord::ready() const {
  std::lock_guard<std::mutex> lock(this->mutex_);
  return profiling_count_ == 0;
}

std::unique_ptr<ProfilingRecord> ProfilingRecord::instrumentGraph(
    const std::shared_ptr<Graph>& graph) {
  auto new_g = graph->copy();

  auto pr = std::unique_ptr<ProfilingRecord>(new ProfilingRecord(new_g));
  auto raw_pr = pr.get();
  unprofileGraphInputs(new_g);
  unprofileBlock(new_g->block());
  pr->instrumentBlock(new_g->block());

  std::function<void(Stack&)> counter = [raw_pr](Stack& stack) {
    int64_t frame_id = 0;
    pop(stack, frame_id);

    std::lock_guard<std::mutex> lock(raw_pr->mutex_);

    if (raw_pr->profiling_count_ > 0) {
      raw_pr->profiling_count_--;
    }
  };

  auto pop = pr->createProfileNode(counter, {});
  new_g->appendNode(pop);
  GRAPH_DUMP("Instrumented Graph: ", new_g);
  return pr;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `void`

**Classes/Structs**: `ProfileRegistry`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/runtime/profiling_record.h`
- `ATen/core/symbol.h`
- `c10/util/irange.h`
- `torch/csrc/jit/codegen/cuda/interface.h`
- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/clear_profiling.h`
- `torch/csrc/jit/passes/tensorexpr_fuser.h`
- `torch/csrc/jit/runtime/autodiff.h`
- `torch/csrc/jit/runtime/graph_executor.h`
- `torch/csrc/jit/runtime/interpreter.h`


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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `profiling_record.cpp_docs.md`
- **Keyword Index**: `profiling_record.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/runtime`):

- [`register_ops_utils.h_docs.md_docs.md`](./register_ops_utils.h_docs.md_docs.md)
- [`register_c10_ops.cpp_docs.md_docs.md`](./register_c10_ops.cpp_docs.md_docs.md)
- [`exception_message.h_kw.md_docs.md`](./exception_message.h_kw.md_docs.md)
- [`register_prim_ops.cpp_kw.md_docs.md`](./register_prim_ops.cpp_kw.md_docs.md)
- [`autodiff.cpp_kw.md_docs.md`](./autodiff.cpp_kw.md_docs.md)
- [`decomposition_registry_util.h_docs.md_docs.md`](./decomposition_registry_util.h_docs.md_docs.md)
- [`slice_indices_adjust.cpp_docs.md_docs.md`](./slice_indices_adjust.cpp_docs.md_docs.md)
- [`graph_iterator.h_kw.md_docs.md`](./graph_iterator.h_kw.md_docs.md)
- [`shape_function_registry.h_docs.md_docs.md`](./shape_function_registry.h_docs.md_docs.md)
- [`symbolic_script.cpp_docs.md_docs.md`](./symbolic_script.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `profiling_record.cpp_docs.md_docs.md`
- **Keyword Index**: `profiling_record.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
