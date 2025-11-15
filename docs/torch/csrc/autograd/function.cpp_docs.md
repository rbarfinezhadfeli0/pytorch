# Documentation: `torch/csrc/autograd/function.cpp`

## File Metadata

- **Path**: `torch/csrc/autograd/function.cpp`
- **Size**: 3,386 bytes (3.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/autograd/function.h>

#include <c10/util/ThreadLocal.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

namespace torch::autograd {

// The current evaluating node. This is useful to assign the current node as a
// parent of new nodes created during the evaluation of this node in anomaly
// mode.
C10_DEFINE_TLS_static(std::shared_ptr<Node>, tls_current_evaluating_node);
#define current_evaluating_node (tls_current_evaluating_node.get())

NodeGuard::NodeGuard(std::shared_ptr<Node> node)
    : last_evaluating_node_(std::move(current_evaluating_node)) {
  current_evaluating_node = std::move(node);
}
NodeGuard::~NodeGuard() {
  // restore the previous evaluating node
  current_evaluating_node = std::move(last_evaluating_node_);
}

std::shared_ptr<Node> get_current_node() {
  return current_evaluating_node;
}

void Node::assign_parent() {
  metadata()->assign_parent(current_evaluating_node);
}

auto Node::name() const -> std::string {
  return c10::demangle(typeid(*this).name());
}

AnomalyMetadata* Node::metadata() noexcept {
  if (!anomaly_metadata_) {
    anomaly_metadata_ = Engine::get_default_engine().make_anomaly_metadata();
  }
  return anomaly_metadata_.get();
}

static void gatherFunctions(
    Node* func,
    std::vector<std::shared_ptr<Node>>& stack) {
  func->release_variables();

  for (auto& edge : func->next_edges()) {
    if (edge.function.use_count() == 1) {
      stack.emplace_back(std::move(edge.function));
    } else {
      edge.function.reset();
    }
  }
}

/*
 * Fix for #5534: prevent stack overflow on deletion of deep computation graph
 *
 * Sometimes one can end up with a very big computation graph of Nodes
 * and Edges. Each std::shared_ptr<Node> contains a list of Edge, and
 * each Edge contains a std::shared_ptr<Node>. Deleting a
 * std::shared_ptr<Node> can trigger the recursive deletion of other
 * std::shared_ptr<Node>'s: this can stack overflow if the graph
 * is deep enough. Here is an example of such a graph:
 *
 * shared_ptr<Node> -> Edge -> shared_ptr<Node> -> Edge -> ... ->
 * shared_ptr<Node>
 *
 * The solution here is to detect when we are decrementing away the last
 * reference to a Node, and when doing so to buffer up the Node's
 * that will be recursively decremented.  We can then decrement (and free)
 * the original Node without causing a recursive cascade, before
 * draining the buffer applying the same behavior.  This is, in effect,
 * converting recursion to a loop, using a heap buffer in place of the
 * recursive call stack.
 */
void deleteNode(Node* function) {
  // To avoid stack overflow on large computational graphs,
  // we need to track reference decrementing and freeing
  // on the heap.
  function->release_variables();
  std::vector<std::shared_ptr<Node>> stack;
  gatherFunctions(function, stack);
  delete function;

  while (!stack.empty()) {
    auto func = std::move(stack.back());
    stack.pop_back();
    gatherFunctions(func.get(), stack);
    // Reference count is decremented on the loop backedge.
  }
}

at::Tensor TypeAndSize::zeros() {
  return at::zeros_symint(sym_sizes, options);
}

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/autograd/function.h`
- `c10/util/ThreadLocal.h`
- `torch/csrc/autograd/engine.h`
- `torch/csrc/autograd/variable.h`
- `ATen/ATen.h`
- `memory`
- `string`
- `utility`
- `vector`
- `ATen/Functions.h`
- `ATen/ops/zeros.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/csrc/autograd`):

- [`graph_task.h_docs.md`](./graph_task.h_docs.md)
- [`python_function.cpp_docs.md`](./python_function.cpp_docs.md)
- [`profiler.h_docs.md`](./profiler.h_docs.md)
- [`TraceTypeManual.cpp_docs.md`](./TraceTypeManual.cpp_docs.md)
- [`python_autograd.h_docs.md`](./python_autograd.h_docs.md)
- [`variable_info.cpp_docs.md`](./variable_info.cpp_docs.md)
- [`jit_decomp_interface.h_docs.md`](./jit_decomp_interface.h_docs.md)
- [`input_buffer.cpp_docs.md`](./input_buffer.cpp_docs.md)
- [`python_variable.h_docs.md`](./python_variable.h_docs.md)
- [`python_nn_functions.h_docs.md`](./python_nn_functions.h_docs.md)


## Cross-References

- **File Documentation**: `function.cpp_docs.md`
- **Keyword Index**: `function.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
