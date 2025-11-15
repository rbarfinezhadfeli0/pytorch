# Documentation: `torch/nativert/executor/ExecutionPlanner.cpp`

## File Metadata

- **Path**: `torch/nativert/executor/ExecutionPlanner.cpp`
- **Size**: 3,410 bytes (3.33 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <unordered_map>

#include <c10/util/Logging.h>

#include <c10/util/Enumerate.h>
#include <torch/nativert/executor/ExecutionPlanner.h>

namespace torch::nativert {

std::unique_ptr<ExecutionPlan> ExecutionPlanner::createPlan() {
  auto plan = std::make_unique<ExecutionPlan>();

  // Current implementation assume that nodes will be executed
  // in the same order as the thrift graph.
  // In the future, we can do execution order plan, as long as it's
  // comply with topological order

  generateDeallocationPlan(*plan);
  return plan;
}

/* static */ c10::FastSet<ValueId> ExecutionPlanner::staticValues(
    const Graph& graph) {
  c10::FastSet<ValueId> staticValues;
  // Filter lastUsedBy by graph inputs
  // parameters/buffer values should not be freed
  // It's a policy decision to whether to free user inputs. For now, we don't
  // free user inputs.
  // TODO: It should be fine to "free" the user inputs. If the user holds a ref
  // to it, it won't be deallocated.
  for (const auto* input : graph.inputs()) {
    if (input) {
      const auto& id = input->id();
      staticValues.insert(id);
    }
  }

  // Filter lastUsedBy by graph outputs, as they are still needed to be returned
  for (const auto& output : graph.outputs()) {
    const auto& id = output->id();
    staticValues.insert(id);
  }

  for (const auto& [id, _] : graph.getConstantSymIntValues()) {
    staticValues.insert(id);
  }

  for (const Node& node : graph.nodes()) {
    if (node.target() == "torch.ops.higher_order.run_const_graph") {
      for (const auto& output : node.outputs()) {
        // Do not free the outputs of run_const_graph, as they are newly
        // produced folded constants
        staticValues.insert(output->id());
      }
    } else {
      for (const auto& input : node.inputs()) {
        if (input.value->isFolded()) {
          staticValues.insert(input.value->id());
        }
      }
    }
  }

  return staticValues;
}

void ExecutionPlanner::generateDeallocationPlan(ExecutionPlan& plan) {
  const auto& nodes = graph_.nodes();
  size_t numNodes = nodes.size();

  std::unordered_map<ValueId, NodeIndex> lastUsedBy;

  // Traverse from the last node to the first node
  // For each Value, find out which is the last node that uses it
  // the Value can freed after executing the node
  size_t nodeIdx = nodes.size() - 1;
  for (auto it = std::rbegin(nodes); it != std::rend(nodes); it++) {
    const auto& inputs = it->inputs();
    for (const auto& input : inputs) {
      const auto& id = input.value->id();
      if (lastUsedBy.find(id) == lastUsedBy.end()) {
        lastUsedBy.insert({id, nodeIdx});
      }
    }
    nodeIdx--;
  }

  std::vector<std::vector<ValueId>> valuesToFree(numNodes);

  const auto& statics = staticValues(graph_);
  for (auto& [id, nodeIndex] : lastUsedBy) {
    if (statics.find(id) == statics.end()) {
      valuesToFree[nodeIndex].push_back(id);
    }
  }

  plan.valuesToFree = std::move(valuesToFree);

  // print allocation plan
  VLOG(2) << plan;

  return;
}

std::ostream& operator<<(std::ostream& out, const ExecutionPlan& plan) {
  out << "****** Deallocation Plan ******\n";
  for (auto&& [i, values] : c10::enumerate(plan.valuesToFree)) {
    out << "Node #" << i << ", valuesToFree = [";
    for (const auto& value : values) {
      out << value << ", ";
    }
    out << "]\n";
  }
  return out;
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/executor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `unordered_map`
- `c10/util/Logging.h`
- `c10/util/Enumerate.h`
- `torch/nativert/executor/ExecutionPlanner.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/nativert/executor`):

- [`OpKernel.cpp_docs.md`](./OpKernel.cpp_docs.md)
- [`AOTInductorDelegateExecutor.cpp_docs.md`](./AOTInductorDelegateExecutor.cpp_docs.md)
- [`ExecutionFrame.cpp_docs.md`](./ExecutionFrame.cpp_docs.md)
- [`ParallelGraphExecutor.h_docs.md`](./ParallelGraphExecutor.h_docs.md)
- [`ExecutionFrame.h_docs.md`](./ExecutionFrame.h_docs.md)
- [`ExecutorConfig.h_docs.md`](./ExecutorConfig.h_docs.md)
- [`SerialGraphExecutor.h_docs.md`](./SerialGraphExecutor.h_docs.md)
- [`Weights.cpp_docs.md`](./Weights.cpp_docs.md)
- [`OpKernelKind.h_docs.md`](./OpKernelKind.h_docs.md)
- [`Executor.h_docs.md`](./Executor.h_docs.md)


## Cross-References

- **File Documentation**: `ExecutionPlanner.cpp_docs.md`
- **Keyword Index**: `ExecutionPlanner.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
