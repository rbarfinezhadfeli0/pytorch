# Documentation: `test/cpp/jit/test_subgraph_utils.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_subgraph_utils.cpp`
- **Size**: 4,744 bytes (4.63 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include "test/cpp/jit/test_utils.h"

#include <torch/csrc/jit/testing/file_check.h>
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"

namespace torch {
namespace jit {

TEST(SubgraphUtilsTest, Basic) {
  auto graph = build_lstm();
  EliminateCommonSubexpression(graph);

  std::vector<Node*> originalNodes(
      graph->nodes().begin(), graph->nodes().end());

  for (bool reverse_iterate : {true, false}) {
    // Merge everything into a single subgraph
    bool first = true;
    Node* subgraph = nullptr;
    auto it =
        reverse_iterate ? graph->nodes().rbegin() : graph->nodes().begin();
    auto end = reverse_iterate ? graph->nodes().rend() : graph->nodes().end();
    for (; it != end;) {
      if (first) {
        subgraph = SubgraphUtils::createSingletonSubgraph(
            *it, prim::DifferentiableGraph);
        it = reverse_iterate ? ++subgraph->reverseIterator()
                             : ++subgraph->iterator();
        first = false;
      }

      SubgraphUtils::mergeNodeIntoSubgraph(*it, subgraph);
      it = reverse_iterate ? ++subgraph->reverseIterator()
                           : ++subgraph->iterator();
    }

    // Unmerge and compare with original node listing
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    SubgraphUtils::unmergeSubgraph(subgraph);
    EliminateCommonSubexpression(graph);

    std::vector<Node*> newNodes(graph->nodes().begin(), graph->nodes().end());
    ASSERT_EQ(originalNodes.size(), newNodes.size());
  }
}

TEST(SubgraphUtilsTest, MergeSubgraphs) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> parse_map;
  parseIR(
      R"IR(
graph(%a : Tensor, %b : Tensor, %c : Tensor):
  %x : Tensor = aten::sigmoid(%a)
  %y : Tensor = aten::mul(%a, %b)
  %p : Tensor = aten::div(%c, %b)
  %q1 : Tensor = aten::mul(%p, %a)
  %q2 : Tensor = aten::tanh(%q1)
  %q3 : Tensor = aten::tanh(%q2)
  %q4 : Tensor = aten::tanh(%q3)
  %q5 : Tensor = aten::hardsigmoid(%q4)
  return (%x, %y, %q5))IR",
      &*graph,
      parse_map);

  std::vector<Node*> originalNodes(
      graph->nodes().begin(), graph->nodes().end());
  for (bool reverse_merge : {true, false}) {
    // Merge everything into two adjacent subgraphs
    Node* graph1 = SubgraphUtils::createSingletonSubgraph(
        *graph->nodes().begin(), prim::DifferentiableGraph);
    while (true) {
      Node* next = graph1->next();
      if (next->kind() == aten::tanh) {
        break;
      }
      SubgraphUtils::mergeNodeIntoSubgraph(next, graph1);
    }
    Node* graph2 = SubgraphUtils::createSingletonSubgraph(
        graph1->next(), prim::DifferentiableGraph);
    while (graph2->next() != *graph->nodes().end()) {
      SubgraphUtils::mergeNodeIntoSubgraph(graph2->next(), graph2);
    }
    Node* subgraph = nullptr;
    if (reverse_merge) {
      SubgraphUtils::mergeNodeIntoSubgraph(graph2, graph1);
      subgraph = graph1;
    } else {
      SubgraphUtils::mergeNodeIntoSubgraph(graph1, graph2);
      subgraph = graph2;
    }
    auto run_file_check = [](std::shared_ptr<Graph> graph) {
      graph->lint();
      testing::FileCheck()
          .check("aten::sigmoid")
          ->check("aten::mul")
          ->check("aten::div")
          ->check("aten::mul")
          ->check_count("aten::tanh", 3)
          ->check("aten::hardsigmoid")
          ->run(*graph);
    };
    run_file_check(subgraph->g(attr::Subgraph));

    // Unmerge and compare with original node listing
    SubgraphUtils::unmergeSubgraph(subgraph);
    EliminateCommonSubexpression(graph);
    run_file_check(graph);

    std::vector<Node*> newNodes(graph->nodes().begin(), graph->nodes().end());
    ASSERT_EQ(originalNodes.size(), newNodes.size());
  }
}

TEST(SubgraphUtilsTest, GraphName) {
  auto graph = std::make_shared<Graph>();

  std::unordered_map<std::string, Value*> parse_map;
  parseIR(
      R"IR(
graph(%a : Tensor, %b : Tensor, %c : Tensor):
  %x : Tensor = aten::tanh(%a)
  %y : Tensor = aten::mul(%a, %b)
  %p : Tensor = aten::div(%c, %b)
  %q1 : Tensor = aten::mul(%p, %a)
  %q2 : Tensor = aten::tanh(%q1)
  %q3 : Tensor = aten::tanh(%q2)
  %q4 : Tensor = aten::tanh(%q3)
  %q5 : Tensor = aten::tanh(%q4)
  return (%x, %y, %q5))IR",
      &*graph,
      parse_map);
  std::string ref_full_name = "graph_tanh_mul_div_mul_tanh_tanh_tanh_tanh";
  std::string full_name =
      SubgraphUtils::generateNameForGraph(graph, 80, "graph");
  ASSERT_EQ(full_name, ref_full_name);

  std::string truncated_name =
      SubgraphUtils::generateNameForGraph(graph, 10, "graph");

  ASSERT_LE(truncated_name.size(), ref_full_name.size());
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `test/cpp/jit/test_utils.h`
- `torch/csrc/jit/testing/file_check.h`
- `torch/csrc/jit/passes/common_subexpression_elimination.h`
- `torch/csrc/jit/passes/utils/subgraph_utils.h`


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

This is a test file. Run it with:

```bash
python test/cpp/jit/test_subgraph_utils.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/jit`):

- [`test_code_template.cpp_docs.md`](./test_code_template.cpp_docs.md)
- [`test_memory_dag.cpp_docs.md`](./test_memory_dag.cpp_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_cleanup_passes.cpp_docs.md`](./test_cleanup_passes.cpp_docs.md)
- [`test_union.cpp_docs.md`](./test_union.cpp_docs.md)
- [`test_subgraph_rewriter.cpp_docs.md`](./test_subgraph_rewriter.cpp_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md)
- [`test_lite_interpreter_direct.cpp_docs.md`](./test_lite_interpreter_direct.cpp_docs.md)
- [`test_save_load.cpp_docs.md`](./test_save_load.cpp_docs.md)
- [`test_module_api.cpp_docs.md`](./test_module_api.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_subgraph_utils.cpp_docs.md`
- **Keyword Index**: `test_subgraph_utils.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
