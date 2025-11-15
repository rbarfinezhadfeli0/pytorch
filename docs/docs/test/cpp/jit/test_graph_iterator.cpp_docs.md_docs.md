# Documentation: `docs/test/cpp/jit/test_graph_iterator.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_graph_iterator.cpp_docs.md`
- **Size**: 8,750 bytes (8.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_graph_iterator.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_graph_iterator.cpp`
- **Size**: 5,982 bytes (5.84 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/jit.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

/**
 * Inverts an unordered map.
 */
template <typename K, typename V>
std::unordered_map<V, K> invert_map(std::unordered_map<K, V>& map) {
  std::unordered_map<V, K> inverted;
  std::for_each(map.begin(), map.end(), [&inverted](const std::pair<K, V>& p) {
    inverted.insert(std::make_pair(p.second, p.first));
  });
  return inverted;
}

/**
 * Traverses the graph using the DepthFirstGraphNodeIterator and
 * returns an array containing the original names in the string
 * graph.
 */
std::vector<std::string> traverse_depth_first(
    std::string graph_string,
    int max_count = 100) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> vmap;
  torch::jit::parseIR(graph_string, graph.get(), vmap);
  auto get_name = invert_map(vmap);

  std::vector<std::string> result;
  DepthFirstGraphNodeIterator graph_it(graph);
  Node* node = graph_it.next();
  int count = 0;
  while (node && count < max_count) {
    std::stringstream buffer;
    std::vector<const torch::jit::Node*> vec;
    node->print(buffer, 0, &vec, false, true, true, false);
    result.push_back(buffer.str());
    node = graph_it.next();
    ++count;
  }
  return result;
}

/** Checks that the iteration order matches the expected/provided order. */
void assert_ordering(
    std::vector<std::string> actual,
    std::initializer_list<std::string> expected_list) {
  auto expected = std::vector<std::string>(expected_list);
  ASSERT_EQ(expected.size(), actual.size())
      << "Got " << actual.size() << " elements (" << actual << ")"
      << " expected " << expected.size() << " elements (" << expected << ")";
  for (unsigned i = 0; i < expected.size(); i++) {
    ASSERT_EQ(expected[i], actual[i])
        << "Difference at index " << i << " in " << actual << " (expected "
        << actual << ")";
  }
}

TEST(GraphIteratorTest, ConstantReturnGraph) {
  const auto graph_string = R"IR(
      graph():
        %1 : int = prim::Constant[value=0]()
        return (%1))IR";
  auto graph = std::make_shared<Graph>();
  torch::jit::parseIR(graph_string, graph.get());
  DepthFirstGraphNodeIterator graph_it(graph);
  ASSERT_EQ(graph_it.next()->kind(), prim::Constant);
  ASSERT_EQ(graph_it.next(), nullptr);
}

TEST(GraphIteratorTest, GraphWithParameters) {
  const auto graph_string = R"IR(
      graph(%0 : Double(2)):
        %1 : int = prim::Constant[value=0]()
        return (%0))IR";
  auto ordering = traverse_depth_first(graph_string);
  assert_ordering(ordering, {"%1 : int = prim::Constant[value=0]()"});
}

TEST(GraphIteratorTest, GraphWithIf) {
  const auto graph_string = R"IR(
graph(%a : Tensor):
  %a : int = prim::Constant[value=30]()
  %b : int = prim::Constant[value=10]()
  %c : bool = aten::Bool(%a)
  %d : int = prim::If(%c)
    block0():
      -> (%a)
    block1():
      -> (%b)
  %e : int = prim::Constant[value=20]()
  return (%d)
)IR";
  auto ordering = traverse_depth_first(graph_string);
  assert_ordering(
      ordering,
      {"%1 : int = prim::Constant[value=30]()",
       "%2 : int = prim::Constant[value=10]()",
       "%3 : bool = aten::Bool(%1)",
       "%4 : int = prim::If(%3)",
       "%5 : int = prim::Constant[value=20]()"});
}

TEST(GraphIteratorTest, GraphWithNestedIf) {
  const auto graph_string = R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %2 : int = prim::Constant[value=10]()
  %3 : int = prim::Constant[value=20]()
  %4 : int = prim::Constant[value=30]()
  %5 : int = prim::Constant[value=40]()
  %6 : bool = aten::Bool(%a.1)
  %7 : int = prim::If(%6)
    block0():
      %8 : bool = aten::Bool(%b.1)
      %9 : int = prim::If(%8)
        block0():
          -> (%2)
        block1():
          -> (%3)
      -> (%9)
    block1():
      %10 : bool = aten::Bool(%b.1)
      %11 : int = prim::If(%10)
        block0():
          -> (%4)
        block1():
          -> (%5)
      -> (%11)
  %8 : bool = aten::Bool(%b.1)
  %9 : int = prim::If(%8)
    block0():
      -> (%2)
    block1():
      -> (%3)
  %10 : bool = aten::Bool(%b.1)
  %11 : int = prim::If(%10)
    block0():
      -> (%4)
    block1():
      -> (%5)
  return (%7)
)IR";
  auto ordering = traverse_depth_first(graph_string);
  assert_ordering(
      ordering,
      {"%2 : int = prim::Constant[value=10]()",
       "%3 : int = prim::Constant[value=20]()",
       "%4 : int = prim::Constant[value=30]()",
       "%5 : int = prim::Constant[value=40]()",
       "%6 : bool = aten::Bool(%a.1)",
       "%7 : int = prim::If(%6)",
       "%8 : bool = aten::Bool(%b.1)",
       "%9 : int = prim::If(%8)",
       "%10 : bool = aten::Bool(%b.1)",
       "%11 : int = prim::If(%10)",
       "%12 : bool = aten::Bool(%b.1)",
       "%13 : int = prim::If(%12)",
       "%14 : bool = aten::Bool(%b.1)",
       "%15 : int = prim::If(%14)"});
}

TEST(GraphIteratorTest, GraphWithLoop) {
  const auto graph_string = R"IR(
graph(%a.1 : Tensor):
  %1 : bool = prim::Constant[value=1]()
  %2 : int = prim::Constant[value=10]()
  %3 : int = prim::Constant[value=1]()
  %4 : Tensor = prim::Loop(%2, %1, %a.1)
    block0(%i : int, %b.9 : Tensor):
      %5 : Tensor = aten::add_(%b.9, %3, %3)
      -> (%1, %5)
  %6 : Tensor = prim::Loop(%2, %1, %a.1)
    block0(%i : int, %b.9 : Tensor):
      -> (%1, %4)
  return (%6)
)IR";
  auto ordering = traverse_depth_first(graph_string);
  assert_ordering(
      ordering,
      {"%1 : bool = prim::Constant[value=1]()",
       "%2 : int = prim::Constant[value=10]()",
       "%3 : int = prim::Constant[value=1]()",
       "%4 : Tensor = prim::Loop(%2, %1, %a.1)",
       "%7 : Tensor = aten::add_(%b.10, %3, %3)",
       "%8 : Tensor = prim::Loop(%2, %1, %a.1)"});
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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

- `iostream`
- `sstream`
- `string`
- `gtest/gtest.h`
- `test/cpp/jit/test_utils.h`
- `torch/csrc/jit/ir/irparser.h`
- `torch/csrc/jit/runtime/graph_iterator.h`
- `torch/jit.h`
- `torch/script.h`
- `torch/torch.h`


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
python test/cpp/jit/test_graph_iterator.cpp
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

- **File Documentation**: `test_graph_iterator.cpp_docs.md`
- **Keyword Index**: `test_graph_iterator.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/jit`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python docs/test/cpp/jit/test_graph_iterator.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/jit`):

- [`test_graph_iterator.cpp_kw.md_docs.md`](./test_graph_iterator.cpp_kw.md_docs.md)
- [`test_qualified_name.cpp_docs.md_docs.md`](./test_qualified_name.cpp_docs.md_docs.md)
- [`test_fuser.cpp_kw.md_docs.md`](./test_fuser.cpp_kw.md_docs.md)
- [`test_utils.cpp_docs.md_docs.md`](./test_utils.cpp_docs.md_docs.md)
- [`test_custom_class_registrations.h_docs.md_docs.md`](./test_custom_class_registrations.h_docs.md_docs.md)
- [`tests_setup.py_docs.md_docs.md`](./tests_setup.py_docs.md_docs.md)
- [`test_exception.cpp_kw.md_docs.md`](./test_exception.cpp_kw.md_docs.md)
- [`test_cs_debug_info_serialization.cpp_docs.md_docs.md`](./test_cs_debug_info_serialization.cpp_docs.md_docs.md)
- [`torch_python_test.cpp_docs.md_docs.md`](./torch_python_test.cpp_docs.md_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_graph_iterator.cpp_docs.md_docs.md`
- **Keyword Index**: `test_graph_iterator.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
