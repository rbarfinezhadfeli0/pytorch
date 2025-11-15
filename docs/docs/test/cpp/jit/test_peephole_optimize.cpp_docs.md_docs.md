# Documentation: `docs/test/cpp/jit/test_peephole_optimize.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_peephole_optimize.cpp_docs.md`
- **Size**: 5,573 bytes (5.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_peephole_optimize.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_peephole_optimize.cpp`
- **Size**: 2,859 bytes (2.79 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/peephole.h>

namespace torch {
namespace jit {

TEST(PeepholeOptimizeTest, IsAndIsNot)
// test is / is not none optimization
{
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%0 : int):
  %1 : None = prim::Constant()
  %2 : bool = aten::__is__(%0, %1)
  %3 : bool = aten::__isnot__(%0, %1)
  return (%2, %3)
  )IR",
      graph.get());
  PeepholeOptimize(graph);
  testing::FileCheck()
      .check_not("aten::__is__")
      ->check_not("aten::__isnot__")
      ->run(*graph);
}

TEST(PeepholeOptimizeTest, IsAndIsNot2) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%0: int?):
  %1 : None = prim::Constant()
  %2 : bool = aten::__is__(%0, %1)
  %3 : bool = aten::__isnot__(%0, %1)
  return (%2, %3)
  )IR",
      graph.get());
  PeepholeOptimize(graph);
  testing::FileCheck()
      .check("aten::__is__")
      ->check("aten::__isnot__")
      ->run(*graph);
}

TEST(PeepholeOptimizeTest, IsAndIsNot3) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%0: int?):
  %1 : Tensor = prim::AutogradZero()
  %2 : None = prim::Constant()
  %4 : bool = aten::__is__(%0, %1)
  %5 : bool = aten::__isnot__(%1, %2)
  return (%4, %5)
  )IR",
      graph.get());
  PeepholeOptimize(graph);
  testing::FileCheck()
      .check("aten::__is__")
      ->check_not("aten::__isnot__")
      ->run(*graph);
}

TEST(PeepholeOptimizeTest, UnwrapOptional)
// test unwrap optional
{
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %1 : Float(*, *, *) = prim::Constant()
  %2 : bool = aten::_unwrap_optional(%1)
  %3 : bool = prim::unchecked_unwrap_optional(%1)
  return (%2, %3)
  )IR",
      graph.get());
  PeepholeOptimize(graph);
  testing::FileCheck().check_not("unwrap")->run(*graph);
}

TEST(PeepholeOptimizeTest, UnwrapOptional2) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%1 : Float(*, *, *)?):
  %2 : bool = aten::_unwrap_optional(%1)
  %3 : bool = prim::unchecked_unwrap_optional(%1)
  return (%2, %3)
  )IR",
      graph.get());
  PeepholeOptimize(graph);
  testing::FileCheck().check_count("unwrap", 2)->run(*graph);
}

TEST(PeepholeOptimizeTest, AddMMFusion) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
      graph(
        %0 : Float(2, 3, 4),
        %1 : Float(2, 3, 4),
        %2 : Float(1, 1, 1)):
        %3 : int = prim::Constant[value=1]()
        %4 : Tensor = aten::mm(%0, %1)
        %5 : Tensor = aten::add(%4, %2, %3)
        %6 : Tensor = aten::add(%5, %2, %3)
        return (%6)
        )IR",
      graph.get());
  FuseAddMM(graph);
  testing::FileCheck().check("addmm")->run(*graph);
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
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/irparser.h`
- `torch/csrc/jit/passes/peephole.h`


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
python test/cpp/jit/test_peephole_optimize.cpp
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

- **File Documentation**: `test_peephole_optimize.cpp_docs.md`
- **Keyword Index**: `test_peephole_optimize.cpp_kw.md`
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
python docs/test/cpp/jit/test_peephole_optimize.cpp_docs.md
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

- **File Documentation**: `test_peephole_optimize.cpp_docs.md_docs.md`
- **Keyword Index**: `test_peephole_optimize.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
